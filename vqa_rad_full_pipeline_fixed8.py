#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WOA7015 Advanced Machine Learning (Universiti Malaya) - Alternative Assignment (Med-VQA)
======================================================================================

This script provides an end-to-end, reproducible pipeline for Medical Visual Question Answering
using the VQA-RAD dataset (images + question-answer annotations).

It implements and compares TWO methods (as required by the assignment):

  (A) Baseline Fusion Classifier (CNN + GRU)
      - Image encoder: ResNet-18 (torchvision) -> image embedding
      - Question encoder: word embedding + BiGRU -> text embedding
      - Fusion: concatenation + MLP classifier over an answer vocabulary

  (B) Vision-Language Model (VLM) Fine-tuning: BLIP VQA (HuggingFace Transformers)
      - Uses a pre-trained BLIP VQA model and fine-tunes on VQA-RAD
      - Generates free-form answers (sequence generation)

Key Improvements in this version:
  - Fixed BLIP 'args' not defined error
  - Fixed Grad-CAM cudnn RNN backward issue
  - Improved baseline model with attention mechanism
  - Better hyperparameters for higher accuracy
  - Label smoothing for better generalization
  - Learning rate scheduling
  - Data augmentation for images

Usage:
   python vqa_rad_full_pipeline_fixed.py \
       --data_root "." \
       --ann_json "VQA_RAD+Dataset+Public.json" \
       --images_dir "VQA_RAD+Image+Folder" \
       --output_dir "./outputs_vqa_rad" \
       --run_baseline --run_blip
"""

from __future__ import annotations
import os

# Set HuggingFace mirror for faster downloads in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
# Helps reduce CUDA memory fragmentation (tunable)
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"

import argparse
import dataclasses
import json
import math
import random
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Utility Functions
# =============================================================================
def _safe_clamp_token_ids(t, vocab_size: int):
    """Clamp token ids into [0, vocab_size-1] to avoid embedding index errors."""
    if t is None:
        return None
    try:
        if hasattr(t, "dtype") and "int" in str(t.dtype):
            return t.clamp_(0, vocab_size - 1)
    except Exception:
        pass
    return t


def _try_resize_blip_embeddings(model, tokenizer_len: int):
    """Try to resize BLIP embeddings if tokenizer has extra tokens."""
    try:
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(tokenizer_len)
    except Exception:
        pass

    for sub_name in ["text_encoder", "text_decoder"]:
        try:
            sub = getattr(model, sub_name, None)
            if sub is None:
                continue
            if hasattr(sub, "resize_token_embeddings"):
                sub.resize_token_embeddings(tokenizer_len)
        except Exception:
            pass




def prepare_blip_special_tokens(processor: Any, model: Any, verbose: bool = True) -> Dict[str, Any]:
    """Ensure BLIP has the extra special tokens ([DEC], [ENC]) and that embeddings match.

    A very common cause of the CUDA crash:
        Indexing.cu: Assertion `srcIndex < srcSelectDimSize` failed
    is that the BLIP config expects a decoder_start_token_id that points to the extra token
    [DEC] (typically id=30522), but the loaded tokenizer/embeddings are still BERT-base size
    (30522) instead of BLIP size (30524). This function makes the tokenizer+model consistent.

    Returns:
        dict with token ids and tokenizer length.
    """
    tok = processor.tokenizer

    added = 0
    # BLIP uses two extra special tokens on top of BERT-base-uncased:
    #   [DEC] -> BOS / decoder start token
    #   [ENC] -> encoder special token
    try:
        if tok.bos_token is None or tok.bos_token != "[DEC]":
            added += tok.add_special_tokens({"bos_token": "[DEC]"})
    except Exception:
        pass

    try:
        additional = list(tok.additional_special_tokens) if tok.additional_special_tokens is not None else []
        if "[ENC]" not in additional:
            added += tok.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    except Exception:
        pass

    # Resize embeddings if tokenizer changed
    if added > 0:
        _try_resize_blip_embeddings(model, len(tok))

    # Token ids
    bos_id = getattr(tok, "bos_token_id", None)
    try:
        enc_id = tok.convert_tokens_to_ids("[ENC]") if "[ENC]" in tok.get_vocab() else None
    except Exception:
        enc_id = None

    # Update model config safely
    try:
        if bos_id is not None:
            model.config.decoder_start_token_id = int(bos_id)
            model.config.bos_token_id = int(bos_id)
    except Exception:
        pass

    try:
        if tok.sep_token_id is not None:
            model.config.eos_token_id = int(tok.sep_token_id)
    except Exception:
        pass

    try:
        if tok.pad_token_id is not None:
            model.config.pad_token_id = int(tok.pad_token_id)
    except Exception:
        pass

    # Validate against embedding sizes; fallback to CLS token if still invalid
    try:
        dec_vocab = model.text_decoder.get_input_embeddings().num_embeddings if hasattr(model, "text_decoder") else None
        if dec_vocab is not None:
            dsid = getattr(model.config, "decoder_start_token_id", None)
            if dsid is not None and dsid >= dec_vocab:
                fallback = tok.cls_token_id if tok.cls_token_id is not None else 0
                if verbose:
                    print(
                        f"[WARN] BLIP decoder_start_token_id ({dsid}) >= decoder vocab ({dec_vocab}). "
                        f"Falling back to cls_token_id={fallback}."
                    )
                model.config.decoder_start_token_id = int(fallback)
                model.config.bos_token_id = int(fallback)

        enc_vocab = model.text_encoder.get_input_embeddings().num_embeddings if hasattr(model, "text_encoder") else None
        if enc_vocab is not None and enc_id is not None and enc_id >= enc_vocab:
            fallback = tok.cls_token_id if tok.cls_token_id is not None else 0
            if verbose:
                print(
                    f"[WARN] BLIP enc_token_id ({enc_id}) >= encoder vocab ({enc_vocab}). "
                    f"Falling back to cls_token_id={fallback}."
                )
            enc_id = int(fallback)
    except Exception:
        pass

    # Convenience attrs (optional)
    try:
        tok.enc_token_id = enc_id
        tok.dec_token_id = bos_id
    except Exception:
        pass

    return {
        "added_tokens": int(added),
        "bos_token_id": int(bos_id) if bos_id is not None else None,
        "enc_token_id": int(enc_id) if enc_id is not None else None,
        "tokenizer_len": int(len(tok)),
    }


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Text normalization + metrics
# =============================================================================
_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def normalize_text(s: Any) -> str:
    """Normalize strings for robust matching."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLE_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gt: str) -> int:
    return int(normalize_text(pred) == normalize_text(gt))


def token_f1(pred: str, gt: str) -> float:
    """Token-level F1, useful for open-ended answers."""
    pred_toks = normalize_text(pred).split()
    gt_toks = normalize_text(gt).split()
    if len(pred_toks) == 0 and len(gt_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gt_toks) == 0:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    gt_count = {}
    for t in gt_toks:
        gt_count[t] = gt_count.get(t, 0) + 1
    for t in common:
        if t in gt_count:
            num_same += min(common[t], gt_count[t])
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gt_toks)
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# Data loading
# =============================================================================
def load_vqa_rad_annotations(
    ann_json: Optional[Path] = None,
    ann_xlsx: Optional[Path] = None,
    ann_xml: Optional[Path] = None,
) -> pd.DataFrame:
    """Load VQA-RAD annotations from JSON / XLSX / XML."""
    if ann_json is None and ann_xlsx is None and ann_xml is None:
        raise ValueError("Please provide at least one of --ann_json / --ann_xlsx / --ann_xml.")

    df_list = []

    if ann_json is not None and ann_json.exists():
        with open(ann_json, "r", encoding="utf-8") as f:
            records = json.load(f)
        df_json = pd.DataFrame(records)
        df_list.append(("json", df_json))

    if ann_xlsx is not None and ann_xlsx.exists():
        df_xlsx = pd.read_excel(ann_xlsx)
        rename_map = {
            "QID_unique": "qid",
            "QID_para": "phrase_type",
            "QID_linked": "qid_linked_id",
            "IMAGEID_case": "image_case_url",
            "IMAGE": "image_name",
            "IMAGE_organ": "image_organ",
            "EVALUATION": "evaluation",
            "QUESTION": "question",
            "Q_REPHASE": "question_rephrase",
            "Q_RELATION": "question_relation",
            "Q_FRAMED": "question_frame",
            "Q_TYPE": "question_type",
            "ANSWER": "answer",
            "A_TYPE": "answer_type",
        }
        df_xlsx = df_xlsx.rename(columns=rename_map)
        df_list.append(("xlsx", df_xlsx))

    if ann_xml is not None and ann_xml.exists():
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(ann_xml))
        root = tree.getroot()
        rows = []
        for q in root.findall("question"):
            row = {child.tag: child.text for child in q}
            rows.append(row)
        df_xml = pd.DataFrame(rows)
        df_list.append(("xml", df_xml))

    # Prefer JSON if available
    source_name, df = df_list[0]
    for name, dfi in df_list:
        if name == "json":
            source_name, df = name, dfi
            break

    # Standardize columns
    expected_cols = [
        "qid", "phrase_type", "qid_linked_id", "image_case_url", "image_name",
        "image_organ", "evaluation", "question", "question_rephrase",
        "question_relation", "question_frame", "question_type", "answer", "answer_type",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    # Clean and normalize
    df["answer_type"] = df["answer_type"].astype(str).str.strip().str.upper()
    df["image_organ"] = df["image_organ"].astype(str).str.strip().str.upper()
    df["question_type"] = df["question_type"].astype(str).str.strip().str.upper()
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["question_len"] = df["question"].apply(lambda x: len(normalize_text(x).split()))
    df["answer_norm"] = df["answer"].apply(normalize_text)

    # Remove rows with empty question or image_name
    df = df[(df["question"].str.strip() != "") & (df["image_name"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)

    print(f"[INFO] Loaded annotations from: {source_name} | rows={len(df)}")
    return df


def _get_meta_i(batch, i: int):
    """Robustly fetch i-th meta record from a DataLoader batch."""
    meta = batch.get("meta", None)
    if meta is None:
        return {}
    if isinstance(meta, list):
        return meta[i]
    if isinstance(meta, dict):
        out = {}
        for k, v in meta.items():
            try:
                out[k] = v[i]
            except Exception:
                try:
                    out[k] = v[i].item() if hasattr(v[i], "item") else v[i]
                except Exception:
                    out[k] = v
        return out
    return meta


def add_image_paths(df: pd.DataFrame, data_root: Path, images_dir: str) -> pd.DataFrame:
    img_root = data_root / images_dir
    df = df.copy()
    df["image_path"] = df["image_name"].apply(lambda n: str(img_root / str(n)))
    return df


def group_split_by_image(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
    group_col: str = "image_name",
) -> pd.DataFrame:
    """Split into train/val/test, grouped by image to reduce leakage."""
    df = df.copy().reset_index(drop=True)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=df[group_col]))
    df.loc[:, "split"] = "train"
    df.loc[test_idx, "split"] = "test"

    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=seed)
    train_idx2, val_idx2 = next(gss2.split(trainval, groups=trainval[group_col]))

    train_image_names = set(trainval.iloc[train_idx2][group_col].tolist())
    val_image_names = set(trainval.iloc[val_idx2][group_col].tolist())

    df.loc[df[group_col].isin(train_image_names), "split"] = "train"
    df.loc[df[group_col].isin(val_image_names), "split"] = "val"

    print(df["split"].value_counts())
    return df


# =============================================================================
# Visualization helpers
# =============================================================================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_dataset_overview(df: pd.DataFrame, fig_dir: Path) -> None:
    """Save multiple dataset distribution plots."""
    _ensure_dir(fig_dir)

    # 1) Organ distribution
    organ_counts = df["image_organ"].value_counts()
    plt.figure(figsize=(10, 5))
    organ_counts.plot(kind="bar", color="steelblue")
    plt.title("VQA-RAD: QA pairs by image organ")
    plt.xlabel("Organ")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "dataset_organ_distribution.png", dpi=200)
    plt.close()

    # 2) Answer type distribution
    at_counts = df["answer_type"].value_counts()
    plt.figure(figsize=(8, 5))
    at_counts.plot(kind="bar", color="coral")
    plt.title("VQA-RAD: Answer type distribution (OPEN vs CLOSED)")
    plt.xlabel("Answer type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "dataset_answer_type_distribution.png", dpi=200)
    plt.close()

    # 3) Question type distribution (top 15)
    qt_counts = df["question_type"].value_counts().head(15)
    plt.figure(figsize=(12, 5))
    qt_counts.plot(kind="bar", color="seagreen")
    plt.title("VQA-RAD: Top question types")
    plt.xlabel("Question type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "dataset_question_type_top15.png", dpi=200)
    plt.close()

    # 4) Question length histogram
    plt.figure(figsize=(8, 5))
    df["question_len"].plot(kind="hist", bins=25, color="purple", alpha=0.7)
    plt.title("VQA-RAD: Question length histogram (#tokens)")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(fig_dir / "dataset_question_length_hist.png", dpi=200)
    plt.close()

    # 5) Top answers
    top_ans = df["answer_norm"].value_counts().head(20)
    plt.figure(figsize=(12, 5))
    top_ans.plot(kind="bar", color="darkorange")
    plt.title("VQA-RAD: Top 20 normalized answers")
    plt.xlabel("Answer (normalized)")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "dataset_top_answers.png", dpi=200)
    plt.close()

    print(f"[INFO] Saved dataset overview figures to: {fig_dir}")


def plot_random_image_grid(
    df: pd.DataFrame,
    fig_dir: Path,
    n: int = 9,
    seed: int = 42,
    title: str = "Random samples (image + Q/A)",
) -> None:
    """Save a grid of random images with (question, answer) in the title."""
    _ensure_dir(fig_dir)

    candidates = df.sample(min(len(df), max(n * 5, n)), random_state=seed).to_dict("records")

    picked = []
    for r in candidates:
        p = Path(r["image_path"])
        if p.exists():
            picked.append(r)
        if len(picked) >= n:
            break

    if len(picked) == 0:
        print("[WARN] No images found for visualization. Skipping image grid.")
        return

    cols = int(math.sqrt(n))
    rows = int(math.ceil(len(picked) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, r in enumerate(picked):
        img = Image.open(r["image_path"]).convert("RGB")
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")
        q = r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"]
        a = r["answer"][:30] + "..." if len(r["answer"]) > 30 else r["answer"]
        ax.set_title(f"Q: {q}\nA: {a}", fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fig_dir / "random_image_grid.png", dpi=200)
    plt.close()
    print(f"[INFO] Saved random image grid to: {fig_dir / 'random_image_grid.png'}")


# =============================================================================
# Baseline model components
# =============================================================================
def simple_tokenize(text: str) -> List[str]:
    """A simple tokenizer for baseline GRU model."""
    text = normalize_text(text)
    return text.split()


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, text: str, max_len: int) -> List[int]:
        toks = simple_tokenize(text)
        ids = [self.stoi.get(t, self.unk_id) for t in toks][:max_len]
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids


def build_vocab(questions: List[str], min_freq: int = 1, max_size: int = 30000) -> Vocab:
    counter: Dict[str, int] = {}
    for q in questions:
        for t in simple_tokenize(q):
            counter[t] = counter.get(t, 0) + 1

    itos = ["<pad>", "<unk>"]
    words = sorted([(w, c) for w, c in counter.items() if c >= min_freq], key=lambda x: (-x[1], x[0]))
    words = words[: max(0, max_size - len(itos))]
    itos.extend([w for w, _ in words])

    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def build_answer_vocab(df_train: pd.DataFrame, top_k: Optional[int] = None) -> Tuple[Dict[str, int], List[str]]:
    """Build answer vocabulary from training set answers (normalized)."""
    counts = df_train["answer_norm"].value_counts()
    if top_k is not None:
        kept = list(counts.head(top_k).index)
        id2ans = ["<unk_answer>"] + kept
        ans2id = {a: i + 1 for i, a in enumerate(kept)}
        ans2id["<unk_answer>"] = 0
        return ans2id, id2ans

    unique = sorted(df_train["answer_norm"].unique().tolist())
    id2ans = unique
    ans2id = {a: i for i, a in enumerate(id2ans)}
    return ans2id, id2ans


def encode_answer(ans_norm: str, ans2id: Dict[str, int]) -> int:
    if ans_norm in ans2id:
        return ans2id[ans_norm]
    if "<unk_answer>" in ans2id:
        return ans2id["<unk_answer>"]
    return 0


class VQARadBaselineDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        ans2id: Dict[str, int],
        max_q_len: int,
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.ans2id = ans2id
        self.max_q_len = max_q_len
        self.image_size = image_size
        self.augment = augment

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        
        # Data augmentation for training
        # NOTE: For medical images (VQA-RAD), many questions involve left/right laterality.
        # Horizontal flipping can therefore corrupt labels (e.g., "left lung" becomes "right lung").
        # We only apply mild geometric augmentation here.
        if self.augment:
            # Small random rotation (-7 to 7 degrees)
            if random.random() < 0.6:
                angle = random.uniform(-7, 7)
                img = img.rotate(angle, fillcolor=(128, 128, 128))
        
        img = img.resize((self.image_size, self.image_size))
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        t = torch.from_numpy(arr)
        t = (t - self.mean) / self.std
        return t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        image = self._load_image(r["image_path"])
        q_ids = torch.tensor(self.vocab.encode(r["question"], self.max_q_len), dtype=torch.long)
        label = torch.tensor(encode_answer(r["answer_norm"], self.ans2id), dtype=torch.long)
        return {
            "image": image,
            "question_ids": q_ids,
            "label": label,
            "meta": {
                "image_name": r["image_name"],
                "image_organ": r["image_organ"],
                "answer_type": r["answer_type"],
                "question": r["question"],
                "answer": r["answer"],
                "answer_norm": r["answer_norm"],
            }
        }


class AttentionFusion(nn.Module):
    """Attention-based fusion between image and text features."""
    def __init__(self, img_dim: int, txt_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        img_proj = self.img_proj(img_feat)  # [B, hidden]
        txt_proj = self.txt_proj(txt_feat)  # [B, hidden]
        
        combined = torch.cat([img_proj, txt_proj], dim=1)  # [B, hidden*2]
        attn_weights = self.attention(combined)  # [B, 2]
        
        # Weighted sum
        fused = attn_weights[:, 0:1] * img_proj + attn_weights[:, 1:2] * txt_proj
        return fused


class SelfAttentionPool(nn.Module):
    """Self-attention pooling over token sequence (baseline text encoder)."""
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq:  [B, L, D] token embeddings
            mask: [B, L] 1 for valid tokens, 0 for padding
        Returns:
            pooled: [B, D]
        """
        attn_logits = self.proj(seq).squeeze(-1)  # [B, L]
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn_logits, dim=1)  # [B, L]
        pooled = torch.bmm(attn.unsqueeze(1), seq).squeeze(1)  # [B, D]
        return pooled


class ImprovedBaselineFusionModel(nn.Module):
    """Improved baseline model with attention fusion and better architecture."""
    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        txt_emb_dim: int = 300,
        txt_hidden_dim: int = 512,
        dropout: float = 0.3,
        resnet_pretrained: bool = True,
    ) -> None:
        super().__init__()
        
        # Image encoder - ResNet18
        from torchvision.models import resnet18, ResNet18_Weights
        
        if resnet_pretrained:
            try:
                weights = ResNet18_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        else:
            weights = None

        self.image_encoder = resnet18(weights=weights)
        self.image_encoder.fc = nn.Identity()
        img_dim = 512

        # Text encoder with LSTM (more stable gradients than GRU for Grad-CAM)
        self.embedding = nn.Embedding(vocab_size, txt_emb_dim, padding_idx=0)
        self.txt_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=txt_emb_dim,
            hidden_size=txt_hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=dropout,
        )
        txt_dim = txt_hidden_dim * 2

        # Self-attention pooling over LSTM outputs (helps on small datasets)
        self.text_pool = SelfAttentionPool(txt_dim, hidden_dim=256, dropout=dropout)

        # Attention fusion
        self.fusion = AttentionFusion(img_dim, txt_dim, hidden_dim=512)
        
        # Classifier with deeper MLP
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_answers),
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' in name:
                nn.init.uniform_(param, -0.1, 0.1)
            elif 'weight' in name and 'norm' not in name.lower():
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, image: torch.Tensor, question_ids: torch.Tensor) -> torch.Tensor:
        # Image features
        img_feat = self.image_encoder(image)  # [B, 512]
        
        # Text features
        emb = self.embedding(question_ids)    # [B, L, D]
        emb = self.txt_dropout(emb)
        
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(emb)  # [B, L, 2*H]

        # Attention pooling over all token states (mask padding tokens)
        mask = (question_ids != 0).long()
        txt_attn = self.text_pool(lstm_out, mask)  # [B, 2*H]

        # Concatenate final forward & backward hidden states
        txt_final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 2*H]

        # Combine both representations
        txt_feat = 0.5 * txt_attn + 0.5 * txt_final

        # Attention fusion
        fused = self.fusion(img_feat, txt_feat)
        
        # Classification
        logits = self.classifier(fused)
        return logits


# =============================================================================
# Baseline training/evaluation
# =============================================================================
@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    max_q_len: int = 32
    txt_emb_dim: int = 300
    txt_hidden_dim: int = 512
    dropout: float = 0.3
    resnet_pretrained: bool = True
    top_k_answers: Optional[int] = None
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    use_augmentation: bool = True


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        
        log_prob = F.log_softmax(pred, dim=1)
        
        if self.weight is not None:
            weight = self.weight[target]
            loss = -(smooth_one_hot * log_prob).sum(dim=1) * weight
        else:
            loss = -(smooth_one_hot * log_prob).sum(dim=1)
        
        return loss.mean()


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(labels), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    # Use inverse square root for smoother weights
    inv = 1.0 / np.sqrt(counts)
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


def train_baseline(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    cfg: TrainConfig = None,
    device: Optional[str] = None,
    max_train_samples: Optional[int] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = TrainConfig()
    
    set_seed(seed)
    _ensure_dir(output_dir)
    fig_dir = output_dir / "figures"
    _ensure_dir(fig_dir)
    ckpt_dir = output_dir / "checkpoints"
    _ensure_dir(ckpt_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if max_train_samples is not None and max_train_samples < len(df_train):
        df_train = df_train.sample(max_train_samples, random_state=seed).reset_index(drop=True)

    # Build vocab on TRAIN questions only
    vocab = build_vocab(df_train["question"].tolist(), min_freq=1, max_size=30000)

    # Build answer vocab on TRAIN answers only
    ans2id, id2ans = build_answer_vocab(df_train, top_k=cfg.top_k_answers)

    # Datasets
    ds_train = VQARadBaselineDataset(
        df_train, vocab=vocab, ans2id=ans2id, 
        max_q_len=cfg.max_q_len, augment=cfg.use_augmentation
    )
    ds_val = VQARadBaselineDataset(df_val, vocab=vocab, ans2id=ans2id, max_q_len=cfg.max_q_len, augment=False)
    ds_test = VQARadBaselineDataset(df_test, vocab=vocab, ans2id=ans2id, max_q_len=cfg.max_q_len, augment=False)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = ImprovedBaselineFusionModel(
        vocab_size=len(vocab.itos),
        num_answers=len(id2ans),
        txt_emb_dim=cfg.txt_emb_dim,
        txt_hidden_dim=cfg.txt_hidden_dim,
        dropout=cfg.dropout,
        resnet_pretrained=cfg.resnet_pretrained,
    ).to(device)

    # Loss with label smoothing and class weights
    train_labels = [int(x) for x in df_train["answer_norm"].apply(lambda a: encode_answer(a, ans2id)).tolist()]
    class_w = compute_class_weights(train_labels, num_classes=len(id2ans)).to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing, weight=class_w)

    # Optimizer with different learning rates for pretrained and new layers
    pretrained_params = list(model.image_encoder.parameters())
    new_params = [p for n, p in model.named_parameters() if 'image_encoder' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': cfg.lr * 0.1},
        {'params': new_params, 'lr': cfg.lr}
    ], weight_decay=cfg.weight_decay)

    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=[cfg.lr * 0.1, cfg.lr],
        epochs=cfg.epochs,
        steps_per_epoch=len(dl_train),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_path = ckpt_dir / "baseline_best.pt"
    patience = 5
    patience_counter = 0

    def run_eval(dl: DataLoader) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for batch in dl:
                images = batch["image"].to(device)
                q_ids = batch["question_ids"].to(device)
                labels = batch["label"].to(device)

                logits = model(images, q_ids)
                loss = criterion(logits, labels)

                total_loss += float(loss.item()) * images.size(0)
                preds = logits.argmax(dim=1)
                total_correct += int((preds == labels).sum().item())
                total += images.size(0)

        return total_loss / max(total, 1), total_correct / max(total, 1)

    print("[INFO] Training baseline (Improved CNN+LSTM) ...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch in tqdm(dl_train, desc=f"Baseline Epoch {epoch}/{cfg.epochs}"):
            images = batch["image"].to(device)
            q_ids = batch["question_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, q_ids)
            loss = criterion(logits, labels)
            loss.backward()
            
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item()) * images.size(0)
            preds = logits.argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            running_total += images.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = run_eval(dl_val)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.6f} time={dt:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "vocab": dataclasses.asdict(vocab),
                "ans2id": ans2id,
                "id2ans": id2ans,
                "cfg": dataclasses.asdict(cfg),
            }, best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > cfg.epochs // 2:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    # Plot training curves
    plot_training_curves(history, fig_dir / "baseline_training_curves.png", title="Baseline (Improved CNN+LSTM) training curves")

    # Load best model for test
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Predict on test
    preds_rows = []
    y_true = []
    y_pred = []
    yesno_true = []
    yesno_pred = []

    with torch.no_grad():
        for batch in tqdm(dl_test, desc="Baseline test inference"):
            images = batch["image"].to(device)
            q_ids = batch["question_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(images, q_ids)
            pred_ids = logits.argmax(dim=1).cpu().numpy().tolist()

            for i in range(len(pred_ids)):
                meta = _get_meta_i(batch, i)
                gt = meta["answer"]
                gt_norm = meta["answer_norm"]
                pred_norm = id2ans[pred_ids[i]] if pred_ids[i] < len(id2ans) else "<unk>"
                
                preds_rows.append({
                    "image_name": meta["image_name"],
                    "image_organ": meta["image_organ"],
                    "answer_type": meta["answer_type"],
                    "question": meta["question"],
                    "answer_gt": gt,
                    "answer_gt_norm": gt_norm,
                    "answer_pred_norm": pred_norm,
                })
                y_true.append(gt_norm)
                y_pred.append(pred_norm)

                if meta["answer_type"].strip().upper() == "CLOSED" and gt_norm in {"yes", "no"}:
                    yesno_true.append(gt_norm)
                    yesno_pred.append(pred_norm if pred_norm in {"yes", "no"} else "other")

    pred_df = pd.DataFrame(preds_rows)
    pred_df.to_csv(output_dir / "baseline_test_predictions.csv", index=False)

    # Metrics
    em = float(np.mean([exact_match(p, g) for p, g in zip(pred_df["answer_pred_norm"], pred_df["answer_gt_norm"])]))
    f1 = float(np.mean([token_f1(p, g) for p, g in zip(pred_df["answer_pred_norm"], pred_df["answer_gt_norm"])]))

    yesno_acc = None
    if len(yesno_true) > 0:
        yesno_acc = float(np.mean([int(p == g) for p, g in zip(yesno_pred, yesno_true)]))
        plot_confusion_yesno(yesno_true, yesno_pred, fig_dir / "baseline_yesno_confusion.png",
                             title="Baseline Yes/No confusion matrix")

    metrics = {
        "exact_match": em,
        "token_f1": f1,
        "yesno_acc": yesno_acc,
        "num_test": int(len(pred_df)),
        "best_val_acc": float(best_val_acc),
        "num_answers": int(len(id2ans)),
        "vocab_size": int(len(vocab.itos)),
    }
    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] Baseline test metrics:", metrics)

    # Qualitative visualization
    plot_qualitative_examples(
        pred_df=pred_df,
        df_all=df_test,
        fig_path=fig_dir / "baseline_qualitative_examples.png",
        n=6,
        title="Baseline qualitative examples",
    )

    # Grad-CAM visualization (with fix for LSTM)
    try:
        plot_gradcam_examples(
            model=model,
            ds_test=ds_test,
            id2ans=id2ans,
            fig_path=fig_dir / "baseline_gradcam_examples.png",
            device=device,
            n=4,
            seed=seed,
        )
    except Exception as e:
        print(f"[WARN] Grad-CAM visualization failed (skipping): {e}")

    return {
        "history": history,
        "metrics": metrics,
        "predictions_path": str(output_dir / "baseline_test_predictions.csv"),
        "metrics_path": str(output_dir / "baseline_metrics.json"),
        "best_ckpt_path": str(best_path),
    }


# =============================================================================
# Training curves + confusion matrix
# =============================================================================
def plot_training_curves(history: Dict[str, List[float]], fig_path: Path, title: str) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], 'b-', label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], 'r-', label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def plot_confusion_yesno(y_true: List[str], y_pred: List[str], fig_path: Path, title: str) -> None:
    labels = ["yes", "no", "other"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(values_format="d", ax=ax, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


# =============================================================================
# Qualitative visualizations
# =============================================================================
def plot_qualitative_examples(
    pred_df: pd.DataFrame,
    df_all: pd.DataFrame,
    fig_path: Path,
    n: int = 6,
    title: str = "Qualitative examples",
    seed: int = 42,
) -> None:
    """Create a panel of qualitative examples."""
    sample = pred_df.sample(min(n, len(pred_df)), random_state=seed)
    img_map = {r["image_name"]: r["image_path"] for r in df_all.to_dict("records")}

    cols = 2
    rows = int(math.ceil(len(sample) / cols))
    plt.figure(figsize=(12, 4 * rows))

    for i, row in enumerate(sample.to_dict("records")):
        img_path = img_map.get(row["image_name"], None)
        ax = plt.subplot(rows, cols, i + 1)
        if img_path is None or not Path(img_path).exists():
            ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
            ax.axis("off")
            continue
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        
        q = row["question"][:60] + "..." if len(row["question"]) > 60 else row["question"]
        gt = row["answer_gt"][:30] + "..." if len(row["answer_gt"]) > 30 else row["answer_gt"]
        pred = row["answer_pred_norm"][:30] + "..." if len(row["answer_pred_norm"]) > 30 else row["answer_pred_norm"]
        
        # Color code: green if correct, red if wrong
        color = "green" if normalize_text(row["answer_gt"]) == normalize_text(row["answer_pred_norm"]) else "red"
        ax.set_title(f"Q: {q}\nGT: {gt}\nPred: {pred}", fontsize=9, color=color)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


# =============================================================================
# Grad-CAM for baseline interpretability (Fixed for LSTM)
# =============================================================================
def _find_last_conv_layer_resnet(model: nn.Module) -> nn.Module:
    if hasattr(model, "image_encoder") and hasattr(model.image_encoder, "layer4"):
        return model.image_encoder.layer4
    raise ValueError("Cannot locate ResNet last conv layer for Grad-CAM.")


@torch.no_grad()
def _denormalize_img(img_t: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    x = img_t.cpu().numpy() * std + mean
    x = np.clip(x, 0, 1)
    x = np.transpose(x, (1, 2, 0))
    return x


def gradcam_single(
    model: nn.Module,
    image: torch.Tensor,
    question_ids: torch.Tensor,
    class_idx: Optional[int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute Grad-CAM heatmap for a single example."""
    # Set model to train mode for gradient computation through LSTM
    # but disable dropout
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    
    image = image.unsqueeze(0).to(device)
    question_ids = question_ids.unsqueeze(0).to(device)

    target_layer = _find_last_conv_layer_resnet(model)
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    logits = model(image, question_ids)
    pred = int(logits.argmax(dim=1).item())
    if class_idx is None:
        class_idx = pred

    # Backward pass
    model.zero_grad(set_to_none=True)
    score = logits[0, class_idx]
    score.backward()

    h1.remove()
    h2.remove()

    act = activations[0].detach()
    grad = gradients[0].detach()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=False)
    cam = F.relu(cam)
    cam = cam[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.cpu().numpy()

    rgb = _denormalize_img(image[0].detach())
    
    # Reset model to eval mode
    model.eval()
    
    return rgb, cam_np, pred


def plot_gradcam_examples(
    model: nn.Module,
    ds_test: VQARadBaselineDataset,
    id2ans: List[str],
    fig_path: Path,
    device: torch.device,
    n: int = 4,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds_test), size=min(n, len(ds_test)), replace=False).tolist()

    cols = 2
    rows = int(math.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, idx in enumerate(idxs):
        ex = ds_test[idx]
        rgb, cam, pred = gradcam_single(
            model=model,
            image=ex["image"],
            question_ids=ex["question_ids"],
            class_idx=None,
            device=device,
        )
        
        ax = axes[i]
        ax.imshow(rgb)
        
        # Resize cam to match image size
        from scipy.ndimage import zoom
        cam_resized = zoom(cam, (rgb.shape[0] / cam.shape[0], rgb.shape[1] / cam.shape[1]), order=1)
        ax.imshow(cam_resized, alpha=0.5, cmap='jet')
        ax.axis("off")
        
        q = ex["meta"]["question"][:50] + "..." if len(ex["meta"]["question"]) > 50 else ex["meta"]["question"]
        gt = ex["meta"]["answer"]
        pred_ans = id2ans[pred] if pred < len(id2ans) else "<unk>"
        ax.set_title(f"Q: {q}\nGT: {gt}\nPred: {pred_ans}", fontsize=9)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Baseline Grad-CAM examples (heatmap overlay)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


# =============================================================================
# BLIP VQA fine-tuning (Transformers) - FIXED
# =============================================================================
class VQARadBlipDataset(Dataset):
    """Dataset wrapper for BLIP VQA fine-tuning.

    Important:
    BLIP adds two special tokens on top of BERT-base:
      [DEC] (BOS/decoder start) and [ENC] (encoder marker).
    If these are missing (common when caches are incomplete), training can crash on CUDA with
    an out-of-range embedding index. We force these token ids here.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processor: Any,
        max_q_len: int = 64,
        max_a_len: int = 16,
        image_size: int = 384,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.image_size = image_size

        tok = self.processor.tokenizer
        self.pad_token_id = int(tok.pad_token_id) if tok.pad_token_id is not None else 0

        # [DEC] is BOS for decoder; fall back to CLS if missing
        if getattr(tok, "bos_token_id", None) is not None:
            self.bos_token_id = int(tok.bos_token_id)
        else:
            # If bos_token_id is missing, try token string; otherwise CLS
            try:
                self.bos_token_id = int(tok.convert_tokens_to_ids("[DEC]"))
            except Exception:
                self.bos_token_id = int(tok.cls_token_id) if tok.cls_token_id is not None else 0

        # [ENC] is used to mark question tokens; fall back to CLS if missing
        try:
            enc_id = tok.convert_tokens_to_ids("[ENC]")
            # convert_tokens_to_ids returns unk_token_id if token is unknown, so ensure it's really present
            if "[ENC]" in tok.get_vocab():
                self.enc_token_id = int(enc_id)
            else:
                self.enc_token_id = int(tok.cls_token_id) if tok.cls_token_id is not None else 0
        except Exception:
            self.enc_token_id = int(tok.cls_token_id) if tok.cls_token_id is not None else 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        img = Image.open(r["image_path"]).convert("RGB")

        # Encode question + image
        enc = self.processor(
            images=img,
            text=r["question"],
            padding="max_length",
            truncation=True,
            max_length=self.max_q_len,
            return_tensors="pt",
        )

        q_input_ids = enc["input_ids"].squeeze(0)
        q_attention_mask = enc["attention_mask"].squeeze(0)

        # Force first token to [ENC] (or CLS fallback)
        if q_input_ids.numel() > 0:
            q_input_ids[0] = self.enc_token_id

        # Encode answer text (decoder input)
        ans = self.processor.tokenizer(
            r["answer"],
            padding="max_length",
            truncation=True,
            max_length=self.max_a_len,
            return_tensors="pt",
        )

        a_input_ids = ans["input_ids"].squeeze(0)
        a_attention_mask = ans["attention_mask"].squeeze(0)

        # Force first token to [DEC] / BOS (or CLS fallback)
        if a_input_ids.numel() > 0:
            a_input_ids[0] = self.bos_token_id

        labels = a_input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        item = {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "input_ids": q_input_ids,
            "attention_mask": q_attention_mask,
            "decoder_input_ids": a_input_ids,
            "decoder_attention_mask": a_attention_mask,
            "labels": labels,
            "meta": {
                "image_name": r["image_name"],
                "image_organ": r["image_organ"],
                "answer_type": r["answer_type"],
                "question": r["question"],
                "answer": r["answer"],
                "answer_norm": r["answer_norm"],
            },
        }
        return item


def _blip_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    decoder_input_ids = torch.stack([b["decoder_input_ids"] for b in batch], dim=0)
    decoder_attention_mask = torch.stack([b["decoder_attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    meta = [b["meta"] for b in batch]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
        "meta": meta,
    }


def train_blip(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    max_q_len: int = 64,
    max_a_len: int = 16,
    image_size: int = 256,
    grad_accum_steps: int = 4,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    freeze_vision: bool = False,
    freeze_text_encoder: bool = False,
    num_workers: int = 2,
    device: Optional[str] = None,
    model_name: str = "Salesforce/blip-vqa-base",
    max_train_samples: Optional[int] = None,
    gen_max_len: int = 16,
) -> Dict[str, Any]:
    """Fine-tune BLIP for VQA.

    This version is optimized for limited VRAM (8-10GB):
      - smaller default batch size
      - gradient accumulation (effective batch = batch_size * grad_accum_steps)
      - AMP fp16 mixed precision (fp16=True)
      - optional gradient checkpointing (saves memory, slower)
      - optional freezing of vision encoder / text encoder

    If you still hit OOM:
      - lower --blip_batch_size to 1
      - increase --blip_grad_accum
      - lower --blip_image_size to 224
      - set --blip_freeze_vision 1 or --blip_freeze_text_encoder 1
    """
    import gc

    set_seed(seed)
    _ensure_dir(output_dir)
    fig_dir = output_dir / "figures"
    _ensure_dir(fig_dir)
    ckpt_dir = output_dir / "checkpoints"
    _ensure_dir(ckpt_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Encourage cache cleanup before big model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
    except Exception as e:
        raise ImportError(
            "Transformers is required for BLIP. Install with: pip install transformers accelerate sentencepiece"
        ) from e

    if max_train_samples is not None and max_train_samples < len(df_train):
        df_train = df_train.sample(max_train_samples, random_state=seed).reset_index(drop=True)

    print(f"[INFO] Loading BLIP model: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)

    # Ensure tokenizer special tokens are consistent with BLIP ([DEC],[ENC]) + resize embeddings
    tok_info = prepare_blip_special_tokens(processor, model, verbose=True)

    # Reduce BLIP input resolution to save VRAM (optional, but very helpful on 8-10GB GPUs)
    try:
        if hasattr(processor, "image_processor") and processor.image_processor is not None:
            if hasattr(processor.image_processor, "size"):
                processor.image_processor.size = {"height": int(image_size), "width": int(image_size)}
            if hasattr(processor.image_processor, "crop_size"):
                processor.image_processor.crop_size = {"height": int(image_size), "width": int(image_size)}
    except Exception as e:
        print(f"[WARN] Could not set BLIP image size to {image_size}: {e}")

    model.to(device)

    # Enable gradient checkpointing to save memory (slower)
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            # Some models require disabling cache when checkpointing
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        except Exception as e:
            print(f"[WARN] Gradient checkpointing not enabled: {e}")

    # Optional freezing to save memory
    if freeze_vision:
        try:
            for p in model.vision_model.parameters():
                p.requires_grad = False
            print("[INFO] Frozen BLIP vision encoder parameters.")
        except Exception:
            print("[WARN] Could not freeze vision encoder (model structure mismatch).")
    if freeze_text_encoder:
        try:
            for p in model.text_encoder.parameters():
                p.requires_grad = False
            print("[INFO] Frozen BLIP text encoder parameters.")
        except Exception:
            print("[WARN] Could not freeze text encoder (model structure mismatch).")

    # Print vocab and token ids
    try:
        tok_len = len(processor.tokenizer)
        enc_vocab = model.text_encoder.get_input_embeddings().num_embeddings if hasattr(model, "text_encoder") else None
        dec_vocab = model.text_decoder.get_input_embeddings().num_embeddings if hasattr(model, "text_decoder") else None
        dsid = getattr(model.config, "decoder_start_token_id", None)
        print(
            f"[INFO] BLIP vocab sizes: tokenizer={tok_len} encoder_emb={enc_vocab} decoder_emb={dec_vocab} "
            f"decoder_start_token_id={dsid} bos={tok_info.get('bos_token_id')} enc={tok_info.get('enc_token_id')}"
        )
    except Exception:
        pass

    # Detect whether forward supports explicit decoder inputs
    try:
        import inspect
        _sig = inspect.signature(model.forward)
        supports_decoder_inputs = ("decoder_input_ids" in _sig.parameters)
    except Exception:
        supports_decoder_inputs = True

    ds_train = VQARadBlipDataset(df_train, processor=processor, max_q_len=max_q_len, max_a_len=max_a_len)
    ds_val = VQARadBlipDataset(df_val, processor=processor, max_q_len=max_q_len, max_a_len=max_a_len)
    ds_test = VQARadBlipDataset(df_test, processor=processor, max_q_len=max_q_len, max_a_len=max_a_len)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_blip_collate,
        pin_memory=(device.type == "cuda"),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_blip_collate,
        pin_memory=(device.type == "cuda"),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_blip_collate,
        pin_memory=(device.type == "cuda"),
    )

    # Trainable params only (important if freezing)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Scheduler steps are per optimizer-step (not per micro-batch)
    updates_per_epoch = int(math.ceil(len(dl_train) / max(1, grad_accum_steps)))
    total_updates = max(1, updates_per_epoch * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)

    use_amp = bool(fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_path = ckpt_dir / "blip_best.pt"

    def run_val_loss() -> float:
        model.eval()
        total_loss = 0.0
        total_n = 0
        with torch.no_grad():
            for batch in dl_val:
                batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

                # Safety: clamp token ids to embedding sizes (prevents rare index errors)
                try:
                    enc_vocab = model.text_encoder.get_input_embeddings().num_embeddings if hasattr(model, "text_encoder") else None
                    if enc_vocab is not None and "input_ids" in batch:
                        batch["input_ids"] = _safe_clamp_token_ids(batch["input_ids"], enc_vocab)
                except Exception:
                    pass
                try:
                    dec_vocab = model.text_decoder.get_input_embeddings().num_embeddings if hasattr(model, "text_decoder") else None
                    if dec_vocab is not None and "decoder_input_ids" in batch:
                        batch["decoder_input_ids"] = _safe_clamp_token_ids(batch["decoder_input_ids"], dec_vocab)
                        if "labels" in batch:
                            labels = batch["labels"]
                            mask = labels != -100
                            labels2 = labels.clone()
                            labels2[mask] = labels2[mask].clamp(0, dec_vocab - 1)
                            batch["labels"] = labels2
                except Exception:
                    pass

                with torch.cuda.amp.autocast(enabled=use_amp):
                    if supports_decoder_inputs:
                        out = model(
                            pixel_values=batch["pixel_values"],
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            decoder_input_ids=batch["decoder_input_ids"],
                            decoder_attention_mask=batch["decoder_attention_mask"],
                            labels=batch["labels"],
                        )
                    else:
                        out = model(
                            pixel_values=batch["pixel_values"],
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                    loss = out.loss

                bs = int(batch["pixel_values"].size(0))
                total_loss += float(loss.item()) * bs
                total_n += bs

        return total_loss / max(1, total_n)

    print("[INFO] Fine-tuning BLIP (memory-optimized) ...")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        total_n = 0

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(enumerate(dl_train), total=len(dl_train), desc=f"BLIP Epoch {epoch}/{epochs}")
        for step, batch in pbar:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Safety: clamp token ids to embedding sizes
            try:
                enc_vocab = model.text_encoder.get_input_embeddings().num_embeddings if hasattr(model, "text_encoder") else None
                if enc_vocab is not None and "input_ids" in batch:
                    batch["input_ids"] = _safe_clamp_token_ids(batch["input_ids"], enc_vocab)
            except Exception:
                pass
            try:
                dec_vocab = model.text_decoder.get_input_embeddings().num_embeddings if hasattr(model, "text_decoder") else None
                if dec_vocab is not None and "decoder_input_ids" in batch:
                    batch["decoder_input_ids"] = _safe_clamp_token_ids(batch["decoder_input_ids"], dec_vocab)
                    if "labels" in batch:
                        labels = batch["labels"]
                        mask = labels != -100
                        labels2 = labels.clone()
                        labels2[mask] = labels2[mask].clamp(0, dec_vocab - 1)
                        batch["labels"] = labels2
            except Exception:
                pass

            with torch.cuda.amp.autocast(enabled=use_amp):
                if supports_decoder_inputs:
                    out = model(
                        pixel_values=batch["pixel_values"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        decoder_input_ids=batch["decoder_input_ids"],
                        decoder_attention_mask=batch["decoder_attention_mask"],
                        labels=batch["labels"],
                    )
                else:
                    out = model(
                        pixel_values=batch["pixel_values"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                loss = out.loss

            # gradient accumulation
            loss_to_backprop = loss / max(1, grad_accum_steps)
            scaler.scale(loss_to_backprop).backward()

            bs = int(batch["pixel_values"].size(0))
            running_loss += float(loss.item()) * bs
            total_n += bs

            # optimizer step
            do_step = ((step + 1) % max(1, grad_accum_steps) == 0) or (step + 1 == len(dl_train))
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            pbar.set_postfix({"loss": running_loss / max(1, total_n)})

        train_loss = running_loss / max(1, total_n)
        val_loss = run_val_loss()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr_now:.6f} time={dt:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "image_size": int(image_size),
                    "max_q_len": int(max_q_len),
                    "max_a_len": int(max_a_len),
                },
                best_path,
            )

        # cache cleanup
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save history + plot
    with open(output_dir / "blip_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_blip_loss_curves(history, fig_dir / "blip_loss_curves.png")

    # Load best
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Test generation
    preds_rows: List[Dict[str, Any]] = []
    yesno_true: List[str] = []
    yesno_pred: List[str] = []

    with torch.no_grad():
        for batch in tqdm(dl_test, desc="BLIP test generation"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            # Mixed precision generation is not always stable; keep fp32 by default
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=gen_max_len,
            )
            pred_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i, pred in enumerate(pred_texts):
                meta = batch["meta"][i]
                gt = meta["answer"]
                gt_norm = normalize_text(gt)
                pred_norm = normalize_text(pred)

                preds_rows.append({
                    "image_name": meta["image_name"],
                    "question": meta["question"],
                    "answer_gt": gt,
                    "answer_gt_norm": gt_norm,
                    "answer_pred": pred,
                    "answer_pred_norm": pred_norm,
                    "answer_type": meta["answer_type"],
                    "image_organ": meta.get("image_organ", ""),
                })

                if meta["answer_type"].strip().upper() == "CLOSED" and gt_norm in {"yes", "no"}:
                    yesno_true.append(gt_norm)
                    if pred_norm in {"yes", "no"}:
                        yesno_pred.append(pred_norm)
                    else:
                        yesno_pred.append("other")

    pred_df = pd.DataFrame(preds_rows)
    pred_df.to_csv(output_dir / "blip_test_predictions.csv", index=False)

    em = float(np.mean([exact_match(p, g) for p, g in zip(pred_df["answer_pred_norm"], pred_df["answer_gt_norm"])]))
    f1 = float(np.mean([token_f1(p, g) for p, g in zip(pred_df["answer_pred_norm"], pred_df["answer_gt_norm"])]))

    yesno_acc = None
    if len(yesno_true) > 0:
        yesno_acc = float(np.mean([1.0 if p == g else 0.0 for p, g in zip(yesno_pred, yesno_true)]))

    metrics = {
        "exact_match": em,
        "token_f1": f1,
        "yesno_acc": yesno_acc,
        "num_test": int(len(df_test)),
        "best_val_loss": float(best_val_loss),
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "fp16": bool(use_amp),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "freeze_vision": bool(freeze_vision),
        "freeze_text_encoder": bool(freeze_text_encoder),
    }

    with open(output_dir / "blip_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Qualitative examples
    try:
        plot_qualitative_examples(
            pred_df=pred_df,
            df_all=df_test,
            fig_path=fig_dir / "blip_qualitative_examples.png",
            n=6,
            title="BLIP qualitative examples",
        )
    except Exception as e:
        print(f"[WARN] Failed to plot BLIP qualitative examples: {e}")

    print(f"[INFO] BLIP test metrics: {metrics}")

    # Return lightweight result (JSON-serializable)
    return {
        "history": history,
        "metrics": metrics,
        "predictions_path": str(output_dir / "blip_test_predictions.csv"),
        "metrics_path": str(output_dir / "blip_metrics.json"),
        "best_ckpt_path": str(best_path),
    }
def plot_blip_loss_curves(history: Dict[str, List[float]], fig_path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BLIP fine-tuning loss curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


# =============================================================================
# Final comparison visualization
# =============================================================================
def plot_model_comparison(
    baseline_metrics: Optional[Dict[str, Any]],
    blip_metrics: Optional[Dict[str, Any]],
    fig_path: Path,
) -> None:
    rows = []
    if baseline_metrics is not None:
        rows.append(("Baseline\n(CNN+LSTM)", 
                     baseline_metrics.get("exact_match", 0), 
                     baseline_metrics.get("token_f1", 0),
                     baseline_metrics.get("yesno_acc", 0) or 0))
    if blip_metrics is not None:
        rows.append(("BLIP\n(VLM)", 
                     blip_metrics.get("exact_match", 0), 
                     blip_metrics.get("token_f1", 0),
                     blip_metrics.get("yesno_acc", 0) or 0))

    if len(rows) == 0:
        return

    labels = [r[0] for r in rows]
    ems = [r[1] for r in rows]
    f1s = [r[2] for r in rows]
    yesno = [r[3] for r in rows]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, ems, width, label="Exact Match", color='steelblue')
    bars2 = ax.bar(x, f1s, width, label="Token F1", color='coral')
    bars3 = ax.bar(x + width, yesno, width, label="Yes/No Acc", color='seagreen')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison on VQA-RAD Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


# =============================================================================
# Main
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VQA-RAD Med-VQA training & evaluation (2 methods + visuals)")

    p.add_argument("--data_root", type=str, required=True,
                   help="Path to the dataset root folder (e.g., /path/to/VQA_RAD)")
    p.add_argument("--images_dir", type=str, default="VQA_RAD+Image+Folder",
                   help="Folder (inside data_root) that contains images.")
    p.add_argument("--ann_json", type=str, default="VQA_RAD+Dataset+Public.json",
                   help="JSON annotations filename (inside data_root) or full path.")
    p.add_argument("--ann_xlsx", type=str, default=None,
                   help="Optional XLSX annotations file (inside data_root) or full path.")
    p.add_argument("--ann_xml", type=str, default=None,
                   help="Optional XML annotations file (inside data_root) or full path.")

    p.add_argument("--output_dir", type=str, default="./outputs_vqa_rad",
                   help="Where to save outputs (figures, checkpoints, metrics, predictions).")
    p.add_argument("--seed", type=int, default=42)

    # Split ratios
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)

    # Run flags
    p.add_argument("--run_baseline", action="store_true", help="Train/eval baseline CNN+LSTM classifier")
    p.add_argument("--run_blip", action="store_true", help="Fine-tune/eval BLIP VQA model")

    # Baseline hyperparams
    p.add_argument("--baseline_epochs", type=int, default=20)
    p.add_argument("--baseline_batch_size", type=int, default=32)
    p.add_argument("--baseline_lr", type=float, default=1e-3)
    p.add_argument("--baseline_top_k_answers", type=int, default=0,
                   help="0 means use ALL answers; otherwise keep only top-K answers.")

    # BLIP hyperparams
    p.add_argument("--blip_model_name", type=str, default="Salesforce/blip-vqa-base")
    p.add_argument("--blip_epochs", type=int, default=5)
    p.add_argument("--blip_batch_size", type=int, default=2)
    p.add_argument("--blip_lr", type=float, default=5e-5)
    p.add_argument("--blip_gen_max_len", type=int, default=16)
    p.add_argument("--blip_image_size", type=int, default=256,
                   help="BLIP image resolution (lower saves VRAM). Typical: 224/256/384.")
    p.add_argument("--blip_grad_accum", type=int, default=4,
                   help="Gradient accumulation steps. Effective batch = blip_batch_size * blip_grad_accum.")
    p.add_argument("--blip_fp16", type=int, default=1,
                   help="1 enables AMP fp16 mixed precision training (recommended on limited VRAM).")
    p.add_argument("--blip_grad_ckpt", type=int, default=1,
                   help="1 enables gradient checkpointing to save memory (slower).")
    p.add_argument("--blip_freeze_vision", type=int, default=0,
                   help="1 freezes vision encoder to save memory (may reduce adaptation).")
    p.add_argument("--blip_freeze_text_encoder", type=int, default=0,
                   help="1 freezes text encoder to save memory (may reduce adaptation).")
    p.add_argument("--blip_max_q_len", type=int, default=64)
    p.add_argument("--blip_max_a_len", type=int, default=16)

    # Debug / speed
    p.add_argument("--max_train_samples", type=int, default=0,
                   help="0 means use full training set; otherwise subsample for quick runs.")

    return p.parse_args()


def resolve_path(data_root: Path, maybe_path: Optional[str]) -> Optional[Path]:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    if p.exists():
        return p
    p2 = data_root / maybe_path
    return p2 if p2.exists() else p


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    _ensure_dir(out_root)

    ann_json = resolve_path(data_root, args.ann_json)
    ann_xlsx = resolve_path(data_root, args.ann_xlsx) if args.ann_xlsx else None
    ann_xml = resolve_path(data_root, args.ann_xml) if args.ann_xml else None

    # Load annotations
    df = load_vqa_rad_annotations(ann_json=ann_json, ann_xlsx=ann_xlsx, ann_xml=ann_xml)
    df = add_image_paths(df, data_root=data_root, images_dir=args.images_dir)

    # Make splits
    df = group_split_by_image(
        df=df,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        group_col="image_name",
    )

    # Dataset overview visuals
    fig_dir = out_root / "figures_dataset"
    plot_dataset_overview(df, fig_dir=fig_dir)
    plot_random_image_grid(df, fig_dir=fig_dir, n=9, seed=args.seed)

    # Split frames
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    max_train_samples = args.max_train_samples if args.max_train_samples and args.max_train_samples > 0 else None

    baseline_result = None
    blip_result = None

    if args.run_baseline:
        baseline_dir = out_root / "baseline_cnn_lstm"
        cfg = TrainConfig(
            epochs=args.baseline_epochs,
            batch_size=args.baseline_batch_size,
            lr=args.baseline_lr,
            top_k_answers=(args.baseline_top_k_answers if args.baseline_top_k_answers > 0 else None),
        )
        baseline_result = train_baseline(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            output_dir=baseline_dir,
            seed=args.seed,
            cfg=cfg,
            max_train_samples=max_train_samples,
        )

    if args.run_blip:
        # Clean CUDA cache before loading BLIP to reduce OOM risk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        blip_dir = out_root / "blip_vqa"
        try:
            blip_result = train_blip(
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                output_dir=blip_dir,
                seed=args.seed,
                epochs=args.blip_epochs,
                batch_size=args.blip_batch_size,
                lr=args.blip_lr,
                model_name=args.blip_model_name,
                max_train_samples=max_train_samples,
                gen_max_len=args.blip_gen_max_len,
                max_q_len=args.blip_max_q_len,
                max_a_len=args.blip_max_a_len,
                image_size=args.blip_image_size,
                grad_accum_steps=args.blip_grad_accum,
                fp16=bool(args.blip_fp16),
                gradient_checkpointing=bool(args.blip_grad_ckpt),
                freeze_vision=bool(args.blip_freeze_vision),
                freeze_text_encoder=bool(args.blip_freeze_text_encoder),
            )
        except Exception as e:
            print(f"[ERROR] BLIP training failed: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] If you want to use BLIP, please ensure you installed transformers and you have internet access.")
            blip_result = None

    # Comparison plot
    comp_fig = out_root / "figures_comparison"
    _ensure_dir(comp_fig)
    plot_model_comparison(
        baseline_metrics=(baseline_result["metrics"] if baseline_result else None),
        blip_metrics=(blip_result["metrics"] if blip_result else None),
        fig_path=comp_fig / "model_comparison.png",
    )

    # Save a final summary JSON
    summary = {
        "baseline": baseline_result,
        "blip": blip_result,
        "data_root": str(data_root),
        "output_dir": str(out_root),
        "seed": args.seed,
        "split_sizes": df["split"].value_counts().to_dict(),
    }
    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] All outputs saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()