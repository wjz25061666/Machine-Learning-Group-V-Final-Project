# Machine-Learning-Group-V-Final-Project

## WOA7015 Advanced Machine Learning (Universiti Malaya) - Alternative Assignment (Med-VQA)

This project provides an end-to-end, reproducible pipeline for Medical Visual Question Answering (Med-VQA) using the VQA-RAD dataset.

### Project Overview
The pipeline implements and compares two methods for the Medical VQA task:
1.  **Baseline Fusion Classifier (CNN + GRU)**: A custom model combining computer vision and natural language processing.
2.  **Vision-Language Model (VLM) Fine-tuning**: Fine-tuning the BLIP VQA model from HuggingFace Transformers.

### Features
- Data augmentation for images.
- Learning rate scheduling (Cosine Annealing / OneCycleLR).
- Improved baseline model with attention mechanism and label smoothing.
- Fixes for common BLIP and CuDNN issues.

### Models

#### A. Baseline Fusion Classifier (CNN + GRU)
- **Image Encoder**: ResNet-18 (torchvision) to extract image embeddings.
- **Question Encoder**: Word embedding followed by a BiGRU to extract text embeddings.
- **Fusion**: Concatenation of image and text features followed by an MLP classifier.

#### B. Vision-Language Model (BLIP VQA)
- Uses a pre-trained BLIP VQA model.
- Fine-tuned on the VQA-RAD dataset to generate free-form answers.

### Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- numpy
- pandas
- matplotlib
- Pillow
- tqdm
- scikit-learn
- torch
- torchvision
- transformers>=4.35.0
- accelerate
- sentencepiece

You can install them using:
```bash
pip install -r requirements.txt
```

### Dataset
The project uses the **VQA-RAD** dataset (images + question-answer annotations).
- `VQA_RAD+Dataset+Public.json`: Annotation file.
- `VQA_RAD+Image+Folder`: Directory containing images.

### Usage
Run the full pipeline using the following command:

```bash
python vqa_rad_full_pipeline_fixed8.py \
    --data_root "." \
    --ann_json "VQA_RAD+Dataset+Public.json" \
    --images_dir "VQA_RAD+Image+Folder" \
    --output_dir "./outputs_vqa_rad" \
    --run_baseline --run_blip
```

This command will train both the baseline and BLIP models and save the outputs to the specified directory.
