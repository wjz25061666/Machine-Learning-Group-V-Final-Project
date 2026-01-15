import os

def write_all_filenames_recursively(folder_path, output_file="output2.txt"):
    with open(output_file, "w", encoding="utf-8") as file:
        for root, dirs, files in os.walk(folder_path):
            for name in files:
                # 获取完整路径（也可以只记录文件名）
                full_path = os.path.join(root, name)
                file.write(full_path + "\n")

    print(f"所有文件路径已写入 {output_file}")

# 示例用法 —— 请修改为你的实际路径QA_RAD
folder_path = r"滤波除噪后数据"  # ←← 修改此处
write_all_filenames_recursively(folder_path)