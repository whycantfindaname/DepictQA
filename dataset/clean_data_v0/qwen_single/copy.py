import json
import os
import shutil

# 源图像数据的根目录
source_root = '/media/liaowenjie/738c79d8-a7b5-43c7-82ce-a07c64929d72/iqadataset'

# 目标目录
target_root = 'dataset/clean_data_v0/spap_images'
if not os.path.exists(target_root):
    os.makedirs(target_root)

# 读取 JSON 文件
with open('dataset/clean_data_v0/qwen_single/train_data.json', 'r') as file:
    data = json.load(file)

# 遍历数据，复制图像到目标目录
for item in data:
    if 'image' in item:
        image_path = item['image']
        
        # 构造源路径和目标路径
        source_path = os.path.join(source_root, image_path)
        target_path = os.path.join(target_root, image_path)
        
        # 确保目标文件夹存在
        target_folder = os.path.dirname(target_path)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # 复制文件
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
            print(f"Copied {source_path} to {target_path}")
        else:
            print(f"Image file {source_path} not found.")

print("Image extraction complete.")
