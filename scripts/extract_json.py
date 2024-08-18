import json
import os
import shutil

image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
json_folder = "./data/json"
json_image = "./data/json_image"  # folder to store json files with image path
os.makedirs(json_image, exist_ok=True)
for json_file in os.listdir(json_folder):
    with open(os.path.join(json_folder, json_file), "r") as f:
        data = json.load(f)
    image_name = data["filename"]
    image_path = os.path.join(image_folder, image_name)

    if os.path.exists(image_path):
        shutil.copy(
            os.path.join(json_folder, json_file), os.path.join(json_image, json_file)
        )
