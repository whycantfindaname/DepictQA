import json
import os

import matplotlib.pyplot as plt
from PIL import Image

json_folder = "./data/clean_bbox_json"
image_folder = "./data/draw_clean_bbox"

bbox_areas = []
image_areas = []

# 遍历 JSON 文件夹中的所有 JSON 文件
for json_file in os.listdir(json_folder):
    if json_file.endswith(".json"):
        with open(os.path.join(json_folder, json_file), "r") as file:
            data = json.load(file)
            image_file = data["filename"]
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path)
            image_width, image_height = image.size
            total_image_area = image_width * image_height

            # 遍历 distortion 字典的键值对
            for distortion_type, boxes in data["distortion"].items():
                for box in boxes:
                    # 计算每个 bbox 的面积
                    tl_x = box["tl"]["x"]
                    tl_y = box["tl"]["y"]
                    br_x = box["br"]["x"]
                    br_y = box["br"]["y"]

                    bbox_width = br_x - tl_x
                    bbox_height = br_y - tl_y
                    bbox_area = bbox_width * bbox_height

                    # 存储每个 bbox 的面积
                    bbox_areas.append(bbox_area)
                    image_areas.append(total_image_area)

# 计算每个 bbox 面积占图像面积的比例
bbox_area_ratios = [
    area / total_area for area, total_area in zip(bbox_areas, image_areas)
]

# 可视化
plt.figure(figsize=(12, 6))
plt.hist(bbox_area_ratios, bins=30, color="skyblue", edgecolor="black")
plt.xlabel("Bounding Box Area Ratio")
plt.ylabel("Frequency")
plt.title("Histogram of Bounding Box Area Ratios")
plt.grid(True)
plt.savefig("./results/clean_bbox_area_ratio_histogram.png")
plt.show()
