import json
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager

json_folder = (
    "/home/liaowenjie/桌面/画质大模型/llava-finetune/gen_prompt/data/convert_json"
)
distortion_counts = {}

# 遍历 JSON 文件夹中的所有 JSON 文件
for json_file in os.listdir(json_folder):
    if json_file.endswith(".json"):
        with open(os.path.join(json_folder, json_file), "r") as file:
            data = json.load(file)
            # 遍历 distortion 字典的键值对
            for distortion_type, boxes in data["distortion"].items():
                # 统计每种 distortion 的 bounding box 数量
                if distortion_type not in distortion_counts:
                    distortion_counts[distortion_type] = len(boxes)
                else:
                    distortion_counts[distortion_type] += len(boxes)

# 打印统计信息
print("Distortion Counts:", distortion_counts)

# 设置字体
font_path = "SimHei.ttf"
if os.path.isfile(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
else:
    print("Font file not found, using default font.")
    font_prop = None

# 绘制柱状图
labels = distortion_counts.keys()
sizes = distortion_counts.values()

plt.figure(figsize=(10, 6))
plt.bar(labels, sizes, color="skyblue")
plt.xlabel("Distortion Types", fontproperties=font_prop)
plt.ylabel("Number of Bounding Boxes", fontproperties=font_prop)
plt.title("Distribution of Distortion Types", fontproperties=font_prop)
plt.xticks(
    rotation=45, ha="right", fontproperties=font_prop
)  # Rotate x labels for better readability
plt.tight_layout()  # Adjust layout to fit labels
plt.savefig("./results/distortion_dist.png")
plt.show()
