import os

import cv2
import matplotlib.pyplot as plt
from PIL import Image

fail_path = []


def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Error processing file {image_path} with PIL: {e}")
        try:
            img = cv2.imread(image_path)
            if img is not None:
                return img.shape[1], img.shape[0]  # width, height
            else:
                fail_path.append(image_path)
                raise ValueError("Image is None")
        except Exception as e:
            print(f"Error processing file {image_path} with cv2: {e}")
            return None


def plot_image_sizes(image_folder, save_path):
    widths = []
    heights = []

    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)

        # 确保是文件
        if not os.path.isfile(file_path):
            continue

        # 确保是图片文件
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        image_size = get_image_size(file_path)
        if image_size is not None:
            width, height = image_size
            widths.append(width)
            heights.append(height)
    plt.scatter(heights, widths)
    plt.xlabel("Height (h)")
    plt.ylabel("Width (w)")
    plt.title("Image Size Distribution")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


# 替换为你图片文件夹的路径
# image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/single_images"
image_folder = "./data/draw_bbox"
os.makedirs("./results", exist_ok=True)
save_path = "./results/image_distribution.png"
plot_image_sizes(image_folder, save_path=save_path)
with open("./results/fail_path.txt", "w") as fw:
    fw.write("\n".join(fail_path))
