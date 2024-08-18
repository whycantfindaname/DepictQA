import json
import os

from PIL import Image, ImageDraw, ImageFont


# 绘制 bounding box
def draw_bounding_box(draw, box, label, font):
    draw.rectangle(
        [box["tl"]["x"], box["tl"]["y"], box["br"]["x"], box["br"]["y"]],
        outline="red",
        width=2,
    )
    text_position = (box["tl"]["x"], box["tl"]["y"])
    draw.text(text_position, label, fill="red", font=font)


# 主程序
def process_all_images(json_folder, images_folder, save_folder):
    # 读取 JSON 数据
    for file in os.listdir(json_folder):
        json_file = os.path.join(json_folder, file)
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # print(data)
        image_file = data["filename"]
        image_path = os.path.join(images_folder, image_file)
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # 加载字体
            try:
                font = ImageFont.truetype("SimHei.ttf", 10)
            except IOError:
                font = ImageFont.load_default()

            for distortion_type, boxes in data["distortion"].items():
                if distortion_type == "filename":
                    continue  # 跳过文件名
                for box in boxes:
                    draw_bounding_box(draw, box, distortion_type, font)

            # 显示结果
            save_path = os.path.join(save_folder, os.path.basename(image_path))
            image.save(save_path)


# JSON 文件和图片文件夹路径
json_folder = "./data/clean_bbox_json"  # JSON 文件夹路径
images_folder = (
    "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"  # 图片文件夹路径
)
save_folder = "./data/draw_clean_bbox"  # 保存图片的文件夹路径
os.makedirs(save_folder, exist_ok=True)
process_all_images(json_folder, images_folder, save_folder)
