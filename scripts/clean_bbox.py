import json
import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from shapely.geometry import Polygon


def calculate_area(bbox):
    width = bbox["br"]["x"] - bbox["tl"]["x"]
    height = bbox["br"]["y"] - bbox["tl"]["y"]
    return width * height


def calculate_iou(bbox1, bbox2):
    poly1 = Polygon(
        [
            (bbox1["tl"]["x"], bbox1["tl"]["y"]),
            (bbox1["br"]["x"], bbox1["tl"]["y"]),
            (bbox1["br"]["x"], bbox1["br"]["y"]),
            (bbox1["tl"]["x"], bbox1["br"]["y"]),
        ]
    )

    poly2 = Polygon(
        [
            (bbox2["tl"]["x"], bbox2["tl"]["y"]),
            (bbox2["br"]["x"], bbox2["tl"]["y"]),
            (bbox2["br"]["x"], bbox2["br"]["y"]),
            (bbox2["tl"]["x"], bbox2["br"]["y"]),
        ]
    )

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union


def filter_bboxes(bboxes, image_width, image_height):
    large_bboxes = []
    small_bboxes = []

    for bbox in bboxes:
        area = calculate_area(bbox)
        area_ratio = area / (image_width * image_height)

        if area_ratio > 0.7:
            large_bboxes.append(bbox)
        else:
            small_bboxes.append(bbox)

    # Step 1: Keep the largest bbox if there are multiple large bboxes
    if len(large_bboxes) > 1:
        large_bboxes = [max(large_bboxes, key=lambda b: calculate_area(b))]

    # Step 2: Filter small bboxes based on IOU
    filtered_bboxes = []
    while small_bboxes:
        bbox = small_bboxes.pop(0)
        add_bbox = True
        for filtered_bbox in filtered_bboxes:
            if calculate_iou(bbox, filtered_bbox) > 0.6:
                if calculate_area(bbox) < calculate_area(filtered_bbox):
                    filtered_bboxes.remove(filtered_bbox)
                    filtered_bboxes.append(bbox)
                add_bbox = False
                break
        if add_bbox:
            filtered_bboxes.append(bbox)

    # Combine large and small bboxes
    return large_bboxes + filtered_bboxes


def process_json(data, image_width, image_height):
    for distortion_type, bboxes in data["distortion"].items():
        filtered_bboxes = filter_bboxes(bboxes, image_width, image_height)

        # Retain only tl and br
        for i in range(len(filtered_bboxes)):
            filtered_bboxes[i] = {
                "tl": filtered_bboxes[i]["tl"],
                "br": filtered_bboxes[i]["br"],
            }

        # Update original data structure with filtered bboxes
        data["distortion"][distortion_type] = filtered_bboxes

    return data


def draw_bboxes(image_path, bboxes, output_path):
    # 打开图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 绘制bbox
    for bbox in bboxes:
        tl = (bbox["tl"]["x"], bbox["tl"]["y"])
        br = (bbox["br"]["x"], bbox["br"]["y"])
        draw.rectangle([tl, br], outline="red", width=2)  # 红色框

    # 保存绘制后的图像
    image.save(output_path)

    # 显示图像
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def extract_bboxes(data):
    bboxes = []
    for distortion_type, bbox_list in data["distortion"].items():
        for bbox in bbox_list:
            bboxes.append(bbox)
    return bboxes


# Example usage
image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
json_folder = "./data/convert_json"
save_folder = "./data/clean_bbox_json"
json_file = "'7392498620597292835.json"
os.makedirs(save_folder, exist_ok=True)
with open(
    os.path.join(json_folder, json_file),
    "r",
    encoding="utf-8",
) as file:
    data = json.load(file)
image_path = os.path.join(image_folder, data["filename"])
image = Image.open(image_path)
image_width, image_height = image.size
print(image_height, image_width)
processed_data = process_json(data, image_width, image_height)
print(processed_data)
# for json_file in os.listdir(json_folder):
#     with open(
#         os.path.join(json_folder, json_file),
#         "r",
#         encoding="utf-8",
#     ) as file:
#         data = json.load(file)
#     image_path = os.path.join(image_folder, data["filename"])
#     image = Image.open(image_path)
#     image_width, image_height = image.size

#     processed_data = process_json(data, image_width, image_height)

#     save_path = os.path.join(save_folder, json_file)
#     with open(save_path, "w", encoding="utf-8") as file:
#         json.dump(processed_data, file, ensure_ascii=False, indent=4)
bboxes = extract_bboxes(processed_data)

draw_bboxes(image_path, bboxes, "1.jpg")
