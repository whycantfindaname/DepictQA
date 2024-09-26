import json
import logging
import os
import random
import shutil
import re
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

categories_list = [
    "Meaningless solid color",  # 纯色无意义
    "Aliasing",  # 锯齿
    "Low clarity",  # 清晰度低
    "Excessive darkness",  # 过暗
    "Compression artifacts",  # 压缩失真块效应
    "Out of focus blur",  # 对焦模糊
    "Overexposure",  # 过曝
    "Noise",  # 噪点
    "Motion blur",  # 运动模糊
    "Underexposure",  # 欠曝
    "Interlaced scanning",  # 隔行扫描
    "Ringing effect",  # 振铃效应
    "Moiré pattern",  # 摩尔纹
    "Banding effect",  # 条带
]

category_translation = {
    "纯色无意义": "Meaningless solid color",
    "锯齿": "Aliasing",
    "清晰度低": "Low clarity",
    "过暗": "Excessive darkness",
    "压缩失真块效应": "Compression artifacts",
    "对焦模糊": "Out of focus blur",
    "过曝": "Overexposure",
    "噪点": "Noise",
    "运动模糊": "Motion blur",
    "欠曝": "Underexposure",
    "隔行扫描": "Interlaced scanning",
    "振铃效应": "Ringing effect",
    "摩尔纹": "Moiré pattern",
    "条带": "Banding effect",
}


categories = [
    {"id": i + 1, "name": name, "supercategory": "distortion"}
    for i, name in enumerate(categories_list)
]


coco_output = {"images": [], "annotations": [], "categories": categories}

annotation_id = image_id = 1


def assign_level(mos_score):
    if mos_score >= 4.2:
        return "excellent"
    elif mos_score >= 3.4:
        return "good"
    elif mos_score >= 2.6:
        return "fair"
    elif mos_score >= 1.8:
        return "poor"
    else:
        return "bad"


def convert_to_coco_format(json_file, output_file, image_folder):
    """
    Converts the dataset to COCO format.

    Parameters:
    - json_file: str - The JSON file containing image metadata and annotations.
    - output_file: str - The path where the COCO formatted JSON will be saved.
    - image_folder: str - The folder where images are stored.
    """
    global image_id, annotation_id

    content = load_json(json_file)

    for data in content:
        image_path = os.path.join(image_folder, data["filename"])

        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            logging.warning(
                f"Image {data['filename']} not found in {image_folder}. Skipping..."
            )
            continue

        image = Image.open(image_path)
        image_width, image_height = image.size

        image_info = {
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": data["filename"],
        }
        coco_output["images"].append(image_info)
        try:
            for category_name, bboxes in data["distortion"].items():
                category_id = categories_list.index(category_name) + 1
                for bbox in bboxes:
                    x_min, y_min = bbox["tl"]["x"], bbox["tl"]["y"]
                    x_max, y_max = bbox["br"]["x"], bbox["br"]["y"]
                    width, height = x_max - x_min, y_max - y_min

                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": [],
                        "area": width * height,
                        "bbox": [x_min, y_min, width, height],
                        "iscrowd": 0,
                    }
                    coco_output["annotations"].append(annotation_info)
                    annotation_id += 1
            image_id += 1
        except AttributeError:
            pass

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)

    logging.info(f"COCO format data saved to {output_file}")


def replace_categories(data):
    """
    Replace the Chinese categories with their English equivalents.

    Parameters:
    - data: dict - The data containing distortion categories in Chinese.

    Returns:
    - data: dict - The data with distortion categories replaced by their English equivalents.
    """
    if "distortion" in data:
        for distortion_type in list(data["distortion"].keys()):
            if distortion_type in category_translation:
                english_name = category_translation[distortion_type]
                data["distortion"][english_name] = data["distortion"].pop(
                    distortion_type
                )
    return data


def merge_json_simple(file1, file2, output_file):
    data1 = load_json(file1)
    data2 = load_json(file2)

    # Ensure data1 and data2 are lists
    if isinstance(data1, dict):
        data1 = list(data1.values())
    if isinstance(data2, dict):
        data2 = list(data2.values())

    # Merge the records from both files into a single list
    merged_data = data1 + data2

    # Write the filtered data to the output file
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(merged_data, f_out, ensure_ascii=False, indent=4)


def merge_json_files(file1, file2, output_file):
    """
    This will merge two JSON files into one and remove any records that do not have both 'filename' and 'mos' fields.
    The merged result will be a list of dictionaries, each representing a record from the original JSON files.

    Parameters:
    - file1: str - The path to the first JSON file.
    - file2: str - The path to the second JSON file.
    - output_file: str - The path where the merged JSON file will be saved.

    The output JSON file will only include records that have both 'filename' and 'mos' fields.
    """
    data1 = load_json(file1)
    data2 = load_json(file2)

    # Ensure data1 and data2 are lists
    if isinstance(data1, dict):
        data1 = list(data1.values())
    if isinstance(data2, dict):
        data2 = list(data2.values())

    # Merge the records from both files into a single list
    merged_data = data1 + data2

    # Filter out records that do not have both 'filename' and 'mos' fields
    filtered_data = [
        record
        for record in merged_data
        if isinstance(record, dict) and "filename" in record and "mos" in record
    ]

    # Write the filtered data to the output file
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(filtered_data, f_out, ensure_ascii=False, indent=4)


def merge_meta_jsons_from_folder(meta_folder, output_file, image_folder):
    """
    This will merge all JSON files in the specified folder into one JSON file by sequentially merging each file's content.
    Before merging, Chinese categories will be replaced with corresponding English ones. Images not in the image_folder will
    be excluded.

    Parameters:
    - meta_folder: str - The path to the folder containing JSON files.
    - output_file: str - The path where the merged JSON file will be saved.
    - image_folder: str - The path which contains all images
    The output JSON file will be a list of dictionaries, each representing a valid record from the JSON files in the folder.
    """
    json_files = [
        os.path.join(meta_folder, f)
        for f in os.listdir(meta_folder)
        if f.endswith(".json")
    ]
    save_dir = os.path.dirname(output_file)
    os.makedirs(save_dir, exist_ok=True)
    if not json_files:
        logging.info("No JSON files found in the specified folder.")
        return

    merged_data = []

    for json_file in json_files:
        print(json_file)
        with open(json_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        for key, data in content.items():
            # Ensure each item is processed only if it contains 'filename' and 'mos'
            if "filename" in data and "mos" in data:
                # Extract 'filename' and 'mos'
                filename = data["filename"]
                if filename.lower().endswith(".bmp"):
                    # Construct the full file path
                    bmp_path = os.path.join(image_folder, filename)
                    # Open the .bmp image
                    with Image.open(bmp_path) as img:
                        # Change the filename extension from .bmp to .png
                        png_filename = filename.rsplit(".", 1)[0] + ".png"
                        png_path = os.path.join(image_folder, png_filename)
                        # Save the image in .png format
                        img.save(png_path, "PNG")
                    print(f"Converted: {filename} to {png_filename}")
                    image_path = png_path
                    filename = png_filename
                else:
                    image_path = os.path.join(image_folder, filename)

                if os.path.exists(image_path):
                    mos = data["mos"]

                    # Extract other key-value pairs into a new dictionary under 'distortion'
                    distortion = {
                        k: v for k, v in data.items() if k not in ["filename", "mos"]
                    }
                    # If distortion is not empty, add to merged data
                    if distortion:
                        new_data = {
                            "filename": filename,
                            "mos": mos,
                            "distortion": distortion,
                        }
                        # Replace Chinese categories with English ones
                        new_data = replace_categories(new_data)
                        merged_data.append(new_data)
                    elif mos >= 4:
                        new_data = {
                            "filename": filename,
                            "mos": mos,
                            "distortion": "There is no distortion in image.",
                        }
                        print(filename)
                        merged_data.append(new_data)

    # Write the merged data to the output file
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(merged_data, f_out, ensure_ascii=False, indent=4)

    logging.info(f"Total number: {len(merged_data)}")
    logging.info(f"All JSON files have been processed and merged into {output_file}")


def merge_jsons_from_folder(json_folder, output_file):
    """Merge JSON files from a folder."""
    merged_data = []
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(json_folder, filename)
            data = load_json(file_path)
            # logging.info(data)
            if isinstance(data, dict):
                merged_data.append(data)
            elif isinstance(data, list):
                merged_data.extend(data)

    # Write the merged data to the output file
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(merged_data, f_out, ensure_ascii=False, indent=4)


def load_json(file_path):
    """Load a JSON file and return its contents."""
    with open(file_path, "r") as file:
        return json.load(file)


def get_images_from_json(json_data):
    """Extract image names from JSON data."""
    return {item["filename"] for item in json_data}


def get_images_from_folder(folder_path):
    """Get a set of image file names from a directory."""
    return {
        file
        for file in os.listdir(folder_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    }


def modify_iqadata(file_path, output_path, keep_mos=False):
    # Load the JSON data
    with open(file_path, "r") as file:
        data = json.load(file)

    # Iterate through each item and apply the modifications
    for item in data:
        # Replace "id" with "mos" and keep only the numeric part
        id_value = item.pop("id")
        mos_value = float(id_value.split("->")[1])

        if keep_mos:
            item["mos"] = mos_value

        # Modify the "image" field to remove the "kadid10k/" part
        item["image"] = item["image"]

        # Update the "value" field in the human conversation

        for conversation in item["conversations"]:
            if conversation["from"] == "human":
                # Modify the conversation value
                conversation["value"] = (
                    "<image>"
                    + conversation["value"].replace("\n", "").replace("<|image|>", "")
                    + "Answer in a single sentence."
                )
        # if mos_value >= 4.5:
        #     dist_question = {
        #         "from": "human",
        #         "value": "What types of distortions are present in the image?",
        #     }
        #     dist_answer = {
        #         "from": "gpt",
        #         "value": "There is no distortion in the image.",
        #     }
        #     item["conversations"].append(dist_question)
        #     item["conversations"].append(dist_answer)
    dir = os.path.dirname(output_path)
    os.makedirs(dir, exist_ok=True)
    # Save the modified data back to a JSON file
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)


follow_up_questions = [
    "Following your evaluation of the image quality, can you outline the distortions detected and provide their bounding box coordinates?",
    "Based on your earlier assessment, could you detail the distortions and return the corresponding bounding boxes?",
    "As a follow-up to your quality review, please list the distortions you identified and their specific bounding box locations.",
    "After reviewing the image, could you now specify the types of distortions and provide their bounding box coordinates?",
    "Could you now elaborate on the distortions you've identified in the image, and include the bounding boxes for each?",
    "Please provide the specific types of distortions in the image along with their bounding box coordinates.",
    "Please specify the types of distortions present and return their bounding box coordinates.",
    "Can you detail the distortions found in the image along with their bounding box locations?",
    "What types of distortions were identified in the image, and could you provide their bounding box coordinates?",
    "Can you give the bounding box coordinates for the identified distortions in the image?",
]

score_questions = [
    "Can you evaluate the quality of the image in a single sentence?",
    "Please describe the overall quality of the image in a single sentence",
    "Give a one-sentence evaluation of the image quality.",
    "How would you summarize the quality of this image in a single sentence?",
    "Can you rate the image quality in just one sentence?",
]


def construct_chat_template(
    desp_file,
    assess_file,
    meta_file,
    image_folder,
    output_file,
    model_name="qwen-vl-chat",
    keep_mos=True,
    bbox_provided=True,
):
    """
        This function processes a dataset by merging descriptions, assessments, and optionally bounding box information
        into a single JSON structure, and copies the corresponding images to a target folder.

        Parameters:
        - desp_file: str - The path to the gpt description JSON file.
        - assess_file: str - The path to the gpt assessment JSON file.
        - meta_file: str - The path to original JSON file. If provided, bounding box data and mos score will be included.
        - image_folder: str - The directory where the original images are stored.
        - output_file: str - The path where the final merged JSON file will be saved.
        - model_name: str  - The model for finetuning. Default is "qwen-vl-chat".
        - bbox_provided: boolean - Indicate if bounding box information should be included
    ):
    """

    fail_image = []
    desp_data = load_json(desp_file)
    assess_data = load_json(assess_file)
    meta_data = load_json(meta_file)
    json_data = []
    image_data = 0
    for meta_item in meta_data:
        img_name = meta_item["filename"]
        desp = next((desp for desp in desp_data if desp["filename"] == img_name), None)
        if desp is None:
            logging.warning(f"{img_name} does not have description, skip")
            continue
        description = desp["gpt4v_description"]

        assess = next(
            (assess for assess in assess_data if assess["filename"] == img_name), None
        )
        if assess is None:
            logging.warning(f"{img_name} does not have assessment, skip.")
            continue
        assessment_text = assess["gpt4v_assessment"]

        # 使用正则表达式查找分隔符
        match = re.search(r'\n*Answer:\s*', assessment_text)

        if match:
            assessment_question = assessment_text[:match.start()].strip()
            assessment_answer = assessment_text[match.end():].strip().replace("\n\n", "")
        else:
            try:
                assessment_question, assessment_answer = assessment_text.split('\n\n', 1)
            except Exception:
                fail_image.append(img_name)
                continue

        assessment_answer = re.sub(r'[\n\-]', ' ', assessment_answer)  
        assessment_answer = re.sub(r'\s+', ' ', assessment_answer).strip()   

        score = meta_item.get("mos", "")

        distortion = meta_item["distortion"]
        if isinstance(distortion, dict):
            distortion_types = distortion.keys()
            distortion_list = ", ".join(distortion_types) if distortion_types else "None"
            bbox = convert_to_qwen_format(distortion)
        else:
            bbox = None
            distortion_list = None
        level = assign_level(score)
        if keep_mos:
            new_data = {
                "image": "QualityLLM_single_2w/" + desp["filename"],
                "system_prompt": "You are an expert in image quality assessment. Your task is to describe an image and evaluate the quality of the image based on your description.",
                "conversations": [],
                "mos": score,
                "level": level,
            }
        else:
            new_data = {
                "image": "QualityLLM_single_2w/" + desp["filename"],
                "system_prompt": "You are an expert in image quality assessment. Your task is to describe an image and evaluate the quality of the image based on your description.",
                "conversations": [],
                "level": level,
            }
        # desp
        description = re.sub(r'[\n\-]', ' ', description)  
        description = re.sub(r'\s+', ' ', description).strip()  
        desp_question = {
            "from": "human",
            "value": "<image>Please provide a brief description of the image, including specific objects and any events.",
        }
        desp_answer = {
            "from": "gpt",
            "value": description                
            .replace("**", "")
            .replace("#", "")
            .lstrip()
        }
 

        new_data["conversations"].append(desp_question)
        new_data["conversations"].append(desp_answer)

        # distortion qa
        dist_question = {
            "from": "human",
            "value": "What types of distortions are present in the image? Answer with names only.",
        }
        if distortion_list is not None:
            dist_answer = {"from": "gpt", "value": distortion_list}
        else:
            dist_answer = {"from": "gpt", "value": distortion}
            print(distortion)
        new_data["conversations"].append(dist_question)
        new_data["conversations"].append(dist_answer)

        # assess
        assess_question = {
            "from": "human",
            "value": assessment_question
            .replace("**", "")
            .replace("\n", "")
            .replace("###", "")
            .replace("Question:", "")
            .replace("Question: ", "")
            .lstrip()
        }
        assess_answer = {
            "from": "gpt",
            "value": (
                assessment_answer
                .replace("**Answer:** ", "")
                .replace("**Answer:**", "")
                .replace("Answer: ", "")
                .replace("**", "")
                .replace("- ", "")
                .replace("#", "")
                .strip()
                .lstrip() 
            ),
        }
        new_data["conversations"].append(assess_question)
        new_data["conversations"].append(assess_answer)

        # bbox
        # print(bbox_provided)
        # print(bbox is not None)
        if bbox is not None and bbox_provided:
            bbox_info = f"{bbox}"
            distortion_info = ", ".join(
                [
                    f"{type} with bounding box {bbox_loc}"
                    for type, bbox_loc in distortion.items()
                ]
            )
            bbox_question = {
                "from": "human",
                "value": random.choice(follow_up_questions),
            }
            bbox_answer = {
                "from": "gpt",
                "value": f"The distortions present in the image and their locations are as follows: {bbox_info if 'qwen-vl' in model_name.lower() else distortion_info}",
            }
            new_data["conversations"].append(bbox_question)
            new_data["conversations"].append(bbox_answer)
        elif bbox_provided:
            bbox_question = {
                "from": "human",
                "value": random.choice(follow_up_questions),
            }
            bbox_answer = {
                "from": "gpt",
                "value": distortion
            }
            new_data["conversations"].append(bbox_question)
            new_data["conversations"].append(bbox_answer)
        else:
            pass


        # instant rating
        new_data["conversations"].append(
            {"from": "human", "value": random.choice(score_questions)}
        )
        new_data["conversations"].append(
            {"from": "gpt", "value": f"The quality of the image is {level}."}
        )

        json_data.append(new_data)
        # Copy the image file
        image_data += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

    (f"Total data: {len(json_data)}")
    logging.info(f"Total images: {image_data}")
    
    print("Fail image:", fail_image)
    assess_data = [item for item in assess_data if item["filename"] not in fail_image]
    with open(assess_file, "w") as f:
        json.dump(assess_data, f, indent=4)
    print("Remove the bad assessment data")


def split_train_and_val(json_file, ratio=0.95):
    json_data = load_json(json_file)
    # Shuffle the data
    random.seed(42)
    random.shuffle(json_data)
    # Split data into training and validation sets (95% training, 5% validation)
    split_index = int(len(json_data) * ratio)
    train_data = json_data[:split_index]
    val_data = json_data[split_index:]

    # Write the data to output JSON files
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file.replace(".json", "_train.json"), "w", encoding="utf-8") as file:
        json.dump(train_data, file, ensure_ascii=False, indent=4)

    with open(json_file.replace(".json", "_val.json"), "w", encoding="utf-8") as file:
        json.dump(val_data, file, ensure_ascii=False, indent=4)

    logging.info(f"Total data: {len(json_data)}")
    logging.info(f"Training data: {len(train_data)}")
    logging.info(f"Validation data: {len(val_data)}")


def convert_to_qwen_format(distortion):
    """
    Converts the distortion data into a formatted string for the conversations.

    Parameters:
    - distortion: dict - The dictionary containing distortion types and their bounding boxes.

    Returns:
    - A formatted string representing the bounding boxes.
    """
    result = ""
    for distortion_type, boxes in distortion.items():
        formatted_boxes = f"<ref>{distortion_type}</ref>"
        for box in boxes:
            tl = box["tl"]
            br = box["br"]
            formatted_boxes += f"<box>({round(tl['x'])},{round(tl['y'])}),({round(br['x'])},{round(br['y'])})</box>"
        result = result + formatted_boxes + ","

    # Remove the last comma
    if result.endswith(","):
        result = result[:-1]

    return result


def check_images(image_folder_path, remove_extra_images=True, *json_paths):
    """
    Check if all images listed in the provided JSON files exist in the image folder.

    Parameters:
    - image_folder_path: str - Path to the folder containing images.
    - *json_paths: str - Paths to the JSON files that need to be checked.
    """
    all_json_images = set()

    # Load and combine images from all provided JSON files
    for json_path in json_paths:
        data = load_json(json_path)
        images = get_images_from_json(data)
        all_json_images = all_json_images.union(images)

    # Get images from the folder
    folder_images = get_images_from_folder(image_folder_path)

    # Check for missing and extra images
    missing_images = all_json_images - folder_images
    extra_images = folder_images - all_json_images

    if not missing_images and not extra_images:
        logging.info("All images in JSON files are accounted for in the image folder.")
    else:
        if missing_images:
            logging.info("Missing images from folder:", missing_images)
        if extra_images:
            logging.info("Extra images in folder:", extra_images)
            if remove_extra_images:
                for image in extra_images:
                    os.remove(os.path.join(image_folder_path, image))
                logging.info("Extra images have been removed from the folder.")


def calculate_area(bbox):
    """
    Calculate the area of a bounding box.

    Parameters:
    - bbox (dict): The bounding box with keys "tl" (top-left) and "br" (bottom-right),
                   each containing "x" and "y" coordinates.

    Returns:
    - float: The area of the bounding box.
    """
    width = bbox["br"]["x"] - bbox["tl"]["x"]
    height = bbox["br"]["y"] - bbox["tl"]["y"]
    return width * height


def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - bbox1 (dict): The first bounding box.
    - bbox2 (dict): The second bounding box.

    Returns:
    - float: The IoU of the two bounding boxes.
    """
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
    """
    Filter bounding boxes based on area and Intersection over Union (IoU).

    Parameters:
    - bboxes (list): A list of bounding boxes to filter.
    - image_width (int): The width of the image.
    - image_height (int): The height of the image.

    Returns:
    - list: A list of filtered bounding boxes.
    """
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


def clean_bbox_json(json_file, image_folder, save_path):
    """
    Process a JSON file containing bounding box data, filter the bounding boxes,
    and update the data structure.

    Parameters:
    - json_file (str): Path to the JSON file containing bounding box data.
    - image_folder (str): Path to the folder containing the images.

    Returns:
    - list: A list of processed entries with filtered bounding boxes.
    """
    data = load_json(json_file)
    error = []
    for entry in data:
        image = entry["filename"]
        # logging.info(image)
        try:
            image_path = os.path.join(image_folder, image)
            image_width, image_height = Image.open(image_path).size
            for distortion_type, bboxes in entry["distortion"].items():
                filtered_bboxes = filter_bboxes(bboxes, image_width, image_height)

                # Retain only tl and br
                for i in range(len(filtered_bboxes)):
                    filtered_bboxes[i] = {
                        "tl": filtered_bboxes[i]["tl"],
                        "br": filtered_bboxes[i]["br"],
                    }

                # Update original data structure with filtered bboxes
                entry["distortion"][distortion_type] = filtered_bboxes
        except Exception as e:
            if str(e) != "'str' object has no attribute 'items'":
                error.append(str(e))
    with open(save_path, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)
    with open("error.json", "w", encoding="utf-8") as e_out:
        json.dump(error, e_out, ensure_ascii=False, indent=4)


def plot_bbox_dist(json_file, image_folder, area_output_path=None, num_output_path=None):
    """
    Calculate bounding box areas as a ratio of image area, and plot the bbox area ratios distribution.

    Parameters:
    - json_file: str - Path to the JSON file containing annotations.
    - image_folder: str - Folder containing the images.
    - output_path: str - Path to save the output plot.

    Returns:
    - List of bounding box area ratios.
    """
    if area_output_path:
        save_folder = os.path.dirname(area_output_path)
        os.makedirs(save_folder, exist_ok=True)
    bbox_areas = []
    image_areas = []
    bbox_num = []
    def process_data(data):
        num = 0
        image_path = os.path.join(image_folder, data["filename"])
        image_width, image_height = Image.open(image_path).size
        total_image_area = image_width * image_height
        try:
            for boxes in data["distortion"].values():
                for box in boxes:
                    tl_x, tl_y = box["tl"]["x"], box["tl"]["y"]
                    br_x, br_y = box["br"]["x"], box["br"]["y"]
                    bbox_area = (br_x - tl_x) * (br_y - tl_y)
                    bbox_areas.append(bbox_area)
                    image_areas.append(total_image_area)
                    num += 1
            bbox_num.append(num)
        except Exception:
            pass

    with open(json_file, "r") as file:
        for data in json.load(file):
            process_data(data)

    bbox_area_ratios = [
        area / total_area for area, total_area in zip(bbox_areas, image_areas)
    ]
    plt.figure(figsize=(12, 6))
    # plt.hist(
    #     bbox_area_ratios, bins=30, color="skyblue", edgecolor="black", range=(0, 1)
    # )
    # plt.xlabel("Bounding Box Area Ratio")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Bounding Box Area Ratios")
    # plt.grid(True)
    # if area_output_path:
    #     plt.savefig(area_output_path)
    # plt.show()
    counter = Counter(bbox_num)
    values = list(counter.keys())
    frequencies = list(counter.values())
    print(counter.items())
    plt.bar(values, frequencies, color="skyblue", edgecolor="black")
    plt.xlabel("Bounding Box Number")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bounding Box Number Per Image")
    plt.grid(True)
    if num_output_path:
        plt.savefig(num_output_path)
    plt.show()
    return bbox_area_ratios


def plot_score_distribution(gpt_score_path, output_path):
    """
    Plot histogram of score distribution.

    Parameters:
    - gpt_score_path: str
    - output_path: str - Path to save the output plot.
    """
    data = load_json(gpt_score_path)
    scores = [entry["score"] for entry in data]
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor="black")
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


def plot_mos_distribution(json_file, output_path):
    """
    Plot histogram of MOS (Mean Opinion Score) distribution.

    Parameters:
    - json_file: str
    - output_path: str - Path to save the output plot.
    """
    dir = os.path.dirname(output_path)
    os.makedirs(dir, exist_ok=True)
    data = load_json(json_file)
    mos_scores = [item["mos"] for item in data if "mos" in item]
    bins = [1, 1.8, 2.6, 3.4, 4.2, 5.0]
    plt.hist(mos_scores, bins=bins, edgecolor="black")
    plt.title("MOS Score Distribution")
    plt.xlabel("MOS")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


def plot_res_distribution(json_file, image_folder, output_path):
    # Step 1: Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Step 2: Extract image resolutions
    widths = []
    heights = []
    for item in data:
        filename = item.get('filename')
        if filename:
            image_path = os.path.join(image_folder, filename)
        else:
            continue

        if os.path.exists(image_path):
            width, height = Image.open(image_path).size
            if width and height:
                widths.append(width)
                heights.append(height)


    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5, edgecolors='b')
    plt.title('Resolution Distribution of Images')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.grid(True)

    plt.savefig(output_path)
    plt.close()

    print(f"Resolution distribution plot saved to: {output_path}")
