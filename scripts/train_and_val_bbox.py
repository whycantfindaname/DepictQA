import json
import os as OS
import random
import shutil

json_data = []
image_data = 0
desp_dir = "./data/description"
assess_dir = "./data/assessment"
bbox_dir = "./data/clean_bbox_json"
image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
save_folder = "./data/images"
OS.makedirs(save_folder, exist_ok=True)


def convert_to_target_format(distortion):
    result = ""
    for distortion_type, boxes in distortion.items():
        # 初始化格式化字符串
        formatted_boxes = f"<ref>{distortion_type}</ref>"
        for box in boxes:
            tl = box["tl"]
            br = box["br"]
            # 追加每个box的坐标到字符串中
            formatted_boxes += f"<box>({tl['x']},{tl['y']}),({br['x']},{br['y']})</box>"
        # 将格式化后的字符串添加到结果列表中
        result += formatted_boxes
    return result


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
for file_name in OS.listdir(assess_dir):
    question = random.choice(follow_up_questions)
    assess_path = OS.path.join(assess_dir, file_name)
    desp_path = OS.path.join(desp_dir, file_name)
    bbox_path = OS.path.join(bbox_dir, file_name)
    # 确保是文件
    if not OS.path.isfile(assess_path):
        continue

    # 读取原始txt文件内容
    try:
        with open(assess_path, "r", encoding="utf-8") as file:
            assess = json.load(file)

        with open(desp_path, "r", encoding="utf-8") as file1:
            desp = json.load(file1)
        with open(bbox_path, "r", encoding="utf-8") as file2:
            meta_data = json.load(file2)

        # 确保 assess 和 desp 的文件名一致
        assert assess["filename"] == desp["filename"] == meta_data["filename"]

        # 分割描述内容，处理换行符情况
        description_parts = desp["gpt4v_description"].split("\n\n")
        if len(description_parts) < 2:
            # 如果找不到双换行符，尝试单换行符分割
            description_parts = desp["gpt4v_description"].split("\n")

        # 分割评估内容，处理换行符情况
        assessment_parts = assess["gpt4v_assessment"].split("\n\n")
        if len(assessment_parts) < 2:
            # 如果找不到双换行符，尝试单换行符分割
            assessment_parts = assess["gpt4v_assessment"].split("\n")

        distortion = meta_data["distortion"]
        bbox = convert_to_target_format(distortion)
        # 构建新格式的JSON结构
        new_data = {
            "image": desp["filename"],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>"
                    + description_parts[0]
                    .replace("**Question:** ", "")
                    .replace("Question: ", ""),
                },
                {
                    "from": "gpt",
                    "value": description_parts[1].replace(
                        "**Answer:** " or "Answer: ", ""
                    )
                    if len(description_parts) > 1
                    else "",
                },
                {
                    "from": "human",
                    "value": assessment_parts[0]
                    .replace("**Question:** ", "")
                    .replace("Question: ", ""),
                },
                {
                    "from": "gpt",
                    "value": assessment_parts[1].replace(
                        "**Answer:** " or "Answer: ", ""
                    )
                    if len(assessment_parts) > 1
                    else "",
                },
                {
                    "from": "human",
                    "value": question,
                },
                {"from": "gpt", "value": bbox},
            ],
        }

        # 复制图像文件
        img_path = OS.path.join(image_folder, desp["filename"])
        save_path = OS.path.join(save_folder, desp["filename"])
        if OS.path.exists(save_path):
            print(f"File {desp['filename']} already exists in {save_folder}")
        else:
            shutil.copy(img_path, save_folder)
            print(f"Copied {desp['filename']} to {save_folder}")
        image_data += 1
        json_data.append(new_data)

    except AssertionError:
        print(f"Filename mismatch between {assess_path} and {desp_path}, skipping...")
    except KeyError as e:
        print(f"Missing key {e} in {file_name}, skipping...")
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        continue

# 打乱数据
random.seed(42)
random.shuffle(json_data)

# 按10:1划分训练集和验证集
split_index = int(len(json_data) * 11 / 12)
train_data = json_data[:split_index]
val_data = json_data[split_index:]

# 写入JSON文件
OS.makedirs("./results", exist_ok=True)
with open("./results/train_data_bbox.json", "w", encoding="utf-8") as file:
    json.dump(train_data, file, ensure_ascii=False, indent=4)

with open("./results/val_data_bbox.json", "w", encoding="utf-8") as file:
    json.dump(val_data, file, ensure_ascii=False, indent=4)

print(f"Total data: {len(json_data)}")
print(f"Training data: {len(train_data)}")
print(f"Validation data: {len(val_data)}")
print(f"Total images copied: {image_data}")

# 检查目标文件夹中的文件数量
actual_image_count = len(OS.listdir(save_folder))
print(f"Actual images in folder: {actual_image_count}")
