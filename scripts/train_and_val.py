import json
import os as OS
import random
import shutil

json_data = []
image_data = 0
desp_dir = "./data/description"
assess_dir = "./data/assessment"
image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
save_folder = "./data/images"
OS.makedirs(save_folder, exist_ok=True)

for file_name in OS.listdir(assess_dir):
    assess_path = OS.path.join(assess_dir, file_name)
    desp_path = OS.path.join(desp_dir, file_name)

    # 确保是文件
    if not OS.path.isfile(assess_path):
        continue

    # 读取原始txt文件内容
    try:
        with open(assess_path, "r", encoding="utf-8") as file:
            assess = json.load(file)

        with open(desp_path, "r", encoding="utf-8") as file1:
            desp = json.load(file1)

        # 确保 assess 和 desp 的文件名一致
        assert assess["filename"] == desp["filename"]

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
            ],
        }

        # 复制图像文件
        img_path = OS.path.join(image_folder, desp["filename"])
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
random.shuffle(json_data)

# 按10:1划分训练集和验证集
split_index = int(len(json_data) * 11 / 12)
train_data = json_data[:split_index]
val_data = json_data[split_index:]

# 写入JSON文件
OS.makedirs("./results", exist_ok=True)
with open("./results/train_data.json", "w", encoding="utf-8") as file:
    json.dump(train_data, file, ensure_ascii=False, indent=4)

with open("./results/val_data.json", "w", encoding="utf-8") as file:
    json.dump(val_data, file, ensure_ascii=False, indent=4)

print(f"Total data: {len(json_data)}")
print(f"Training data: {len(train_data)}")
print(f"Validation data: {len(val_data)}")
print(f"Total images copied: {image_data}")

# 检查目标文件夹中的文件数量
actual_image_count = len(OS.listdir(save_folder))
print(f"Actual images in folder: {actual_image_count}")
