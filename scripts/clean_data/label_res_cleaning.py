import json
import numpy as np
from collections import Counter
from tqdm import tqdm


def mos_acc(scores):
    """判断MOS的最大值与最小值之间是否差值不超过1"""
    scores = [score for score in scores if score is not None]
    if len(scores) < 5:
        raise ValueError("Score list must contain at least five elements.")
    
    max_score = max(scores)
    min_score = min(scores)
    
    return max_score - min_score <= 1


def pickup_common_rect(rect_n_list):
    """挑选出两轮以上公共的画质子维度"""
    add_rect = sum(rect_n_list, [])
    count_rect = dict(Counter(add_rect))
    
    return count_rect


def calculate_iou(box1, box2):
    """计算两个框的 IoU 值"""
    x1, y1, w1, h1 = box1["tl"]["x"], box1["tl"]["y"], box1["br"]["x"] - box1["tl"]["x"], box1["br"]["y"] - box1["tl"]["y"]
    x2, y2, w2, h2 = box2["tl"]["x"], box2["tl"]["y"], box2["br"]["x"] - box2["tl"]["x"], box2["br"]["y"] - box2["tl"]["y"]

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area, box2_area = w1 * h1, w2 * h2
    iou = inter_area / min(box1_area, box2_area) if min(box1_area, box2_area) != 0 else 0
    
    return iou, 0 if box1_area < box2_area else 1


def is_valid(round_1, round_2, va_rect, filtered_item):
    """检查是否两个轮次的框有效，并且 IoU 大于 0.6"""
    for i in range(len(round_1)):
        for j in range(len(round_2)):
            iou, index = calculate_iou(round_1[i], round_2[j])
            if iou > 0.6:
                if index == 0 and round_1[i] not in filtered_item[va_rect]:
                    filtered_item[va_rect].append(round_1[i])
                if index == 1 and round_2[j] not in filtered_item[va_rect]:
                    filtered_item[va_rect].append(round_2[j])


def quality_acc(item_data, filtered_item, label_round):
    """检查每轮的质量子维度，提取出公共维度"""
    rect_n_list = []
    for round_key in item_data:
        if round_key == "filename":
            continue
        rect_n = list(item_data[round_key].keys())
        rect_n.remove("mos")
        rect_n_list.append(rect_n)

    valid_rect = pickup_common_rect(rect_n_list)

    for va_rect in valid_rect:
        common_round_n = []
        if valid_rect[va_rect] < 2:
            continue
        else:
            filtered_item[va_rect] = []
            common_idx = [i for i in range(label_round) if va_rect in rect_n_list[i]]
            for idx in common_idx:
                common_round_n.append(item_data[str(idx + 1)][va_rect])
            for i in range(len(common_round_n)):
                round_1 = common_round_n[i]
                for j in range(i + 1, len(common_round_n)):
                    round_2 = common_round_n[j]
                    is_valid(round_1, round_2, va_rect, filtered_item)

            if len(filtered_item[va_rect]) == 0:
                filtered_item.pop(va_rect)



# 主程序
label_round = 5
json_path = "dataset/clean_image_step1_replace_mos.json"

with open(json_path, "r", encoding='utf-8') as f:
    res_dict = json.load(f)

filtered_res_list = []  # 使用列表存储最终结果
mos_acc_count = 0
removed_items = []
for item_data in tqdm(res_dict):
    filename = item_data["filename"]

    # 提取MOS分数
    scores = [item_data[round]["mos"] for round in item_data if round != "filename"]

    # 初始化过滤后的字典项
    filtered_item = {"filename": filename}
    try:
        # 判断MOS分数的有效性
        mos_acc(scores)
    except ValueError:
        print(f"Invalid MOS score: {filename}")
        removed_items.append(f"Invalid MOS score: {filename}")
        continue

    # 判断MOS分数的有效性
    if mos_acc(scores):
        mos_acc_count += 1
        filtered_item["mos"] = np.mean(scores)
        # 根据scores中众数生成level, 众数为1，levle是bad
        counts = np.bincount(scores)
        mode = np.argmax(counts)
        if mode == 1:
            filtered_item["level"] = "bad"
        elif mode == 2:
            filtered_item["level"] = "poor"
        elif mode == 3:
            filtered_item["level"] = "fair"
        elif mode == 4:
            filtered_item["level"] = "good"
        else:
            filtered_item["level"] = "excellent"

        # 判断画质子维度 & 拉框结果
        quality_acc(item_data, filtered_item, label_round)
    
    # **检查是否满足删除条件**：
    # 1. 如果 filtered_item 长度 < 2（即只有 filename）
    # 2. 或者 MOS 分数 < 4 且没有有效的画质子维度（即长度仍为 2）
    if len(filtered_item) < 2 or (
        "mos" in filtered_item and filtered_item["mos"] < 4 and len(filtered_item) == 2
    ):
        removed_items.append(item_data)  # 记录到删除列表中
    else:
        # 只在通过条件后将 filtered_item 添加到列表
        filtered_res_list.append(filtered_item)

# 保存结果为JSON文件
output_path = "dataset/clean_data_v1/step2_filtered.json"
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(filtered_res_list, json_file, ensure_ascii=False, indent=4)

# 保存被删除的项为 JSON 文件
removed_items_path = "dataset/step2_removed.json"
with open(removed_items_path, "w", encoding="utf-8") as json_file:
    json.dump(removed_items, json_file, ensure_ascii=False, indent=4)

print(f"saved: {len(filtered_res_list)} / {len(res_dict)}")
print(f"removed: {len(removed_items)} / {len(res_dict)}")
