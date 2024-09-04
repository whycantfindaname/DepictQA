import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def csv2json(csv_path, json_path):
    res_dict = {}

    res_df = pd.read_csv(csv_path)

    for index, row in res_df.iterrows():
        csvRes = json.loads(row["答案"])

        if row["题目ID"] not in res_dict:
            res_dict[row["题目ID"]] = {}
        res_dict[row["题目ID"]]["filename"] = csvRes["item"]["filename"]

        if row["轮次"] not in res_dict[row["题目ID"]]:
            res_dict[row["题目ID"]][row["轮次"]] = {}
        else:
            continue

        # res_dict[row['题目ID']][row['轮次']]['操作人'] = row['[标注]操作人']
        res_dict[row["题目ID"]][row["轮次"]]["mos"] = int(
            csvRes["data"]["pic_type"].split("分")[0]
        )

        # 画质子维度 & 拉框结果
        qualityRect = csvRes["data"]["items"]

        for quality in qualityRect:
            if quality["rect_type"] not in res_dict[row["题目ID"]][row["轮次"]]:
                res_dict[row["题目ID"]][row["轮次"]][quality["rect_type"]] = []
            res_dict[row["题目ID"]][row["轮次"]][quality["rect_type"]].append(
                quality["region"]
            )

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(res_dict, json_file, ensure_ascii=False, indent=4)

        print(f"{index} / {len(res_df)}")

    return res_dict


def mos_acc(scores):
    if len(scores) < 3:
        raise ValueError("Score list must contain at least three elements.")

    scores.remove(max(scores))
    scores.remove(min(scores))

    max_score = max(scores)
    min_score = min(scores)

    return max_score - min_score <= 1


def pickup_common_rect(rect_n_list):
    add_rect = sum(rect_n_list, [])
    count_rect = dict(Counter(add_rect))

    return count_rect


def calculate_iou(box1, box2):
    """
    计算两个框的 IoU 值：两个框的 overlap 面积 / 小框的面积
    """
    x1 = box1["tl"]["x"]
    y1 = box1["tl"]["y"]
    w1 = box1["br"]["x"] - x1
    h1 = box1["br"]["y"] - y1

    x2 = box2["tl"]["x"]
    y2 = box2["tl"]["y"]
    w2 = box2["br"]["x"] - x2
    h2 = box2["br"]["y"] - y2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    # iou = inter_area / union_area if union_area != 0 else 0
    iou = (
        inter_area / min(box1_area, box2_area)
        if union_area != 0 and min(box1_area, box2_area) != 0
        else 0
    )

    return iou, 0 if box1_area < box2_area else 1


def is_valid(round_1, round_2, va_rect):
    for i in range(len(round_1)):
        for j in range(len(round_2)):
            iou, index = calculate_iou(round_1[i], round_2[j])
            if iou > 0.6:
                # quality_acc_flag = True
                if index == 0 and round_1[i] not in filtered_res_dict[item_id][va_rect]:
                    filtered_res_dict[item_id][va_rect].append(round_1[i])
                if index == 1 and round_2[j] not in filtered_res_dict[item_id][va_rect]:
                    filtered_res_dict[item_id][va_rect].append(round_2[j])


def quality_acc(res_dict):
    rect_n_list = []
    for round in res_dict[item_id]:
        if round == "filename":
            continue
        rect_n = list(res_dict[item_id][round].keys())
        rect_n.remove("mos")
        rect_n_list.append(rect_n)

    # 挑选出两轮以上公共的画质子维度
    valid_rect = pickup_common_rect(rect_n_list)

    # 遍历三轮的公共子维度
    for va_rect in valid_rect:
        common_round_n = []

        if valid_rect[va_rect] < 2:
            continue
        else:
            filtered_res_dict[item_id][va_rect] = []
            common_idx = [i for i in range(label_round) if va_rect in rect_n_list[i]]

            for idx in common_idx:
                common_round_n.append(res_dict[item_id][str(idx + 1)][va_rect])

            for i in range(len(common_round_n)):
                round_1 = common_round_n[i]
                for j in range(i + 1, len(common_round_n)):
                    round_2 = common_round_n[j]
                    is_valid(round_1, round_2, va_rect)

            if len(filtered_res_dict[item_id][va_rect]) == 0:
                filtered_res_dict[item_id].pop(va_rect)


label_round = 5
csv_path = "dataset/meta_json/正式5轮_第四批2k标注结果.csv"
json_path = f"dataset/meta_json/{csv_path.split('/')[-1].split('.')[0]}.json"
print(json_path)
if not os.path.exists(json_path):
    res_dict = csv2json(csv_path, json_path)
else:
    res_dict = json.load(open(json_path, "r"))

mos_acc_count = 0
filtered_res_dict = {}

# 遍历 res_dict
for item_id, item_data in tqdm(res_dict.items()):
    # 提取 MOS 分数
    scores = [item_data[round]["mos"] for round in item_data if round != "filename"]

    # 初始化过滤后的字典项
    filtered_res_dict[item_id] = {"filename": item_data["filename"]}

    # 判断 MOS 分数的有效性
    if mos_acc(scores):
        mos_acc_count += 1
        # 去掉最高分和最低分
        scores = [
            score for score in scores if score != max(scores) and score != min(scores)
        ]

        # 计算平均分并更新字典
        filtered_res_dict[item_id]["mos"] = np.mean(scores)

        # 判断画质子维度 & 拉框结果
        quality_acc(item_data)

    # 如果没有 MOS 分数，或者 MOS 分数小于 4 且没有有效画质小项，则标注不合格
    if len(filtered_res_dict[item_id]) < 2 or (
        "mos" in filtered_res_dict[item_id] and filtered_res_dict[item_id]["mos"] < 4
    ):
        filtered_res_dict.pop(item_id)

# 保存结果为 JSON 文件
output_path = f"dataset/meta_json_clean_v1/{csv_path.split('/')[-1].split('.')[0]}严格正确结果.json"
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(filtered_res_dict, json_file, ensure_ascii=False, indent=4)

print(f"mos acc: {mos_acc_count} / {len(res_dict)}")
