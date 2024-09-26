from typing import List
import argparse
import json
parse = argparse.ArgumentParser()
parse.add_argument("--pred_json", type=str, help="Path to the predicted JSON file")
parse.add_argument("--gt_json", type=str, help="Path to the actual JSON file")

from typing import List, Tuple

def calculate_tp_fp_fn(predicted: List[str], actual: List[str], all_types: List[str]) -> Tuple[int, int, int]:
    """
    计算单个样本的 TP, FP, FN。

    参数:
    - predicted (List[str]): 模型预测的退化类型列表。
    - actual (List[str]): 实际存在的退化类型列表。
    - all_types (List[str]): 所有可能的退化类型列表。

    返回:
    - Tuple[int, int, int]: TP, FP, FN 的数量。
    """
    tp = sum(1 for t in all_types if t in predicted and t in actual)  # True Positives
    fp = sum(1 for t in all_types if t in predicted and t not in actual)  # False Positives
    fn = sum(1 for t in all_types if t not in predicted and t in actual)  # False Negatives
    return tp, fp, fn

def evaluate_overall_performance(predicted_data: List[List[str]], actual_data: List[List[str]]) -> dict:
    """
    汇总多个数据样本的预测和实际结果，计算整体指标。

    参数:
    - predicted_data (List[List[str]]): 每个样本的模型预测的退化类型列表。
    - actual_data (List[List[str]]): 每个样本的实际存在的退化类型列表。

    返回:
    - dict: 包含整体精确度、召回率、F1 分数和 Jaccard 相似系数的字典。
    """
    # 获取所有可能的退化类型
    all_types = list(set([t for sublist in predicted_data + actual_data for t in sublist]))

    # 初始化 TP, FP, FN 总数
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # 逐组计算 TP, FP, FN 并累加
    for predicted, actual in zip(predicted_data, actual_data):
        tp, fp, fn = calculate_tp_fp_fn(predicted, actual, all_types)
        # print(f"Predicted: {predicted}, Actual: {actual}, TP: {tp}, FP: {fp}, FN: {fn}")
        total_tp += tp
        total_fp += fp
        total_fn += fn

    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    # 根据总的 TP, FP, FN 计算指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 1

    return {
        'Overall Precision': precision,
        'Overall Recall': recall,
        'Overall F1 Score': f1,
        'Overall Jaccard Index (IoU)': iou
    }

# 示例使用
predicted_data_samples = [
    ['Motion blur', 'Noise', 'Low clarity'],
    ['Underexposure', 'Low clarity'],
    ['Noise', 'Low clarity']
]

actual_data_samples = [
    ['Motion blur', 'Low clarity', 'Underexposure'],
    ['Motion blur', 'Underexposure'],
    ['Low clarity']
]
if __name__ == "__main__":
    args = parse.parse_args()
    pred_dist = []
    gt_dist = []
    with open(args.pred_json, 'r') as f:
        pred = json.load(f)
    for i in range(len(pred)):
        dist_item = list(pred[i]["conversation"][1]["value"].split(","))
        pred_dist.append(dist_item)
    
    with open(args.gt_json, 'r') as f:
        gt = json.load(f)
    for i in range(len(gt)):
        dist_item = list(gt[i]["conversations"][3]["value"].split(","))
        gt_dist.append(dist_item)

    print("pred_dist:", pred_dist[0])
    print("gt_dist:", gt_dist[0])
    
    result = evaluate_overall_performance(pred_dist, gt_dist)
    print(result)