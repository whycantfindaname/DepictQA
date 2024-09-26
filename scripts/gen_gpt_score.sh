# 初始化 conda
eval "$(conda shell.bash hook)"

# 激活指定的 conda 环境
conda activate datadepictqa

python scripts/cal_gpt4_score_detail.py \
    --pred_path results-8k/qwen_with_bbox_sample2_val_pred.json \
    --gt_path ../datasets/val_json/qwen_with_bbox_val.json \
    --save_path results-8k/qwen_with_bbox_val_gpt4_score.json \
