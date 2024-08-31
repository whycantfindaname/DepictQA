# 初始化 conda
eval "$(conda shell.bash hook)"

# 激活指定的 conda 环境
conda activate datadepictqa

python scripts/cal_gpt4_score_detail.py \
    --pred_path qwen_with_bbox_val_pred.json \
    --gt_path results/gpt_score/qwen/qwen_val_no_bbox.json \
    --save_path qwen_with_bbox_val_gpt4_score.json \
