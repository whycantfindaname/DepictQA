# 初始化 conda
eval "$(conda shell.bash hook)"

# 激活指定的 conda 环境
conda activate datadepictqa

python scripts/modify_iqadata.py \
    --train_path dataset/spaq_json/train_spaq.json \
    --test_path dataset/spaq_json/test_spaq.json \
    --train_output dataset/spaq_json/train_spaq_with_bbox.json \
    --val_output dataset/spaq_json/val_spaq_with_bbox.json 

python scripts/modify_iqadata.py \
    --train_path dataset/koniq_json/train_koniq.json \
    --test_path dataset/koniq_json/test_koniq.json \
    --train_output dataset/koniq_json/train_koniq_with_bbox.json \
    --val_output dataset/koniq_json/val_koniq_with_bbox.json 
