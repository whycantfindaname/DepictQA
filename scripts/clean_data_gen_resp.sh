#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate datadepictqa

# Define common paths
meta_folder="dataset"
image_folder="/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"

# Run scripts with simplified variable usage
python scripts/construct_clean_dataset.py \
    --meta_folder $meta_folder/meta_json \
    --json_save_path $meta_folder/meta_data.json \
    --clean_json_path $meta_folder/clean_data.json \
    --image_folder $image_folder 

python scripts/gen_resp_gpt4v_desp.py \
    --meta_file $meta_folder/clean_data.json \
    --image_folder $image_folder \
    --desp_file $meta_folder/description.json \
    --desp_fail_dir $meta_folder/desp_fail

python scripts/gen_resp_gpt4v_assess.py \
    --meta_file $meta_folder/clean_data.json \
    --desp_file $meta_folder/description.json \
    --assess_file $meta_folder/assessment.json \
    --image_folder $image_folder \
    --assess_fail_dir $meta_folder/assess_fail

python scripts/train_and_val.py \
    --meta_file $meta_folder/clean_data.json \
    --desp_file $meta_folder/description.json \
    --assess_file $meta_folder/assessment.json \
    --image_folder $image_folder \
    --meta_folder $meta_folder/single6k \
    --output_file $meta_folder/qwen_single/qwen_bbox.json \
    --keep_mos True \
    --bbox_provided True
