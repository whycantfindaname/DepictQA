meta_folder="dataset/meta_json_clean_v0"
image_folder="/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
save_folder='dataset/clean_data_v0'

python scripts/gen_resp_gpt4v_assess.py \
    --meta_file $save_folder/clean_data.json \
    --desp_file $save_folder/description.json \
    --assess_file $save_folder/assessment.json \
    --image_folder $image_folder \
    --assess_fail_dir $save_folder/assess_fail