from utils import merge_json_simple

qwen_single_train = "dataset/qwen_single/qwen_bbox_train.json"
qwen_single_val = "dataset/qwen_single/qwen_bbox_val.json"
kadid_train = "dataset/kadid_json/train_kadid_no_mos.json"
kadid_val = "dataset/kadid_json/val_kadid_no_mos.json"

merge_json_simple(qwen_single_train, kadid_train, "dataset/qwen_train.json")
merge_json_simple(qwen_single_val, kadid_val, "dataset/qwen_val.json")
