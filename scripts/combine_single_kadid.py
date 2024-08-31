from utils import merge_json_simple


# Function to merge JSON files with a given prefix
def merge_datasets(prefix, base_path, single_path, kadid_train, output_suffixes):
    files = {
        "train": [
            f"{prefix}_single_no_bbox_train.json",
            f"{prefix}_single_with_bbox_train.json",
        ],
        "val": [
            f"{prefix}_single_no_bbox_val.json",
            f"{prefix}_single_with_bbox_val.json",
        ],
    }

    for mode, file_names in files.items():
        for i, file_name in enumerate(file_names):
            input_file = f"{single_path}{file_name}"
            output_file = f"{base_path}{prefix}_{mode}{output_suffixes[i]}"
            merge_json_simple(input_file, kadid_train, output_file)


# Paths
base_path = "results/{prefix}_single5k_kadid/"
single_path = "results/{prefix}_single/"
kadid_train = "train_kadid_no_mos.json"
output_suffixes = ["_no_bbox.json", "_with_bbox.json"]

# Prefixes to process
prefixes = ["llava", "qwen"]

# Merge for each prefix
for prefix in prefixes:
    merged_base_path = base_path.format(prefix=prefix)
    merged_single_path = single_path.format(prefix=prefix)
    merge_datasets(
        prefix,
        merged_base_path,
        merged_single_path,
        f"{merged_base_path}{kadid_train}",
        output_suffixes,
    )
