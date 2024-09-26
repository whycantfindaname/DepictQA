from utils import convert_to_coco_format

if __name__ == "__main__":
    json_file = "dataset/clean_data_v0/meta_data.json"
    save_path = "dataset/clean_data_v0/coco_data.json"
    # if not os.path.exists(meta_path):
    data = convert_to_coco_format(
        json_file,
        save_path,
        "../datasets/images/QualityLLM_single_2w",
    )
