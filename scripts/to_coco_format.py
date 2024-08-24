from utils import convert_to_coco_format

if __name__ == "__main__":
    json_file = "data/clean_data.json"
    save_path = "data/coco_format_all.json"
    # if not os.path.exists(meta_path):
    data = convert_to_coco_format(
        json_file,
        save_path,
        "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w",
    )
