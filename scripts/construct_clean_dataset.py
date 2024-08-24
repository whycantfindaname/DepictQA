from utils import check_images, clean_bbox_json, merge_meta_jsons_from_folder

if __name__ == "__main__":
    meta_folder = "data/meta_json"
    meta_path = "data/meta_data.json"
    clean_path = "data/clean_data.json"
    image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
    save_path = "data/images"
    # if not os.path.exists(meta_path):
    merge_meta_jsons_from_folder(meta_folder, meta_path, image_folder, save_path)
    data = clean_bbox_json(
        meta_path,
        image_folder,
        clean_path,
    )
    check_images(save_path, True, clean_path)
