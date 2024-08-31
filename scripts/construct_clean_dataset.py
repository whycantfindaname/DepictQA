import argparse

from utils import clean_bbox_json, merge_meta_jsons_from_folder

parser = argparse.ArgumentParser(description="construct clean dataset")

parser.add_argument("--meta_folder", type=str, required=True)
parser.add_argument("--json_save_path", type=str, required=True)
parser.add_argument("--clean_json_path", type=str, required=True)
parser.add_argument("--image_folder", type=str, required=True)

if __name__ == "__main__":
    # meta_folder = "data/meta_json"
    # meta_path = "data/meta_data.json"
    # clean_path = "data/clean_data.json"
    # image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
    # save_path = "data/images"
    args = parser.parse_args()
    meta_folder = args.meta_folder
    json_save_path = args.json_save_path
    clean_json_path = args.clean_json_path
    image_folder = args.image_folder
    # if not os.path.exists(meta_path):
    merge_meta_jsons_from_folder(meta_folder, json_save_path, image_folder)
    data = clean_bbox_json(
        json_save_path,
        image_folder,
        clean_json_path,
    )
