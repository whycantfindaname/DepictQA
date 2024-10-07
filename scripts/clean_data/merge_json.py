from utils import merge_jsons_from_folder

if __name__ == "__main__":
    desp_folder = "dataset/meta_json"
    desp_file = "dataset/total_image_no_clean.json"
    merge_jsons_from_folder(desp_folder, desp_file)
    assess_folder = "dataset/meta_json_clean_v0"
    assess_file = "dataset/image_clean_v0.json"
    merge_jsons_from_folder(assess_folder, assess_file)
