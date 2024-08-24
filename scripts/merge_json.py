from utils import merge_jsons_from_folder

if __name__ == "__main__":
    desp_folder = "data/description"
    desp_file = "data/description.json"
    merge_jsons_from_folder(desp_folder, desp_file)
    assess_folder = "data/assessment"
    assess_file = "data/assessment.json"
    merge_jsons_from_folder(assess_folder, assess_file)
