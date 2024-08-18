import json
import os

image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
json_folder = "./data/json_image"
save_folder = "./data/convert_json"  # folder to store json files with image path


os.makedirs(save_folder, exist_ok=True)

for json_file in os.listdir(json_folder):
    json_file_path = os.path.join(json_folder, json_file)
    with open(json_file_path, "r") as f:
        data = json.load(f)

    filename = data.get("filename")
    mos = data.get("mos")

    # Only process if filename and mos are present
    if filename and mos:
        # Extract other key-value pairs into a new dictionary under 'distortion'
        distortion = {k: v for k, v in data.items() if k not in ["filename", "mos"]}
        if distortion:
            # Create a new JSON structure
            new_data = {"filename": filename, "mos": mos, "distortion": distortion}

            # Save the new JSON file to the json_image folder
            new_json_file_path = os.path.join(save_folder, json_file)
            with open(new_json_file_path, "w") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

    else:
        print(f"Skipping {json_file} because filename or mos is missing.")
