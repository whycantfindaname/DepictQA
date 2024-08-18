import json
import os


def load_json(file_path):
    """Load a JSON file and return its contents."""
    with open(file_path, "r") as file:
        return json.load(file)


def get_images_from_json(json_data):
    """Extract image names from JSON data."""
    return {item["image"] for item in json_data}


def get_images_from_folder(folder_path):
    """Get a set of image file names from a directory."""
    return {
        file
        for file in os.listdir(folder_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    }


def check_images(json_file_path1, json_file_path2, image_folder_path):
    """Check if all images listed in both JSON files exist in the image folder."""
    # Load JSON data
    json_data1 = load_json(json_file_path1)
    json_data2 = load_json(json_file_path2)

    # Get images from JSON files
    json_images1 = get_images_from_json(json_data1)
    json_images2 = get_images_from_json(json_data2)

    # Combine images from both JSON files
    all_json_images = json_images1.union(json_images2)

    # Get images from folder
    folder_images = get_images_from_folder(image_folder_path)

    # Check for missing images
    missing_images = all_json_images - folder_images
    extra_images = folder_images - all_json_images

    if not missing_images and not extra_images:
        print("All images in JSON files are accounted for in the image folder.")
    else:
        if missing_images:
            print("Missing images from folder:", missing_images)
        if extra_images:
            print("Extra images in folder:", extra_images)


# Example usage
json_file_path1 = "./results/train_data.json"
json_file_path2 = "./results/val_data.json"
image_folder_path = "./data/images"

check_images(json_file_path1, json_file_path2, image_folder_path)
