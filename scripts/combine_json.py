import json
import os

def combine_json_files(input_folder, output_file):
    combined_data = {}

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # 确保只处理文件
        if not os.path.isfile(file_path):
            continue

        # 读取 JSON 文件内容
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 确保数据是字典类型
                if isinstance(data, dict):
                    combined_data.update(data)
                else:
                    print(
                        f"Warning: The content of file {file_name} is not a dictionary. Skipping..."
                    )
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON in file {file_name}. Skipping...")
        except Exception as e:
            print(f"Error: {e} - Skipping file {file_name}.")

    # 将合并后的数据写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)

# 示例用法
input_folder = "./data/meta_json"  # JSON 文件所在的文件夹路径
output_file = "./data/meta_json/combined_json.json"  # 合并后的 JSON 文件路径
combine_json_files(input_folder, output_file)
