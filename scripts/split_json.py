import json
import os

json_path = "./data/meta_json/combined_json.json"
save_folder = "./data/json"
os.makedirs(save_folder, exist_ok=True)

# 读取原始JSON文件内容
with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 遍历JSON对象中的每个键
for key, value in data.items():
    # 创建每个新JSON文件的文件名
    filename = f"{key}.json"
    path = os.path.join(save_folder, filename)
    # 将数据写入新的JSON文件
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(value, outfile, ensure_ascii=False, indent=4)

print("拆分完成！")
