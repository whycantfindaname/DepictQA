import json

import json

# 读取并处理 JSON 数据
with open('dataset/clean_image_step1.json', 'r', encoding='utf-8') as f:
    ori_data= json.load(f)

# 打印总共加载了多少条数据
print(f"Total number of entries loaded: {len(ori_data)}")

# 读取MOS数据
with open('dataset/mos分回扫第一批3691.json', 'r', encoding='utf-8') as f:
    mos_data = list(json.load(f).values())
l = 0
# 遍历MOS数据，查找并更新原始数据
for item in mos_data:
    img_name = item['filename'].strip()  # 去掉可能的空格
    print(f"Looking for image: {img_name}")  # 打印当前处理的img_name
    
    # 查找匹配的 ori_item
    ori_item = next((x for x in ori_data if x['filename'].strip() == img_name), None)
    
    if ori_item is None:
        print(f"Could not find image: {img_name} in original data.")
    else:
        print(f"Found image: {img_name}")
        # 遍历新数据中的键，更新原始数据中的 'mos' 值
        for key in item:
            if key in ori_item and isinstance(item[key], dict) and 'mos' in item[key]:
                ori_item[key]['mos'] = item[key]['mos']
        
        # 打印检查

        l += 1
    
print(f"Updated {l} items.")

# 保存更新后的原始数据
with open('dataset/clean_image_step1_replace_mos.json', 'w', encoding='utf-8') as f:
    json.dump(ori_data, f, indent=4, ensure_ascii=False)
