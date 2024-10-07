import json
import shutil
import os
image_folder = '/media/liaowenjie/738c79d8-a7b5-43c7-82ce-a07c64929d72/QualityLLM_single_2w'
save_folder = '../datasets/single_1w'
os.makedirs(save_folder, exist_ok=True)
with open('dataset/total_image_no_clean.json', 'r', encoding='utf-8') as f:
    ori_data_raw = json.load(f)

# 初始化一个空列表，用于存储所有的文件数据
ori_data = []

# 遍历整个列表，获取每个字典中的值
for entry in ori_data_raw:
    for key, value in entry.items():
        ori_data.append(value)
print(f'Total number of images: {len(ori_data)}')
l = 0
previous_l = -1  # 用来记录上一轮的l值，初始为-1以确保第一次循环不会提前退出

while True:
    for item in ori_data[:]:  # 使用[:]复制列表，避免在循环中直接修改列表产生错误
        image_name = item['filename']
        image_path = f'{image_folder}/{image_name}'

        if not os.path.exists(image_path):
            print(f'{image_path} not exists')
            l += 1
            ori_data.remove(item)  # 移除不存在的图片项
            continue

        # 复制图片到目标文件夹
        shutil.copy(image_path, f'{save_folder}/{image_name}')
    
    print(f'{l} images not exist')

    # 如果本轮的l和上一轮的l没有变化，说明图片缺失的数量不再变化，退出循环
    if l == previous_l:
        break  # 退出循环，所有图片都已经处理完成

    # 更新previous_l以用于下一轮的比较
    previous_l = l


with open(f'dataset/clean_image_step1.json', 'w', encoding='utf-8') as f:
    json.dump(ori_data, f, ensure_ascii=False, indent=4)