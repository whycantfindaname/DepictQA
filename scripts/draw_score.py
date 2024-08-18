import json

import matplotlib.pyplot as plt

# 假设你的 JSON 数据保存在一个名为 data.json 的文件中
with open("data.json", "r") as file:
    data = json.load(file)

# 提取所有的 score 值
scores = [entry["score"] for entry in data]

# 绘制 score 的分布
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=20, edgecolor="black")
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
