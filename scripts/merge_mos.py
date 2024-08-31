import json
import os

from utils import load_json

mos_dir = "data/mos"
save_path = os.path.join(mos_dir, "mos_with_kadid.json")
mos = []
for mos_file in os.listdir(mos_dir):
    mos_path = os.path.join(mos_dir, mos_file)
    data = load_json(mos_path)
    mos.append(data)

with open(save_path, "w") as file:
    json.dump(mos, file, indent=4)
