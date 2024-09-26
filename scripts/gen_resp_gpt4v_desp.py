import argparse
import base64
import json
import os

import weave
from openai import OpenAI
from utils import load_json

OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"
OPENAI_API_BASE = "https://api.gpt.ge/v1"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

parser = argparse.ArgumentParser(description="To Prompt GPT-4 for Image Descriptions")
parser.add_argument("--meta_file", type=str, required=True)
parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--desp_file", type=str, required=True)
parser.add_argument("--desp_fail_dir", type=str, required=True)


def encode_img(img_path):
    ext = os.path.splitext(img_path)[1].lower()
    print(ext)
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"
    elif ext == ".bmp":
        mime_type = "image/bmp"
    else:
        raise ValueError("Unsupported image format")

    with open(img_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    return mime_type, img_base64


@weave.op()
def gpt4v(img_path, query):
    mime_type, img_base64 = encode_img(img_path)
    print(f"Encoded image data: {img_base64[:30]}...")  # 添加调试信息，打印前30个字符

    resp = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{img_base64}",
                    },
                ],
            }
        ],
        temperature=0.5,
        max_tokens=200,
    )
    content = resp.choices[0].message.content
    return content


if __name__ == "__main__":
    weave.init("image description")
    args = parser.parse_args()
    idx_meta_start = 0
    idx_meta_end = -1

    meta_file = args.meta_file
    desp_file = args.desp_file
    fail_dir = args.desp_fail_dir
    image_folder = args.image_folder
    # description_query
    dist_paths_error = []
    meta_data = load_json(meta_file)
    if os.path.exists(desp_file):
        desp_data = load_json(desp_file)
    else:
        desp_data = []

    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:]):
        # print("=" * 100)
        # print(idx_meta + idx_meta_start)

        img_name = meta_item["filename"]
        if img_name in [item["filename"] for item in desp_data]:
            print(f"{img_name} has been generated, skip.")
            continue
        img_path = os.path.join(image_folder, img_name)

        # description_query
        description_query = "Please provide a brief description of the image, including specific objects and any events. If you do not have enough confidence in recognizing the content, please directly return the reason for the inability to recognize it."

        try:
            content = gpt4v(img_path, description_query)
            meta_item["gpt4v_description"] = content
            desp_data.append(meta_item)
            print(content)
            with open(desp_file, "w") as fw:
                json.dump(desp_data, fw, indent=4, ensure_ascii=False)
        except:
            import sys

            except_type, except_value, except_traceback = sys.exc_info()
            except_file = os.path.split(except_traceback.tb_frame.f_code.co_filename)[1]
            exc_dict = {
                "error type": except_type,
                "error info": except_value,
                "error file": except_file,
                "error line": except_traceback.tb_lineno,
            }
            print(exc_dict)
            dist_paths_error.append(img_name)

    os.makedirs(fail_dir, exist_ok=True)
    fail_path = os.path.join(fail_dir, "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(dist_paths_error))
