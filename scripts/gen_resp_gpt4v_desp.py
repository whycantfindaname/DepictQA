import argparse
import base64
import glob
import json
import os

import weave
from openai import OpenAI

OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"
OPENAI_API_BASE = "https://api.gpt.ge/v1"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

parser = argparse.ArgumentParser(
    description="To Prompt GPT-4 for Image Quality Assessment"
)
parser.add_argument("--meta_dir", type=str, required=True)
parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--desp_dir", type=str, required=True)
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

    meta_dir = args.meta_dir
    meta_paths = sorted(glob.glob(os.path.join(meta_dir, "*.json")))
    desp_dir = args.desp_dir
    fail_dir = args.desp_fail_dir
    image_folder = args.image_folder
    os.makedirs(desp_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    # description_query
    dist_paths_error = []
    for idx_meta, meta_path in enumerate(meta_paths[idx_meta_start:idx_meta_end]):
        # print("=" * 100)
        # print(idx_meta + idx_meta_start)

        meta_name = os.path.basename(meta_path)
        desp_path = os.path.join(desp_dir, meta_name)
        save_path = os.path.join(desp_dir, meta_name)
        if os.path.exists(save_path):
            # print(f"{save_path} has been generated, skip.")
            continue

        with open(meta_path) as fr:
            meta = json.load(fr)
        img_name = meta["filename"]
        img_path = os.path.join(image_folder, img_name)
        dist_class = meta["distortion"]
        score = meta["mos"]

        # description_query
        description_query = (
            "Please provide a brief description of the image, including specific objects and any events. "
            + "Please arrange your answer in the following format: "
            + "Question: the question I asked. "
            + "Answer: your answer to the question. "
        )

        try:
            content = gpt4v(img_path, description_query)
            meta["gpt4v_description"] = content
            with open(save_path, "w") as fw:
                fw.write(json.dumps(meta, indent=4))
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

    fail_path = os.path.join(fail_dir, "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(dist_paths_error))
