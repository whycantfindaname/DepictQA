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
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--fail_dir", type=str, required=True)


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


# Weave will track the inputs, outputs and code of this function
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
                    # {
                    #     "type": "text",
                    #     "text": (
                    #         "How can i give you the base64 encoded image which is encoded by 'img_base64 = base64.b64encode(img_file.read()).decode('utf-8')?"
                    #         + "Can i give you like {'type': 'image_url','url': 'data:{mime_type};base64,{img_base64}'}?"
                    #     ),
                    # },
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
    weave.init("image quality assessment")
    args = parser.parse_args()
    idx_meta_start = 0
    idx_meta_end = -1

    meta_dir = args.meta_dir
    meta_paths = sorted(glob.glob(os.path.join(meta_dir, "*.json")))
    save_dir = args.save_dir
    fail_dir = args.fail_dir
    desp_dir = "./data/description"
    image_folder = args.image_folder
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    os.makedirs(desp_dir, exist_ok=True)
    # question pool
    question_pool = [
        "Can you provide a detailed evaluation of the image’s quality",
        "Please evaluate the image’s quality and provide your reasons.",
        "What is your opinion on the quality of the image? Explain your viewpoint.",
        "Analyze the image’s quality, and detail your findings.",
        "How would you rate the overall quality of the image, and why?",
        "What is your opinion on the image’s quality? Elaborate on your evaluation.",
        "How does the image’s quality impact its overall effectiveness or appeal?",
        "How would you rate the image’s quality, and what factors contribute to your assessment?",
        "How do you perceive the quality of the image, and what aspects influence your judgment?",
        "Offer an assessment of the image’s quality, highlighting any strengths or weaknesses.",
        "Assess the quality of the image with detailed reasons.",
        "Evaluate the image’s quality and justify your evaluation.",
        "How about the overall quality of the image, and why?",
        "Provide a thorough evaluation of the image’s quality.",
        "Provide a comprehensive assessment of the image’s quality, including both strengths and areas for improvement.",
        "Assess the image’s quality from a professional standpoint.",
    ]

    # description_query
    dist_paths_error = []
    for idx_meta, meta_path in enumerate(meta_paths[idx_meta_start:idx_meta_end]):
        print("=" * 100)
        print(idx_meta + idx_meta_start)

        meta_name = os.path.basename(meta_path)
        desp_path = os.path.join(desp_dir, meta_name)
        save_path = os.path.join(save_dir, meta_name)
        if os.path.exists(save_path):
            print(f"{save_path} has been generated, skip.")
            continue

        with open(meta_path) as fr:
            meta = json.load(fr)
        img_name = meta["filename"]
        img_path = os.path.join(image_folder, img_name)
        dist_class = meta["distortion"]
        score = meta["mos"]

        try:
            with open(desp_path) as fr:
                description = json.load(fr)["gpt4v_description"]
        except Exception as e:
            print(f"Error: {e}")
            dist_paths_error.append(desp_path)
            continue

        # assess_query
        assess_query = (
            "You are an expert in image quality assessment. You receive an image and a detailed description of the image. "
            + f"The detailed description is: {description}. "
            + "The image is generated by adding one or more distortions in different regions of the image. "
            + f"The Mean Opinion Score of the image is {score} (out of 5). "
            + "The higher the score, the lower the overall degree of degradation. "
            + "You will also receive the bounding box information for the distorted regions. "
            + "The bounding box information is in the form of [{'tl':{x, y}, 'tr':{x, y}, 'br':{x, y}, 'bl':{x, y}}], where tl is the top-left corner, tr is the top-right corner, br is the bottom-right corner, and bl is the bottom-left corner. "
            + f"The distortions present in the image and their locations are as follows: {', '.join([f'{name} with bounding box {bbox}' for name, bbox in dist_class.items()])}. "
            + "Please create a plausible question about the image and provide the answer in detail. "
            + f"Sample one question from the following list of pool: {question_pool} as the input question"
            + "Please arrange your answer in the following format: "
            + "Question: the question you sampled from the pool "
            + "Answer: your answer to the question, which should cover the following three areas. "
            + "First, a short description of the image content which summaries the detailed description, please answer in one sentence. "
            + "Second, The location of the distortions in the image and how they affect the quality of specific objects. Describe the distortions' positions using natural language, and explain their impact on the objects near the distortions. "
            + "Instead of directly mentioning the bounding box coordinates, utilize this data to describe the location of distortions using natural language. Include details like relative position between distortions and near normal regions."
            + "Third, a summary of the overall quality of the evaluated image. Use the mean opinion score as a basis for your assessment but do not tell anything related to the score."
            + "When using the information from the caption and coordinates, directly explain the scene, and do not mention that the information source is the caption or the bounding box. "
            + "The whole response must be below 150 words."
        )

        try:
            content = gpt4v(img_path, assess_query)
            meta["gpt4v_assessment"] = content
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
