import argparse
import base64
import json
import os

import weave
from openai import OpenAI
from utils import assign_level, load_json

OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"
OPENAI_API_BASE = "https://api.gpt.ge/v1/"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

parser = argparse.ArgumentParser(
    description="To Prompt GPT-4 for Image Quality Assessment"
)
parser.add_argument("--meta_file", type=str, required=True)
parser.add_argument("--desp_file", type=str, required=True)
parser.add_argument("--assess_file", type=str, required=True)
parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--assess_fail_dir", type=str, required=True)


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
    print("Encoded image data length:", len(img_base64))  # 添加调试信息，打印前30个字符

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
        max_tokens=230,
    )
    content = resp.choices[0].message.content
    return content


if __name__ == "__main__":
    weave.init("image quality assessment")
    args = parser.parse_args()
    idx_meta_start = 0
    idx_meta_end = -1

    meta_file = args.meta_file
    desp_file = args.desp_file
    assess_file = args.assess_file
    image_folder = args.image_folder
    fail_dir = args.assess_fail_dir

    meta_data = load_json(meta_file)
    if os.path.exists(desp_file):
        desp_data = load_json(desp_file)
    else:
        print("Please generate description first")
    if os.path.exists(assess_file):
        assess_data = load_json(assess_file)
    else:
        assess_data = []

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
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:]):
        img_name = meta_item["filename"]
        if img_name in [item["filename"] for item in assess_data]:
            # print(f"{img_name} has been generated, skip.")
            continue
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)
        desp_item = next(
            (item for item in desp_data if item["filename"] == img_name), None
        )
        if desp_item:
            description_parts = desp_item["gpt4v_description"].split("\n\n")
            description_parts = (
                description_parts
                if len(description_parts) > 1
                else desp_item["gpt4v_description"].split("\n")
            )
            description = (
                description_parts[1]
                .replace("**Answer:** ", "")
                .replace("**Answer:**", "")
                .replace("Answer: ", "")
                .strip()  # This removes leading or trailing spaces
                if len(description_parts) > 1
                else description_parts
            )
            print(description)
        else:
            print(f"{img_name} has no description, please generate description first.")
            continue

        dist_class = meta_item["distortion"]
        score = meta_item["mos"]
        level = assign_level(score)

        if isinstance(dist_class, dict):
            assess_query = (
                "You are an expert in image quality assessment. Your task is to evaluate an image based on a detailed description provided to you. "
                + f"The detailed description is: {description}. "
                + f"The image contains one or more distortions applied to different regions, with a Mean Opinion Score (MOS) of {score} out of 5,where a higher score indicates a lower degree of degradation."
                + "Use only the MOS as the basis for your assessment, and do not output any content related to the MOS."
                + "You will also receive the bounding box information for the distorted regions. "
                + "The bounding box information is in the form of [{'tl':{x, y}, 'br':{x, y}}], where tl is the top-left corner, and br is the bottom-right corner. "
                + f"The distortions present in the image and their locations are as follows: {', '.join([f'{name} with bounding box {bbox}' for name, bbox in dist_class.items()])}. "
                + f"Sample one question from the following list of pool: {question_pool} as the input question."
                + "Please arrange your answer in the following format: "
                + "Question: the question you sampled from the pool "
                + "Answer: your answer to the question."
                + "To answer the question, you should think step by step"
                + "First step, provide a brief description of the image content summarizing the detailed description in one sentence. "
                + "If the detailed description is unavailable, please provide a one-sentence summary explaining why the image content cannot be described."
                + "The description should be in a declarative form, without using the first person."
                + "Second step, describe the location of the distortions in the image and how they affect the quality of specific objects within the bounding box areas. Focus on the local impact of each distortion, describing how it degrades the quality of objects in its vicinity. "
                + "Instead of directly mentioning the bounding box coordinates, utilize this data to describe the location of distortions using natural language. Include details like relative positions between distortions and near-normal regions."
                + "Third step, analyze the severity of each distortion based on the MOS and how they affect the overall image quality. Describe which distortions are most severe and how they contribute to the global degradation, considering their interactions, overlaps, or influence on each other. Discuss how these combined effects impact the viewer's perception of the image."
                + f"Final step, end your answer with this sentence: Thus, the quality of the image is {level}."
                + f"Note that your analysis should be consistent with the provided MOS and the quality level {level} assigned to the image, but do not output any content including the MOS."
                + "When you describe the distortions, you must strictly use the exact distortion names provided, and do not use any variations, synonyms, or alternative expressions."
                + "Your response must be concise and logically consistent, not exceeding 150 words."
            )
        else:
            assess_query = (
                "You are an expert in image quality assessment. Your task is to evaluate an image based on a detailed description provided to you. "
                + f"The detailed description is: {description}. "
                + f"The image contains no distortion or artifact, with a Mean Opinion Score (MOS) of {score} out of 5, where a higher score indicates a lower degree of degradation."
                + "Use only the MOS as the basis for your assessment, and do not output any content related to the MOS."
                + f"Sample one question from the following list of pool: {question_pool} as the input question."
                + "Please arrange your answer in the following format: "
                + "Question: the question you sampled from the pool "
                + "Answer: your answer to the question."
                + "To answer the question, you should think step by step"
                + "First step, provide a brief description of the image content summarizing the detailed description in one sentence. "
                + "Second step,analyze the overall image quality by highlighting the strengths of the image quality."
                + "Focus on the impact on visual quality and clarity, considering details such as texture, color accuracy, and sharpness. Emphasize the absence of artifacts. "
                + f"Final step, end your answer with this sentence: Thus, the quality of the image is {level}."
                + "Your response must be concise and logically consistent, not exceeding 150 words."
            )

        try:
            content = gpt4v(img_path, assess_query)
            meta_item["gpt4v_assessment"] = content
            assess_data.append(meta_item)
            with open(assess_file, "w") as fw:
                json.dump(assess_data, fw, indent=4, ensure_ascii=False)
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
