import argparse
import json
import os

import weave
from openai import OpenAI

OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"
OPENAI_API_BASE = "https://api.gpt.ge/v1"


parser = argparse.ArgumentParser(description="evaluation parameters for DepictQA")
parser.add_argument("--pred_path", type=str, required=True)
parser.add_argument("--gt_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)


DEFAULT_SETTINGS = {
   "system_prompt": "You are a helpful and understanding assistant for providing a balanced assessment of the answer quality.",
    "prompt": "We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. "
    + "The user asks a question about assessing image quality. "
    + "The ground truth is provided for reference, but your evaluation should focus on the overall usefulness and relevance of the assistant's response. "
    + "While evaluating, consider the image description, distortion analysis, and quality judgment, but allow for flexibility and reasonable variations in interpretation. "
    + "You should also pay attention to logic of the assistant's response, and ensure that the assistant is able to provide a clear and concise response. "
    + "The assistant should be given an overall score on a scale of 0 to 10, where a higher score reflects better performance. "
    + "Note that if the response contains inconsistencies in language (e.g., switching between different languages), unnecessary repetition, or unnatural phrasing, the score must be 0. "
    + "If the response lacks logical coherence, or is incomplete, the score must also be 0. "
    + "Please output a single line containing ONLY ONE INT NUMBER indicating the score of the assistant."
    # + "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.\n",
}


def parse_score(review):
    try:
        score = review.split("\n")[0].strip()
        try:
            return float(score)
        except:
            print("error", review)
            return -1
    except Exception as e:
        print(e)
        print("error", review)
        return -1


@weave.op()
def gen_res_from_gpt(content):
    gpt_model = "gpt-4-turbo"

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {
                "role": "system",
                "content": DEFAULT_SETTINGS["system_prompt"],
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        temperature=0.0,
    )
    res = response.choices[0].message.content
    return res


if __name__ == "__main__":
    weave.init("generate gpt score")
    args = parser.parse_args()

    # load predict results
    pred_path = args.pred_path
    pred_images = []
    pred_answers = []
    with open(pred_path) as fr:
        pred_metas = json.load(fr)

    for entry in pred_metas:
        if os.path.dirname(entry["image"]) == "QualityLLM_single_2w":
            pred_images.append(entry["image"])
            pred_answers.append(entry["conversation"][3]["value"])
    # load gt results
    with open(args.gt_path) as fr:
        gt_metas = json.load(fr)
    gt_images = []
    gt_answers = []
    gt_questions = []
    for entry in gt_metas:
        if os.path.dirname(entry["image"]) == "QualityLLM_single_2w":
            gt_images.append(entry["image"])
            gt_answers.append(entry["conversations"][5]["value"])
            gt_questions.append(entry["conversations"][4]["value"])


    # check if the two lists are the same
    assert pred_images == gt_images
    assert len(pred_answers) == len(gt_answers), f"{len(pred_answers)} vs {len(gt_answers)}"
    # Read existing review file content
    try:
        with open(args.save_path, "r") as review_file:
            content = json.load(review_file)
            handled_images = {line["image"] for line in content}
            scores = [line["score"] for line in content]
            print(f"Handled images: {handled_images}")
    except FileNotFoundError:
        handled_images = set()

    # generate review results
    scores = []
    save_path = args.save_path

    review_file = open(save_path, "a")

    results = []

    for idx in range(len(pred_answers)):
        assert pred_images[idx] == gt_images[idx]
        pred = {"image": pred_images[idx], "answer": pred_answers[idx]}
        gt = {"question": gt_questions[idx], "answer": gt_answers[idx]}
        image = pred["image"]

        if image in handled_images:
            print(f"Image {image} has been handled")
            continue

        print("=" * 100)
        print(f"Handling {image}")

        question = gt["question"]
        answer_gt = gt["answer"]
        answer_pred = pred["answer"]

        content = (
            f"[Question]\n{question}\n\n"
            f"[Ground Truth]\n{answer_gt}\n\n[End of Ground Truth]\n\n"
            f"[Assistant]\n{answer_pred}\n\n[End of Assistant]\n\n"
            f"[System]\n{DEFAULT_SETTINGS['prompt']}\n\n"
        )

        cur_js = {
            "image": image,
            "question": question,
            "answer_gt": answer_gt,
            "answer_pred": answer_pred,
        }

        review = gen_res_from_gpt(content)
        score = float(review)
        scores.append(score)
        # cur_js["content"] = review
        cur_js["score"] = score
        print(cur_js)

        # 将当前结果添加到结果列表中
        results.append(cur_js)

        # 将所有结果保存为列表形式的JSON文件
        with open(save_path, "w") as review_file:
            json.dump(results, review_file, ensure_ascii=False, indent=4)

    scores.extend([_ for _ in scores if _ >= 0])
    score = sum(scores) / len(scores)
    print(len(scores))
    print(f"GPT4 Score: {score}")
