import argparse
import base64
import glob
import json
import os

from openai import OpenAI

API_KEY = ""

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument("--meta_dir", type=str, required=True)
parser.add_argument("--pred_json", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--fail_dir", type=str, required=True)


def encode_img(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def gpt4v(
    ref_path,
    imgA_path,
    imgB_path,
    distA_classes,
    severities_A,
    distB_classes,
    severities_B,
    res_compare,
):
    grades_multi = ["slight", "obvious", "serious"]
    grades_multi_str = "[slight, obvious, serious]"
    grades_single = ["slight", "moderate", "obvious", "serious", "catastrophic"]
    grades_single_str = "[slight, moderate, obvious, serious, catastrophic]"

    ref_base64 = encode_img(ref_path)
    imgA_base64 = encode_img(imgA_path)
    imgB_base64 = encode_img(imgB_path)

    if severities_A == 0:
        assert distA_classes is None
        str_distA = "Image A is a high quality image with no distortions. "
    elif len(severities_A) == 1:
        assert len(distA_classes) == 1
        gradeA = grades_single[severities_A[0] - 1]
        str_distA = f"In Image A, the added distortion is {distA_classes[0]} with grade {gradeA} (out of {grades_single_str}). "
    else:
        assert len(severities_A) == 2 and len(distA_classes) == 2
        gradeA0 = grades_multi[severities_A[0] - 1]
        gradeA1 = grades_multi[severities_A[1] - 1]
        str_distA = (
            f"In Image A, the added distortions are (1) {distA_classes[0]} with grade {gradeA0} (out of {grades_multi_str}), "
            + f"and (2) {distA_classes[1]} with grade {gradeA1} (out of {grades_multi_str}). "
        )
    if severities_B == 0:
        assert distB_classes is None
        str_distB = "Image B is a high quality image with no distortions. "
    elif len(severities_B) == 1:
        gradeB = grades_single[severities_B[0] - 1]
        str_distB = f"In Image B, the added distortion is {distB_classes[0]} with grade {gradeB} (out of {grades_single_str}). "
    else:
        assert len(severities_B) == 2 and len(distB_classes) == 2
        gradeB0 = grades_multi[severities_B[0] - 1]
        gradeB1 = grades_multi[severities_B[1] - 1]
        str_distB = (
            f"In Image B, the added distortions are (1) {distB_classes[0]} with grade {gradeB0} (out of {grades_multi_str}), "
            + f"and (2) {distB_classes[1]} with grade {gradeB1} (out of {grades_multi_str}). "
        )
    query = (
        "You are an expert in image quality assessment. "
        + "The first image is a reference image, the second image is Image A, and the third image is Image B. "
        + "Image A and Image B are generated by adding distortions into the reference. "
        + str_distA
        + str_distB
        + f"The comparison result of an assisstant model is Image {res_compare} has better quality, and ignore this result if it is obviously wrong. "
        + "Please compare the quality of Image A and Image B. "
        + "The response should cover three areas. "
        + "First, a short description of the image content. "
        + "Second, distortion identification in Image A and Image B and discussion on how these distortions affect the image content. "
        + "Third, compare the quality of Image A and Image B with an explicit conclusion that which one is better. "
        + "The response must not show that you were given the reference. "
        + "The whole response must be below 170 words."
    )

    client = OpenAI(api_key=API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpg;base64,{ref_base64}",
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpg;base64,{imgA_base64}",
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpg;base64,{imgB_base64}",
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
    args = parser.parse_args()
    idx_meta_start = 0
    idx_meta_end = 5

    meta_dir = args.meta_dir
    meta_paths = sorted(glob.glob(os.path.join(meta_dir, "*.json")))

    pred_dict = {}
    pred_json = args.pred_json
    with open(pred_json) as fr:
        preds = json.load(fr)
    for pred in preds:
        assert pred["id"] not in pred_dict
        if pred["text"].strip() == "Image A":
            pred_dict[pred["id"]] = "A"
        else:
            assert pred["text"].strip() == "Image B"
            pred_dict[pred["id"]] = "B"
    for meta_path in meta_paths:
        meta_id = os.path.splitext(os.path.basename(meta_path))[0]
        assert meta_id in pred_dict

    save_dir = args.save_dir
    fail_dir = args.fail_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    meta_paths_error = []
    for idx_meta, meta_path in enumerate(meta_paths[idx_meta_start:idx_meta_end]):
        print("=" * 100)
        print(idx_meta + idx_meta_start)

        meta_name = os.path.basename(meta_path)
        save_path = os.path.join(save_dir, meta_name)
        if os.path.exists(save_path):
            print(f"{save_path} has been generated, skip.")
            continue

        with open(meta_path) as fr:
            meta = json.load(fr)

        ref_path = meta["img_ref"]
        imgA_path = meta["img_lq_A"]["img_path"]
        imgB_path = meta["img_lq_B"]["img_path"]
        distA_class = meta["img_lq_A"]["distortion_classes"]
        severityA = meta["img_lq_A"]["severities"]
        distB_class = meta["img_lq_B"]["distortion_classes"]
        severityB = meta["img_lq_B"]["severities"]
        res_compare = pred_dict[meta["id"]]

        try:
            content = gpt4v(
                ref_path,
                imgA_path,
                imgB_path,
                distA_class,
                severityA,
                distB_class,
                severityB,
                res_compare,
            )
            meta["text"] = content
            with open(save_path, "w") as fw:
                fw.write(json.dumps(meta, indent=4))
            print(meta)
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
            meta_paths_error.append(meta_path)

    fail_path = os.path.join(fail_dir, "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(meta_paths_error))
