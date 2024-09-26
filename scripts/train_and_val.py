import argparse

from utils import construct_chat_template, split_train_and_val

parser = argparse.ArgumentParser(description="split train and val data")
parser.add_argument("--meta_file", type=str, required=True)
parser.add_argument("--desp_file", type=str, required=True)
parser.add_argument("--assess_file", type=str, required=True)
parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--keep_mos", type=str, required=True)
parser.add_argument("--bbox_provided", required=True)

if __name__ == "__main__":
    # desp_file = "data/description.json"
    # assess_file = "data/assessment.json"
    # meta_file = "data/clean_data.json"
    # save_folder = "data/single5k_kadid10k"
    # image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
    # output_file = "results/qwen_single/qwen_mos.json"

    args = parser.parse_args()
    meta_file = args.meta_file
    desp_file = args.desp_file
    assess_file = args.assess_file
    image_folder = args.image_folder
    output_file = args.output_file
    keep_mos = args.keep_mos
    bbox_provided = args.bbox_provided
    construct_chat_template(
        desp_file,
        assess_file,
        meta_file,
        image_folder,
        output_file,
        "qwen-vl",
        keep_mos,
        bbox_provided,
    )
    split_train_and_val(output_file)
