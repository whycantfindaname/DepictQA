from utils import construct_chat_template

if __name__ == "__main__":
    desp_file = "data/description.json"
    assess_file = "data/assessment.json"
    meta_file = "data/clean_data.json"
    save_folder = "data/test_image"
    image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
    output_file = "results/test_llava/llava.json"

    construct_chat_template(
        desp_file,
        assess_file,
        meta_file,
        image_folder,
        save_folder,
        output_file,
        "llava-interleave",
    )
    # split_train_and_val(output_file)
