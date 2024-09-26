from utils import plot_bbox_dist, plot_mos_distribution, plot_score_distribution

if __name__ == "__main__":
    # image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/images/QualityLLM_single_2w"
    # json_file = "dataset/clean_data_v0/clean_data.json"
    # bbox_dist_path = "results-8k/vis_clean_v0/single_bbox_dist.png"
    # mos_path = "results-8k/vis_clean_v1/single_mos_dist.png"
    # plot_bbox_dist(json_file, image_folder)
    # #plot_mos_distribution(json_file, mos_path)

    gpt_score_path = 'results-8k/qwen_with_bbox_val_gpt4_score_sample2.json'
    score_path = "results-8k/vis_clean_v0/qwen_with_bbox_score_dist_sample2.png"
    plot_score_distribution(gpt_score_path, score_path)