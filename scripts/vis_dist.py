from utils import plot_bbox_dist, plot_mos_distribution

if __name__ == "__main__":
    image_folder = "/home/liaowenjie/桌面/画质大模型/datasets/QualityLLM_single_2w"
    json_file = "dataset/clean_data_v1/clean_data.json"
    bbox_dist_path = "results-8k/vis_clean_v1/single_bbox_dist.png"
    mos_path = "results-8k/vis_clean_v1/single_mos_dist.png"
    plot_bbox_dist(json_file, image_folder, bbox_dist_path)
    plot_mos_distribution(json_file, mos_path)
