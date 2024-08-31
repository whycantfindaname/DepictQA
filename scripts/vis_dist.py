from utils import plot_mos_distribution

if __name__ == "__main__":
    train_file = "data/kadid_json/train_kadid_with_mos.json"
    val_file = "data/kadid_json/val_kadid_with_mos.json"
    train_path = "./results/vis/train_kadid_mos_dist.png"
    val_path = "./results/vis/val_kadid_mos_dist.png"
    # plot_bbox_dist(json_file, image_folder, bbox_dist_path)
    # plot_mos_distribution(train_file, train_path)
    # plot_mos_distribution(val_file, val_path)
    plot_mos_distribution("data/clean_data.json", "results/vis/single_mos_dist.png")
    # plot_score_distribution(
    #     "./results/llava/llava_gpt4_score_detail.json",
    #     "./results/llava/llava_score_dist.png",
    # )
