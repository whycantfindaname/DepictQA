from utils import modify_kadid

if __name__ == "__main__":
    # folder_path = "data/kadid10k"

    # for i in range(1, 82):
    #     file_name = f"I{i:02}.png"
    #     file_path = os.path.join(folder_path, file_name)

    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    #         print(f"Deleted: {file_name}")
    #     else:
    #         print(f"File not found: {file_name}")

    train_path = "data/kadid_json/train_kadid.json"
    test_path = "data/kadid_json/test_kadid.json"
    train_output = "data/kadid_json/train_kadid_no_mos.json"
    val_output = "data/kadid_json/val_kadid_no_mos.json"
    modify_kadid(train_path, train_output)
    modify_kadid(test_path, val_output)
