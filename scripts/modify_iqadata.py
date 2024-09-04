import argparse

from utils import modify_iqadata

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, required=True)
parser.add_argument("--test_path", type=str, required=True)
parser.add_argument(
    "--train_output", type=str, required=True
)  # Removed the extra space
parser.add_argument("--val_output", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    train_path = args.train_path
    test_path = args.test_path
    train_output = args.train_output
    val_output = args.val_output
    modify_iqadata(train_path, train_output, True)
    modify_iqadata(test_path, val_output, True)
