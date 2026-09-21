from pathlib import Path

from src.data.image_dataset_train_test_split import image_dataset_train_test_split;
from src.utils.seed import set_seed

if __name__ == "__main__":
    set_seed()
    image_dataset_train_test_split(
        src_path=Path("../data/processed"),
        dest_path=Path("../data/training"),
        test_size=0.2
    )