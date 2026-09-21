from pathlib import Path

from src.data.preprocessing import preprocess
from src.utils.seed import set_seed

if __name__ == "__main__":
    set_seed()
    preprocess(
        src_dir=Path("../data/raw"),
        dst_dir=Path("../data/processed"),
        img_size=224
    )