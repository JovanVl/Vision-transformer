from pathlib import Path
import shutil
from PIL import Image
from utils.seed import set_seed

set_seed()

def preprocess(src_dir: Path, dst_dir: Path, img_size: int = 224) -> dict:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    classes = sorted(d for d in src_dir.iterdir() if d.is_dir())
    if not classes:
        raise ValueError(f'No subdirectories in {src_dir}')

    summary = {}
    for class_dir in classes:
        class_name = class_dir.name

        images = list(class_dir.rglob("*.jpg")) + list(class_dir.rglob("*.JPG")) \
            + list(class_dir.rglob("*.jpeg")) + list(class_dir.rglob("*.png"))
        images = sorted(set(images))

        if not images:
            print(f'No images found for class {class_name}')
            continue

        dst_class_dir = dst_dir / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, img_path in enumerate(images):
            try:
                img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
                img.save(dst_class_dir / f"{i:04d}.jpg")
                saved += 1
            except Exception as e:
                print(f"  Skipping unreadable image {img_path}: {e}")

        summary[class_name] = saved
        print(f"{class_name}: {len(images)} images found -> {saved} saved")

    return summary


def clear_processed_data(dst_dir: Path):
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)