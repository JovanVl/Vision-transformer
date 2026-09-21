
from pathlib import Path
import shutil
import random

def copy_images(image_paths, out_dir: Path) -> int:
    saved = 0
    for i, img_path in enumerate(image_paths):
        try:
            shutil.copy2(img_path, out_dir / f"{i:04d}.jpg")
            saved += 1
        except Exception as e:
            print(f"Unreadable image: {img_path}: {e} | SKIP")
    return saved

def image_dataset_train_test_split(src_path: str, dest_path: str, test_size: float = 0.2):
    assert src_path != dest_path
    src_dir = Path(src_path)
    dst_dir = Path(dest_path)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = dst_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir = dst_dir / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)
    
    classes = sorted(d for d in src_dir.iterdir() if d.is_dir())
    if not classes:
        raise ValueError(f"No subdirectories in {src_dir}")
 
    summary = {}
    for class_dir in classes:
        class_name = class_dir.name
 
        images = list(class_dir.rglob("*.jpg")) + list(class_dir.rglob("*.JPG")) \
            + list(class_dir.rglob("*.jpeg")) + list(class_dir.rglob("*.png"))
        images = sorted(set(images))
 
        if not images:
            print(f"No images found for class {class_name}")
            continue
 
        n_test = int(len(images) * test_size)
        test_images = random.sample(images, n_test)
        test_set = set(test_images)
        train_images = [img for img in images if img not in test_set]
 
        train_class_dir = train_dir / class_name
        test_class_dir = test_dir / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)
 
        train_saved = copy_images(train_images, train_class_dir)
        test_saved = copy_images(test_images, test_class_dir)
 
        summary[class_name] = {"train": train_saved, "test": test_saved}
        print(f"{class_name}: {len(images)} found -> {train_saved} train, {test_saved} test")
 
    return summary