"""
Prepare YOLO segmentation dataset: split into train/val and create dataset.yaml.
Usage: python prepare_yolo_dataset.py
"""
import shutil
import random
from pathlib import Path

SEED = 42
VAL_RATIO = 0.2

SRC_DIR = Path(r"V:\HondaPlus\napchai_yolo\scratch")
DATASET_DIR = Path(r"V:\HondaPlus\napchai_yolo\scratch_dataset")

CLASS_NAMES = {0: "scratch"}


def main():
    random.seed(SEED)

    images = sorted((SRC_DIR / "images").glob("*.png"))
    random.shuffle(images)

    n_val = int(len(images) * VAL_RATIO)
    val_images = images[:n_val]
    train_images = images[n_val:]

    print(f"Total: {len(images)} | Train: {len(train_images)} | Val: {len(val_images)}")

    for split, img_list in [("train", train_images), ("val", val_images)]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            shutil.copy2(img_path, img_dir / img_path.name)
            lbl_path = SRC_DIR / "labels" / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dir / lbl_path.name)

    yaml_content = f"""path: {DATASET_DIR}
train: train/images
val: val/images

names:
  0: scratch
"""
    (DATASET_DIR / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\nDataset ready: {DATASET_DIR}")
    print(f"dataset.yaml created")


if __name__ == "__main__":
    main()
