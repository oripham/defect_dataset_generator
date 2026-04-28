"""
Prepare synthetic B6/MKA data for YOLO segmentation training.
Creates train/val split (80/20) + dataset.yaml from b6_full_100.
"""
import shutil
import random
from pathlib import Path

SRC = Path(r"V:\defect_samples\results\cap\b6_full_100")
DST = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\b6_yolo_dataset")

CLASSES = ["dark", "dent", "plastic", "scratch", "thread"]
VAL_RATIO = 0.2
SEED = 42


def main():
    random.seed(SEED)

    for split in ["train", "val"]:
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    images = sorted(SRC.glob("images/*.png"))
    if not images:
        images = sorted(SRC.glob("images/*.jpg"))
    print(f"Found {len(images)} images")

    random.shuffle(images)
    n_val = int(len(images) * VAL_RATIO)
    val_imgs = images[:n_val]
    train_imgs = images[n_val:]

    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
        for img_path in img_list:
            label_path = SRC / "labels" / (img_path.stem + ".txt")
            shutil.copy2(img_path, DST / split / "images" / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, DST / split / "labels" / label_path.name)

    yaml_content = f"""path: {DST}
train: train/images
val: val/images

names:
"""
    for i, name in enumerate(CLASSES):
        yaml_content += f"  {i}: {name}\n"

    (DST / "dataset.yaml").write_text(yaml_content)
    print(f"dataset.yaml written to {DST / 'dataset.yaml'}")

    for split in ["train", "val"]:
        n = len(list((DST / split / "images").iterdir()))
        print(f"  {split}: {n} images")


if __name__ == "__main__":
    main()
