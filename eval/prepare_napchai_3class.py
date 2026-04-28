"""
Prepare 3-class napchai dataset for YOLO-seg and PaddleSeg.
Classes: 0=scratch, 1=mc_deform, 2=ring_fracture
Creates train/val split (80/20) with proper class IDs.
Also generates PaddleSeg index masks (0=bg, 1=scratch, 2=mc_deform, 3=ring_fracture).
"""
import shutil
import random
import numpy as np
import cv2
from pathlib import Path

NAPCHAI = Path(r"V:\HondaPlus\napchai_yolo")
DST_YOLO = Path(r"V:\HondaPlus\napchai_yolo\dataset_3class")
DST_PADDLE = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\paddleseg_napchai_3class")
CLASSES = {"scratch": 0, "mc_deform": 1, "ring_fracture": 2}
VAL_RATIO = 0.2
SEED = 42


def remap_class_id(label_path, new_class_id):
    text = label_path.read_text().strip()
    if not text:
        return ""
    lines = []
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 7:
            parts[0] = str(new_class_id)
            lines.append(" ".join(parts))
    return "\n".join(lines) + "\n" if lines else ""


def mask_to_index(mask_path, class_id):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    idx = np.zeros_like(mask, dtype=np.uint8)
    idx[mask > 127] = class_id
    return idx


def main():
    random.seed(SEED)

    for split in ["train", "val"]:
        (DST_YOLO / split / "images").mkdir(parents=True, exist_ok=True)
        (DST_YOLO / split / "labels").mkdir(parents=True, exist_ok=True)
        (DST_PADDLE / split / "images").mkdir(parents=True, exist_ok=True)
        (DST_PADDLE / split / "masks").mkdir(parents=True, exist_ok=True)

    all_data = []
    for cls_name, cls_id in CLASSES.items():
        img_dir = NAPCHAI / cls_name / "images"
        # Use v2 labels/masks for ring_fracture (re-annotated)
        if cls_name == "ring_fracture":
            lbl_dir = NAPCHAI / cls_name / "labels_v2"
            mask_dir = NAPCHAI / cls_name / "masks_v2"
        else:
            lbl_dir = NAPCHAI / cls_name / "labels"
            mask_dir = NAPCHAI / cls_name / "masks"
        images = sorted(img_dir.glob("*"))
        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            mask = mask_dir / (img.stem + "_mask.png")
            all_data.append((img, lbl, mask, cls_name, cls_id))

    random.shuffle(all_data)
    n_val = int(len(all_data) * VAL_RATIO)
    val_data = all_data[:n_val]
    train_data = all_data[n_val:]

    print(f"Total: {len(all_data)}, Train: {len(train_data)}, Val: {len(val_data)}")

    for split, data in [("train", train_data), ("val", val_data)]:
        for img_path, lbl_path, mask_path, cls_name, cls_id in data:
            out_name = f"{cls_name}_{img_path.stem}"
            shutil.copy2(img_path, DST_YOLO / split / "images" / (out_name + img_path.suffix))

            if lbl_path.exists():
                remapped = remap_class_id(lbl_path, cls_id)
                (DST_YOLO / split / "labels" / (out_name + ".txt")).write_text(remapped, encoding="utf-8")
            else:
                (DST_YOLO / split / "labels" / (out_name + ".txt")).write_text("", encoding="utf-8")

            shutil.copy2(img_path, DST_PADDLE / split / "images" / (out_name + img_path.suffix))
            if mask_path.exists():
                idx = mask_to_index(mask_path, cls_id + 1)
                if idx is not None:
                    cv2.imwrite(str(DST_PADDLE / split / "masks" / (out_name + ".png")), idx)

    yaml = f"""path: {DST_YOLO}
train: train/images
val: val/images

names:
  0: scratch
  1: mc_deform
  2: ring_fracture
"""
    (DST_YOLO / "dataset.yaml").write_text(yaml)

    # PaddleSeg file lists
    for split in ["train", "val"]:
        imgs = sorted((DST_PADDLE / split / "images").glob("*"))
        lines = []
        for img in imgs:
            mask = DST_PADDLE / split / "masks" / (img.stem + ".png")
            lines.append(f"{split}/images/{img.name} {split}/masks/{mask.name}")
        (DST_PADDLE / f"{split}.txt").write_text("\n".join(lines) + "\n")

    # Stats
    for split in ["train", "val"]:
        per_class = {c: 0 for c in CLASSES}
        empty = 0
        for lbl in (DST_YOLO / split / "labels").glob("*.txt"):
            text = lbl.read_text().strip()
            if not text:
                empty += 1
                continue
            for line in text.split("\n"):
                cid = int(line.split()[0])
                for name, idx in CLASSES.items():
                    if idx == cid:
                        per_class[name] += 1
        print(f"\n{split}:")
        for name, cnt in per_class.items():
            print(f"  {name}: {cnt} labels")
        print(f"  empty (bg): {empty}")


if __name__ == "__main__":
    main()
