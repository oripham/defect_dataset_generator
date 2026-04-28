"""Train individual YOLO-seg models per napchai class (CPU workaround).
Multi-class YOLO crashes on CPU (SIGILL), but single-class works for ~200 iterations.
Each class: 80 train / 20 val, 10 epochs, batch=4 → ~200 iterations.
"""
import shutil
import random
from pathlib import Path

NAPCHAI = Path(r"V:\HondaPlus\napchai_yolo")
RESULTS_BASE = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\results")
SEED = 42
VAL_RATIO = 0.2


def prepare_single_class_dataset(cls_name, dst_dir):
    random.seed(SEED)
    img_dir = NAPCHAI / cls_name / "images"
    lbl_dir = NAPCHAI / cls_name / "labels"
    images = sorted(img_dir.glob("*"))
    random.shuffle(images)

    n_val = int(len(images) * VAL_RATIO)
    val_imgs = images[:n_val]
    train_imgs = images[n_val:]

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        (dst_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, dst_dir / split / "images" / img.name)
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                text = lbl.read_text().strip()
                lines = []
                for line in text.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        parts[0] = "0"
                        lines.append(" ".join(parts))
                (dst_dir / split / "labels" / (img.stem + ".txt")).write_text(
                    "\n".join(lines) + "\n" if lines else "", encoding="utf-8")
            else:
                (dst_dir / split / "labels" / (img.stem + ".txt")).write_text("", encoding="utf-8")

    yaml_text = f"""path: {dst_dir}
train: train/images
val: val/images

names:
  0: {cls_name}
"""
    (dst_dir / "dataset.yaml").write_text(yaml_text)
    print(f"  {cls_name}: {len(train_imgs)} train, {len(val_imgs)} val")
    return dst_dir / "dataset.yaml"


def train_class(cls_name):
    from ultralytics import YOLO

    dst_dir = NAPCHAI / f"dataset_{cls_name}"
    results_dir = RESULTS_BASE / f"yolo_napchai_{cls_name}"

    if (dst_dir / "dataset.yaml").exists():
        print(f"Dataset for {cls_name} already exists")
    else:
        prepare_single_class_dataset(cls_name, dst_dir)

    yaml_path = str(dst_dir / "dataset.yaml")
    save_dir = str(results_dir / "train_v1")

    if Path(save_dir).exists():
        print(f"Results already exist at {save_dir}, skipping {cls_name}")
        return

    print(f"\nTraining YOLO-seg on {cls_name}...")
    model = YOLO("yolov8n-seg.pt")
    results = model.train(
        data=yaml_path,
        epochs=10,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(results_dir),
        name="train_v1",
        exist_ok=False,
        patience=0,
        verbose=True,
    )
    print(f"  {cls_name} done!")
    return results


if __name__ == "__main__":
    for cls in ["mc_deform", "ring_fracture"]:
        try:
            train_class(cls)
        except Exception as e:
            print(f"  {cls} FAILED: {e}")
