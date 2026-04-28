"""
Train YOLOv8-seg on circular metal scratch dataset.
Usage: python train_yolo_scratch.py [--epochs 50] [--imgsz 640]
"""
import argparse
from ultralytics import YOLO
from pathlib import Path

DATASET_YAML = Path(r"V:\HondaPlus\napchai_yolo\scratch_dataset\dataset.yaml")
RESULTS_DIR = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\results\yolo_scratch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== YOLO Segmentation Training: Scratch ===")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch: {args.batch}")
    print(f"  Dataset: {DATASET_YAML}")
    print()

    model = YOLO(args.model)

    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(RESULTS_DIR),
        name="train_v3",
        exist_ok=False,
        patience=10,
        save=True,
        plots=True,
        device="cpu",
    )

    print("\n=== Validation ===")
    val_results = model.val(data=str(DATASET_YAML))

    print(f"\nResults saved to {RESULTS_DIR / 'train'}")
    print("Best model: best.pt")


if __name__ == "__main__":
    main()
