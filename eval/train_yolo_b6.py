"""
Train YOLOv8-seg on synthetic B6/MKA dataset (5 defect classes).
"""
import argparse
from ultralytics import YOLO
from pathlib import Path

DATASET_YAML = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\b6_yolo_dataset\dataset.yaml")
RESULTS_DIR = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\results\yolo_b6")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== YOLO-seg Training: B6/MKA (5 classes) ===")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch: {args.batch}")
    print(f"  Dataset: {DATASET_YAML}")
    print()

    model = YOLO(args.model)
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(RESULTS_DIR),
        name="train_v1",
        exist_ok=False,
        patience=10,
        save=True,
        plots=True,
        device="cpu",
    )

    print("\n=== Validation ===")
    model.val(data=str(DATASET_YAML))


if __name__ == "__main__":
    main()
