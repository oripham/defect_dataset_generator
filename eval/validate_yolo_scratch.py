"""Validate saved YOLO model and generate predictions."""
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\results\yolo_scratch\train\weights\best.pt")
DATASET_YAML = Path(r"V:\HondaPlus\napchai_yolo\scratch_dataset\dataset.yaml")
RESULTS_DIR = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\results\yolo_scratch")

print(f"Loading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

print("\n=== Validation ===")
results = model.val(data=str(DATASET_YAML), device="cpu")

print(f"\nBox  mAP50: {results.box.map50:.4f}")
print(f"Box  mAP50-95: {results.box.map:.4f}")
print(f"Mask mAP50: {results.seg.map50:.4f}")
print(f"Mask mAP50-95: {results.seg.map:.4f}")

print("\n=== Predictions on val set ===")
val_imgs = sorted((Path(r"V:\HondaPlus\napchai_yolo\scratch_dataset\val\images")).glob("*.png"))
pred_dir = RESULTS_DIR / "predictions"
pred_dir.mkdir(parents=True, exist_ok=True)

for img_path in val_imgs[:10]:
    preds = model.predict(str(img_path), save=True, project=str(pred_dir), name="vis",
                          exist_ok=True, device="cpu", conf=0.25)
    n_masks = len(preds[0].masks.data) if preds[0].masks is not None else 0
    print(f"  {img_path.name}: {n_masks} detections")

print(f"\nResults saved to {RESULTS_DIR}")
print(f"Best model: {MODEL_PATH}")
