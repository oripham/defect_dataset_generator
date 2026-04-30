"""Optimized YOLO-seg training for demo-quality results."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(r"V:\HondaPlus\defect_dataset_generator")
BATCH_ROOT = ROOT / "batch_output"
RESULTS_ROOT = ROOT / "eval" / "output" / "results" / "yolo_batch"

CONFIGS = [
    {
        "name": "napchai_optimized",
        "dataset_yaml": BATCH_ROOT / "yolo" / "napchai" / "dataset.yaml",
    },
    {
        "name": "mka_optimized",
        "dataset_yaml": BATCH_ROOT / "yolo" / "mka" / "dataset.yaml",
    },
]

MODEL = "yolov8s-seg.pt"
EPOCHS = 100
PATIENCE = 20
IMGSZ = 640
BATCH = 4


def train_one(cfg: dict) -> None:
    from ultralytics import YOLO

    name = cfg["name"]
    dataset_yaml = cfg["dataset_yaml"]
    project_dir = RESULTS_ROOT
    run_dir = project_dir / name

    marker = run_dir / "weights" / "best.pt"
    if marker.exists():
        print(f"SKIP {name} — already done ({marker})", flush=True)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"START {name}  model={MODEL}  epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}", flush=True)
    print(f"{'='*60}", flush=True)

    model = YOLO(MODEL)
    t0 = time.time()

    model.train(
        data=str(dataset_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(project_dir),
        name=name,
        exist_ok=True,
        patience=PATIENCE,
        save=True,
        plots=True,
        workers=0,
        device=0,
        # augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.2,
    )

    elapsed = time.time() - t0
    print(f"\nDONE training {name} in {elapsed/60:.1f} min", flush=True)

    print(f"Running validation on best weights...", flush=True)
    best_model = YOLO(str(marker))
    val_results = best_model.val(
        data=str(dataset_yaml), split="val", imgsz=IMGSZ, batch=BATCH, device=0
    )

    metrics = {
        "mAP50_box": float(val_results.box.map50),
        "mAP50-95_box": float(val_results.box.map),
        "mAP50_mask": float(val_results.seg.map50),
        "mAP50-95_mask": float(val_results.seg.map),
        "epochs": EPOCHS,
        "early_stop_patience": PATIENCE,
        "model": MODEL,
        "imgsz": IMGSZ,
        "training_time_min": elapsed / 60,
    }

    per_class = {}
    class_names = val_results.names
    for i, name_cls in class_names.items():
        per_class[name_cls] = {
            "mAP50_box": float(val_results.box.maps[i]) if i < len(val_results.box.maps) else None,
            "mAP50_mask": float(val_results.seg.maps[i]) if i < len(val_results.seg.maps) else None,
        }
    metrics["per_class"] = per_class

    out_path = run_dir / "optimized_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"Metrics saved → {out_path}", flush=True)
    print(json.dumps(metrics, indent=2, default=str), flush=True)


def main():
    for cfg in CONFIGS:
        try:
            train_one(cfg)
        except Exception as e:
            print(f"FAILED {cfg['name']}: {e}", flush=True)
    print("\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
