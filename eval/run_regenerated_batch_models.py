from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"V:\HondaPlus\defect_dataset_generator")
EVAL_DIR = ROOT / "eval"
BATCH_ROOT = ROOT / "batch_output"
OUT_DIR = EVAL_DIR / "output"
RESULTS_ROOT = OUT_DIR / "results"
LOG_DIR = RESULTS_ROOT / "batch_pipeline_logs"
ANOM_DATA_ROOT = OUT_DIR / "anomalib_batch_data"
PYTHON = sys.executable


def run_subprocess_step(name: str, payload: str, log_path: Path, success_marker: Path | None = None) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if success_marker is not None and success_marker.exists():
        print(f"SKIP {name} marker={success_marker}", flush=True)
        return True
    print(f"START {name}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            [PYTHON, "-c", payload],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        print(f"FAILED {name} rc={proc.returncode} log={log_path}", flush=True)
        return False
    print(f"DONE {name} log={log_path}", flush=True)
    return True


def run_logged_step(name: str, log_path: Path, fn, success_marker: Path | None = None) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if success_marker is not None and success_marker.exists():
        print(f"SKIP {name} marker={success_marker}", flush=True)
        return True
    print(f"START {name}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                fn()
    except Exception as exc:
        print(f"FAILED {name} error={exc} log={log_path}", flush=True)
        return False
    print(f"DONE {name} log={log_path}", flush=True)
    return True


def build_yolo_payload(*, dataset_yaml: Path, project_dir: Path, run_name: str, model_path: str, epochs: int, imgsz: int, batch: int) -> str:
    return rf'''
from ultralytics import YOLO

model = YOLO(r"{model_path}")
model.train(
    data=r"{dataset_yaml}",
    epochs={epochs},
    imgsz={imgsz},
    batch={batch},
    project=r"{project_dir}",
    name="{run_name}",
    exist_ok=True,
    patience=10,
    save=True,
    plots=True,
    workers=0,
    device=0,
)
model.val(data=r"{dataset_yaml}", split="val", imgsz={imgsz}, batch={batch}, device=0)
'''


def run_yolo_staged(name: str, dataset_yaml: Path, base_run_name: str, stages: int, imgsz: int = 512, batch: int = 2) -> bool:
    project_dir = RESULTS_ROOT / "yolo_batch"
    model_path = "yolov8n-seg.pt"
    ok = True
    for stage in range(1, stages + 1):
        run_name = f"{base_run_name}_stage{stage:02d}"
        run_dir = project_dir / run_name
        success_marker = run_dir / "weights" / "best.pt"
        if success_marker.exists():
            print(f"SKIP {name}_stage{stage:02d} marker={success_marker}", flush=True)
            model_path = str(success_marker)
            continue
        payload = build_yolo_payload(
            dataset_yaml=dataset_yaml,
            project_dir=project_dir,
            run_name=run_name,
            model_path=model_path,
            epochs=50,
            imgsz=imgsz,
            batch=batch,
        )
        step_ok = run_subprocess_step(
            f"{name}_stage{stage:02d}",
            payload,
            LOG_DIR / f"{name}_stage{stage:02d}.log",
            success_marker=success_marker,
        )
        if not step_ok or not success_marker.exists():
            ok = False
            break
        model_path = str(success_marker)
    return ok


NAPCHAI_PADDLE_PAYLOAD = rf'''
import sys
from pathlib import Path
sys.path.insert(0, r"{EVAL_DIR}")
import run_paddleseg_napchai_3class as m
m.DATA_DIR = Path(r"{BATCH_ROOT / 'paddleseg' / 'napchai'}")
m.RESULTS_DIR = Path(r"{RESULTS_ROOT / 'paddleseg_batch' / 'napchai'}")
m.EPOCHS = 30
m.main()
'''


MKA_PADDLE_PAYLOAD = rf'''
import sys
from pathlib import Path
sys.path.insert(0, r"{EVAL_DIR}")
import run_paddleseg_napchai_3class as m
m.DATA_DIR = Path(r"{BATCH_ROOT / 'paddleseg' / 'mka'}")
m.RESULTS_DIR = Path(r"{RESULTS_ROOT / 'paddleseg_batch' / 'mka'}")
m.NUM_CLASSES = 7
m.CLASS_NAMES = {{
    0: "background",
    1: "scratch",
    2: "dent",
    3: "dark_spots",
    4: "thread",
    5: "can_mieng",
    6: "plastic_flow",
}}
m.EPOCHS = 30
m.generate_predictions = lambda model, results_dir: print("Skipped prediction export for MKA")
m.main()
'''


def copy_pngs(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("*.png")):
        shutil.copy2(path, dst / path.name)


def prepare_napchai_anomalib_dataset(class_name: str) -> Path:
    source_root = BATCH_ROOT / "anomalib" / "napchai"
    dataset_root = ANOM_DATA_ROOT / "napchai" / class_name
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    train_good = dataset_root / "train" / "good"
    test_good = dataset_root / "test" / "good"
    test_defective = dataset_root / "test" / "defective"
    gt_defective = dataset_root / "ground_truth" / "defective"

    copy_pngs(source_root / "train" / "good", train_good)
    copy_pngs(source_root / "test" / "good", test_good)
    copy_pngs(source_root / "test" / class_name, test_defective)
    copy_pngs(source_root / "ground_truth" / class_name, gt_defective)
    return dataset_root


def prepare_mka_anomalib_dataset() -> Path:
    source_root = BATCH_ROOT / "anomalib" / "mka"
    dataset_root = ANOM_DATA_ROOT / "mka"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    train_good = dataset_root / "train" / "good"
    test_good = dataset_root / "test" / "good"
    test_defective = dataset_root / "test" / "defective"
    gt_defective = dataset_root / "ground_truth" / "defective"

    copy_pngs(source_root / "train" / "good", train_good)
    copy_pngs(source_root / "test" / "good", test_good)
    test_defective.mkdir(parents=True, exist_ok=True)
    gt_defective.mkdir(parents=True, exist_ok=True)

    for defect_dir in sorted((source_root / "test").iterdir()):
        if not defect_dir.is_dir() or defect_dir.name == "good":
            continue
        for img_path in sorted(defect_dir.glob("*.png")):
            shutil.copy2(img_path, test_defective / img_path.name)
        gt_dir = source_root / "ground_truth" / defect_dir.name
        for gt_path in sorted(gt_dir.glob("*.png")):
            shutil.copy2(gt_path, gt_defective / gt_path.name)

    return dataset_root


def run_patchcore(dataset_root: Path, results_dir: Path, dataset_name: str) -> dict:
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.metrics import AUROC, F1Max, Evaluator, create_anomalib_metric
    from anomalib.models import Patchcore
    from torchmetrics.classification import BinaryJaccardIndex

    results_dir.mkdir(parents=True, exist_ok=True)
    iou_metric = create_anomalib_metric(BinaryJaccardIndex)
    evaluator = Evaluator(
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Max(fields=["pred_score", "gt_label"], prefix="image_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            iou_metric(fields=["pred_mask", "gt_mask"], prefix="pixel_"),
        ],
    )
    datamodule = Folder(
        name=dataset_name,
        root=str(dataset_root),
        normal_dir="train/good",
        abnormal_dir="test/defective",
        normal_test_dir="test/good",
        mask_dir="ground_truth/defective",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
        test_split_mode="from_dir",
        val_split_mode="from_test",
        val_split_ratio=0.5,
    )
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        evaluator=evaluator,
    )
    engine = Engine(
        default_root_dir=str(results_dir),
        devices=1,
        accelerator="auto",
    )
    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(model=model, datamodule=datamodule)
    metrics = test_results[0] if test_results else {}
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    return metrics


def run_anomalib_napchai() -> None:
    results_base = RESULTS_ROOT / "anomalib_batch" / "napchai"
    all_metrics: dict[str, dict] = {}
    for class_name in ["scratch", "mc_deform", "ring_fracture"]:
        dataset_root = prepare_napchai_anomalib_dataset(class_name)
        metrics = run_patchcore(dataset_root, results_base / class_name, f"napchai_{class_name}")
        all_metrics[class_name] = metrics
    (results_base / "all_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, default=str), encoding="utf-8"
    )


def run_anomalib_mka() -> None:
    results_dir = RESULTS_ROOT / "anomalib_batch" / "mka"
    dataset_root = prepare_mka_anomalib_dataset()
    run_patchcore(dataset_root, results_dir, "mka_batch")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Anomalib Napchai
    run_logged_step(
        "anomalib_napchai",
        LOG_DIR / "anomalib_napchai.log",
        run_anomalib_napchai,
        success_marker=RESULTS_ROOT / "anomalib_batch" / "napchai" / "all_metrics.json",
    )
    # 2. Anomalib MKA
    run_logged_step(
        "anomalib_mka",
        LOG_DIR / "anomalib_mka.log",
        run_anomalib_mka,
        success_marker=RESULTS_ROOT / "anomalib_batch" / "mka" / "metrics.json",
    )
    # 3. YOLO Napchai
    run_yolo_staged(
        name="yolo_napchai",
        dataset_yaml=BATCH_ROOT / "yolo" / "napchai" / "dataset.yaml",
        base_run_name="napchai_batch_safe",
        stages=1,
        imgsz=512,
        batch=2,
    )
    # 4. YOLO MKA
    run_yolo_staged(
        name="yolo_mka",
        dataset_yaml=BATCH_ROOT / "yolo" / "mka" / "dataset.yaml",
        base_run_name="mka_batch_safe",
        stages=1,
        imgsz=512,
        batch=2,
    )
    # 5. PaddleSeg Napchai
    run_subprocess_step(
        "paddleseg_napchai",
        NAPCHAI_PADDLE_PAYLOAD,
        LOG_DIR / "paddleseg_napchai.log",
        success_marker=RESULTS_ROOT / "paddleseg_batch" / "napchai" / "metrics.json",
    )
    # 6. PaddleSeg MKA
    run_subprocess_step(
        "paddleseg_mka",
        MKA_PADDLE_PAYLOAD,
        LOG_DIR / "paddleseg_mka.log",
        success_marker=RESULTS_ROOT / "paddleseg_batch" / "mka" / "metrics.json",
    )

    print(f"PIPELINE_DONE logs={LOG_DIR}", flush=True)


if __name__ == "__main__":
    main()
