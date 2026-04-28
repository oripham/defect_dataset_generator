"""
Anomalib evaluation — PatchCore on B6/MKA defect dataset.
Unsupervised anomaly detection: trains on OK images, tests on defect images.
Outputs pixel-level AUROC, F1Max, IoU.
"""
import sys
import json
from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.metrics import AUROC, F1Max, Evaluator, create_anomalib_metric
from torchmetrics.classification import BinaryJaccardIndex

from eval_config import ANOMALIB_DIR, RESULTS_DIR as _RESULTS_BASE

DATASET_ROOT = ANOMALIB_DIR
RESULTS_DIR  = _RESULTS_BASE / "anomalib"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Anomalib PatchCore Evaluation ===\n")

    # 1. Dataset
    print("Setting up Folder dataset...")
    datamodule = Folder(
        name="mka_defect",
        root=str(DATASET_ROOT),
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

    # 2. Metrics
    IoU = create_anomalib_metric(BinaryJaccardIndex)

    evaluator = Evaluator(
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Max(fields=["pred_score", "gt_label"], prefix="image_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            IoU(fields=["pred_mask", "gt_mask"], prefix="pixel_"),
        ],
    )

    # 3. Model — PatchCore (memory bank, no gradient training)
    print("Initializing PatchCore...")
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        evaluator=evaluator,
    )

    # 4. Engine
    engine = Engine(
        default_root_dir=str(RESULTS_DIR),
        devices=1,
        accelerator="auto",
    )

    # 5. Train (builds memory bank — 1 epoch)
    print("\nTraining (building memory bank)...")
    engine.fit(model=model, datamodule=datamodule)

    # 6. Test
    print("\nTesting...")
    test_results = engine.test(model=model, datamodule=datamodule)

    # 7. Print results
    print("\n" + "=" * 50)
    print("RESULTS — Anomalib PatchCore")
    print("=" * 50)
    if test_results:
        results = test_results[0]
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        (RESULTS_DIR / "metrics.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nMetrics saved to {RESULTS_DIR / 'metrics.json'}")
    else:
        print("No results returned!")

    return test_results


if __name__ == "__main__":
    main()
