"""
run_anomalib_benchmark.py — PatchCore benchmark on Napchai + MKA (new data).
Uses batch_output/anomalib/ dataset structure (MVTec-style multi-class).
"""
import json, time
from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.metrics import AUROC, F1Max, Evaluator, create_anomalib_metric
from torchmetrics.classification import BinaryJaccardIndex

ANOMALIB_BASE = Path(r"V:\HondaPlus\defect_dataset_generator\batch_output\anomalib")
RESULTS_DIR = Path(__file__).parent / "output" / "results" / "anomalib_benchmark"


def run_product(product_name):
    dataset_root = ANOMALIB_BASE / product_name
    results_dir = RESULTS_DIR / product_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Dataset: {dataset_root}")
    print(f"  Results: {results_dir}")

    datamodule = Folder(
        name=f"{product_name}_defect",
        root=str(dataset_root),
        normal_dir="train/good",
        abnormal_dir="test",
        normal_test_dir="test_good",
        mask_dir="ground_truth",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
        test_split_mode="from_dir",
        val_split_mode="from_test",
        val_split_ratio=0.5,
    )

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

    print("  Training (building memory bank)...")
    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    train_time = time.time() - t0

    print("  Testing...")
    t0 = time.time()
    test_results = engine.test(model=model, datamodule=datamodule)
    test_time = time.time() - t0

    if test_results:
        results = test_results[0]
        results["train_time_s"] = round(train_time, 1)
        results["test_time_s"] = round(test_time, 1)
        (results_dir / "metrics.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        return results
    return None


def main():
    print("=" * 60)
    print("Anomalib PatchCore Benchmark — Napchai + MKA (new data)")
    print("=" * 60)

    all_results = {}

    for product in ["napchai", "mka"]:
        print(f"\n{'━' * 60}")
        print(f"  {product.upper()}")
        print(f"{'━' * 60}")

        results = run_product(product)
        if results:
            all_results[product] = results
            print(f"\n  Results for {product}:")
            for k, v in sorted(results.items()):
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")

    print(f"\n{'=' * 60}")
    print("SUMMARY — Anomalib PatchCore")
    print(f"{'=' * 60}")
    header = f"{'Product':<12} {'img_AUROC':>10} {'img_F1Max':>10} {'pix_AUROC':>10} {'pix_IoU':>10} {'train(s)':>10} {'test(s)':>10}"
    print(header)
    print("-" * len(header))
    for prod, r in all_results.items():
        print(f"{prod:<12} "
              f"{r.get('image_AUROC', 0):>10.4f} "
              f"{r.get('image_F1Max', 0):>10.4f} "
              f"{r.get('pixel_AUROC', 0):>10.4f} "
              f"{r.get('pixel_BinaryJaccardIndex', 0):>10.4f} "
              f"{r.get('train_time_s', 0):>10.1f} "
              f"{r.get('test_time_s', 0):>10.1f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "all_metrics.json").write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nSaved to {RESULTS_DIR / 'all_metrics.json'}")


if __name__ == "__main__":
    main()
