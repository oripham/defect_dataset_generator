"""
Anomalib PatchCore evaluation on napchai defect classes.
Runs PatchCore independently per class (scratch, mc_deform, ring_fracture).

Dataset setup:
  - OK images: single ok_001.jpg replicated (same approach as MKA)
  - Defective images: synthetic images from napchai_yolo/{class}/images/
  - Ground truth masks: binarized from napchai_yolo/{class}/masks/
"""
import sys
import json
import cv2
import numpy as np
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent
OUT_DIR = EVAL_DIR / "output"
RESULTS_BASE = OUT_DIR / "results" / "anomalib_napchai"
ANOMALIB_BASE = OUT_DIR / "anomalib"

NAPCHAI_YOLO = Path(r"V:\HondaPlus\napchai_yolo")
OK_IMG_PATH = Path(r"V:\defect_samples\Napchai\Xước\ok\ok_001.jpg")

CLASSES = {
    "scratch": {"img_dir": NAPCHAI_YOLO / "scratch" / "images",
                "mask_dir": NAPCHAI_YOLO / "scratch" / "masks"},
    "mc_deform": {"img_dir": NAPCHAI_YOLO / "mc_deform" / "images",
                  "mask_dir": NAPCHAI_YOLO / "mc_deform" / "masks"},
    "ring_fracture": {"img_dir": NAPCHAI_YOLO / "ring_fracture" / "images",
                      "mask_dir": NAPCHAI_YOLO / "ring_fracture" / "masks_v2"},
}

N_TRAIN_OK = 50
N_TEST_OK = 20
TEST_RATIO = 0.2
SEED = 42


def prepare_anomalib_dataset(class_name, class_cfg):
    """Create Anomalib Folder-format dataset for one napchai class."""
    dataset_dir = ANOMALIB_BASE / f"napchai_{class_name}"

    train_good = dataset_dir / "train" / "good"
    test_def = dataset_dir / "test" / "defective"
    test_good = dataset_dir / "test" / "good"
    gt_def = dataset_dir / "ground_truth" / "defective"

    for d in [train_good, test_def, test_good, gt_def]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Read OK image
    ok_img = cv2.imread(str(OK_IMG_PATH))
    if ok_img is None:
        print(f"  ERROR: Cannot read OK image: {OK_IMG_PATH}")
        return None

    # Replicate OK images for training
    for i in range(N_TRAIN_OK):
        cv2.imwrite(str(train_good / f"good_{i:04d}.png"), ok_img)

    # Replicate OK images for test
    for i in range(N_TEST_OK):
        cv2.imwrite(str(test_good / f"good_{i:04d}.png"), ok_img)

    # Collect defective images
    img_dir = class_cfg["img_dir"]
    mask_dir = class_cfg["mask_dir"]

    images = sorted(img_dir.glob("*.png"))
    if not images:
        print(f"  ERROR: No images in {img_dir}")
        return None

    # Use last TEST_RATIO as test (deterministic)
    np.random.seed(SEED)
    indices = np.random.permutation(len(images))
    n_test = max(int(len(images) * TEST_RATIO), 1)
    test_indices = set(indices[:n_test])

    # For Anomalib, we use ALL images as test (since PatchCore trains only on OK)
    # But we still need to provide masks. Use all 100 defective for testing.
    n_with_mask = 0
    n_empty_mask = 0

    for img_path in images:
        stem = img_path.stem
        mask_name = stem + "_mask.png"
        mask_path = mask_dir / mask_name

        # Copy defective image
        dst_img = test_def / (stem + ".png")
        src_img = cv2.imread(str(img_path))
        if src_img is None:
            continue
        cv2.imwrite(str(dst_img), src_img)

        # Copy + binarize mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                binary = (mask > 10).astype(np.uint8) * 255
                if binary.sum() > 0:
                    cv2.imwrite(str(gt_def / (stem + ".png")), binary)
                    n_with_mask += 1
                else:
                    # Empty mask — Anomalib needs a mask file, write zeros
                    cv2.imwrite(str(gt_def / (stem + ".png")), binary)
                    n_empty_mask += 1
        else:
            # No mask file — write all-zeros mask
            h, w = src_img.shape[:2]
            cv2.imwrite(str(gt_def / (stem + ".png")), np.zeros((h, w), dtype=np.uint8))
            n_empty_mask += 1

    print(f"  Dataset: {N_TRAIN_OK} train OK, {N_TEST_OK} test OK, "
          f"{len(images)} test defective ({n_with_mask} with mask, {n_empty_mask} empty)")
    return dataset_dir


def run_patchcore(class_name, dataset_dir):
    """Run Anomalib PatchCore on prepared dataset."""
    from anomalib.data import Folder
    from anomalib.models import Patchcore
    from anomalib.engine import Engine
    from anomalib.metrics import AUROC, F1Max, Evaluator, create_anomalib_metric
    from torchmetrics.classification import BinaryJaccardIndex

    results_dir = RESULTS_BASE / class_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Setting up Folder dataset from {dataset_dir}...")
    datamodule = Folder(
        name=f"napchai_{class_name}",
        root=str(dataset_dir),
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

    print("  Initializing PatchCore...")
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
    engine.fit(model=model, datamodule=datamodule)

    print("  Testing...")
    test_results = engine.test(model=model, datamodule=datamodule)

    if test_results:
        results = test_results[0]
        (results_dir / "metrics.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        return results
    return None


def main():
    print("=" * 60)
    print("Anomalib PatchCore — Napchai Defect Classes")
    print("=" * 60)

    all_results = {}

    for class_name, class_cfg in CLASSES.items():
        print(f"\n{'─' * 60}")
        print(f"Class: {class_name}")
        print(f"{'─' * 60}")

        # Check if mask_dir exists; fall back to regular masks
        if not class_cfg["mask_dir"].exists():
            fallback = class_cfg["mask_dir"].parent / "masks"
            print(f"  Warning: {class_cfg['mask_dir']} not found, using {fallback}")
            class_cfg["mask_dir"] = fallback

        # 1. Prepare dataset
        print("  Preparing dataset...")
        dataset_dir = prepare_anomalib_dataset(class_name, class_cfg)
        if dataset_dir is None:
            print(f"  SKIPPED: dataset preparation failed")
            continue

        # 2. Run PatchCore
        results = run_patchcore(class_name, dataset_dir)
        if results:
            all_results[class_name] = results
            print(f"\n  Results for {class_name}:")
            for k, v in sorted(results.items()):
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY — Anomalib PatchCore Napchai")
    print(f"{'=' * 60}")
    if all_results:
        header = f"{'Class':<16} {'img_AUROC':>10} {'pix_AUROC':>10} {'pix_IoU':>10}"
        print(header)
        print("-" * len(header))
        for cls, r in all_results.items():
            img_auroc = r.get("image_AUROC", 0)
            pix_auroc = r.get("pixel_AUROC", 0)
            pix_iou = r.get("pixel_BinaryJaccardIndex", 0)
            print(f"{cls:<16} {img_auroc:>10.4f} {pix_auroc:>10.4f} {pix_iou:>10.4f}")

        # Save combined results
        RESULTS_BASE.mkdir(parents=True, exist_ok=True)
        (RESULTS_BASE / "all_metrics.json").write_text(
            json.dumps(all_results, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nAll metrics saved to {RESULTS_BASE / 'all_metrics.json'}")


if __name__ == "__main__":
    main()
