"""
Prepare B6/MKA defect dataset for segmentation model evaluation.
Creates train/val splits using file lists (no copying — saves disk space).

Input:  b6_full_100/{images, masks}
Output: C:/eval_b6/{split_lists, anomalib_config, paddleseg_config, ...}
"""
import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

from eval_config import (
    SRC_DIR, OK_IMG, OUT_DIR, SPLIT_DIR, ANOMALIB_DIR,
    CLASS_NAMES, NUM_CLASSES, CV_PREFIXES, VAL_RATIO, SEED,
)


def collect_cv_samples():
    img_dir = SRC_DIR / "images"
    mask_dir = SRC_DIR / "masks"
    samples = []
    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem
        prefix = None
        for p in CV_PREFIXES:
            if stem.startswith(p + "_"):
                prefix = p
                break
        if prefix is None:
            continue
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        classes_present = set(np.unique(mask)) - {0}
        if not classes_present:
            continue
        samples.append({
            "img": str(img_path),
            "mask": str(mask_path),
            "stem": stem,
            "classes": list(classes_present),
            "prefix": prefix,
        })
    return samples


def prepare_split(samples):
    prefixes = [s["prefix"] for s in samples]
    train_idx, val_idx = train_test_split(
        range(len(samples)), test_size=VAL_RATIO,
        random_state=SEED, stratify=prefixes
    )
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    return train, val


def write_split_lists(train, val):
    lists_dir = SPLIT_DIR
    lists_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val)]:
        lines = []
        for s in data:
            # Relative paths (relative to SRC_DIR) — portable across machines
            img_rel  = Path(s["img"]).relative_to(SRC_DIR).as_posix()
            mask_rel = Path(s["mask"]).relative_to(SRC_DIR).as_posix()
            lines.append(f"{img_rel}\t{mask_rel}")
        (lists_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "source": str(SRC_DIR),
        "ok_image": str(OK_IMG),
        "num_classes": NUM_CLASSES,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "train_count": len(train),
        "val_count": len(val),
        "train_class_dist": {},
        "val_class_dist": {},
    }
    for name, data in [("train", train), ("val", val)]:
        dist = {}
        for s in data:
            for c in s["classes"]:
                cname = CLASS_NAMES.get(c, f"class_{c}")
                dist[cname] = dist.get(cname, 0) + 1
        meta[f"{name}_class_dist"] = dist

    (lists_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Split lists: {lists_dir}")
    return lists_dir


def setup_anomalib_config(train, val):
    """Create Anomalib folder-style dataset on C: (only OK images need copying)."""
    anom_dir = ANOMALIB_DIR

    train_good = anom_dir / "train" / "good"
    test_def = anom_dir / "test" / "defective"
    test_good = anom_dir / "test" / "good"
    gt_def = anom_dir / "ground_truth" / "defective"

    for d in [train_good, test_def, test_good, gt_def]:
        d.mkdir(parents=True, exist_ok=True)

    ok_img = cv2.imread(str(OK_IMG))
    if ok_img is None:
        print("ERROR: Cannot read OK image")
        return None

    h, w = ok_img.shape[:2]
    # Write a subset of OK images for training (avoid too many duplicates)
    n_train_ok = min(len(train), 50)
    for i in range(n_train_ok):
        cv2.imwrite(str(train_good / f"good_{i:04d}.png"), ok_img)
    print(f"Anomalib train/good: {n_train_ok} OK images")

    for s in val:
        src_img = cv2.imread(s["img"])
        if src_img is not None:
            cv2.imwrite(str(test_def / (Path(s["img"]).stem + ".png")), src_img)
        mask = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            binary = (mask > 0).astype(np.uint8) * 255
            cv2.imwrite(str(gt_def / (Path(s["mask"]).stem + ".png")), binary)

    n_test_good = min(20, len(val) // 3)
    for i in range(n_test_good):
        cv2.imwrite(str(test_good / f"good_{i:04d}.png"), ok_img)

    print(f"Anomalib test: {len(val)} defective + {n_test_good} good")
    return anom_dir


def print_stats(train, val):
    for name, data in [("Train", train), ("Val", val)]:
        class_counts = {}
        for s in data:
            for c in s["classes"]:
                cname = CLASS_NAMES.get(c, f"class_{c}")
                class_counts[cname] = class_counts.get(cname, 0) + 1
        print(f"\n{name} ({len(data)} images):")
        for cname, cnt in sorted(class_counts.items()):
            print(f"  {cname}: {cnt}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting CV samples...")
    samples = collect_cv_samples()
    print(f"Found {len(samples)} valid samples")

    print("\nSplitting train/val (80/20)...")
    train, val = prepare_split(samples)
    print_stats(train, val)

    print("\n--- Writing split lists ---")
    write_split_lists(train, val)

    print("\n--- Setting up Anomalib dataset ---")
    setup_anomalib_config(train, val)

    print(f"\nDone! Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
