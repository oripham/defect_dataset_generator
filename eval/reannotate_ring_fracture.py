"""
Re-annotate ring_fracture masks by image differencing against OK base.

Original masks cover the entire ring surface (~44% of image) because they are
blending masks from synthesis, not defect masks. This script generates proper
masks by finding where visible distortion actually occurs.

Approach:
  1. Compute absolute pixel difference between defective and OK image
  2. Threshold to isolate significant changes (actual fractures/distortion)
  3. Morphological cleanup: remove JPEG noise, connect nearby fragments
  4. Also regenerate YOLO label files from the new masks
"""
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
OK_IMG = Path(r"V:\defect_samples\Napchai\Vỡ_vòng\ok\ok_001.jpg")
RF_DIR = Path(r"V:\HondaPlus\napchai_yolo\ring_fracture")
IMG_DIR = RF_DIR / "images"
OLD_MASK_DIR = RF_DIR / "masks"
OLD_LABEL_DIR = RF_DIR / "labels"

NEW_MASK_DIR = RF_DIR / "masks_v2"
NEW_LABEL_DIR = RF_DIR / "labels_v2"

# ── Parameters ─────────────────────────────────────────────────────────────────
DIFF_THRESHOLD = 15        # minimum pixel diff to consider as defect
MIN_AREA_PX = 50           # minimum connected component area (pixels)
MORPH_CLOSE_K = 7          # closing kernel to connect nearby fragments
MORPH_OPEN_K = 3           # opening kernel to remove noise
DILATE_K = 5               # dilate final mask slightly for better coverage
YOLO_CLASS_ID = 2          # ring_fracture class ID in the 3-class YOLO setup

# Also limit to the ring region using the old mask as a region-of-interest
USE_OLD_MASK_AS_ROI = True
OLD_MASK_THRESHOLD = 10    # pixels > this in old mask = ring region


def load_ok_image():
    ok = cv2.imread(str(OK_IMG))
    if ok is None:
        raise FileNotFoundError(f"Cannot read OK image: {OK_IMG}")
    return ok.astype(np.float32)


def compute_diff_mask(ok_f32, defect_path, old_mask_path=None):
    defect = cv2.imread(str(defect_path))
    if defect is None:
        return None

    diff = np.abs(defect.astype(np.float32) - ok_f32)
    diff_gray = diff.mean(axis=2)

    # Threshold
    binary = (diff_gray > DIFF_THRESHOLD).astype(np.uint8) * 255

    # Restrict to ring region using old mask as ROI
    if USE_OLD_MASK_AS_ROI and old_mask_path and old_mask_path.exists():
        old_mask = cv2.imread(str(old_mask_path), cv2.IMREAD_GRAYSCALE)
        if old_mask is not None:
            roi = (old_mask > OLD_MASK_THRESHOLD).astype(np.uint8) * 255
            # Dilate ROI slightly to not clip edges
            roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            binary = cv2.bitwise_and(binary, roi)

    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_K, DILATE_K))

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    binary = cv2.dilate(binary, kernel_dilate, iterations=1)

    # Remove small connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA_PX:
            binary[labels == i] = 0

    return binary


def mask_to_yolo_polygons(mask, class_id, min_area_ratio=0.0001):
    """Convert binary mask to YOLO segmentation format polygons."""
    H, W = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area / (H * W) < min_area_ratio:
            continue
        if len(cnt) < 3:
            continue
        # Simplify contour
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        if len(cnt) < 3:
            continue

        coords = []
        for pt in cnt.reshape(-1, 2):
            coords.append(f"{pt[0] / W:.6f}")
            coords.append(f"{pt[1] / H:.6f}")
        lines.append(f"{class_id} " + " ".join(coords))

    return lines


def main():
    NEW_MASK_DIR.mkdir(parents=True, exist_ok=True)
    NEW_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    ok_f32 = load_ok_image()
    print(f"OK image loaded: {ok_f32.shape}")

    images = sorted(IMG_DIR.glob("*.png"))
    print(f"Found {len(images)} ring_fracture images\n")

    stats = defaultdict(int)
    area_ratios = []

    for img_path in images:
        stem = img_path.stem
        mask_name = stem + "_mask.png"
        old_mask_path = OLD_MASK_DIR / mask_name

        new_mask = compute_diff_mask(ok_f32, img_path, old_mask_path)
        if new_mask is None:
            stats["error"] += 1
            continue

        nonzero_ratio = (new_mask > 0).sum() / new_mask.size

        if nonzero_ratio < 0.0001:
            stats["empty"] += 1
            # Still save empty mask and label
            cv2.imwrite(str(NEW_MASK_DIR / mask_name), new_mask)
            (NEW_LABEL_DIR / (stem + ".txt")).write_text("", encoding="utf-8")
            continue

        stats["has_defect"] += 1
        area_ratios.append(nonzero_ratio * 100)

        # Save new mask
        cv2.imwrite(str(NEW_MASK_DIR / mask_name), new_mask)

        # Generate YOLO labels
        yolo_lines = mask_to_yolo_polygons(new_mask, YOLO_CLASS_ID)
        (NEW_LABEL_DIR / (stem + ".txt")).write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
            encoding="utf-8",
        )

    print("=" * 50)
    print("Re-annotation Results")
    print("=" * 50)
    print(f"  Total images:   {len(images)}")
    print(f"  Has defect:     {stats['has_defect']}")
    print(f"  Empty (no diff): {stats['empty']}")
    print(f"  Errors:         {stats['error']}")

    if area_ratios:
        print(f"\n  Defect area (% of image):")
        print(f"    Mean:   {np.mean(area_ratios):.2f}%")
        print(f"    Median: {np.median(area_ratios):.2f}%")
        print(f"    Min:    {np.min(area_ratios):.2f}%")
        print(f"    Max:    {np.max(area_ratios):.2f}%")

    # Compare with old masks
    old_areas = []
    for mp in sorted(OLD_MASK_DIR.glob("*.png")):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            old_areas.append((m > 10).sum() / m.size * 100)
    if old_areas:
        print(f"\n  Old mask area (for comparison):")
        print(f"    Mean: {np.mean(old_areas):.2f}%")

    print(f"\nNew masks saved to: {NEW_MASK_DIR}")
    print(f"New labels saved to: {NEW_LABEL_DIR}")


if __name__ == "__main__":
    main()
