"""
Crop patches around defect regions to fix class imbalance.
Input: full images + masks from b6_full_100
Output: 512x512 patches with ~15-30% defect pixel ratio
"""
import cv2
import numpy as np
from pathlib import Path
import random

from eval_config import SRC_DIR, SPLIT_DIR, OUT_DIR, NUM_CLASSES, CLASS_NAMES, CV_PREFIXES

PATCH_SIZE = 256
PATCHES_PER_DEFECT = 4
BG_PATCHES_RATIO = 0.15
SEED = 42

PATCH_DIR = OUT_DIR / "patches"
PATCH_IMG_DIR = PATCH_DIR / "images"
PATCH_MASK_DIR = PATCH_DIR / "masks"
PATCH_SPLIT_DIR = PATCH_DIR / "split_lists"


def extract_defect_patches(img, mask, base_name, patch_size=PATCH_SIZE, n_patches=PATCHES_PER_DEFECT):
    """Extract patches centered around defect regions."""
    h, w = mask.shape[:2]
    patches = []

    defect_mask = (mask > 0).astype(np.uint8)
    if np.count_nonzero(defect_mask) == 0:
        return patches

    ys, xs = np.where(defect_mask > 0)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))

    half = patch_size // 2
    rng = random.Random(hash(base_name))

    for i in range(n_patches):
        if i == 0:
            px, py = cx, cy
        else:
            jx = rng.randint(-half // 3, half // 3)
            jy = rng.randint(-half // 3, half // 3)
            px, py = cx + jx, cy + jy

        x1 = max(0, min(px - half, w - patch_size))
        y1 = max(0, min(py - half, h - patch_size))
        x2 = x1 + patch_size
        y2 = y1 + patch_size

        if x2 > w:
            x1, x2 = w - patch_size, w
        if y2 > h:
            y1, y2 = h - patch_size, h

        if x1 < 0 or y1 < 0:
            continue

        img_patch = img[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]

        defect_ratio = np.count_nonzero(mask_patch) / mask_patch.size
        if defect_ratio < 0.001:
            continue

        patches.append((img_patch, mask_patch, f"{base_name}_p{i}"))

    return patches


def extract_bg_patches(img, mask, base_name, patch_size=PATCH_SIZE, n_patches=1):
    """Extract background-only patches."""
    h, w = mask.shape[:2]
    if h < patch_size or w < patch_size:
        return []

    patches = []
    rng = random.Random(hash(base_name) + 999)

    for i in range(n_patches * 3):
        x1 = rng.randint(0, w - patch_size)
        y1 = rng.randint(0, h - patch_size)
        mask_patch = mask[y1:y1+patch_size, x1:x1+patch_size]

        if np.count_nonzero(mask_patch) == 0:
            img_patch = img[y1:y1+patch_size, x1:x1+patch_size]
            patches.append((img_patch, mask_patch, f"{base_name}_bg{len(patches)}"))
            if len(patches) >= n_patches:
                break

    return patches


def main():
    random.seed(SEED)

    for d in [PATCH_IMG_DIR, PATCH_MASK_DIR, PATCH_SPLIT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=== Crop Patches for Training ===\n")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Patches per defect: {PATCHES_PER_DEFECT}")

    all_patches = {"train": [], "val": []}

    for split in ["train", "val"]:
        src = SPLIT_DIR / f"{split}.txt"
        lines = src.read_text(encoding="utf-8").strip().split("\n")
        defect_count = 0
        bg_count = 0

        for line in lines:
            img_rel, mask_rel = line.split("\t")
            img_path = SRC_DIR / img_rel
            mask_path = SRC_DIR / mask_rel

            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            base_name = Path(img_rel).stem

            defect_patches = extract_defect_patches(img, mask, base_name)
            for img_p, mask_p, name in defect_patches:
                img_fname = f"{name}.png"
                mask_fname = f"{name}_mask.png"
                cv2.imwrite(str(PATCH_IMG_DIR / img_fname), img_p)
                cv2.imwrite(str(PATCH_MASK_DIR / mask_fname), mask_p)
                all_patches[split].append(f"images/{img_fname}\tmasks/{mask_fname}")
                defect_count += 1

            n_bg = max(1, int(len(defect_patches) * BG_PATCHES_RATIO))
            bg_patches = extract_bg_patches(img, mask, base_name, n_patches=n_bg)
            for img_p, mask_p, name in bg_patches:
                img_fname = f"{name}.png"
                mask_fname = f"{name}_mask.png"
                cv2.imwrite(str(PATCH_IMG_DIR / img_fname), img_p)
                cv2.imwrite(str(PATCH_MASK_DIR / mask_fname), mask_p)
                all_patches[split].append(f"images/{img_fname}\tmasks/{mask_fname}")
                bg_count += 1

        print(f"\n{split}: {defect_count} defect patches + {bg_count} bg patches = {defect_count+bg_count} total")

    for split in ["train", "val"]:
        random.shuffle(all_patches[split])
        out_path = PATCH_SPLIT_DIR / f"{split}.txt"
        out_path.write_text("\n".join(all_patches[split]) + "\n", encoding="utf-8")

    # Stats
    print("\n--- Defect pixel ratio in patches ---")
    sample_masks = list(PATCH_MASK_DIR.glob("*_mask.png"))[:100]
    total_px = 0
    defect_px = 0
    for mp in sample_masks:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            total_px += m.size
            defect_px += np.count_nonzero(m)

    if total_px > 0:
        ratio = defect_px / total_px * 100
        print(f"  Defect ratio: {ratio:.1f}% (was 0.4% on full images)")

    print(f"\nPatches saved to: {PATCH_DIR}")


if __name__ == "__main__":
    main()
