# scripts/mask_sampler.py
import os, random, cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def sample_mask(mask_root, defect_type, image_size,
                rotation_range=(0.0, 0.0),
                offset_range=(-15, 15)):
    """
    Load + augment a mask for the given defect class.

    Augmentations applied (all optional via config):
      1. Random rotation (rotation_range degrees, around image center)
      2. Random XY translation (offset_range pixels) — adds positional diversity
         so 20 images with the same base mask are NOT all pixel-identical.

    offset_range : (int, int)  — pixel shift range, e.g. (-15, 15).
                   Set to (0, 0) to disable.
    """
    base_dir = os.path.join(mask_root, defect_type)
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"[ERROR] Mask folder not found: {base_dir}")

    subfolders = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not subfolders:
        raise RuntimeError(f"[ERROR] No mask subfolders in {base_dir}")

    chosen_dir = random.choice(subfolders)
    mask_files = [f for f in os.listdir(chosen_dir) if f.lower().endswith(IMG_EXT)]
    if not mask_files:
        raise RuntimeError(f"[ERROR] No mask images in {chosen_dir}")

    mask_path = os.path.join(chosen_dir, random.choice(mask_files))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"[ERROR] Failed to read mask: {mask_path}")

    # Resize & binarize
    mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8) * 255

    h, w = mask.shape

    # ── 1. Rotation ──────────────────────────────────────────────────────────
    rot_min = float(rotation_range[0])
    rot_max = float(rotation_range[1])
    if rot_min != 0.0 or rot_max != 0.0:
        angle = random.uniform(rot_min, rot_max) if rot_min != rot_max else rot_min
        center = (w / 2.0, h / 2.0)
        # Negate: OpenCV positive = CCW, but UI convention positive = CW (clockwise)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        mask = cv2.warpAffine(mask, M, (w, h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ── 2. XY translation (positional diversity) ──────────────────────────────
    off_min = int(offset_range[0])
    off_max = int(offset_range[1])
    if off_max != off_min:
        dx = random.randint(off_min, off_max)
        dy = random.randint(off_min, off_max)
        if dx != 0 or dy != 0:
            M_t = np.float32([[1, 0, dx], [0, 1, dy]])
            mask = cv2.warpAffine(mask, M_t, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            print(f"[MASK] offset dx={dx:+d} dy={dy:+d}")

    return mask
