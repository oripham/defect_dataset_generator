"""
engines/structure_adapt.py — Structure-Aware Mask Adaptation (Vuong)
=====================================================================

Phát hiện ring structure từ base image (Hough Circle) rồi:
  - Snap mask lên ring circumference (chip, dent, scratch, ...)
  - Orient scratch/crack theo tangent của ring
  - Tính radial light direction tại điểm đặt lỗi
  - Tính clock position cho SDXL prompt ("at 9 o'clock position on rim")

Reuses: scripts/geometry/ring_detector.detect_ring()

Entry point: structure_adapt(base_rgb, mask, defect_type)
  → (mask_out, light_dir, clock_str)
  Falls back to (mask, None, None) if detect_ring fails.
"""

import sys
import os
import numpy as np
import cv2

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCRIPTS))

from geometry.ring_detector import detect_ring

# Defect types cần snap lên ring edge (outer rim)
# scratch/dent/bulge nam tren be mat phang ben trong — KHONG snap ra rim ngoai
_SNAP_TYPES    = {"crack", "chip"}
# Defect types cần orient theo tangent — CHI line-shaped defects
_TANGENT_TYPES = {"crack"}


# ── Helpers ───────────────────────────────────────────────────────────────────

_BG_BRIGHTNESS_THRESHOLD = 40   # snap skip: pixel > threshold → on part, skip snap
_BG_CLIP_THRESHOLD       = 15   # mask clip: pixel < threshold → pure black background, remove from mask
                                 # (inner gray surface ~40-80, background ~0-15)

def _snap_mask_to_ring(mask: np.ndarray, cx: int, cy: int, r: int,
                       base_gray: np.ndarray = None) -> np.ndarray:
    """
    Translate mask centroid onto ring circumference.
    Chi snap neu centroid dang o vung toi (background).
    Neu da o tren be mat kim loai (brightness > threshold) → giu nguyen.
    """
    ys, xs = np.where(mask > 127)
    if not len(ys):
        return mask
    my, mx = float(ys.mean()), float(xs.mean())

    # Kiem tra brightness tai centroid — neu da o tren part thi khong snap
    if base_gray is not None:
        bx = int(np.clip(mx, 0, base_gray.shape[1]-1))
        by = int(np.clip(my, 0, base_gray.shape[0]-1))
        brightness = int(base_gray[by, bx])
        if brightness > _BG_BRIGHTNESS_THRESHOLD:
            print(f"[SNAP] Skip snap — centroid ({int(mx)},{int(my)}) brightness={brightness} (on part)")
            return mask
        print(f"[SNAP] Snapping — centroid ({int(mx)},{int(my)}) brightness={brightness} (background)")

    angle = np.arctan2(my - cy, mx - cx)
    dx = int(round(cx + r * np.cos(angle) - mx))
    dy = int(round(cy + r * np.sin(angle) - my))
    M  = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _orient_tangential(mask: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Rotate mask so long axis aligns with ring tangent at centroid."""
    ys, xs = np.where(mask > 127)
    if not len(ys):
        return mask
    my, mx  = float(ys.mean()), float(xs.mean())
    tangent = np.degrees(np.arctan2(my - cy, mx - cx)) + 90.0
    M = cv2.getRotationMatrix2D((mx, my), tangent, 1.0)
    return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _radial_light_dir(mask: np.ndarray, cx: int, cy: int):
    """Radial outward normal at mask centroid — [lx, ly, lz] or None."""
    ys, xs = np.where(mask > 127)
    if not len(ys):
        return None
    dx = float(xs.mean()) - cx
    dy = float(ys.mean()) - cy
    n  = np.sqrt(dx**2 + dy**2) + 1e-6
    return [dx / n, -dy / n, 0.7]  # outward radial + ~53° elevation


def _angle_to_clock(deg: float) -> str:
    """Convert polar angle (degrees, 0=right) to clock position string."""
    # 0° (3 o'clock) → +3 offset, wrap 12
    h = int(round(((deg / 360.0) * 12 + 3) % 12)) or 12
    return f"{h} o'clock"


# ── Main entry point ──────────────────────────────────────────────────────────

def structure_adapt(base_rgb: np.ndarray, mask: np.ndarray, defect_type: str):
    """
    Detect ring structure in base image, then adapt mask + compute spatial info.

    Parameters
    ----------
    base_rgb    : uint8 RGB (H, W, 3)
    mask        : uint8 grayscale (H, W), white = defect region
    defect_type : str

    Returns
    -------
    mask_out  : uint8 (H, W) — adapted mask (snapped + oriented)
    light_dir : list [lx, ly, lz] or None (radial normal at defect centroid)
    clock_str : str like "9 o'clock" or None (for SDXL prompt injection)

    Falls back to (mask, None, None) if ring detection fails.
    """
    gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape

    # Resize mask to match base image if canvas sent different resolution
    if mask.shape[:2] != (H, W):
        print(f"[STRUCT] Resizing mask {mask.shape[:2]} → ({H},{W})")
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    try:
        cx, cy, r = detect_ring(gray, tag=f"_{defect_type}")
    except Exception as e:
        print(f"[STRUCT] detect_ring failed: {e}")
        return mask, None, None

    # Fallback for non-circular parts:
    # r/min(H,W) < 0.3 → detected circle is an internal feature, not product rim
    # Skip snap/orient/light_dir to avoid incorrect shading direction
    r_ratio = r / max(min(H, W), 1)
    if r_ratio < 0.3:
        print(f"[STRUCT] Non-circular part detected (r={r}, r/min(H,W)={r_ratio:.2f} < 0.3) — skipping structure_adapt")
        return mask, None, None

    mask_out = mask

    # Snap centroid onto ring edge
    if defect_type in _SNAP_TYPES:
        mask_out = _snap_mask_to_ring(mask_out, cx, cy, r, base_gray=gray)

    # Rotate line-shaped defects to follow ring tangent
    if defect_type in _TANGENT_TYPES:
        mask_out = _orient_tangential(mask_out, cx, cy)

    # Radial light direction (all defect types)
    light_dir = _radial_light_dir(mask_out, cx, cy)

    # Clock position for prompt
    clock_str = None
    ys, xs = np.where(mask_out > 127)
    if len(ys):
        deg = np.degrees(np.arctan2(float(ys.mean()) - cy,
                                    float(xs.mean()) - cx)) % 360
        clock_str = _angle_to_clock(deg)

    # Clip mask to product region — zero out pure black background pixels only
    # Dung _BG_CLIP_THRESHOLD (15), KHONG dung _BG_BRIGHTNESS_THRESHOLD (40)
    # vi inner gray surface co brightness ~40-80, khong phai background
    product_region = gray > _BG_CLIP_THRESHOLD
    before_px = int(np.sum(mask_out > 127))
    mask_out[~product_region] = 0
    after_px  = int(np.sum(mask_out > 127))
    if before_px != after_px:
        print(f"[STRUCT] Clipped {before_px - after_px}px background from mask")

    print(f"[STRUCT] {defect_type}: ring r={r}  light={light_dir}  pos={clock_str}")
    return mask_out, light_dir, clock_str
