"""
compositing/blend.py
Composite a rendered patch back into the base image.

Strategy (in order):
  1. Poisson MIXED_CLONE  — gradient-domain, preserves ring texture gradients
  2. Gaussian alpha blend — fallback if Poisson fails (mask touching edge etc.)

Poisson MIXED_CLONE is preferred because:
  - It enforces gradient continuity across the boundary
  - MIXED mode blends gradients from both src (defect) and dst (ring texture)
  - No halo at boundary; lighting gradient is continuous

Blend mask (for Poisson) is deliberately separate from the render mask:
  - Render mask: dilated 4 px (renderer gets wider context)
  - Blend mask:  original + dilate 5 px + GaussianBlur soften
    → irregular, non-circular boundary → harder to detect
"""
import cv2
import numpy as np


def build_blend_mask(
    mask_orig: np.ndarray,
    H: int,
    W: int,
    y1: int,
    x1: int,
    dilation: int = 5,
) -> np.ndarray:
    """
    Build full-image blend mask for seamlessClone.

    Parameters
    ----------
    mask_orig : (crop_h, crop_w) uint8  original (non-dilated) mask in crop coords
    H, W      : full image size
    y1, x1    : crop origin in full image
    dilation  : ellipse dilation in pixels

    Returns
    -------
    mask_full : (H, W) uint8  binary blend mask, zeros at 5 px image border
    """
    h_c, w_c = mask_orig.shape[:2]

    # Dilate in crop space
    if dilation > 0:
        ks = dilation * 2 + 1
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        m  = cv2.dilate(mask_orig, k, iterations=1)
    else:
        m = mask_orig.copy()

    # Soften boundary: GaussianBlur + re-threshold → irregular, not circular
    m = cv2.GaussianBlur(m, (7, 7), 0)
    _, m = cv2.threshold(m, 15, 255, cv2.THRESH_BINARY)

    # Place in full-image canvas
    mask_full = np.zeros((H, W), dtype=np.uint8)
    y2 = min(H, y1 + h_c)
    x2 = min(W, x1 + w_c)
    mask_full[y1:y2, x1:x2] = m[:y2 - y1, :x2 - x1]

    # Enforce seamlessClone border safety (mask must not touch image edge)
    border = 5
    mask_full[:border, :]  = 0
    mask_full[-border:, :] = 0
    mask_full[:, :border]  = 0
    mask_full[:, -border:] = 0

    return mask_full


def composite(
    base_arr: np.ndarray,
    rendered_crop: np.ndarray,
    blend_mask_full: np.ndarray,
    alpha_crop: np.ndarray,
    y1: int,
    x1: int,
    mode: str = "poisson",
) -> np.ndarray:
    """
    Composite rendered_crop into base_arr.

    Parameters
    ----------
    base_arr        : (H, W, 3) uint8  original base image
    rendered_crop   : (h_c, w_c, 3) uint8  processed patch
    blend_mask_full : (H, W) uint8  mask for Poisson blend
    alpha_crop      : (h_c, w_c) float32  alpha for Gaussian fallback
    y1, x1          : crop origin
    mode            : "poisson" | "gaussian"

    Returns
    -------
    output : (H, W, 3) uint8
    """
    if mode == "poisson" and blend_mask_full.max() > 0:
        result = _poisson(base_arr, rendered_crop, blend_mask_full, y1, x1)
        if result is not None:
            return result

    return _gaussian(base_arr, rendered_crop, alpha_crop, y1, x1)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _poisson(
    base_arr: np.ndarray,
    rendered_crop: np.ndarray,
    blend_mask_full: np.ndarray,
    y1: int,
    x1: int,
) -> np.ndarray | None:
    """Attempt Poisson MIXED_CLONE. Returns None on failure."""
    h_c, w_c = rendered_crop.shape[:2]

    # src = full base with rendered patch pasted in
    src = base_arr.copy()
    src[y1:y1 + h_c, x1:x1 + w_c] = rendered_crop

    ys_m, xs_m = np.where(blend_mask_full > 0)
    if len(ys_m) == 0:
        return None
    center = (int(xs_m.mean()), int(ys_m.mean()))

    try:
        result = cv2.seamlessClone(src, base_arr, blend_mask_full,
                                   center, cv2.MIXED_CLONE)
        print(f"[BLEND] Poisson MIXED_CLONE OK  center={center}")
        return result
    except Exception as e:
        print(f"[BLEND] Poisson failed ({e})")
        return None


def _gaussian(
    base_arr: np.ndarray,
    rendered_crop: np.ndarray,
    alpha_crop: np.ndarray,
    y1: int,
    x1: int,
) -> np.ndarray:
    """Gaussian alpha composite fallback."""
    h_c, w_c = rendered_crop.shape[:2]
    blend     = alpha_crop[:, :, np.newaxis]
    orig      = base_arr[y1:y1 + h_c, x1:x1 + w_c].astype(np.float32)
    merged    = (rendered_crop.astype(np.float32) * blend +
                 orig * (1.0 - blend)).astype(np.uint8)
    output = base_arr.copy()
    output[y1:y1 + h_c, x1:x1 + w_c] = merged
    print("[BLEND] Gaussian alpha fallback")
    return output
