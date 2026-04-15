"""
photometry/illumination.py
Estimate local illumination from a polar image and provide
modulation utilities for defect synthesis.

Convention
----------
All functions operate on polar-space images (rows=radius, cols=angle).
"""
import cv2
import numpy as np


def estimate_illumination(
    polar: np.ndarray,
    sigma: float = 20.0,
) -> np.ndarray:
    """
    Estimate smoothed illumination map from polar image.

    Method:
      1. Convert to grayscale (luminance)
      2. Gaussian blur (sigma controls smoothness, default 20 px)
      3. Normalize to [0, 1] per-image

    Returns
    -------
    illum : (H, W) float32   normalized illumination map
    """
    if polar.ndim == 3:
        gray = cv2.cvtColor(polar, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = polar.astype(np.float32)

    ksize = int(sigma * 6) | 1  # odd kernel, 6σ wide
    illum = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    illum /= (illum.max() + 1e-6)
    return illum


def ring_band_illumination(
    polar_illum: np.ndarray,
    r_row: int,
    band_half: int = 5,
) -> np.ndarray:
    """
    Extract mean illumination profile along the ring band (per column).

    Returns
    -------
    profile : (polar_w,) float32  per-column mean illumination at ring band
    """
    polar_h, polar_w = polar_illum.shape[:2]
    r_lo = max(0, r_row - band_half)
    r_hi = min(polar_h, r_row + band_half + 1)
    profile = polar_illum[r_lo:r_hi, :].mean(axis=0).astype(np.float32)
    return profile


def affine_illumination_match(
    rendered_crop: np.ndarray,
    orig_crop: np.ndarray,
    mask_f: np.ndarray,
) -> np.ndarray:
    """
    Align rendered patch illumination to original base crop (Cartesian space).

    Steps:
      1. Sample background pixels (mask_f < 0.1) → compute mean/std per channel
      2. Global affine correction applied to entire rendered crop
         (r_c − μ_r) / σ_r × σ_o + μ_o
      3. Hard-reset context pixels (mask_f < 0.05) to exactly equal original

    This eliminates global color/brightness offset before Poisson blending.
    """
    r = rendered_crop.astype(np.float32)
    o = orig_crop.astype(np.float32)
    bg = mask_f < 0.1

    if bg.sum() >= 10:
        for c in range(3):
            r_mean = float(r[:, :, c][bg].mean())
            o_mean = float(o[:, :, c][bg].mean())
            r_std  = float(r[:, :, c][bg].std())  + 1e-6
            o_std  = float(o[:, :, c][bg].std())  + 1e-6
            r[:, :, c] = (r[:, :, c] - r_mean) / r_std * o_std + o_mean

    # Hard-reset true background to original (no seam at all for context pixels)
    outside = (mask_f < 0.05)[:, :, np.newaxis]
    r = r * (1.0 - outside) + o * outside

    return np.clip(r, 0, 255).astype(np.uint8)
