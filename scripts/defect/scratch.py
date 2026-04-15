"""defect/scratch.py  v5 — height-field based scratch (キズ)
=============================================================

warpPolar convention: rows=θ (angle), cols=r (radius)

Physical model:
    Surface height: h(row, col) = -A * groove_profile(dcol) * in_range(row) * noise(row)
        groove_profile = Gaussian depression at col_ring
        dcol = col - col_path(row)   (signed distance from scratch path)

    Shading from normals:
        nx = -∂h/∂col  → on LEFT of groove: positive (faces light) → BRIGHT
                         on RIGHT of groove: negative (away from light) → DARK
        ny = -∂h/∂row  → angular entry/exit get highlight/shadow

    Result (without drawing anything):
        - Groove center: deep shadow (∂h/∂col ≈ 0 but depth absorbs light)
        - Left wall:  bright specular highlight
        - Right wall: shadow
        - Entry/exit: perpendicular highlight+shadow

    This matches the real scratch appearance on metallic surfaces.
"""

import cv2
import numpy as np
from ._render import shade_from_height


def _smooth_1d(arr: np.ndarray, k: int) -> np.ndarray:
    k = max(int(k), 1) | 1
    return cv2.GaussianBlur(arr.reshape(-1, 1).astype(np.float32), (1, k), 0).flatten()


def synthesize_scratch(
    polar:        np.ndarray,
    polar_illum:  np.ndarray,
    r_ring:       float,
    r_max:        float,
    theta_center: float,
    theta_span:   float,
    seed:         int   = 42,
    groove_depth: float = 18.0,   # col units; depth of groove
    groove_sigma: float = 1.6,    # col units; width half-sigma
    k_diffuse:    float = 240.0,
    k_specular:   float = 130.0,
    gamma:        float = 1.8,
    shininess:    int   = 8,
) -> tuple[np.ndarray, np.ndarray]:
    H, W = polar.shape[:2]
    rng  = np.random.RandomState(seed)

    row_idx = np.arange(H, dtype=np.float32)
    col_idx = np.arange(W, dtype=np.float32)
    ROW, COL = np.meshgrid(row_idx, col_idx, indexing="ij")

    col_ring = float(np.clip(r_ring / r_max * W, 1, W - 2))

    # Angular range — f(θ) only
    row_center = (theta_center % (2 * np.pi)) / (2 * np.pi) * H
    drow       = np.abs(ROW - row_center)
    drow       = np.minimum(drow, H - drow)
    row_half   = (theta_span / (2 * np.pi)) * H / 2.0
    taper_rows = max(row_half * 0.15, 3.0)
    in_range   = np.clip((row_half - drow) / (taper_rows + 1e-6), 0.0, 1.0)

    # Path wobble — f(θ) only
    raw_noise  = rng.normal(0, 1.0, H).astype(np.float32)
    path_noise = _smooth_1d(raw_noise, H // 20)
    path_noise = (path_noise - path_noise.mean()) * groove_sigma * 0.5
    col_path_g = (col_ring + path_noise)[:, np.newaxis]

    # Signed distance from scratch path
    dcol = COL - col_path_g

    # Intensity variation — f(θ) only
    raw_intens = rng.normal(0, 1.0, H).astype(np.float32)
    intens_1d  = _smooth_1d(raw_intens, H // 12)
    intens_1d  = 0.7 + 0.5 * (intens_1d - intens_1d.min()) / (intens_1d.ptp() + 1e-6)
    intens_g   = intens_1d[:, np.newaxis]

    # Height field: h(r,θ) = -depth * Gaussian(dcol) * angular_range * intensity
    h = -groove_depth * np.exp(-0.5 * (dcol / groove_sigma) ** 2) * in_range * intens_g

    # Shading from height field normals (handles dark/bright edges automatically)
    delta = shade_from_height(h, polar_illum,
                              k_diffuse=k_diffuse, k_specular=k_specular,
                              gamma=gamma, shininess=shininess)

    out_f     = polar.astype(np.float32) + delta * in_range[:, :, np.newaxis]
    polar_out = np.clip(out_f, 0, 255).astype(np.uint8)

    groove_peak = np.exp(-0.5 * (dcol / groove_sigma) ** 2)
    mask_polar  = np.clip(groove_peak * in_range, 0, 1).astype(np.float32)
    mask_polar  = cv2.GaussianBlur(mask_polar, (5, 5), 0)

    return polar_out, mask_polar


# Alias used by generator.py
synthesize_scratch_specular = synthesize_scratch
