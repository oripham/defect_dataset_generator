"""defect/dent.py  v5 — height-field based dent (整形変形)
=========================================================

warpPolar convention: rows=θ (angle), cols=r (radius)

Physical model:
    1. Height field h(row, col) = -A * bump_r(col) * bump_t(row) * noise(row)
       [depression centered on ring band at angular position theta_center]

    2. Geometric warp: col displacement d = -h (inward radial shift)
       → ring band shifts toward smaller r at the dent location
       → This is the geometric deformation (surface physically displaced)

    3. Shading from height normals:
       nx = -∂h/∂col: radial slope → leading/trailing shadow-highlight
       ny = -∂h/∂row: angular slope → entry/exit edges
       → Produces highlight on approaching edge, shadow in dent, highlight on exit

    Result: dent looks like it physically deforms the surface, not a dark patch.
    The bright ring band is locally displaced + shadowed with directional shading.
"""

import cv2
import numpy as np
from ._render import shade_from_height


def _smooth_1d(arr: np.ndarray, k: int) -> np.ndarray:
    k = max(int(k), 1) | 1
    return cv2.GaussianBlur(arr.reshape(-1, 1).astype(np.float32), (1, k), 0).flatten()


def synthesize_dent(
    polar:         np.ndarray,
    polar_illum:   np.ndarray,
    r_ring:        float,
    r_max:         float,
    theta_center:  float,
    theta_span:    float,
    seed:          int   = 42,
    amplitude:     float = 8.0,    # col units; radial inward shift at peak
    h_sigma_r:     float = 6.0,    # col units; radial extent of dent
    k_diffuse:     float = 220.0,
    k_specular:    float = 80.0,
    gamma:         float = 1.8,
    shininess:     int   = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    amplitude  : radial shift in polar col units (8 cols ≈ 8 * r_max/W metres)
    h_sigma_r  : Gaussian half-width in radial direction [col units]
    """
    H, W = polar.shape[:2]
    rng  = np.random.RandomState(seed)

    row_idx = np.arange(H, dtype=np.float32)
    col_idx = np.arange(W, dtype=np.float32)
    ROW, COL = np.meshgrid(row_idx, col_idx, indexing="ij")

    col_ring = float(np.clip(r_ring / r_max * W, 1, W - 2))

    # ── Radial envelope — f(r) only: Gaussian centred at ring ────────────
    bump_r = np.exp(-0.5 * ((COL - col_ring) / (h_sigma_r + 1e-6)) ** 2)

    # ── Angular envelope — f(θ) only: smooth taper ───────────────────────
    row_center = (theta_center % (2 * np.pi)) / (2 * np.pi) * H
    drow       = np.abs(ROW - row_center)
    drow       = np.minimum(drow, H - drow)
    row_half   = (theta_span / (2 * np.pi)) * H / 2.0
    taper_rows = max(row_half * 0.20, 3.0)
    bump_t     = np.clip((row_half - drow) / (taper_rows + 1e-6), 0.0, 1.0)

    # ── Stochastic variation — f(θ) only ─────────────────────────────────
    raw_n  = rng.normal(0, 1.0, H).astype(np.float32)
    noise  = _smooth_1d(raw_n, H // 10)
    noise  = 0.75 + 0.5 * (noise - noise.min()) / (noise.ptp() + 1e-6)   # [0.75, 1.25]
    noise_g = noise[:, np.newaxis]                                         # (H, 1)

    # ── Height field (col units, negative = inward depression) ───────────
    # h(r,θ) = -amplitude * f(r) * f(θ) * noise(θ)
    h = -amplitude * bump_r * bump_t * noise_g   # (H, W)

    # ── Geometric warp: shift col by -h (inward radial displacement) ─────
    # map_col = col - h = col + |h|  (fetch from outer r → ring pulls inward)
    map_row = ROW.astype(np.float32)
    map_col = np.clip((COL - h).astype(np.float32), 0, W - 1)
    warped  = cv2.remap(polar, map_col, map_row, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

    # ── Shading from height normals (applied to warped image) ────────────
    # This creates highlight on leading edge, shadow in dent valley,
    # highlight on trailing edge — physically correct for a dent.
    delta  = shade_from_height(h, polar_illum,
                               k_diffuse=k_diffuse, k_specular=k_specular,
                               gamma=gamma, shininess=shininess)

    # Envelope for blending
    envelope = np.clip(bump_r * bump_t * 2.5, 0.0, 1.0)

    out_f     = warped.astype(np.float32) + delta * envelope[:, :, np.newaxis]
    polar_out = np.clip(out_f, 0, 255).astype(np.uint8)

    mask_polar = envelope.astype(np.float32) * np.clip(-h / (amplitude + 1e-6), 0, 1)
    mask_polar = cv2.GaussianBlur(mask_polar.astype(np.float32), (5, 5), 0)

    return polar_out, mask_polar
