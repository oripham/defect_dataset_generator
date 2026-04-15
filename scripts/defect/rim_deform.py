"""defect/rim_deform.py  v5 — height-field based rim deformation (MC変形)
=======================================================================

warpPolar convention: rows=θ (angle), cols=r (radius)

Physical model:
    The outer bright rim at r_ring has locally collapsed / been dented inward.
    This is modeled as:

    1. Height field h(row, col) = -A * band_r(col) * taper_t(row) * noise(row)
       band_r: narrow Gaussian centered at col_ring [the rim band]
       taper_t: smooth angular window at theta_center

    2. The rim pixels' surface normals change because the band is depressed:
       - The bright specular highlight disappears (normal tilts away from camera)
       - Entry/exit of the collapsed zone have steep slopes → strong highlight+shadow

    3. Shading from height normals automatically produces:
       - Angular entry edge: bright highlight (slope toward light)
       - Collapsed center: dark (normal away from camera/light)
       - Angular exit edge: shadow (slope away from light)

    This is NOT a dark patch — it is a surface deformation with directional shading.
"""

import cv2
import numpy as np
from ._render import shade_from_height


def _smooth_1d(arr: np.ndarray, k: int) -> np.ndarray:
    k = max(int(k), 1) | 1
    return cv2.GaussianBlur(arr.reshape(-1, 1).astype(np.float32), (1, k), 0).flatten()


def synthesize_rim_deform(
    polar:          np.ndarray,
    polar_illum:    np.ndarray,
    r_ring:         float,
    r_max:          float,
    theta_center:   float,
    theta_span:     float,
    seed:           int   = 42,
    collapse_depth: float = 14.0,   # col units; how far rim is pushed inward
    rim_sigma_col:  float = 4.0,    # col units; radial half-width of rim band
    k_diffuse:      float = 260.0,  # Lambert scale — larger → more visible shading
    k_specular:     float = 120.0,  # Specular scale — bright highlights at edges
    gamma:          float = 2.0,
    shininess:      int   = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    collapse_depth  : radial inward shift at peak of collapse [col units]
    rim_sigma_col   : Gaussian σ for rim band width [col units]
    """
    H, W = polar.shape[:2]
    rng  = np.random.RandomState(seed)

    row_idx = np.arange(H, dtype=np.float32)
    col_idx = np.arange(W, dtype=np.float32)
    ROW, COL = np.meshgrid(row_idx, col_idx, indexing="ij")

    col_ring = float(np.clip(r_ring / r_max * W, 1, W - 2))

    # ── Radial band — f(r) only: Gaussian centred at rim ─────────────────
    band_r = np.exp(-0.5 * ((COL - col_ring) / (rim_sigma_col + 1e-6)) ** 2)

    # ── Angular window — f(θ) only: smooth trapezoid ─────────────────────
    row_center = (theta_center % (2 * np.pi)) / (2 * np.pi) * H
    drow       = np.abs(ROW - row_center)
    drow       = np.minimum(drow, H - drow)
    row_half   = (theta_span / (2 * np.pi)) * H / 2.0
    taper_rows = max(row_half * 0.25, 4.0)
    taper_t    = np.clip((row_half - drow) / (taper_rows + 1e-6), 0.0, 1.0)

    # ── Stochastic variation — f(θ) only ─────────────────────────────────
    raw_n  = rng.normal(0, 1.0, H).astype(np.float32)
    noise  = _smooth_1d(raw_n, H // 12)
    noise  = 0.7 + 0.6 * (noise - noise.min()) / (noise.ptp() + 1e-6)   # [0.7, 1.3]
    noise_g = noise[:, np.newaxis]                                        # (H, 1)

    # ── Height field: rim band depression in angular window ───────────────
    # h(r,θ) = -collapse_depth * band_r(r) * taper_t(θ) * noise(θ)
    # Negative = inward depression of the rim surface
    h = -collapse_depth * band_r * taper_t * noise_g   # (H, W)

    # ── Shading from height field normals ─────────────────────────────────
    # nx = -∂h/∂col: on the radial slope of the rim band → bright/dark walls
    # ny = -∂h/∂row: on the angular entry/exit → strong highlight+shadow
    # This produces the characteristic "dent in the rim" appearance:
    #   entry highlight → collapsed dark zone → exit shadow
    delta = shade_from_height(h, polar_illum,
                              k_diffuse=k_diffuse, k_specular=k_specular,
                              gamma=gamma, shininess=shininess)

    # Envelope (where the defect applies)
    envelope = np.clip(band_r * taper_t * 2.0, 0.0, 1.0)

    out_f     = polar.astype(np.float32) + delta * envelope[:, :, np.newaxis]
    polar_out = np.clip(out_f, 0, 255).astype(np.uint8)

    mask_polar = envelope.astype(np.float32) * np.clip(-h / (collapse_depth + 1e-6), 0, 1)
    mask_polar = cv2.GaussianBlur(mask_polar.astype(np.float32), (5, 5), 0)

    return polar_out, mask_polar
