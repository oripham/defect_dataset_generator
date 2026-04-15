"""defect/_render.py — shared height-field shading engine
=========================================================

Physical model:
    1. Defect defined as height field h(row, col)  [depression → negative]
    2. Surface normals from gradient: n ≈ (-∂h/∂col, -∂h/∂row, 1)
    3. Light direction from local illumination gradient: L = normalize(∇I)
    4. Lambert diffuse: D = n · L
    5. Blinn-Phong specular: S = (n · H)^shininess
    6. Illumination modulation: output += (D*k_d + S*k_s) * (I_local/I_mean)^γ

warpPolar convention: rows=θ (angle), cols=r (radius)
"""

import numpy as np
import cv2


def shade_from_height(
    h: np.ndarray,
    polar_illum: np.ndarray,
    k_diffuse: float  = 220.0,
    k_specular: float = 100.0,
    gamma: float      = 1.8,
    shininess: int    = 6,
) -> np.ndarray:
    """
    Convert height field to shading delta (add to polar_img.float32).

    Parameters
    ----------
    h            : (H, W) float32  Surface height, negative = depression.
                   Units are polar-pixel radial distance (col units).
    polar_illum  : (H, W) float32  Local illumination in [0, 1].
    k_diffuse    : Diffuse shading scale  [intensity units, ~0-255].
    k_specular   : Specular shading scale [intensity units].
    gamma        : Power-law exponent for illumination modulation.
    shininess    : Blinn-Phong specular exponent.

    Returns
    -------
    delta : (H, W, 3) float32  Shading delta to add to polar_img.float32.
            Positive = brighter, negative = darker.
    """
    # ── Surface normals (outward from surface) ────────────────────────────
    # n ≈ (-∂h/∂col, -∂h/∂row, 1)  (not normalized; z=1 toward viewer)
    nx = -np.gradient(h, axis=1)   # radial component   (H, W)
    ny = -np.gradient(h, axis=0)   # angular component  (H, W)

    # ── Light direction from illumination gradient ────────────────────────
    # The gradient of the local illumination approximates the dominant
    # light direction projected onto the surface plane.
    Lx = np.gradient(polar_illum, axis=1)   # (H, W)
    Ly = np.gradient(polar_illum, axis=0)   # (H, W)
    Lmag = np.sqrt(Lx ** 2 + Ly ** 2) + 1e-8
    Lx = Lx / Lmag                          # normalized
    Ly = Ly / Lmag

    # ── Lambert diffuse: D = n · L ────────────────────────────────────────
    # n = (nx, ny, 1), L = (Lx, Ly, 0) (in-plane light)
    diffuse = nx * Lx + ny * Ly             # [-inf, +inf] but typically [-1, 1]

    # ── Blinn-Phong specular ──────────────────────────────────────────────
    # View direction V = (0, 0, 1)  (camera looking straight down)
    # Half-vector H = normalize(L + V) ≈ normalize(Lx, Ly, 1)
    # n · H ≈ (nz + 0.5*(nx*Lx + ny*Ly)) / sqrt(...)
    # Simplified: n·H ≈ clip(0.707 + 0.5*diffuse, 0, 1)
    nH = np.clip(0.707 + 0.5 * diffuse, 0.0, 1.0)
    specular = nH ** shininess              # [0, 1]

    # ── Local illumination power-law modulation ───────────────────────────
    mean_illum = float(polar_illum.mean()) + 1e-6
    illum_mod = np.clip((polar_illum / mean_illum) ** gamma, 0.02, 4.0)

    # ── Combined shading delta ────────────────────────────────────────────
    # diffuse contribution: creates dark-on-one-side, bright-on-other-side
    # specular contribution: bright highlight at groove walls / ridge peaks
    delta_2d = (diffuse * k_diffuse + specular * k_specular) * illum_mod

    return delta_2d[:, :, np.newaxis].repeat(3, axis=-1).astype(np.float32)


def apply_height_shading(
    polar: np.ndarray,
    polar_illum: np.ndarray,
    h: np.ndarray,
    envelope: np.ndarray,
    **shade_kwargs,
) -> np.ndarray:
    """
    Apply height-field shading to polar image within an envelope mask.

    Parameters
    ----------
    polar      : (H, W, 3) uint8  Input polar image.
    polar_illum: (H, W) float32
    h          : (H, W) float32  Height field (col units).
    envelope   : (H, W) float32  Blend weight [0, 1].
    **shade_kwargs: passed to shade_from_height

    Returns
    -------
    (H, W, 3) uint8
    """
    delta = shade_from_height(h, polar_illum, **shade_kwargs)
    out = polar.astype(np.float32) + delta * envelope[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)
