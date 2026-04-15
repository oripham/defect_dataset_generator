"""
geometry/polar_transform.py
Cartesian ↔ Polar using cv2.warpPolar (WARP_POLAR_LINEAR).

cv2.warpPolar convention
------------------------
  polar output rows  → angle  θ  ∈ [0, 2π)   row  i  ↔  θ = i/H * 2π
  polar output cols  → radius r  ∈ [0, r_max] col  j  ↔  r = j/W * r_max

In polar space:
  - Ring at r = r_ring → VERTICAL BAND at col  = r_ring/r_max * W
  - Scratch along ring (arc at r_ring, θ range) → near-VERTICAL SEGMENT
  - Scratch along radius (constant θ)          → HORIZONTAL SEGMENT
  - θ=0 = 3 o'clock, θ=π/2 = 6 o'clock, θ=π = 9 o'clock, θ=3π/2 = 12 o'clock

This convention is strictly preserved by the WARP_INVERSE_MAP flag —
no coordinate arithmetic is needed for back-projection.
"""
import cv2
import numpy as np


def to_polar(
    img: np.ndarray,
    cx: float,
    cy: float,
    r_max: float,
    polar_h: int | None = None,
    polar_w: int | None = None,
) -> np.ndarray:
    """
    Cartesian → Polar via cv2.warpPolar (LINEAR).

    polar_h : θ resolution.  Default = int(2π × r_max)  (≈1 px per arc-px at r_max)
    polar_w : r resolution.  Default = int(r_max)        (1 px per radius unit)

    Returns
    -------
    polar : same dtype/channels as img, shape (polar_h, polar_w[, C])
    """
    if polar_h is None:
        polar_h = max(int(2 * np.pi * r_max), 32)
    if polar_w is None:
        polar_w = max(int(r_max), 16)

    polar = cv2.warpPolar(
        img,
        dsize=(polar_w, polar_h),
        center=(float(cx), float(cy)),
        maxRadius=float(r_max),
        flags=cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR,
    )
    return polar


def from_polar(
    polar: np.ndarray,
    cx: float,
    cy: float,
    r_max: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """
    Polar → Cartesian via cv2.warpPolar WARP_INVERSE_MAP.

    Exact inverse of to_polar — center MUST match exactly (no drift).
    """
    polar_h, polar_w = polar.shape[:2]
    return cv2.warpPolar(
        polar,
        dsize=(out_w, out_h),
        center=(float(cx), float(cy)),
        maxRadius=float(r_max),
        flags=cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP,
    )


def ring_col_idx(r_ring: float, r_max: float, polar_w: int) -> int:
    """Column index in polar image corresponding to ring radius r_ring."""
    return int(np.clip(r_ring / r_max * polar_w, 1, polar_w - 2))


def angle_to_row(theta: float, polar_h: int) -> int:
    """Row index in polar image for angle theta ∈ [0, 2π)."""
    return int((theta % (2 * np.pi)) / (2 * np.pi) * polar_h) % polar_h
