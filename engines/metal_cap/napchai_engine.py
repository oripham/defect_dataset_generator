"""
engines/napchai_engine.py — Napchai (Nắp Chai / Bottle Cap) Defect Engine
==========================================================================

CV-only synthesis — direct port from notebook pipelines:
  pipeline_mc.ipynb      → synthesize_rim_deform     → defect_type "mc_deform"
  pipeline_ring.ipynb    → synthesize_ring_fractures  → defect_type "ring_fracture"
  pipeline_scratch.ipynb → synthesize_scratch_procedural → defect_type "scratch"
  (custom)               → synthesize_dent            → defect_type "dent"

All synthesis uses Polar Transform:
  1. detect_main_circle  → Hough Circle → (cx, cy, r)
  2. to_polar            → warpPolar to (720, 512)
  3. synthesize_*        → modify in polar space
  4. from_polar          → warpPolar inverse back to original size

No SDXL / GPU required — pure NumPy + OpenCV.
"""

from __future__ import annotations

import math
import random
import base64

import cv2
import numpy as np

from ..utils import encode_b64, decode_b64


# ── Polar transform constants ──────────────────────────────────────────────────

POLAR_H = 720   # θ axis (rows)
POLAR_W = 512   # r axis (cols)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES (ported from all 3 notebooks)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_main_circle(img_gray: np.ndarray) -> tuple[int, int, int]:
    """Hough Circle detection — identical across all 3 notebooks."""
    blurred = cv2.GaussianBlur(img_gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=80, param2=40,
        minRadius=int(min(img_gray.shape) * 0.2),
        maxRadius=int(min(img_gray.shape) * 0.48),
    )
    if circles is not None:
        c = circles[0][0]
        return int(c[0]), int(c[1]), int(c[2])
    # Fallback: image center
    return img_gray.shape[1] // 2, img_gray.shape[0] // 2, min(img_gray.shape) // 3


def to_polar(img: np.ndarray, center: tuple, max_radius: int,
             size: tuple = (POLAR_H, POLAR_W)) -> np.ndarray:
    """Cartesian → Polar (same in all 3 notebooks)."""
    return cv2.warpPolar(
        img, size, center, max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.INTER_LANCZOS4,
    )


def from_polar(img: np.ndarray, center: tuple, max_radius: int,
               osize: tuple) -> np.ndarray:
    """Polar → Cartesian inverse (same in all 3 notebooks)."""
    return cv2.warpPolar(
        img, osize, center, max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LANCZOS4,
    )


def _detect_rim_col(polar_img: np.ndarray) -> int:
    """
    Find the outermost rim column in polar space.
    From pipeline_scratch Cell 5: last col with brightness > threshold.
    """
    H, W = polar_img.shape[:2]
    if len(polar_img.shape) == 3:
        polar_gray = cv2.cvtColor(polar_img, cv2.COLOR_BGR2GRAY)
    else:
        polar_gray = polar_img
    profile = polar_gray.mean(axis=0)
    threshold = profile.min() + (profile.max() - profile.min()) * 0.2
    rim_indices = np.where(profile > threshold)[0]
    if len(rim_indices) > 0:
        return int(rim_indices[-1])
    return int(W * 0.8)


def _img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# SHADING ENGINE v2 — from pipeline_mc Cell 2
# ═══════════════════════════════════════════════════════════════════════════════

def _shade_from_height(h: np.ndarray, polar_illum: np.ndarray,
                       k_diffuse=220.0, k_specular=250.0,
                       gamma=1.8, shininess=20, specular_boost=3.0) -> np.ndarray:
    """Metallic specular shading delta from height field (pipeline_mc Cell 2)."""
    nx = -np.gradient(h, axis=1)
    ny = -np.gradient(h, axis=0)
    Lx = np.gradient(polar_illum, axis=1)
    Ly = np.gradient(polar_illum, axis=0)
    Lmag = np.sqrt(Lx ** 2 + Ly ** 2) + 1e-8
    Lx, Ly = Lx / Lmag, Ly / Lmag
    diffuse = nx * Lx + ny * Ly
    nH = np.clip(0.707 + 0.5 * diffuse, 0.0, 1.0)
    specular = (nH ** shininess) * specular_boost
    mean_illum = float(polar_illum.mean()) + 1e-6
    illum_mod = np.clip((polar_illum / mean_illum) ** gamma, 0.02, 4.0)
    delta_2d = (diffuse * k_diffuse + specular * k_specular) * illum_mod
    return delta_2d[:, :, np.newaxis].repeat(3, axis=-1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS: MC DEFORM — from pipeline_mc Cells 2-3
# ═══════════════════════════════════════════════════════════════════════════════

def synthesize_rim_deform(
    polar: np.ndarray,
    polar_illum: np.ndarray,
    r_ring: int,
    r_max: int,
    theta_center: float,
    theta_span: float,
    seed: int = 42,
    deform_strength: float = 18.0,
    rim_sigma_col: float = 4.0,
    warp_factor: float = 1.0,
    k_diffuse: float = 260.0,
    k_specular: float = 300.0,
    gamma: float = 2.0,
    shininess: int = 25,
    r_mask_center: float | None = None,
    r_mask_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rim deformation in polar space with physical warp + jagged edges + shading.
    Returns (polar_out, mask_polar_float).
    Exact port of pipeline_mc Cell 3.
    """
    H, W = polar.shape[:2]
    rng = np.random.RandomState(seed)
    row_idx = np.arange(H, dtype=np.float32)
    col_idx = np.arange(W, dtype=np.float32)
    ROW, COL = np.meshgrid(row_idx, col_idx, indexing="ij")

    # Radial position of rim in polar space
    if r_mask_center is not None:
        col_ring = float(np.clip(r_mask_center / r_max * W, 1, W - 2))
    else:
        col_ring = float(np.clip(r_ring / r_max * W, 1, W - 2))
    if r_mask_width is not None:
        rim_sigma_col = float(max(r_mask_width / r_max * W / 2.0, 3.0))

    dist_r = np.abs((COL - col_ring) / (rim_sigma_col + 1e-6))
    band_r = np.exp(-0.5 * (dist_r ** 2))

    # Angular position
    row_center = (theta_center % (2 * math.pi)) / (2 * math.pi) * H
    drow = np.abs(ROW - row_center)
    drow = np.minimum(drow, H - drow)
    row_half = (theta_span / (2 * math.pi)) * H / 2.0
    taper_rows = max(row_half * 0.15, 2.0)
    taper_t = np.clip((row_half - drow) / (taper_rows + 1e-6), 0.0, 1.0)

    # Jagged noise
    raw_n = rng.normal(0, 1.5, H).astype(np.float32)
    noise_v = cv2.GaussianBlur(raw_n.reshape(-1, 1), (1, 3), 0).flatten()
    noise_v = 0.5 + 1.2 * (noise_v - noise_v.min()) / (np.ptp(noise_v) + 1e-6)

    # Height field + warp
    h = deform_strength * band_r * taper_t * noise_v[:, np.newaxis]
    map_x = COL - h * warp_factor
    map_y = ROW
    polar_warped = cv2.remap(
        polar, map_x, map_y,
        cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
    )

    # Shading
    delta = _shade_from_height(
        h, polar_illum,
        k_diffuse=k_diffuse, k_specular=k_specular,
        gamma=gamma, shininess=shininess,
    )
    envelope = np.clip(band_r * taper_t * 2.5, 0.0, 1.0)
    polar_out = np.clip(
        polar_warped.astype(np.float32) + delta * envelope[:, :, np.newaxis],
        0, 255,
    ).astype(np.uint8)
    mask_polar = cv2.GaussianBlur(envelope.astype(np.float32), (3, 3), 0)
    return polar_out, np.clip(mask_polar, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS: RING FRACTURE — from pipeline_ring Cell 7 + Cell 8
# ═══════════════════════════════════════════════════════════════════════════════

def synthesize_ring_fractures(
    polar: np.ndarray,
    r_ring_col: int,
    seed: int | None = None,
    jitter_amplitude: float = 6.0,
    falloff_width: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random Walk rim distortion in polar space.
    Returns (result_rgb, mask_uint8).
    Exact port of pipeline_ring Cells 7-8.
    """
    if seed is None:
        seed = random.randint(0, 999999)

    H, W = polar.shape[:2]
    ROW, COL = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rng = np.random.RandomState(seed)

    # 1. Random Walk noise profile
    steps = rng.normal(0, 1.0, H)
    noise_profile = np.cumsum(steps)
    lin_trend = np.linspace(noise_profile[0], noise_profile[-1], H)
    noise_profile = noise_profile - lin_trend  # detrend
    noise_profile = noise_profile * (jitter_amplitude / (np.std(noise_profile) + 1e-6))

    # 2. Displacement mapping
    dist_from_rim = np.abs(COL - r_ring_col)
    influence = np.exp(-(dist_from_rim ** 2) / (2 * (12 ** 2)))

    shift_val = (influence * noise_profile[:, np.newaxis]).astype(np.float32)
    map_x = (COL - shift_val).astype(np.float32)
    map_y = ROW.astype(np.float32)

    # 3. Remap
    polar_distorted = cv2.remap(
        polar, map_x, map_y,
        cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
    )

    # 4. Glints along rim
    glint_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        cx_rim = int(r_ring_col + noise_profile[i])
        if 0 <= cx_rim < W and rng.rand() > 0.85:
            sz = rng.randint(1, 3)
            cv2.circle(glint_mask, (cx_rim, i), sz, 255, -1)
    if len(polar_distorted.shape) == 3:
        polar_distorted[glint_mask > 0] = 255
    else:
        polar_distorted[glint_mask > 0] = 255

    # 5. Soft alpha blend (Cell 8 logic)
    mask_input = (influence * 255).astype(np.float32) / 255.0
    soft_mask = np.power(np.clip(mask_input, 0, 1), 1.0 / falloff_width)
    soft_mask = cv2.GaussianBlur(soft_mask, (9, 9), 0)

    return polar_distorted, (soft_mask * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS: SCRATCH — from pipeline_scratch Cell 6
# ═══════════════════════════════════════════════════════════════════════════════

def synthesize_scratch_procedural(
    polar_img: np.ndarray,
    polar_mask: np.ndarray,
    rim_col: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Procedural scratch: slash, scuff, pitting events in polar space.
    Returns (polar_scratched, clean_mask_uint8).
    Exact port of pipeline_scratch Cell 6.
    """
    random.seed(seed)
    np.random.seed(seed)

    H, W = polar_img.shape[:2]
    clean_mask = polar_mask.copy()
    clean_mask[:, rim_col:] = 0   # only inside rim
    mask_f = (clean_mask.astype(np.float32) / 255.0)
    if len(polar_img.shape) == 3:
        mask_f_3d = mask_f[:, :, np.newaxis]
    else:
        mask_f_3d = mask_f

    base_f = polar_img.astype(np.float32) / 255.0
    trench_layer = np.zeros((H, W), dtype=np.float32)
    glint_layer  = np.zeros((H, W), dtype=np.float32)

    y_coords, x_coords = np.where(clean_mask > 0)
    if len(x_coords) == 0:
        return polar_img, clean_mask

    num_events = random.randint(4, 7)
    for _ in range(num_events):
        event_type = random.choice(["slash", "scuff", "pitting"])
        idx = random.randint(0, len(x_coords) - 1)
        sx, sy = int(x_coords[idx]), int(y_coords[idx])

        if event_type == "slash":
            angle  = random.uniform(0, 2 * math.pi)
            length = random.randint(40, 150)
            cx, cy = float(sx), float(sy)
            for _ in range(length):
                cx += math.cos(angle) + random.uniform(-0.2, 0.2)
                cy += math.sin(angle) + random.uniform(-0.2, 0.2)
                tx, ty = int(cx), int(cy)
                if 0 <= tx < W and 0 <= ty < H:
                    cv2.circle(trench_layer, (tx, ty), random.randint(1, 2), 1.0, -1)
                    if random.random() > 0.8:
                        glint_layer[ty, tx] = 0.5

        elif event_type == "scuff":
            base_angle = random.uniform(0, 2 * math.pi)
            num_lines  = random.randint(3, 6)
            for _ in range(num_lines):
                lx = sx + random.randint(-5, 5)
                ly = sy + random.randint(-5, 5)
                length = random.randint(15, 40)
                for i in range(length):
                    tx = int(lx + i * math.cos(base_angle))
                    ty = int(ly + i * math.sin(base_angle))
                    if 0 <= tx < W and 0 <= ty < H:
                        trench_layer[ty, tx] = 0.7

        elif event_type == "pitting":
            for _ in range(random.randint(5, 15)):
                tx = sx + random.randint(-10, 10)
                ty = sy + random.randint(-10, 10)
                if 0 <= tx < W and 0 <= ty < H:
                    cv2.circle(trench_layer, (tx, ty), 1, 0.9, -1)
                    if random.random() > 0.5:
                        glint_layer[ty, tx] = 0.7

    trench_layer = cv2.GaussianBlur(trench_layer, (3, 3), 0.6)

    if len(base_f.shape) == 3:
        tl = trench_layer[:, :, np.newaxis]
        gl = glint_layer[:, :, np.newaxis]
    else:
        tl = trench_layer
        gl = glint_layer

    res_f = base_f * (1.0 - tl * 0.65 * mask_f_3d)
    res_f = np.clip(res_f + gl * 0.35 * mask_f_3d, 0, 1)
    return (res_f * 255.0).astype(np.uint8), clean_mask


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS: DENT (Lõm) — custom polar inward dent (no notebook)
# Similar to mc_deform but inward + smaller span + Gaussian depth
# ═══════════════════════════════════════════════════════════════════════════════

def synthesize_dent(
    polar: np.ndarray,
    r_ring: int,
    r_max: int,
    theta_center: float,
    seed: int = 42,
    intensity: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inward dent on the cap surface in polar space.
    Returns (polar_out, mask_polar_float).
    """
    H, W = polar.shape[:2]
    rng = np.random.RandomState(seed)
    row_idx = np.arange(H, dtype=np.float32)
    col_idx = np.arange(W, dtype=np.float32)
    ROW, COL = np.meshgrid(row_idx, col_idx, indexing="ij")

    col_ring = float(np.clip(r_ring / r_max * W, 1, W - 2))
    # Smaller span for dent vs MC deform
    theta_span = rng.uniform(math.pi / 18, math.pi / 8)
    deform_strength = intensity * 12.0

    rim_sigma_col = float(max(W * 0.05, 3.0))
    dist_r = np.abs((COL - col_ring) / (rim_sigma_col + 1e-6))
    band_r = np.exp(-0.5 * (dist_r ** 2))

    row_center = (theta_center % (2 * math.pi)) / (2 * math.pi) * H
    drow = np.abs(ROW - row_center)
    drow = np.minimum(drow, H - drow)
    row_half = (theta_span / (2 * math.pi)) * H / 2.0
    sigma_row = max(row_half / 1.5, 2.0)
    taper_g = np.exp(-0.5 * (drow / (sigma_row + 1e-6)) ** 2)

    envelope = band_r * taper_g

    # Inward push (positive shift = move pixels left = inward dent)
    noise_raw = rng.normal(0, 1.0, (H, 1)).astype(np.float32)
    noise_v = cv2.GaussianBlur(noise_raw, (1, 5), 0)
    noise_v = (noise_v - noise_v.min()) / (np.ptp(noise_v) + 1e-6) * 2.0 - 1.0
    jagged_f = 1.0 + (noise_v * envelope * 0.4)

    shift_val = (envelope * jagged_f * deform_strength * 4.0).astype(np.float32)
    map_x = (COL + shift_val).astype(np.float32)   # +shift = inward
    map_y = ROW.astype(np.float32)

    polar_out = cv2.remap(
        polar, map_x, map_y,
        cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
    )
    # Darken dented area slightly
    dark_factor = np.clip(1.0 - envelope * 0.25, 0.6, 1.0)
    if len(polar_out.shape) == 3:
        polar_out = np.clip(
            polar_out.astype(np.float32) * dark_factor[:, :, np.newaxis], 0, 255
        ).astype(np.uint8)
    else:
        polar_out = np.clip(
            polar_out.astype(np.float32) * dark_factor, 0, 255
        ).astype(np.uint8)

    mask_polar = cv2.GaussianBlur(envelope.astype(np.float32), (3, 3), 0)
    return polar_out, np.clip(mask_polar, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-GENERATE POLAR MASK for scratch (no user mask needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_scratch_polar_mask(H: int, W: int, rim_col: int,
                              theta_center: float, theta_span: float) -> np.ndarray:
    """
    Generate a polar-space mask at a given angular position + radial band near rim.
    Used when user doesn't draw a mask.
    """
    row_center = (theta_center % (2 * math.pi)) / (2 * math.pi) * H
    row_half   = max((theta_span / (2 * math.pi)) * H / 2.0, 10.0)

    mask = np.zeros((H, W), dtype=np.uint8)
    r_start = max(0,       int(rim_col * 0.5))
    r_end   = min(W - 1,   int(rim_col - 2))
    if r_end <= r_start:
        r_end = min(W - 1, r_start + 30)

    y0 = int(max(0,     row_center - row_half))
    y1 = int(min(H - 1, row_center + row_half))
    if y0 < y1:
        mask[y0:y1, r_start:r_end] = 255
    # Handle wrap-around
    if y0 < 0:
        mask[0:int(row_center + row_half), r_start:r_end] = 255
        mask[int(H + row_center - row_half):, r_start:r_end] = 255

    return cv2.GaussianBlur(mask, (5, 5), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# MASK UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def mask_xy_to_polar_params(
    mask: np.ndarray,
    center: tuple[int, int],
    radius: int,
) -> tuple[float, float, float, float]:
    """
    Convert a Cartesian binary mask to polar defect parameters.
    Returns (theta_center, theta_span, r_mask_center, r_mask_width).
    Exact port of pipeline_mc Cell 4.
    """
    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        return 0.0, math.pi / 6, float(radius), 20.0
    dx = xs.astype(float) - center[0]
    dy = ys.astype(float) - center[1]
    angles = np.arctan2(dy, dx) % (2 * math.pi)
    radii  = np.sqrt(dx ** 2 + dy ** 2)
    sorted_a = np.sort(angles)
    diffs    = np.append(
        np.diff(sorted_a),
        (2 * math.pi - sorted_a[-1] + sorted_a[0]),
    )
    gap_idx     = int(np.argmax(diffs))
    start_angle = sorted_a[gap_idx + 1] if gap_idx < len(sorted_a) - 1 else sorted_a[0]
    end_angle   = sorted_a[gap_idx]
    span        = (end_angle - start_angle) % (2 * math.pi)
    return (
        (start_angle + span / 2) % (2 * math.pi),
        max(span, math.pi / 18),
        float(np.mean(radii)),
        float(max(float(np.ptp(radii)), 10.0)),
    )


def _decode_mask_b64(mask_b64: str, target_w: int, target_h: int) -> np.ndarray:
    """Decode base64 mask to grayscale uint8, resized to target dimensions."""
    rgb  = decode_b64(mask_b64)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if len(rgb.shape) == 3 else rgb
    gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    _, out = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC: generate()
# ═══════════════════════════════════════════════════════════════════════════════

def generate(
    base_image_b64: str,
    defect_type: str,
    params: dict,
) -> dict:
    """
    Generate one Napchai defect image.

    Parameters
    ----------
    base_image_b64 : base64 PNG — OK image (RGB)
    defect_type    : "scratch" | "mc_deform" | "ring_fracture" | "dent"
    params         : dict with seed, intensity, and optional defect-specific keys

    Returns
    -------
    dict:
        result_image : base64 PNG
        mask_b64     : base64 PNG (defect region mask)
        engine       : "cv"
        metadata     : dict
    """
    seed         = int(params.get("seed", 42))
    intensity    = float(params.get("intensity", 0.7))
    mask_b64_str = params.get("mask_b64")   # optional user-drawn Cartesian mask
    rng          = np.random.RandomState(seed)

    # ── Decode image ────────────────────────────────────────────────────────
    img_rgb = decode_b64(base_image_b64)            # H×W×3 RGB uint8
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = img_bgr.shape[:2]

    # ── Detect circle ───────────────────────────────────────────────────────
    cx, cy, radius = detect_main_circle(img_gray)
    max_radius = int(radius * 1.3)
    center     = (cx, cy)

    # ── Polar transform ─────────────────────────────────────────────────────
    polar_img = to_polar(img_bgr, center, max_radius, (POLAR_H, POLAR_W))
    p_h, p_w = polar_img.shape[:2]   # actual output shape (may differ from POLAR_H/W due to cv2 dsize convention)

    # ── Rim column detection ────────────────────────────────────────────────
    rim_col = _detect_rim_col(polar_img)

    # ── Random angular position ──────────────────────────────────────────────
    theta_center = float(params.get("theta_center", rng.uniform(0, 2 * math.pi)))
    theta_span   = float(params.get("theta_span",   rng.uniform(math.pi / 10, math.pi / 4)))

    # ── Dispatch synthesis ──────────────────────────────────────────────────
    try:
        if defect_type == "scratch":
            if mask_b64_str:
                # User drew a position mask — project Cartesian → polar
                # Must use same dsize=(POLAR_H,POLAR_W) as the base image so shapes match
                cart_mask = _decode_mask_b64(mask_b64_str, orig_w, orig_h)
                polar_mask = to_polar(cart_mask, center, max_radius, (POLAR_H, POLAR_W))
            else:
                polar_mask = _make_scratch_polar_mask(
                    p_h, p_w, rim_col, theta_center, theta_span
                )
            polar_out, defect_mask_polar = synthesize_scratch_procedural(
                polar_img, polar_mask, rim_col, seed=seed,
            )

        elif defect_type == "mc_deform":
            deform_strength = float(params.get("deform_strength", intensity * 25.0))
            polar_illum = cv2.cvtColor(polar_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            # If user drew a mask, extract polar params from it
            r_mask_c = r_mask_w = None
            if mask_b64_str:
                cart_mask = _decode_mask_b64(mask_b64_str, orig_w, orig_h)
                theta_center, theta_span, r_mask_c, r_mask_w = mask_xy_to_polar_params(
                    cart_mask, (cx, cy), radius
                )
            polar_out, mask_float = synthesize_rim_deform(
                polar_img, polar_illum,
                r_ring=rim_col,
                r_max=max_radius,
                theta_center=theta_center,
                theta_span=theta_span,
                seed=seed,
                deform_strength=deform_strength,
                r_mask_center=r_mask_c,
                r_mask_width=r_mask_w,
            )
            defect_mask_polar = (mask_float * 255).astype(np.uint8)

        elif defect_type == "ring_fracture":
            jitter = float(params.get("jitter_amplitude", intensity * 12.0))
            polar_out, defect_mask_polar = synthesize_ring_fractures(
                polar_img, rim_col,
                seed=seed,
                jitter_amplitude=jitter,
                falloff_width=float(params.get("falloff_width", 1.0)),
            )

        elif defect_type == "dent":
            polar_out, mask_float = synthesize_dent(
                polar_img, rim_col, max_radius,
                theta_center=theta_center,
                seed=seed,
                intensity=intensity,
            )
            defect_mask_polar = (mask_float * 255).astype(np.uint8)

        else:
            return {"error": f"Unknown defect_type: {defect_type!r}"}

    except Exception as e:
        return {"error": f"Synthesis error: {e}"}

    # ── Inverse polar transform ──────────────────────────────────────────────
    result_bgr = from_polar(polar_out, center, max_radius, (orig_w, orig_h))

    # Blend result — ring_fracture uses soft blend (Cell 8 of pipeline_ring)
    if defect_type == "ring_fracture":
        mask_cart = from_polar(defect_mask_polar, center, max_radius, (orig_w, orig_h))
        if len(mask_cart.shape) == 3:
            soft = mask_cart[:, :, 0].astype(np.float32) / 255.0
        else:
            soft = mask_cart.astype(np.float32) / 255.0
        soft = soft[:, :, np.newaxis]
        alpha_max = 0.9
        blended = (
            img_bgr.astype(np.float32) * (1 - soft * alpha_max)
            + result_bgr.astype(np.float32) * (soft * alpha_max)
        )
        result_bgr = np.clip(blended, 0, 255).astype(np.uint8)

    # ── Build output mask (cartesian) ────────────────────────────────────────
    mask_cart = from_polar(defect_mask_polar, center, max_radius, (orig_w, orig_h))
    if len(mask_cart.shape) == 3:
        mask_gray = mask_cart[:, :, 0]
    else:
        mask_gray = mask_cart
    _, mask_out = cv2.imencode(".png", mask_gray)
    mask_b64 = base64.b64encode(mask_out).decode("utf-8")

    # ── Encode result ────────────────────────────────────────────────────────
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return {
        "result_image": encode_b64(result_rgb),
        "mask_b64":     mask_b64,
        "engine":       "cv",
        "metadata": {
            "defect_type":   defect_type,
            "circle":        [cx, cy, radius],
            "rim_col":       rim_col,
            "theta_center":  round(theta_center, 3),
            "theta_span":    round(theta_span, 3),
            "params":        params,
        },
    }
