"""
cap_experiments.py — CV Synthesis for Circular Cap Products (MKA)
=================================================================
Extracted and refactored from:
  pipeline_scratch.ipynb   → synth_scratch()
  pipeline_mc.ipynb        → synth_mc_deform()
  pipeline_ring.ipynb      → synth_ring_fracture()

All functions operate on BGR numpy arrays (OpenCV convention).
"""
from __future__ import annotations
import cv2
import math
import random
import numpy as np


# ── Circle detection + Polar helpers ──────────────────────────────────────────

def detect_circle(img_gray: np.ndarray) -> tuple[int, int, int]:
    """Hough Circle detection. Returns (cx, cy, r)."""
    blurred = cv2.GaussianBlur(img_gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=80, param2=40,
        minRadius=int(min(img_gray.shape) * 0.2),
        maxRadius=int(min(img_gray.shape) * 0.48),
    )
    if circles is not None:
        c = circles[0][0]
        return (int(c[0]), int(c[1]), int(c[2]))
    # Fallback: image center
    return (img_gray.shape[1] // 2, img_gray.shape[0] // 2, min(img_gray.shape) // 3)


def to_polar(img: np.ndarray, center: tuple, max_radius: int,
             size: tuple = (720, 512)) -> np.ndarray:
    return cv2.warpPolar(img, size, center, max_radius,
                         cv2.WARP_POLAR_LINEAR + cv2.INTER_LANCZOS4)


def from_polar(img: np.ndarray, center: tuple, max_radius: int,
               osize: tuple) -> np.ndarray:
    return cv2.warpPolar(img, osize, center, max_radius,
                         cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LANCZOS4)


def find_rim_col(polar_gray: np.ndarray, min_frac: float = 0.35) -> int:
    """Find the outermost bright column (rim position) in polar space."""
    profile = polar_gray.mean(axis=0)
    W = len(profile)
    threshold = profile.max() * 0.5
    for i in range(W - 1, int(W * min_frac), -1):
        if profile[i] > threshold:
            return i
    return W - 1


# ── Scratch synthesis ──────────────────────────────────────────────────────────

def synth_scratch(
    polar_img: np.ndarray,
    r_col:     int,
    seed:      int   = 42,
    intensity: float = 0.7,
    n_events:  int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Procedural scratch in polar space. Damage types: slash, scuff, pitting.
    Returns (polar_result_bgr, mask_gray).
    """
    H, W = polar_img.shape[:2]
    rng    = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    if n_events is None:
        n_events = rng.randint(3, 6)

    trench = np.zeros((H, W), dtype=np.float32)
    glint  = np.zeros((H, W), dtype=np.float32)

    for _ in range(n_events):
        event = rng.choice(['slash', 'scuff', 'pitting'])

        if event == 'slash':
            y0     = rng.randint(0, H)
            x0     = rng.randint(W // 4, max(r_col - 20, W // 4 + 1))
            length = rng.randint(20, min(H // 3, 120))
            width  = rng.randint(1, 3)
            angle  = rng.uniform(-25, 25)
            cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            for t in range(length):
                py = int(y0 + t * sin_a)
                px = int(x0 + t * cos_a)
                if 0 <= px < r_col:
                    for dy in range(-width, width + 1):
                        ny = (py + dy) % H
                        trench[ny, px] = max(trench[ny, px], 0.55 * intensity)
                        if px + 1 < W:
                            glint[ny, px + 1] = max(glint[ny, px + 1], 0.35 * intensity)

        elif event == 'scuff':
            yc = rng.randint(H // 4, 3 * H // 4)
            xc = rng.randint(W // 4, max(r_col - 40, W // 4 + 1))
            rh = rng.randint(15, 50)
            rw = rng.randint(30, 80)
            for _ in range(rng.randint(6, 16)):
                ly = rng.randint(yc - rh // 2, yc + rh // 2)
                lx = rng.randint(xc - rw // 2, xc + rw // 2)
                ll = rng.randint(8, rw)
                for t in range(ll):
                    px = lx + t
                    if 0 <= ly < H and 0 <= px < r_col:
                        trench[ly % H, px] = max(trench[ly % H, px], 0.28 * intensity)

        elif event == 'pitting':
            for _ in range(rng.randint(5, 20)):
                py = rng.randint(0, H)
                px = rng.randint(0, max(r_col - 5, 1))
                cv2.circle(trench, (px, py), rng.randint(1, 4), 0.45 * intensity, -1)

    # Apply to polar image
    base_f = polar_img.astype(np.float32) / 255.0
    result = base_f.copy()
    if result.ndim == 3:
        result -= trench[:, :, np.newaxis] * 0.9
        result += glint[:, :, np.newaxis]  * 0.4
    else:
        result -= trench * 0.9
        result += glint  * 0.4
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    out_mask = np.clip((trench + glint) * 255 * 2.5, 0, 255).astype(np.uint8)
    out_mask = cv2.GaussianBlur(out_mask, (5, 5), 1)
    return result, out_mask


# ── MC rim deformation ─────────────────────────────────────────────────────────

def synth_mc_deform(
    polar_img:    np.ndarray,
    polar_illum:  np.ndarray,
    r_col:        int,
    r_max:        int,
    theta_center: float = math.pi,
    theta_span:   float = math.pi / 8,
    seed:         int   = 42,
    deform_strength: float = 15.0,
    warp_factor:  float = 1.0,
    k_diffuse:    float = 220.0,
    k_specular:   float = 250.0,
    shininess:    int   = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Polar-space rim deformation: Gaussian warp + jagged noise + specular shading.
    Returns (polar_result_bgr, mask_gray).
    """
    H, W   = polar_img.shape[:2]
    rng    = np.random.RandomState(seed)

    ROW, COL = np.meshgrid(np.arange(H, dtype=np.float32),
                            np.arange(W, dtype=np.float32), indexing='ij')

    # Angular (theta) Gaussian envelope
    theta      = (ROW / H) * 2 * math.pi
    angle_diff = np.abs(theta - theta_center)
    angle_diff = np.minimum(angle_diff, 2 * math.pi - angle_diff)
    env_theta  = np.exp(-0.5 * (angle_diff / (theta_span / 2 + 1e-6)) ** 2)

    # Radial Gaussian envelope (around rim column)
    rim_sigma  = max(deform_strength * 0.5, 3.0)
    col_ring   = float(np.clip(r_col, 1, W - 2))
    band_r     = np.exp(-0.5 * ((COL - col_ring) / (rim_sigma + 1e-6)) ** 2)

    envelope   = env_theta * band_r

    # Jagged noise along theta
    noise_raw = rng.normal(0, 1.0, (H, 1)).astype(np.float32)
    noise_v   = cv2.GaussianBlur(noise_raw, (1, 5), 0)
    noise_v   = (noise_v - noise_v.min()) / (np.ptp(noise_v) + 1e-6) * 2.0 - 1.0
    jagged_factor   = 1.0 + noise_v * envelope * 0.4
    jagged_envelope = envelope * jagged_factor

    binary_mask = (envelope > 0.05).astype(np.float32)
    shift_val   = jagged_envelope * deform_strength * warp_factor * binary_mask

    map_x = (COL - shift_val).astype(np.float32)
    map_y = ROW.astype(np.float32)
    polar_warped = cv2.remap(polar_img, map_x, map_y, cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REFLECT)

    # Specular shading from height field
    h_field = (shift_val / (deform_strength * warp_factor + 1e-6)).astype(np.float32)
    nx = -np.gradient(h_field, axis=1)
    ny = -np.gradient(h_field, axis=0)
    Lx = np.gradient(polar_illum, axis=1)
    Ly = np.gradient(polar_illum, axis=0)
    Lmag    = np.sqrt(Lx**2 + Ly**2) + 1e-8
    Lx, Ly  = Lx / Lmag, Ly / Lmag
    diffuse  = nx * Lx + ny * Ly
    specular = np.clip(diffuse, 0, None) ** shininess
    shading  = np.clip(k_diffuse * diffuse + k_specular * specular * 3.0, -80, 120)
    shading *= binary_mask

    result = polar_warped.astype(np.float32)
    if result.ndim == 3:
        result += shading[:, :, np.newaxis]
    else:
        result += shading
    result = np.clip(result, 0, 255).astype(np.uint8)

    out_mask = np.clip(binary_mask * envelope * 255, 0, 255).astype(np.uint8)
    return result, out_mask


# ── Ring fracture synthesis ────────────────────────────────────────────────────

def synth_ring_fracture(
    polar_img:        np.ndarray,
    r_col:            int,
    seed:             int   = 42,
    jitter_amplitude: float = 6.0,
    falloff_power:    float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random Walk rim jitter — jagged/fractured rim appearance.
    Returns (polar_result_bgr, mask_gray).
    """
    H, W = polar_img.shape[:2]
    rng  = np.random.RandomState(seed)

    # Random Walk detrended profile
    steps   = rng.normal(0, 1.0, H)
    profile = np.cumsum(steps)
    profile -= np.linspace(profile[0], profile[-1], H)
    profile  = profile * (jitter_amplitude / (np.std(profile) + 1e-6))

    ROW, COL = np.meshgrid(np.arange(H, dtype=np.float32),
                            np.arange(W, dtype=np.float32), indexing='ij')

    influence = np.exp(-((COL - r_col) ** 2) / (2 * 12.0 ** 2))
    shift_val = (influence * profile[:, np.newaxis]).astype(np.float32)

    map_x = (COL - shift_val).astype(np.float32)
    map_y = ROW.astype(np.float32)
    polar_distorted = cv2.remap(polar_img, map_x, map_y, cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REFLECT)

    # Sparse glints along fractured rim
    glint_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        x = int(r_col + profile[i])
        if 0 <= x < W and rng.rand() > 0.85:
            sz = int(rng.randint(1, 3))
            cv2.circle(polar_distorted, (x, i), sz,
                       (255, 255, 255) if polar_distorted.ndim == 3 else 255, -1)
            cv2.circle(glint_mask, (x, i), sz, 255, -1)

    # Blend mask (dùng để alpha-blend — đều 360°)
    p_mask_f = np.power(np.clip(influence, 0, 1), 1.0 / max(falloff_power, 0.1))
    p_mask_f = cv2.GaussianBlur(p_mask_f.astype(np.float32), (9, 9), 0)
    alpha_max = 0.9

    orig_f = polar_img.astype(np.float32)
    dist_f = polar_distorted.astype(np.float32)
    if orig_f.ndim == 3:
        combined = (orig_f * (1 - p_mask_f[:, :, np.newaxis] * alpha_max) +
                    dist_f * (p_mask_f[:, :, np.newaxis] * alpha_max))
    else:
        combined = orig_f * (1 - p_mask_f * alpha_max) + dist_f * (p_mask_f * alpha_max)

    result   = np.clip(combined, 0, 255).astype(np.uint8)

    # Defect mask — dựa trên displacement thực tế (thay đổi theo góc)
    displacement = np.abs(shift_val)
    if displacement.max() > 1e-6:
        defect_mask = displacement / displacement.max()
    else:
        defect_mask = np.zeros((H, W), dtype=np.float32)
    defect_mask = cv2.GaussianBlur(defect_mask.astype(np.float32), (5, 5), 0)
    defect_mask[defect_mask < 0.15] = 0
    defect_mask[glint_mask > 0] = 1.0
    out_mask = np.clip(defect_mask * 255, 0, 255).astype(np.uint8)
    return result, out_mask
