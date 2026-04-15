"""
engines/scratch_napchai_engine.py — Scratch Napchai (Xước vành polar)
=======================================================================
Port of pipeline_scratch.ipynb.

Pipeline (exactly as notebook):
  1. Hough Circle → center, max_radius = radius * 1.3
  2. Polar transform 720×512
  3. r_ring_col = last col where profile > min + (max-min)*0.2  ← outer boundary
     (pipeline_scratch cell-6 — different from mc_deform/ring_fracture)
  4. User mask (base64) → to_polar, or auto band around outer rim
  5. synthesize_scratch_procedural: slash / scuff / pitting in polar space
  6. Inverse polar → cv_res, mask_res
  7. SDXL ControlNet-Depth + IP-Adapter Plus refine
     - TARGET_SIZE = (768, 768)
     - ip_image at (224, 224), ip_scale=0.8 (0.5 if no ref)
     - strength=0.14, guidance=7.5, steps=30
  8. ai_res_low.resize(original_size, LANCZOS)  ← explicit LANCZOS

API endpoint: POST /api/metal_cap/preview  {defect_type: "scratch"}
"""
from __future__ import annotations

import math
import random as _random
import base64 as _b64

import cv2
import numpy as np
from PIL import Image as _PIL

from .utils import encode_b64, decode_b64
from ._napchai_models import get_pipe, get_depth_est, get_lock

# ── Constants ─────────────────────────────────────────────────────────────────
POLAR_H = 720
POLAR_W = 512

# ── SDXL config (pipeline_scratch cell-10) ───────────────────────────────────
_TARGET   = (768, 768)
_PROMPT   = (
    "jagged metallic fracture, deep industrial micro-cracks, "
    "weathered steel texture, hyper-realistic, 8k, industrial damage"
)
_NEG      = "color, bronze, gold, plastic, blurry, smooth, artificial seam, lowres, noise"
_STRENGTH = 0.14
_GUIDANCE = 7.5
_STEPS    = 30
# ip_scale: 0.8 if ref provided, 0.5 if no ref (pipeline_scratch cell-10 exact)


# ── Polar helpers ─────────────────────────────────────────────────────────────

def _detect_circle(gray):
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 1.2, 100,
        param1=80, param2=40,
        minRadius=int(min(gray.shape) * 0.2),
        maxRadius=int(min(gray.shape) * 0.48),
    )
    if circles is None:
        return (gray.shape[1] // 2, gray.shape[0] // 2, min(gray.shape) // 3)
    c = circles[0][0]
    return (int(c[0]), int(c[1]), int(c[2]))


def _to_polar(img, center, max_radius):
    return cv2.warpPolar(img, (POLAR_W, POLAR_H), center, max_radius,
                         cv2.WARP_POLAR_LINEAR + cv2.INTER_LANCZOS4)


def _from_polar(polar, center, max_radius, osize):
    return cv2.warpPolar(polar, osize, center, max_radius,
                         cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LANCZOS4)


def _find_rim_col_outer(polar_gray: np.ndarray) -> int:
    """
    pipeline_scratch cell-6: last col where mean > min + (max-min)*0.2.
    Finds the outermost meaningful boundary of the rim.
    """
    profile   = polar_gray.mean(axis=0)
    threshold = profile.min() + (profile.max() - profile.min()) * 0.2
    indices   = np.where(profile > threshold)[0]
    if len(indices) > 0:
        return int(indices[-1])
    return int(POLAR_W * 0.8)


# ── CV synthesis (pipeline_scratch cell-7) ───────────────────────────────────

def _synthesize_scratch_procedural(polar_img, polar_mask, rim_col):
    """
    Diverse procedural damage: Slashes, Scuffs, Pits — filtered to mask region.
    Exactly as notebook cell-7.
    """
    H, W = polar_img.shape[:2]
    clean_mask = polar_mask.copy()
    clean_mask[:, rim_col:] = 0   # zero out beyond outer rim

    mask_f = clean_mask.astype(np.float32) / 255.0
    if len(polar_img.shape) == 3:
        mask_f3 = mask_f[:, :, np.newaxis]
    else:
        mask_f3 = mask_f

    base_f       = polar_img.astype(np.float32) / 255.0
    trench_layer = np.zeros((H, W), dtype=np.float32)
    glint_layer  = np.zeros((H, W), dtype=np.float32)

    y_coords, x_coords = np.where(clean_mask > 0)
    if len(x_coords) > 0:
        num_events = _random.randint(4, 7)
        for _ in range(num_events):
            event_type = _random.choice(["slash", "scuff", "pitting"])
            idx        = _random.randint(0, len(x_coords) - 1)
            sx, sy     = int(x_coords[idx]), int(y_coords[idx])

            if event_type == "slash":
                angle  = _random.uniform(0, 2 * math.pi)
                length = _random.randint(40, 150)
                curr_x, curr_y = float(sx), float(sy)
                for _ in range(length):
                    curr_x += math.cos(angle) + _random.uniform(-0.2, 0.2)
                    curr_y += math.sin(angle) + _random.uniform(-0.2, 0.2)
                    tx, ty = int(curr_x), int(curr_y)
                    if 0 <= tx < W and 0 <= ty < H:
                        cv2.circle(trench_layer, (tx, ty), _random.randint(1, 2), 1.0, -1)
                        if _random.random() > 0.8:
                            glint_layer[ty, tx] = 0.5

            elif event_type == "scuff":
                base_angle = _random.uniform(0, 2 * math.pi)
                num_lines  = _random.randint(3, 6)
                for _ in range(num_lines):
                    lx = sx + _random.randint(-5, 5)
                    ly = sy + _random.randint(-5, 5)
                    for i in range(_random.randint(15, 40)):
                        tx = int(lx + i * math.cos(base_angle))
                        ty = int(ly + i * math.sin(base_angle))
                        if 0 <= tx < W and 0 <= ty < H:
                            trench_layer[ty, tx] = 0.7

            elif event_type == "pitting":
                for _ in range(_random.randint(5, 15)):
                    tx = sx + _random.randint(-10, 10)
                    ty = sy + _random.randint(-10, 10)
                    if 0 <= tx < W and 0 <= ty < H:
                        cv2.circle(trench_layer, (tx, ty), 1, 0.9, -1)
                        if _random.random() > 0.5:
                            glint_layer[ty, tx] = 0.7

        trench_layer = cv2.GaussianBlur(trench_layer, (3, 3), 0.6)
        if len(base_f.shape) == 3:
            trench_3 = trench_layer[:, :, np.newaxis]
            glint_3  = glint_layer[:, :, np.newaxis]
        else:
            trench_3 = trench_layer
            glint_3  = glint_layer

        res_f = base_f * (1.0 - trench_3 * 0.65 * mask_f3)
        res_f = np.clip(res_f + glint_3 * 0.35 * mask_f3, 0, 1)
        return (res_f * 255.0).astype(np.uint8), clean_mask

    return polar_img, clean_mask


# ── CV step ───────────────────────────────────────────────────────────────────

def _cv_step(img_rgb: np.ndarray, params: dict):
    seed = int(params.get("seed", 42))
    _random.seed(seed)
    np.random.seed(seed)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cx, cy, radius = _detect_circle(gray)
    center     = (cx, cy)
    max_radius = int(radius * 1.3)

    polar_img  = _to_polar(img_rgb, center, max_radius)
    polar_gray = cv2.cvtColor(polar_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r_ring_col = _find_rim_col_outer(polar_gray)

    # Polar mask: user-drawn (base64) or auto-band around rim
    user_mask_b64 = params.get("mask_b64") or params.get("user_mask_b64")
    if user_mask_b64:
        arr       = np.frombuffer(_b64.b64decode(user_mask_b64), np.uint8)
        user_mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if user_mask is not None:
            user_mask      = cv2.resize(user_mask, (img_rgb.shape[1], img_rgb.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            polar_mask_raw = _to_polar(user_mask, center, max_radius)
        else:
            polar_mask_raw = None
    else:
        polar_mask_raw = None

    if polar_mask_raw is None:
        # Auto-band around outer rim
        col_idx        = np.arange(POLAR_W, dtype=np.float32)
        band_sigma     = max(POLAR_W * 0.08, 8.0)
        band           = np.exp(-0.5 * ((col_idx - r_ring_col) / band_sigma) ** 2)
        polar_mask_raw = (band[None, :] * 255).astype(np.uint8).repeat(POLAR_H, axis=0)

    if polar_mask_raw.ndim == 3:
        polar_mask_raw = polar_mask_raw[:, :, 0]

    polar_scratched, polar_mask_out = _synthesize_scratch_procedural(
        polar_img, polar_mask_raw, r_ring_col,
    )

    osize       = (img_rgb.shape[1], img_rgb.shape[0])
    cv_res      = _from_polar(polar_scratched, center, max_radius, osize)
    mask_result = _from_polar(polar_mask_out,  center, max_radius, osize)
    if mask_result.ndim == 3:
        mask_result = mask_result[:, :, 0]

    return cv_res, mask_result


# ── SDXL refine step — pipeline_scratch cell-10 ──────────────────────────────

def _sdxl_step(cv_res_rgb, mask_gray, ref_rgb, seed):
    """
    Returns final_rgb (same HW as cv_res_rgb).
    Notebook: ai_res_low.resize(good_image.size, Image.LANCZOS)
    ip_scale: 0.8 if ref provided, 0.5 if not.
    """
    import torch
    import gc

    orig_hw = (cv_res_rgb.shape[1], cv_res_rgb.shape[0])

    with get_lock():
        pipe      = get_pipe()
        depth_est = get_depth_est()

        gc.collect()
        torch.cuda.empty_cache()

        cv_low   = _PIL.fromarray(cv_res_rgb).resize(_TARGET)
        mask_low = _PIL.fromarray(mask_gray).resize(_TARGET)
        depth_map = depth_est(cv_low)["depth"].convert("RGB").resize(_TARGET)

        if ref_rgb is not None:
            ip_image = _PIL.fromarray(ref_rgb).convert("RGB").resize((224, 224))
            ip_scale = 0.8   # High influence — ref provided
        else:
            ip_image = cv_low
            ip_scale = 0.5   # Lower — prompt-only mode

        pipe.set_ip_adapter_scale(ip_scale)

        with torch.inference_mode():
            pipe.to("cuda" if __import__("torch").cuda.is_available() else "cpu")
            ai_res_low = pipe(
                prompt=_PROMPT,
                negative_prompt=_NEG,
                image=cv_low,
                mask_image=mask_low,
                control_image=depth_map,
                ip_adapter_image=ip_image,
                strength=_STRENGTH,
                num_inference_steps=_STEPS,
                guidance_scale=_GUIDANCE,
            ).images[0]

        # Notebook: ai_res_low.resize(good_image.size, Image.LANCZOS)
        final = ai_res_low.resize(orig_hw, _PIL.LANCZOS)

        torch.cuda.empty_cache()
        gc.collect()

    return np.array(final.convert("RGB"))


# ── Public generate() ─────────────────────────────────────────────────────────

def generate(base_image_b64: str, params: dict) -> dict:
    """
    Generate one Scratch Napchai defect image.

    params keys:
      seed             int        (default 42)
      sdxl_refine      bool       (default True)
      ref_image_b64    str        — base64 NG crop for IP-Adapter
      mask_b64         str        — user-drawn mask (optional; auto-band if absent)
    """
    img_rgb = decode_b64(base_image_b64)

    try:
        cv_res, mask_res = _cv_step(img_rgb, params)
    except Exception as e:
        return {"error": f"Scratch CV error: {e}"}

    _, buf = cv2.imencode(".png", cv2.cvtColor(cv_res, cv2.COLOR_RGB2BGR))
    pre_b64  = _b64.b64encode(buf).decode()
    _, mbuf = cv2.imencode(".png", mask_res)
    mask_b64 = _b64.b64encode(mbuf).decode()

    do_refine = params.get("sdxl_refine", True)
    seed      = int(params.get("seed", 42))
    ref_b64   = params.get("ref_image_b64")

    if do_refine:
        ref_rgb = decode_b64(ref_b64) if ref_b64 else None
        try:
            final_rgb  = _sdxl_step(cv_res, mask_res, ref_rgb, seed)
            result_b64 = encode_b64(final_rgb)
            engine     = "cv+sdxl"
        except Exception as e:
            print(f"[scratch_napchai] SDXL failed: {e} — returning CV result")
            result_b64 = encode_b64(cv_res)
            engine     = "cv"
    else:
        result_b64 = encode_b64(cv_res)
        engine     = "cv"

    return {
        "result_image":      result_b64,
        "result_pre_refine": pre_b64,
        "mask_b64":          mask_b64,
        "engine":            engine,
        "metadata": {"defect_type": "scratch", "sdxl_refine": do_refine, "params": params},
    }
