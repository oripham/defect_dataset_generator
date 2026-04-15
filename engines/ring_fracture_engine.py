"""
engines/ring_fracture_engine.py — Ring Fracture (Vỡ vòng)
===========================================================
Port of pipeline_ring.ipynb.

Pipeline (exactly as notebook):
  1. Hough Circle → center, max_radius = radius * 1.3
  2. Polar transform 720×512
  3. r_col = rightmost col where profile > max*0.5   ← pipeline_ring cell-5 exact
     (NOT argmax — that was a bug in the old engine)
  4. synthesize_ring_fractures: Random Walk displacement + glints
  5. Inverse polar → cv_distorted, cv_mask  (Cartesian)
  6. Cartesian soft-mask blend:
       soft_mask = cv_mask^(1/falloff_width) → GaussianBlur(9,9)
       cv_res = good_img*(1-soft*0.9) + cv_distorted*(soft*0.9)
     ← blended in Cartesian space with ORIGINAL good image, NOT in polar
  7. SDXL ControlNet-Depth + IP-Adapter Plus refine
     - TARGET_SIZE = (512, 512)    ← smallest of the 3 (T4 memory constraint)
     - ip_adapter_image at (224, 224)
     - strength=0.08, guidance=15.0, steps=20, ip_scale=1.0 (no explicit cn_scale)
  8. ai_res.convert('L').resize(original_size)  → convert('RGB')

API endpoint: POST /api/metal_cap/preview  {defect_type: "ring_fracture"}
"""
from __future__ import annotations

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

# ── SDXL config (pipeline_ring cell-8) ───────────────────────────────────────
_TARGET   = (512, 512)
_PROMPT   = (
    "extremely sharp industrial metal surface, hyper-detailed steel grain, "
    "microscopic metallic scratches, high contrast, 8k, ultra sharp focus"
)
_NEG      = (
    "blur, soft, out of focus, bokeh, smooth, plastic, paint, fog, "
    "glowing edge, noise, compression artifacts"
)
_STRENGTH = 0.08
_GUIDANCE = 15.0
_STEPS    = 20
_IP_SCALE = 1.0
# NOTE: pipeline_ring does NOT set controlnet_conditioning_scale → diffusers default (1.0)


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


def _find_rim_col(polar_gray: np.ndarray) -> int:
    """
    pipeline_ring cell-5: scan from right, return rightmost col > max*0.5.
    Different from mc_deform (argmax) and scratch (last > 20% range).
    """
    profile   = polar_gray.mean(axis=0)
    threshold = profile.max() * 0.5
    r_col     = POLAR_W - 1
    for i in range(POLAR_W - 1, POLAR_W // 3, -1):
        if profile[i] > threshold:
            r_col = i
            break
    return r_col


# ── CV synthesis ──────────────────────────────────────────────────────────────

def _synthesize_ring_fractures(polar, r_ring_col, seed=None, jitter_amplitude=4.0):
    """
    pipeline_ring cell-6 (synthesize_ring_fractures).
    Random Walk displacement + glints along distorted rim.
    """
    if seed is None:
        seed = _random.randint(0, 999999)
    H, W = polar.shape[:2]
    ROW, COL = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rng = np.random.RandomState(seed)

    steps         = rng.normal(0, 1.0, H)
    noise_profile = np.cumsum(steps)
    lin_trend     = np.linspace(noise_profile[0], noise_profile[-1], H)
    noise_profile = noise_profile - lin_trend
    noise_profile = noise_profile * (jitter_amplitude / (np.std(noise_profile) + 1e-6))

    dist_from_rim = np.abs(COL - r_ring_col)
    influence     = np.exp(-(dist_from_rim ** 2) / (2 * (12 ** 2)))
    shift_val     = (influence * noise_profile[:, np.newaxis]).astype(np.float32)

    map_x = (COL - shift_val).astype(np.float32)
    map_y = ROW.astype(np.float32)
    polar_distorted = cv2.remap(polar, map_x, map_y,
                                cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

    glint_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        cx = int(r_ring_col + noise_profile[i])
        if 0 <= cx < W and rng.rand() > 0.85:
            cv2.circle(glint_mask, (cx, i), int(rng.randint(1, 3)), 255, -1)

    p_mask = (influence * 255).astype(np.uint8)
    return polar_distorted, p_mask, glint_mask


# ── CV step ───────────────────────────────────────────────────────────────────

def _cv_step(img_rgb: np.ndarray, params: dict):
    """
    Returns (cv_res_rgb, m_res_gray) — Cartesian blended result.
    Blend is done in Cartesian space with original good image (cell-7).
    """
    seed            = int(params.get("seed", 42))
    intensity       = float(params.get("intensity", 0.7))
    jitter          = float(params.get("jitter_amplitude", 6.0)) * intensity
    falloff_width   = float(params.get("falloff_width", 1.0))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cx, cy, radius = _detect_circle(gray)
    center     = (cx, cy)
    max_radius = int(radius * 1.3)

    polar_img  = _to_polar(img_rgb, center, max_radius)
    polar_gray = cv2.cvtColor(polar_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r_col      = _find_rim_col(polar_gray)

    p_distorted, p_mask, p_glints = _synthesize_ring_fractures(
        polar_img, r_ring_col=r_col, seed=seed, jitter_amplitude=jitter,
    )
    p_distorted[p_glints > 0] = 255

    # ── Cartesian reconstruct (cell-7 step-2) ──────────────────────────────
    osize        = (img_rgb.shape[1], img_rgb.shape[0])
    cv_distorted = _from_polar(p_distorted, center, max_radius, osize)
    cv_mask      = _from_polar(p_mask,       center, max_radius, osize)

    if cv_mask.ndim == 3:
        mask_input = cv_mask[:, :, 0].astype(np.float32) / 255.0
    else:
        mask_input = cv_mask.astype(np.float32) / 255.0

    # ── Cartesian soft-mask blend (cell-7 step-3) ──────────────────────────
    # soft_mask = mask^(1/falloff_width) → GaussianBlur
    soft_mask = np.power(mask_input, 1.0 / max(falloff_width, 0.05))
    soft_mask = cv2.GaussianBlur(soft_mask, (9, 9), 0)
    soft_mask = soft_mask[:, :, np.newaxis]

    img_orig_f   = img_rgb.astype(np.float32)          # original good image
    cv_dist_f    = cv_distorted.astype(np.float32)

    alpha_max = 0.9
    combined  = (img_orig_f * (1 - soft_mask * alpha_max)
                 + cv_dist_f * (soft_mask * alpha_max))

    cv_res  = np.clip(combined, 0, 255).astype(np.uint8)
    m_res   = (mask_input * 255).astype(np.uint8)       # 2D

    return cv_res, m_res


# ── SDXL refine step — pipeline_ring cell-8 ──────────────────────────────────

def _sdxl_step(cv_res_rgb, m_res_gray, ref_rgb, seed):
    """
    Returns final_rgb (same HW as cv_res_rgb).
    Notebook: ai_res.convert('L').resize(good_image.size)
    """
    import torch
    import gc

    orig_hw = (cv_res_rgb.shape[1], cv_res_rgb.shape[0])

    with get_lock():
        pipe      = get_pipe()
        depth_est = get_depth_est()

        gc.collect()
        torch.cuda.empty_cache()

        cv_low  = _PIL.fromarray(cv_res_rgb).resize(_TARGET)
        m_low   = _PIL.fromarray(m_res_gray).resize(_TARGET)
        d_low   = depth_est(cv_low)["depth"].convert("RGB").resize(_TARGET)

        # ip_image: ng_patch at (224, 224) or cv_low fallback
        if ref_rgb is not None:
            ip_image = _PIL.fromarray(ref_rgb).convert("RGB").resize((224, 224))
        else:
            ip_image = cv_low

        pipe.set_ip_adapter_scale(_IP_SCALE)

        with torch.inference_mode():
            pipe.to("cuda" if __import__("torch").cuda.is_available() else "cpu")
            ai_res = pipe(
                prompt=_PROMPT,
                negative_prompt=_NEG,
                image=cv_low,
                mask_image=m_low,
                control_image=d_low,
                ip_adapter_image=ip_image,
                strength=_STRENGTH,
                num_inference_steps=_STEPS,
                guidance_scale=_GUIDANCE,
                generator=torch.manual_seed(seed),
            ).images[0]

        # Notebook: ai_res.convert('L').resize(good_image.size)
        final_gray = ai_res.convert("L").resize(orig_hw, _PIL.LANCZOS)
        final      = final_gray.convert("RGB")

        torch.cuda.empty_cache()
        gc.collect()

    return np.array(final)


# ── Public generate() ─────────────────────────────────────────────────────────

def generate(base_image_b64: str, params: dict) -> dict:
    """
    Generate one Ring Fracture defect image.

    params keys:
      intensity        float 0-1  (default 0.7)
      seed             int        (default 42)
      jitter_amplitude float      (default 6.0, scaled by intensity)
      falloff_width    float      (default 1.0 — softness of blend edge)
      sdxl_refine      bool       (default True)
      ref_image_b64    str        — base64 NG crop for IP-Adapter
    """
    img_rgb = decode_b64(base_image_b64)

    try:
        cv_res, m_res = _cv_step(img_rgb, params)
    except Exception as e:
        return {"error": f"Ring Fracture CV error: {e}"}

    _, buf = cv2.imencode(".png", cv2.cvtColor(cv_res, cv2.COLOR_RGB2BGR))
    pre_b64  = _b64.b64encode(buf).decode()
    _, mbuf = cv2.imencode(".png", m_res)
    mask_b64 = _b64.b64encode(mbuf).decode()

    do_refine = params.get("sdxl_refine", True)
    seed      = int(params.get("seed", 42))
    ref_b64   = params.get("ref_image_b64")

    if do_refine:
        ref_rgb = decode_b64(ref_b64) if ref_b64 else None
        try:
            final_rgb  = _sdxl_step(cv_res, m_res, ref_rgb, seed)
            result_b64 = encode_b64(final_rgb)
            engine     = "cv+sdxl"
        except Exception as e:
            print(f"[ring_fracture] SDXL failed: {e} — returning CV result")
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
        "metadata": {"defect_type": "ring_fracture", "sdxl_refine": do_refine, "params": params},
    }
