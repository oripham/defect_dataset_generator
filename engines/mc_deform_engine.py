"""
engines/mc_deform_engine.py — MC Deform (Biến dạng MC / Crushed Rim)
======================================================================
Port of pipeline_mc.ipynb — bulk-generation cell (cell-16) is the canonical
production code. Single-image cell-15 has same logic.

Pipeline (exactly as notebook):
  1. Hough Circle → center, max_radius = radius * 1.3
  2. Polar transform 720×512 (RGB, warpPolar WARP_POLAR_LINEAR INTER_LANCZOS4)
  3. r_ring_col = argmax of mean brightness in outer-half (col > W//3)
  4. Jagged-proportional warp: Gaussian radial + Gaussian angular taper,
     proportional noise * 0.4, s_amt = depth * 4.0
  5. Inverse polar → cv_res, m_res (envelope)
  6. SDXL ControlNet-Depth + IP-Adapter Plus refine
     - TARGET_SIZE = (768, 768)
     - ip_adapter_image resized to (256, 256)    ← cell-16 exact
     - strength=0.98, guidance=12.0, cn_scale=0.2, steps=30, ip_scale=1.0
  7. blend_with_original_clean(cv_p, ai_out, m_p, alpha=1.0)
     background = cv_p (CV result at 768), NOT original good image  ← fixed bug
  8. Resize back to original size, convert('L') → convert('RGB')    ← notebook grayscale save

API endpoint: POST /api/metal_cap/preview  {defect_type: "mc_deform"}
"""
from __future__ import annotations

import math
import base64 as _b64

import cv2
import numpy as np
from PIL import Image as _PIL

from .utils import encode_b64, decode_b64
from ._napchai_models import get_pipe, get_depth_est, get_lock

# ── Constants (same as all 3 notebooks) ──────────────────────────────────────
POLAR_H = 720
POLAR_W = 512

# ── SDXL config (pipeline_mc cell-15 / cell-16) ───────────────────────────────
_TARGET  = (768, 768)
_PROMPT  = (
    "irregular industrial metal defect, crushed rim, jagged metallic edges, "
    "deep dent, heavy specular reflections, polished chrome, photorealistic, "
    "high contrast, non-geometric damage"
)
_NEG     = "smooth, perfect circle, plastic, matte, flat, low quality, sphere"
_STRENGTH  = 0.98
_GUIDANCE  = 12.0
_CN_SCALE  = 0.2
_STEPS     = 30
_IP_SCALE  = 1.0


# ── Polar helpers ─────────────────────────────────────────────────────────────

def _detect_circle(gray: np.ndarray):
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
    """Argmax brightness in outer half — pipeline_mc cell-5."""
    profile      = polar_gray.mean(axis=0)
    search_start = POLAR_W // 3
    return search_start + int(np.argmax(profile[search_start:]))


# ── CV synthesis — matches cell-16 bulk-gen exactly ──────────────────────────

def _jagged_warp(polar_img, max_radius, t_center, t_span, r_c, r_w,
                 seed, collapse_depth):
    """
    pipeline_mc cell-16 step-2 (inline warp logic).
    Gaussian radial band × Gaussian angular taper × proportional jagged noise.
    """
    H, W = polar_img.shape[:2]
    row_idx = np.arange(H, dtype=np.float32)
    col_idx = np.arange(W, dtype=np.float32)
    ROW, COL = np.meshgrid(row_idx, col_idx, indexing="ij")

    col_ring  = float(np.clip(r_c / max_radius * W, 1, W - 2))
    rim_sigma = float(max(r_w / max_radius * W / 2.5, 3.0))
    band_r    = np.exp(-0.5 * ((COL - col_ring) / (rim_sigma + 1e-6)) ** 2)

    row_center = (t_center % (2 * math.pi)) / (2 * math.pi) * H
    drow       = np.minimum(np.abs(ROW - row_center), H - np.abs(ROW - row_center))
    row_half   = (t_span / (2 * math.pi)) * H / 2.0
    sigma_row  = max(row_half / 1.5, 2.0)
    taper      = np.exp(-0.5 * (drow / (sigma_row + 1e-6)) ** 2)

    env = band_r * taper

    np.random.seed(seed)
    noise_raw = np.random.normal(0, 1.0, (H, 1)).astype(np.float32)
    noise_v   = cv2.GaussianBlur(noise_raw, (1, 5), 0)
    noise_v   = (noise_v - noise_v.min()) / (np.ptp(noise_v) + 1e-6) * 2.0 - 1.0
    jagged_f  = 1.0 + (noise_v * env * 0.4)

    s_amt = abs(collapse_depth) * 4.0
    map_x = (COL - (env * jagged_f * s_amt)).astype(np.float32)
    map_y = ROW.astype(np.float32)

    polar_deformed = cv2.remap(polar_img, map_x, map_y,
                               cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    return polar_deformed, env.astype(np.float32)


# ── Blend helper — pipeline_mc cell-14 ───────────────────────────────────────

def _blend_with_original_clean(original_good_pil, sdxl_res_pil, mask_pil, alpha=1.0):
    """
    Strict binary mask blend.
    original_good = cv_p (CV result at 768px)  ← NOT the raw good image
    Inside mask: alpha-blend AI + cv.
    Outside mask: 100% cv result.
    """
    size     = sdxl_res_pil.size
    clean_bg = np.array(original_good_pil.convert("RGB").resize(size)).astype(np.float32)
    ai_arr   = np.array(sdxl_res_pil.convert("RGB")).astype(np.float32)
    mask_arr = np.array(mask_pil.convert("L")).astype(np.float32)

    _, binary = cv2.threshold(mask_arr, 1, 1.0, cv2.THRESH_BINARY)
    binary    = binary[:, :, np.newaxis]

    blended = ai_arr * alpha + clean_bg * (1.0 - alpha)
    final   = clean_bg * (1.0 - binary) + blended * binary
    return _PIL.fromarray(np.clip(final, 0, 255).astype(np.uint8))


# ── CV step ───────────────────────────────────────────────────────────────────

def _cv_step(img_rgb: np.ndarray, params: dict):
    """Returns (cv_result_rgb, envelope_mask_gray)."""
    seed      = int(params.get("seed", 42))
    intensity = float(params.get("intensity", 0.7))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cx, cy, radius = _detect_circle(gray)
    center     = (cx, cy)
    max_radius = int(radius * 1.3)

    polar_img  = _to_polar(img_rgb, center, max_radius)
    polar_gray = cv2.cvtColor(polar_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    _find_rim_col(polar_gray)   # for logging / debugging; unused in warp

    rng = np.random.RandomState(seed)

    # --- Optional: user-drawn mask to pick polar params (notebook Cell 7) ---
    # Studio sends mask_b64 at top-level and the Flask route copies it into params["mask_b64"].
    mask_b64 = params.get("mask_b64") or params.get("user_mask_b64")
    t_center = t_span = None
    r_c = r_w = None
    if mask_b64:
        arr = np.frombuffer(_b64.b64decode(mask_b64), np.uint8)
        user_mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if user_mask is not None:
            user_mask = cv2.resize(
                user_mask, (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            _, user_mask = cv2.threshold(user_mask, 127, 255, cv2.THRESH_BINARY)
            ys, xs = np.where(user_mask > 127)
            if ys.size > 0:
                dx = xs.astype(float) - center[0]
                dy = ys.astype(float) - center[1]
                angles = np.arctan2(dy, dx) % (2 * math.pi)
                radii  = np.sqrt(dx**2 + dy**2)

                sorted_a = np.sort(angles)
                diffs = np.append(
                    np.diff(sorted_a),
                    (2 * math.pi - sorted_a[-1] + sorted_a[0]),
                )
                gap_idx = int(np.argmax(diffs))
                start_angle = sorted_a[gap_idx + 1] if gap_idx < len(sorted_a) - 1 else sorted_a[0]
                end_angle   = sorted_a[gap_idx]
                span = (end_angle - start_angle) % (2 * math.pi)

                t_center = float((start_angle + span / 2) % (2 * math.pi))
                t_span   = float(max(span, math.pi / 18))
                r_c      = float(np.mean(radii))
                r_w      = float(max(float(np.ptp(radii)), 10.0))

    # --- Fallback: explicit params or random defaults (notebook ranges) ---
    if t_center is None:
        t_center = float(params.get("theta_center", rng.uniform(0, 2 * math.pi)))
    if t_span is None:
        # Notebook when no mask: uniform(pi/10, pi/4)
        t_span = float(params.get("theta_span", rng.uniform(math.pi / 10, math.pi / 4)))
    if r_c is None:
        r_c = float(params.get("r_center", radius * rng.uniform(0.98, 1.05)))
    if r_w is None:
        # Match notebook: rim sigma ~3-6 cols → cart width about 9-18 px for this product scale
        r_w = float(params.get("r_width", rng.uniform(9.0, 18.0)))

    # Deform strength: use explicit knob directly (do not re-scale by intensity).
    if "deform_strength" in params:
        collapse_depth = float(params.get("deform_strength") or 18.0)
    else:
        collapse_depth = rng.uniform(12.0, 25.0) * intensity

    rim_offset = int(params.get("rim_offset", 0) or 0)
    if rim_offset:
        # Rim col isn't directly used in warp, but keep for debugging parity
        pass

    polar_deformed, envelope = _jagged_warp(
        polar_img, max_radius, t_center, t_span, r_c, r_w,
        seed, collapse_depth,
    )

    osize       = (img_rgb.shape[1], img_rgb.shape[0])
    cv_result   = _from_polar(polar_deformed, center, max_radius, osize)
    env_u8      = (envelope * 255).astype(np.uint8)
    mask_result = _from_polar(env_u8, center, max_radius, osize)
    if mask_result.ndim == 3:
        mask_result = mask_result[:, :, 0]

    return cv_result, mask_result, (cx, cy, radius, max_radius)


# ── SDXL refine step — pipeline_mc cell-15 / cell-16 ─────────────────────────

def _sdxl_step(cv_result_rgb, mask_gray, ref_rgb, seed):
    """
    Returns final_rgb (same HW as cv_result_rgb).
    Background for blend = cv_p (the CV result at 768px), NOT original good image.
    Final notebook output: convert('L').convert('RGB').
    """
    import torch
    import gc

    orig_hw = (cv_result_rgb.shape[1], cv_result_rgb.shape[0])  # (W, H)

    with get_lock():
        pipe      = get_pipe()
        depth_est = get_depth_est()

        gc.collect()
        torch.cuda.empty_cache()

        # Inputs at TARGET_SIZE
        cv_pil   = _PIL.fromarray(cv_result_rgb).convert("RGB").resize(_TARGET)
        m_pil    = _PIL.fromarray(mask_gray).resize(_TARGET)
        depth_pil = depth_est(cv_pil)["depth"].convert("RGB").resize(_TARGET)

        # IP-Adapter image: ref crop at (256, 256) — cell-16 exact
        # Fallback to cv_pil when no ref (same pattern as ring/scratch engines)
        if ref_rgb is not None:
            ip_image = _PIL.fromarray(ref_rgb).convert("RGB").resize((256, 256))
        else:
            ip_image = cv_pil

        pipe.set_ip_adapter_scale(_IP_SCALE)

        with torch.inference_mode():
            ai_out = pipe(
                prompt=_PROMPT,
                negative_prompt=_NEG,
                image=cv_pil,
                mask_image=m_pil,
                control_image=depth_pil,
                ip_adapter_image=ip_image,
                controlnet_conditioning_scale=_CN_SCALE,
                num_inference_steps=_STEPS,
                guidance_scale=_GUIDANCE,
                strength=_STRENGTH,
                generator=torch.manual_seed(seed),
            ).images[0]

        # blend_with_original_clean(cv_p, ai_out, m_p, alpha=1.0)
        # background = cv_p  (not original good image)
        blended = _blend_with_original_clean(cv_pil, ai_out, m_pil, alpha=1.0)

        # Notebook saves as grayscale: final_bw = final_rgb.convert('L')
        # We return RGB for the webapp but go through L to match tonality
        final = blended.convert("L").resize(orig_hw, _PIL.LANCZOS).convert("RGB")

        torch.cuda.empty_cache()
        gc.collect()

    return np.array(final)


# ── Public generate() ─────────────────────────────────────────────────────────

def generate(base_image_b64: str, params: dict) -> dict:
    """
    Generate one MC Deform defect image.

    params keys:
      intensity       float 0-1  (default 0.7)
      seed            int        (default 42)
      sdxl_refine     bool       (default True) — False = CV-only fast mode
      ref_image_b64   str        — base64 NG crop for IP-Adapter (recommended)
    """
    img_rgb = decode_b64(base_image_b64)

    try:
        cv_result, defect_mask, _ = _cv_step(img_rgb, params)
    except Exception as e:
        return {"error": f"MC Deform CV error: {e}"}

    # Encode pre-refine
    _, buf = cv2.imencode(".png", cv2.cvtColor(cv_result, cv2.COLOR_RGB2BGR))
    pre_b64 = _b64.b64encode(buf).decode()
    _, mbuf = cv2.imencode(".png", defect_mask)
    mask_b64 = _b64.b64encode(mbuf).decode()

    do_refine = params.get("sdxl_refine", True)
    seed      = int(params.get("seed", 42))
    ref_b64   = params.get("ref_image_b64")

    if do_refine:
        ref_rgb = decode_b64(ref_b64) if ref_b64 else None
        try:
            final_rgb  = _sdxl_step(cv_result, defect_mask, ref_rgb, seed)
            result_b64 = encode_b64(final_rgb)
            engine     = "cv+sdxl"
        except Exception as e:
            print(f"[mc_deform] SDXL failed: {e} — returning CV result")
            result_b64 = encode_b64(cv_result)
            engine     = "cv"
    else:
        result_b64 = encode_b64(cv_result)
        engine     = "cv"

    return {
        "result_image":      result_b64,
        "result_pre_refine": pre_b64,
        "mask_b64":          mask_b64,
        "engine":            engine,
        "metadata": {"defect_type": "mc_deform", "sdxl_refine": do_refine, "params": params},
    }
