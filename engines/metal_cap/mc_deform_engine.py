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

from ..utils import encode_b64, decode_b64
from ..models._napchai_models import get_pipe, get_depth_est, get_lock

# ── Constants (same as all 3 notebooks) ──────────────────────────────────────
POLAR_H = 720
POLAR_W = 512

# ── SDXL config (pipeline_mc cell-15 / cell-16) ───────────────────────────────
_TARGET  = (768, 768)
_PROMPT  = "irregular industrial metal defect, crushed rim, jagged metallic edges, deep dent, heavy specular reflections, polished chrome, photorealistic, high contrast, non-geometric damage"
_NEG     = "smooth, perfect circle, plastic, matte, flat, low quality, sphere"
_STRENGTH  = 0.5
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


# ── CV synthesis — matches cell-16 bulk-gen ──────────────────────────────────

def _jagged_warp(polar_img, max_radius, t_center, t_span, r_c, r_w,
                 seed, collapse_depth, custom_env=None):
    """
    pipeline_mc cell-16 step-2 (inline warp logic).
    Gaussian radial band × Gaussian angular taper (or custom mask envelope) × proportional jagged noise.
    """
    H, W = polar_img.shape[:2]
    row_idx = np.arange(H, dtype=np.float32)
    col_idx = np.arange(W, dtype=np.float32)
    ROW, COL = np.meshgrid(row_idx, col_idx, indexing="ij")

    # 1. Geometry Mask (Envelope)
    if custom_env is not None:
        env = custom_env
    else:
        col_ring_p = float(np.clip(r_c / max_radius * W, 1, W - 2))
        rim_sigma_p = float(max(r_w / max_radius * W / 2.5, 3.0))
        band_r_p = np.exp(-0.5 * ((COL - col_ring_p) / (rim_sigma_p + 1e-6))**2)

        row_center_p = (t_center % (2 * np.pi)) / (2 * np.pi) * H
        drow_p = np.abs(ROW - row_center_p)
        drow_p = np.minimum(drow_p, H - drow_p)
        row_half_p = (t_span / (2 * np.pi)) * H / 2.0
        taper_p = np.exp(-0.5 * (drow_p / (max(row_half_p/1.5, 2.0) + 1e-6))**2)

        # ── Envelope (Mask) ──────────────────────────────────────────────────
        env = band_r_p * taper_p

    # 2. Jagged Noise (Proportional to envelope)
    np.random.seed(seed)
    noise_raw = np.random.normal(0, 1.0, (H, 1)).astype(np.float32)
    noise_v = cv2.GaussianBlur(noise_raw, (1, 5), 0)
    # Normalize noise to [-1.0, 1.0]
    noise_v = (noise_v - noise_v.min()) / (np.ptp(noise_v) + 1e-6) * 2.0 - 1.0
    # Multiplier 0.4 as per confirmed snippet
    jagged_f = 1.0 + (noise_v * env * 0.4)

    # 3. Geometric Warp (Remap)
    s_amt = abs(collapse_depth) * 4.0
    map_x = (COL - (env * jagged_f * s_amt)).astype(np.float32)
    map_y = ROW.astype(np.float32)

    polar_deformed = cv2.remap(polar_img, map_x, map_y,
                               cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

    return polar_deformed, env


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
    rng       = np.random.RandomState(seed)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cx, cy, radius = _detect_circle(gray)
    center     = (cx, cy)
    max_radius = int(radius * 1.3)

    polar_img  = _to_polar(img_rgb, center, max_radius)

    # 1. Parameter Extraction (Priority: Mask > UI Fixed > RNG)
    t_center = t_span = r_c = r_w = None
    custom_env = None
    mask_b64 = params.get("mask_b64") or params.get("user_mask_b64")

    if mask_b64:
        arr = np.frombuffer(_b64.b64decode(mask_b64), np.uint8)
        user_mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if user_mask is not None:
            user_mask = cv2.resize(user_mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            _, user_mask = cv2.threshold(user_mask, 127, 255, cv2.THRESH_BINARY)
            
            ys, xs = np.where(user_mask > 127)
            if ys.size > 0:
                dx, dy = xs.astype(float) - cx, ys.astype(float) - cy
                angles = np.arctan2(dy, dx) % (2 * math.pi)
                radii  = np.sqrt(dx**2 + dy**2)

                # Find the center angle of the mask
                sorted_a = np.sort(angles)
                diffs = np.append(np.diff(sorted_a), (2 * math.pi - sorted_a[-1] + sorted_a[0]))
                gap_idx = int(np.argmax(diffs))
                start_angle = sorted_a[gap_idx + 1] if gap_idx < len(sorted_a) - 1 else sorted_a[0]
                span_tmp = (2 * math.pi - diffs[gap_idx]) % (2 * math.pi)
                
                t_center = float((start_angle + span_tmp / 2) % (2 * math.pi))
                r_c      = float(np.mean(radii))
                r_w      = float(max(float(np.ptp(radii)), 10.0))

                print(f"[mc_deform] MASK USED FOR POSITION ONLY: t_center={t_center:.3f}, r_c={r_c:.1f}")

    # 2. Parameter Extraction (Strictly from UI Sliders as requested)
    if t_center is None:
        # If no mask, check for fixed UI position or random
        ui_theta = params.get("theta_center")
        if ui_theta is not None and ui_theta != "random":
            t_center = float(ui_theta)
        else:
            t_center = rng.uniform(0, 2 * math.pi)

    # 3. Apply Seed-based Jitter (Makes even masked positions dynamic)
    jitter_amt = float(params.get("position_jitter", 0.1)) # default 0.1 rad (~6 deg)
    t_center  += rng.uniform(-1, 1) * jitter_amt
    t_center  %= (2 * math.pi)

    # Always take Span and Depth from UI sliders
    t_span = float(params.get("theta_span", 0.39))
    base_depth = float(params.get("depth") or params.get("deform_strength") or 15.0)
    collapse_depth = base_depth * intensity
    
    if r_c is None:
        r_c = float(params.get("r_center", radius))
    
    # Slight radial jitter too
    r_c += rng.uniform(-1, 1) * (jitter_amt * 20.0) 

    if r_w is None:
        r_w = float(params.get("r_width", 25.0))

    print(f"[mc_deform] FINAL LOC: t_center={t_center:.3f} ({math.degrees(t_center):.1f}deg), r_c={r_c:.1f}")
    print(f"[mc_deform] Warp Params: depth={collapse_depth:.1f}, span={t_span:.3f}, intensity={intensity}")

    # 3. Execution (Always use analytical Gaussian envelope for quality)
    polar_deformed, envelope = _jagged_warp(
        polar_img, max_radius, t_center, t_span, r_c, r_w,
        seed, collapse_depth, custom_env=None
    )

    osize       = (img_rgb.shape[1], img_rgb.shape[0])
    cv_result   = _from_polar(polar_deformed, center, max_radius, osize)
    env_u8      = (envelope * 255).astype(np.uint8)
    mask_result = _from_polar(env_u8, center, max_radius, osize)
    if mask_result.ndim == 3:
        mask_result = mask_result[:, :, 0]

    return cv_result, mask_result, (cx, cy, radius, max_radius)


# ── SDXL refine step — pipeline_mc cell-15 / cell-16 ─────────────────────────

def _sdxl_step(cv_result_rgb, mask_gray, ref_rgb, seed, prompt=None, negative_prompt=None, params=None):
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
        # Robust depth extraction
        depth_out = depth_est(cv_pil)
        if isinstance(depth_out, dict):
            depth_image = depth_out["depth"]
        elif isinstance(depth_out, (list, tuple)):
            depth_image = depth_out[0]
        else:
            depth_image = depth_out
        depth_pil = depth_image.convert("RGB").resize(_TARGET)

        # IP-Adapter image: ref crop at (256, 256) — cell-16 exact
        # Fallback to cv_pil when no ref (same pattern as ring/scratch engines)
        if ref_rgb is not None:
            ip_image = _PIL.fromarray(ref_rgb).convert("RGB").resize((256, 256))
        else:
            ip_image = cv_pil.resize((256, 256))

        _p = params or {}
        ip_scale = float(_p.get("ip_scale", _IP_SCALE))
        s_strength = float(_p.get("strength", _STRENGTH))
        s_guidance = float(_p.get("guidance_scale", _GUIDANCE))
        s_steps = int(_p.get("steps", _STEPS))
        s_cn_scale = float(_p.get("controlnet_scale", _CN_SCALE))

        pipe.set_ip_adapter_scale(ip_scale)

        print(f"[mc_deform] SDXL inpaint: strength={s_strength}, guidance={s_guidance}, "
              f"steps={s_steps}, ip_scale={ip_scale}, cn_scale={s_cn_scale}")

        with torch.inference_mode():
            pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            ai_out = pipe(
                prompt=prompt or _PROMPT,
                negative_prompt=negative_prompt or _NEG,
                image=cv_pil,
                mask_image=m_pil,
                control_image=depth_pil,
                ip_adapter_image=ip_image,
                controlnet_conditioning_scale=s_cn_scale,
                num_inference_steps=s_steps,
                guidance_scale=s_guidance,
                strength=s_strength,
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

def generate(base_image_b64: str, params: dict, mask_b64: str | None = None) -> dict:
    """
    Generate one MC Deform defect image.

    params keys:
      intensity       float 0-1  (default 0.7)
      seed            int        (default 42)
      sdxl_refine     bool       (default True) — False = CV-only fast mode
      ref_image_b64   str        — base64 NG crop for IP-Adapter (recommended)
    """
    if mask_b64:
        params["mask_b64"] = mask_b64
    img_rgb = decode_b64(base_image_b64)

    # DEBUG: Log received seed
    received_seed = params.get("seed")
    print(f"\n[mc_deform] Generate called with seed={received_seed}")

    try:
        cv_result, defect_mask, _ = _cv_step(img_rgb, params)
    except Exception as e:
        return {"error": f"MC Deform CV error: {e}"}

    # Encode pre-refine
    _, buf = cv2.imencode(".png", cv2.cvtColor(cv_result, cv2.COLOR_RGB2BGR))
    pre_b64 = _b64.b64encode(buf).decode()
    _, mbuf = cv2.imencode(".png", defect_mask)
    mask_b64 = _b64.b64encode(mbuf).decode()

    do_refine = params.get("use_sdxl") or params.get("sdxl_refine") or False
    seed      = int(params.get("seed", 42))
    ref_b64   = params.get("ref_image_b64")

    if do_refine:
        ref_rgb = decode_b64(ref_b64) if ref_b64 else None
        try:
            final_rgb  = _sdxl_step(cv_result, defect_mask, ref_rgb, seed,
                                     prompt=params.get("prompt"),
                                     negative_prompt=params.get("negative_prompt"),
                                     params=params)
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
