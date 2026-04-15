"""
engines/metal_cap_engine.py — Metal Cap (Napchai) Polar-Transform Pipeline
===========================================================================
Full port of the 3 Colab notebooks:
  pipeline_mc.ipynb      → mc_deform    (Rim deform + SDXL refine)
  pipeline_ring.ipynb    → ring_fracture (Ring fracture + SDXL refine)
  pipeline_scratch.ipynb → scratch       (Polar scratch + SDXL refine)

Pipeline per defect (matches notebook bulk-gen cells exactly):
  Step 1  Hough Circle detect → center, max_radius = radius * 1.3
  Step 2  Polar transform  720×512
  Step 3  CV synthesis in polar space
  Step 4  Inverse polar → cv_result + mask_result
  Step 5  SDXL ControlNet-Depth + IP-Adapter-Plus refine
  Step 6  Blend / resize to original size

Set params["sdxl_refine"] = False to skip Step 5 (fast CV-only mode).

Interface:
  generate(base_image_b64, defect_type, params) -> dict
"""

from __future__ import annotations

import math
import threading
import random as _random

import cv2
import numpy as np

from .utils import encode_b64, decode_b64


# ── Polar Transform Constants ─────────────────────────────────────────────────

POLAR_H = 720   # θ rows  (360° × 2)
POLAR_W = 512   # r  cols


# ── Lazy SDXL model state ─────────────────────────────────────────────────────

_mc_pipe          = None   # StableDiffusionXLControlNetInpaintPipeline
_mc_depth_est     = None   # transformers depth-estimation pipeline
_mc_model_lock    = None   # threading.Lock — shared with deep_generative if possible


def _get_lock() -> threading.Lock:
    global _mc_model_lock
    if _mc_model_lock is None:
        try:
            from .deep_generative import _GEN_LOCK
            _mc_model_lock = _GEN_LOCK
        except Exception:
            _mc_model_lock = threading.Lock()
    return _mc_model_lock


def _load_mc_models():
    """
    Lazy-load SDXL + ControlNet Depth + IP-Adapter Plus.
    Mirrors notebook cell-6 / cell-2 / cell-2 model loading.
    """
    global _mc_pipe, _mc_depth_est
    if _mc_pipe is not None:
        return

    import torch
    from transformers import CLIPVisionModelWithProjection
    from transformers import pipeline as _tfpipe
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    # ── IPAdapterMixin compat patch (diffusers < 0.26) ──────────────────────
    try:
        from diffusers.loaders import IPAdapterMixin as _IPA
        if not hasattr(StableDiffusionXLControlNetInpaintPipeline, "load_ip_adapter"):
            StableDiffusionXLControlNetInpaintPipeline.__bases__ = (
                (_IPA,) + StableDiffusionXLControlNetInpaintPipeline.__bases__)
    except Exception:
        pass

    print("[metal_cap] Loading ControlNet Depth...")
    depth_model = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=dtype
    ).to(device)

    print("[metal_cap] Loading CLIP image encoder (IP-Adapter)...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=dtype
    ).to(device)

    print("[metal_cap] Loading SDXL Inpainting pipeline...")
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet=depth_model,
        image_encoder=image_encoder,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    )

    print("[metal_cap] Loading IP-Adapter Plus weights...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        image_encoder_folder=None,
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[metal_cap] xformers enabled")
    except Exception:
        pass

    pipe = pipe.to(device)
    pipe.enable_vae_slicing()

    print("[metal_cap] Loading depth estimator...")
    _mc_depth_est = _tfpipe(
        "depth-estimation",
        device=0 if device == "cuda" else -1,
    )

    _mc_pipe = pipe
    print("[metal_cap] ALL MODELS LOADED")


# ── SDXL params per defect ────────────────────────────────────────────────────
# Taken directly from the final SDXL-run cell of each notebook.

_SDXL_CFG = {
    "mc_deform": {
        "target":   (768, 768),
        "strength": 0.98,
        "guidance": 12.0,
        "cn_scale": 0.2,
        "steps":    30,
        "ip_scale": 1.0,
        "prompt":   (
            "irregular industrial metal defect, crushed rim, jagged metallic edges, "
            "deep dent, heavy specular reflections, polished chrome, photorealistic, "
            "high contrast, non-geometric damage"
        ),
        "negative": "smooth, perfect circle, plastic, matte, flat, low quality, sphere",
    },
    "ring_fracture": {
        "target":   (512, 512),
        "strength": 0.08,
        "guidance": 15.0,
        "cn_scale": 1.0,    # notebook doesn't override → default
        "steps":    20,
        "ip_scale": 1.0,
        "prompt":   (
            "extremely sharp industrial metal surface, hyper-detailed steel grain, "
            "microscopic metallic scratches, high contrast, 8k, ultra sharp focus"
        ),
        "negative": (
            "blur, soft, out of focus, bokeh, smooth, plastic, paint, fog, "
            "glowing edge, noise, compression artifacts"
        ),
    },
    "scratch": {
        "target":   (768, 768),
        "strength": 0.14,
        "guidance": 7.5,
        "cn_scale": 1.0,
        "steps":    30,
        "ip_scale": 0.8,    # 0.5 if no ref_image_b64
        "prompt":   (
            "jagged metallic fracture, deep industrial micro-cracks, "
            "weathered steel texture, hyper-realistic, 8k, industrial damage"
        ),
        "negative": (
            "color, bronze, gold, plastic, blurry, smooth, "
            "artificial seam, lowres, noise"
        ),
    },
}


# ── Blend helper (pipeline_mc cell-14) ───────────────────────────────────────

def _blend_with_original(original_rgb_pil, ai_result_pil, mask_pil, alpha: float = 1.0):
    """
    Strict binary mask blend.
    Vùng mask: alpha-blend AI + original.
    Vùng ngoài: 100% original.
    alpha=1.0 → inside mask = 100% AI (notebook default).
    """
    from PIL import Image as _PIL
    import numpy as _np

    size = ai_result_pil.size
    clean_bg = _np.array(original_rgb_pil.convert("RGB").resize(size)).astype(_np.float32)
    ai_arr   = _np.array(ai_result_pil.convert("RGB")).astype(_np.float32)
    mask_arr = _np.array(mask_pil.convert("L")).astype(_np.float32)

    _, binary = cv2.threshold(mask_arr, 1, 1.0, cv2.THRESH_BINARY)
    binary = binary[:, :, _np.newaxis]

    blended = ai_arr * alpha + clean_bg * (1.0 - alpha)
    final   = clean_bg * (1.0 - binary) + blended * binary
    return _PIL.fromarray(_np.clip(final, 0, 255).astype(_np.uint8))


# ── SDXL refine step ─────────────────────────────────────────────────────────

def _sdxl_refine(
    cv_result_rgb: np.ndarray,    # HxWx3 RGB
    mask_result:   np.ndarray,    # HxW grayscale
    original_rgb:  np.ndarray,    # HxWx3 RGB (for mc_deform blend)
    defect_type:   str,
    ref_rgb:       np.ndarray | None,  # NG reference (IP-Adapter input)
    seed:          int,
) -> np.ndarray:
    """
    Run SDXL ControlNet-Depth + IP-Adapter refine.
    Returns refined image as HxWx3 RGB numpy array (original size).
    """
    import torch
    import gc
    from PIL import Image as _PIL

    cfg = _SDXL_CFG[defect_type]
    target = cfg["target"]

    with _get_lock():
        _load_mc_models()

        gc.collect()
        torch.cuda.empty_cache()

        # --- Prepare inputs ---
        cv_pil   = _PIL.fromarray(cv_result_rgb).convert("RGB").resize(target)
        mask_pil = _PIL.fromarray(mask_result).resize(target)
        depth_pil = _mc_depth_est(cv_pil)["depth"].convert("RGB").resize(target)

        # IP-Adapter image: NG ref crop → 224×224 (notebook uses 256×256 for mc_deform)
        if ref_rgb is not None:
            ip_size = (256, 256) if defect_type == "mc_deform" else (224, 224)
            ip_image = _PIL.fromarray(ref_rgb).convert("RGB").resize(ip_size)
            ip_scale = cfg["ip_scale"]
        else:
            # Fallback: use cv_result as IP ref (prompt-only mode)
            ip_image = cv_pil
            ip_scale = 0.5

        _mc_pipe.set_ip_adapter_scale(ip_scale)

        generator = torch.manual_seed(seed)

        with torch.inference_mode():
            ai_out = _mc_pipe(
                prompt=cfg["prompt"],
                negative_prompt=cfg["negative"],
                image=cv_pil,
                mask_image=mask_pil,
                control_image=depth_pil,
                ip_adapter_image=ip_image,
                controlnet_conditioning_scale=cfg["cn_scale"],
                num_inference_steps=cfg["steps"],
                guidance_scale=cfg["guidance"],
                strength=cfg["strength"],
                generator=generator,
            ).images[0]

        # --- Post-process (match notebook exactly) ---
        orig_hw = (cv_result_rgb.shape[1], cv_result_rgb.shape[0])  # (W, H)

        if defect_type == "mc_deform":
            # blend_with_original_clean alpha=1.0 → strict mask binary
            orig_pil = _PIL.fromarray(original_rgb).convert("RGB").resize(target)
            blended  = _blend_with_original(orig_pil, ai_out, mask_pil, alpha=1.0)
            result   = np.array(blended.resize(orig_hw).convert("RGB"))
        elif defect_type == "ring_fracture":
            # notebook: ai_res.convert('L').resize(good_image.size)
            result = np.array(ai_out.convert("L").resize(orig_hw).convert("RGB"))
        else:
            # scratch: ai_res_low.resize(good_image.size)
            result = np.array(ai_out.resize(orig_hw).convert("RGB"))

        torch.cuda.empty_cache()
        gc.collect()

    return result


# ── Polar Transform Utilities ─────────────────────────────────────────────────

def _detect_main_circle(gray: np.ndarray):
    """Hough Circle detection → (cx, cy, radius). Same params as all 3 notebooks."""
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


def _to_polar(img: np.ndarray, center, max_radius: int) -> np.ndarray:
    return cv2.warpPolar(
        img, (POLAR_W, POLAR_H), center, max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.INTER_LANCZOS4,
    )


def _from_polar(polar: np.ndarray, center, max_radius: int, osize) -> np.ndarray:
    return cv2.warpPolar(
        polar, osize, center, max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LANCZOS4,
    )


def _find_rim_col_peak(polar_gray: np.ndarray) -> int:
    """Brightest peak in outer half — used by mc_deform and ring_fracture."""
    profile = polar_gray.mean(axis=0)
    search_start = POLAR_W // 3
    return search_start + int(np.argmax(profile[search_start:]))


def _find_rim_col_outer(polar_gray: np.ndarray) -> int:
    """
    Outer boundary: last index above 20% of (max-min) range.
    Used by scratch (pipeline_scratch cell-6) to find the outermost edge.
    """
    profile = polar_gray.mean(axis=0)
    threshold = profile.min() + (profile.max() - profile.min()) * 0.2
    indices = np.where(profile > threshold)[0]
    if len(indices) > 0:
        return int(indices[-1])
    return int(POLAR_W * 0.8)


# ── Synthesis: MC Deform ──────────────────────────────────────────────────────
# Matches pipeline_mc cell-16 (bulk generation) exactly.
# Uses jagged proportional noise warp with Gaussian taper.

def synthesize_rim_deform_jagged(
    polar_img:   np.ndarray,
    max_radius:  int,
    theta_center: float,
    theta_span:   float,
    r_center:     float,
    r_width:      float,
    seed:         int,
    collapse_depth: float = 18.0,   # positive = warp inward (bulge outward in Cartesian)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jagged-proportional warp from pipeline_mc cell-8 + cell-9 + cell-16.

    Returns (polar_deformed, envelope_float32 HxW)
    """
    H, W = polar_img.shape[:2]
    row_idx = np.arange(H, dtype=np.float32)
    col_idx = np.arange(W, dtype=np.float32)
    ROW, COL = np.meshgrid(row_idx, col_idx, indexing="ij")

    # Radial band (sigma from r_width)
    col_ring   = float(np.clip(r_center / max_radius * W, 1, W - 2))
    rim_sigma  = float(max(r_width / max_radius * W / 2.5, 3.0))
    band_r     = np.exp(-0.5 * ((COL - col_ring) / (rim_sigma + 1e-6)) ** 2)

    # Angular taper (Gaussian — matches cell-16)
    row_center = (theta_center % (2 * math.pi)) / (2 * math.pi) * H
    drow       = np.abs(ROW - row_center)
    drow       = np.minimum(drow, H - drow)
    row_half   = (theta_span / (2 * math.pi)) * H / 2.0
    sigma_row  = max(row_half / 1.5, 2.0)
    taper      = np.exp(-0.5 * (drow / (sigma_row + 1e-6)) ** 2)

    envelope = band_r * taper  # float32 [0, 1]

    # Jagged proportional noise
    np.random.seed(seed)
    noise_raw = np.random.normal(0, 1.0, (H, 1)).astype(np.float32)
    noise_v   = cv2.GaussianBlur(noise_raw, (1, 5), 0)
    noise_v   = (noise_v - noise_v.min()) / (np.ptp(noise_v) + 1e-6) * 2.0 - 1.0  # [-1, 1]
    jagged_f  = 1.0 + (noise_v * envelope * 0.4)

    s_amt  = abs(collapse_depth) * 4.0
    map_x  = (COL - (envelope * jagged_f * s_amt)).astype(np.float32)
    map_y  = ROW.astype(np.float32)

    polar_deformed = cv2.remap(
        polar_img, map_x, map_y,
        cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
    )
    return polar_deformed, envelope.astype(np.float32)


# ── Synthesis: Ring Fracture ──────────────────────────────────────────────────
# pipeline_ring cell-7 (synthesize_ring_fractures) + cell-8 blending.

def synthesize_ring_fractures(polar, r_ring_col, seed=None, jitter_amplitude=4.0):
    """Random Walk displacement along rim → natural wavy fracture + glints."""
    if seed is None:
        seed = _random.randint(0, 999999)
    H, W = polar.shape[:2]
    ROW, COL = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rng = np.random.RandomState(seed)

    steps = rng.normal(0, 1.0, H)
    noise_profile = np.cumsum(steps)
    lin_trend = np.linspace(noise_profile[0], noise_profile[-1], H)
    noise_profile = noise_profile - lin_trend
    noise_profile = noise_profile * (jitter_amplitude / (np.std(noise_profile) + 1e-6))

    dist_from_rim = np.abs(COL - r_ring_col)
    influence = np.exp(-(dist_from_rim ** 2) / (2 * (12 ** 2)))
    shift_val = (influence * noise_profile[:, np.newaxis]).astype(np.float32)

    map_x = (COL - shift_val).astype(np.float32)
    map_y = ROW.astype(np.float32)
    polar_distorted = cv2.remap(polar, map_x, map_y, cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REFLECT)

    glint_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        current_rim_x = int(r_ring_col + noise_profile[i])
        if 0 <= current_rim_x < W and rng.rand() > 0.85:
            size = rng.randint(1, 3)
            cv2.circle(glint_mask, (current_rim_x, i), size, 255, -1)

    p_mask = (influence * 255).astype(np.uint8)
    return polar_distorted, p_mask, glint_mask


# ── Synthesis: Scratch ────────────────────────────────────────────────────────
# pipeline_scratch cell-7 (unchanged from original port).

def synthesize_scratch_procedural(polar_img, polar_mask, rim_col):
    """Diverse procedural damage: Slashes, Scuffs, Pits — filtered to mask region."""
    H, W = polar_img.shape[:2]
    clean_mask = polar_mask.copy()
    clean_mask[:, rim_col:] = 0
    mask_f = clean_mask.astype(np.float32) / 255.0
    if len(polar_img.shape) == 3:
        mask_f3 = mask_f[:, :, np.newaxis]
    else:
        mask_f3 = mask_f

    base_f = polar_img.astype(np.float32) / 255.0
    trench_layer = np.zeros((H, W), dtype=np.float32)
    glint_layer  = np.zeros((H, W), dtype=np.float32)

    y_coords, x_coords = np.where(clean_mask > 0)
    if len(x_coords) > 0:
        num_events = _random.randint(4, 7)
        for _ in range(num_events):
            event_type = _random.choice(["slash", "scuff", "pitting"])
            idx = _random.randint(0, len(x_coords) - 1)
            sx, sy = int(x_coords[idx]), int(y_coords[idx])

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
                    length = _random.randint(15, 40)
                    for i in range(length):
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
        res_f = np.clip(res_f + (glint_3 * 0.35 * mask_f3), 0, 1)
        return (res_f * 255.0).astype(np.uint8), clean_mask

    return polar_img, clean_mask


# ── Dispatch Runners (CV step only) ──────────────────────────────────────────

def _run_mc_deform(img_rgb: np.ndarray, params: dict):
    """
    pipeline_mc cell-16 (bulk-gen) exact replica.
    Uses jagged-proportional warp with Gaussian angular taper.
    Returns (cv_result_rgb, mask_result_gray).
    """
    seed      = int(params.get("seed", 42))
    intensity = float(params.get("intensity", 0.7))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cx, cy, radius = _detect_main_circle(gray)
    center     = (cx, cy)
    max_radius = int(radius * 1.3)

    polar_img  = _to_polar(img_rgb, center, max_radius)
    polar_gray = cv2.cvtColor(polar_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r_ring_col = _find_rim_col_peak(polar_gray)
    rim_offset = int(params.get("rim_offset", 0) or 0)
    if rim_offset:
        r_ring_col = int(np.clip(r_ring_col + rim_offset, 0, POLAR_W - 1))
    r_ring     = r_ring_col / POLAR_W * max_radius

    rng = np.random.RandomState(seed)

    # --- Optional: use user-drawn mask to pick location/coverage (notebook-like) ---
    # Studio sends mask_b64 at top-level (also duplicated into params["mask_b64"] by API).
    theta_center = None
    theta_span   = None
    r_center     = None
    r_width      = None

    user_mask_b64 = params.get("mask_b64") or params.get("user_mask_b64")
    if user_mask_b64:
        import base64 as _b64
        arr = np.frombuffer(_b64.b64decode(user_mask_b64), np.uint8)
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

                theta_center = float((start_angle + span / 2) % (2 * math.pi))
                theta_span   = float(max(span, math.pi / 18))
                r_center     = float(np.mean(radii))
                r_width      = float(max(float(np.ptp(radii)), 10.0))

    # --- Fallback to explicit params, then random defaults (matches notebook intent) ---
    if theta_center is None:
        theta_center = float(params.get("theta_center", rng.uniform(0, 2 * math.pi)))
    if theta_span is None:
        # Notebook default when no mask: uniform(pi/10, pi/4)
        theta_span = float(params.get("theta_span", rng.uniform(math.pi / 10, math.pi / 4)))
    if r_center is None:
        r_center = float(params.get("r_center", radius * rng.uniform(0.98, 1.05)))
    if r_width is None:
        # Match notebook behavior: when no user mask, rim sigma in polar is ~3-6 cols.
        # Our engine parameterizes width in Cartesian pixels; convert by sampling a range
        # that maps to similar polar sigma: r_width ≈ rim_sigma * max_radius * 2.5 / POLAR_W.
        r_width = float(params.get("r_width", rng.uniform(9.0, 18.0)))

    # Deformation magnitude:
    # - If studio provides deform_strength, treat it as an absolute knob (do NOT re-scale by intensity),
    #   otherwise output becomes weaker than notebook for the same slider value.
    # - If not provided, fall back to a notebook-like random range scaled by intensity.
    if "deform_strength" in params:
        collapse_depth = float(params.get("deform_strength") or 18.0)
    else:
        collapse_depth = rng.uniform(12.0, 25.0) * intensity

    polar_deformed, envelope = synthesize_rim_deform_jagged(
        polar_img, max_radius,
        theta_center, theta_span, r_center, r_width,
        seed=seed, collapse_depth=collapse_depth,
    )

    osize       = (img_rgb.shape[1], img_rgb.shape[0])
    cv_result   = _from_polar(polar_deformed, center, max_radius, osize)
    env_uint8   = (envelope * 255).astype(np.uint8)
    mask_result = _from_polar(env_uint8, center, max_radius, osize)
    if len(mask_result.shape) == 3:
        mask_result = mask_result[:, :, 0]

    return cv_result, mask_result


def _run_ring_fracture(img_rgb: np.ndarray, params: dict):
    """
    pipeline_ring cell-7 + cell-8 exact replica.
    Includes Gaussian falloff alpha-blend in polar space (alpha_max=0.9).
    Returns (cv_result_rgb, mask_result_gray).
    """
    seed      = int(params.get("seed", 42))
    intensity = float(params.get("intensity", 0.7))
    jitter    = float(params.get("jitter_amplitude", 6.0)) * intensity

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cx, cy, radius = _detect_main_circle(gray)
    center     = (cx, cy)
    max_radius = int(radius * 1.3)

    polar_img  = _to_polar(img_rgb, center, max_radius)
    polar_gray = cv2.cvtColor(polar_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r_ring_col = _find_rim_col_peak(polar_gray)

    # Optional: user-drawn mask to pick rim column (notebook-like manual rim selection)
    user_mask_b64 = params.get("mask_b64") or params.get("user_mask_b64")
    if user_mask_b64:
        import base64 as _b64
        arr = np.frombuffer(_b64.b64decode(user_mask_b64), np.uint8)
        user_mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if user_mask is not None:
            user_mask = cv2.resize(
                user_mask, (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            polar_m = _to_polar(user_mask, center, max_radius)
            if len(polar_m.shape) == 3:
                polar_m = polar_m[:, :, 0]
            ys, xs = np.where(polar_m > 127)
            if xs.size > 0:
                r_ring_col = int(np.clip(int(xs.mean()), 0, POLAR_W - 1))

    rim_offset = int(params.get("rim_offset", 0) or 0)
    if rim_offset:
        r_ring_col = int(np.clip(r_ring_col + rim_offset, 0, POLAR_W - 1))

    polar_distorted, p_mask, glint_mask = synthesize_ring_fractures(
        polar_img, r_ring_col, seed=seed, jitter_amplitude=jitter,
    )
    polar_distorted[glint_mask > 0] = 255

    # ── Cell-8 Gaussian falloff blend ─────────────────────────────────────────
    # soft_mask = mask^(1/falloff_width), GaussianBlur, then alpha_max=0.9 blend
    falloff_width = float(params.get("falloff_width", 1.0))
    mask_f        = p_mask.astype(np.float32) / 255.0
    soft_mask     = np.power(mask_f, 1.0 / max(falloff_width, 0.05))
    soft_mask     = cv2.GaussianBlur(soft_mask, (9, 9), 0)

    polar_dist_f = polar_distorted.astype(np.float32)

    # blend in polar space
    alpha_max = 0.9
    blended_polar = (
        polar_dist_f * (soft_mask[:, :, None] * alpha_max) +
        polar_img.astype(np.float32) * (1.0 - soft_mask[:, :, None] * alpha_max)
    )
    blended_polar = np.clip(blended_polar, 0, 255).astype(np.uint8)

    osize       = (img_rgb.shape[1], img_rgb.shape[0])
    cv_result   = _from_polar(blended_polar, center, max_radius, osize)
    mask_result = _from_polar(p_mask, center, max_radius, osize)
    if len(mask_result.shape) == 3:
        mask_result = mask_result[:, :, 0]

    return cv_result, mask_result


def _run_scratch(img_rgb: np.ndarray, params: dict):
    """
    pipeline_scratch cell-8 + cell-9 + cell-10 exact replica.
    Outer-boundary rim detection (last index > 20% threshold).
    Returns (cv_result_rgb, mask_result_gray).
    """
    seed = int(params.get("seed", 42))
    _random.seed(seed)
    np.random.seed(seed)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cx, cy, radius = _detect_main_circle(gray)
    center     = (cx, cy)
    max_radius = int(radius * 1.3)

    polar_img  = _to_polar(img_rgb, center, max_radius)
    polar_gray = cv2.cvtColor(polar_img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Outer-boundary rim detection (pipeline_scratch cell-6)
    r_ring_col = _find_rim_col_outer(polar_gray)
    rim_offset = int(params.get("rim_offset", 0) or 0)
    if rim_offset:
        r_ring_col = int(np.clip(r_ring_col + rim_offset, 0, POLAR_W - 1))

    # Build polar mask from user-drawn mask or full-band
    user_mask_b64 = params.get("mask_b64") or params.get("user_mask_b64")
    if user_mask_b64:
        import base64 as _b64
        arr       = np.frombuffer(_b64.b64decode(user_mask_b64), np.uint8)
        user_mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if user_mask is not None:
            user_mask = cv2.resize(
                user_mask, (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            polar_mask_raw = _to_polar(user_mask, center, max_radius)
        else:
            polar_mask_raw = None
    else:
        polar_mask_raw = None

    if polar_mask_raw is None:
        # Band around outer rim
        col_idx    = np.arange(POLAR_W, dtype=np.float32)
        band_sigma = max(POLAR_W * 0.08, 8.0)
        band       = np.exp(-0.5 * ((col_idx - r_ring_col) / band_sigma) ** 2)
        polar_mask_raw = (band[None, :] * 255).astype(np.uint8).repeat(POLAR_H, axis=0)

    if len(polar_mask_raw.shape) == 3:
        polar_mask_raw = polar_mask_raw[:, :, 0]

    polar_scratched, polar_mask_out = synthesize_scratch_procedural(
        polar_img, polar_mask_raw, r_ring_col,
    )

    osize       = (img_rgb.shape[1], img_rgb.shape[0])
    cv_result   = _from_polar(polar_scratched, center, max_radius, osize)
    mask_result = _from_polar(polar_mask_out, center, max_radius, osize)
    if len(mask_result.shape) == 3:
        mask_result = mask_result[:, :, 0]

    return cv_result, mask_result


_DISPATCH = {
    "mc_deform":     _run_mc_deform,
    "ring_fracture": _run_ring_fracture,
    "scratch":       _run_scratch,
}


# ── Public: generate() ────────────────────────────────────────────────────────

def generate(
    base_image_b64: str,
    defect_type:    str,
    params:         dict,
) -> dict:
    """
    Generate one Metal Cap defect image.

    Parameters
    ----------
    base_image_b64 : base64 PNG — OK image (RGB)
    defect_type    : "mc_deform" | "ring_fracture" | "scratch"
    params         : dict with:
        intensity       float 0-1
        seed            int
        sdxl_refine     bool  (default True)  — set False for fast CV-only
        ref_image_b64   str   — base64 NG crop for IP-Adapter (recommended)
        jitter_amplitude float — ring_fracture only
        falloff_width    float — ring_fracture softness (default 1.0)
        mask_b64         str  — scratch only, user-drawn mask

    Returns
    -------
    dict:
        result_image      : base64 PNG  (SDXL refined if sdxl_refine=True)
        result_pre_refine : base64 PNG  (CV-only result, always present)
        mask_b64          : base64 PNG  (defect mask)
        engine            : "cv" | "cv+sdxl"
        metadata          : dict
    """
    fn = _DISPATCH.get(defect_type)
    if fn is None:
        return {"error": f"Unknown defect_type: {defect_type!r}. "
                         f"Valid: {list(_DISPATCH)}"}

    img_rgb = decode_b64(base_image_b64)

    try:
        cv_result_rgb, defect_mask = fn(img_rgb, params)
    except Exception as e:
        return {"error": f"CV synthesis error ({defect_type}): {e}"}

    if cv_result_rgb.shape[:2] != img_rgb.shape[:2]:
        cv_result_rgb = cv2.resize(cv_result_rgb,
                                   (img_rgb.shape[1], img_rgb.shape[0]))

    # Encode CV result
    import base64 as _b64
    _, buf = cv2.imencode(".png", cv2.cvtColor(cv_result_rgb, cv2.COLOR_RGB2BGR))
    pre_refine_b64 = _b64.b64encode(buf).decode("utf-8")

    _, mbuf = cv2.imencode(".png", defect_mask)
    mask_b64_out = _b64.b64encode(mbuf).decode("utf-8")

    # ── SDXL refine ───────────────────────────────────────────────────────────
    do_refine = params.get("sdxl_refine", True)
    ref_b64   = params.get("ref_image_b64")
    seed      = int(params.get("seed", 42))

    if do_refine:
        ref_rgb = decode_b64(ref_b64) if ref_b64 else None
        try:
            refined_rgb = _sdxl_refine(
                cv_result_rgb, defect_mask, img_rgb,
                defect_type, ref_rgb, seed,
            )
            result_b64 = encode_b64(refined_rgb)
            engine_str = "cv+sdxl"
        except Exception as e:
            print(f"[metal_cap] SDXL refine failed ({defect_type}): {e} — returning CV result")
            result_b64 = encode_b64(cv_result_rgb)
            engine_str = "cv"
    else:
        result_b64 = encode_b64(cv_result_rgb)
        engine_str = "cv"

    return {
        "result_image":      result_b64,
        "result_pre_refine": pre_refine_b64,
        "mask_b64":          mask_b64_out,
        "engine":            engine_str,
        "metadata": {
            "defect_type": defect_type,
            "sdxl_refine": do_refine,
            "params":      params,
        },
    }
