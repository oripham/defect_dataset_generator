"""
engines/other_engine.py — "Other" (Unclassified) Defect Engine
================================================================

Pipeline for defects that don't fit the hardcoded cap/pharma/metal_cap
categories.  Uses a **reference-image-based** approach:

  1.  User provides OK image + NG reference image + mask
  2.  IP-Adapter learns the *style* of the defect from the NG image
  3.  SDXL inpaints the defect region on the OK image

If GenAI is not available (no GPU), falls back to a simple alpha-blend
paste of the NG crop onto the masked region of the OK image.

generate(
    base_image_b64,   # OK product image (base64 PNG)
    mask_b64,         # user-drawn mask on OK image (base64 PNG grayscale)
    ref_image_b64,    # NG reference image (base64 PNG) — style source
    params,           # {seed, intensity, ...}
) -> dict
"""

from __future__ import annotations

import base64
import traceback

import cv2
import numpy as np

from ..utils import encode_b64, decode_b64, decode_b64_gray


# ── Helpers ──────────────────────────────────────────────────────────────────

def _b64_to_bgr(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _bgr_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


def _make_debug_panel(ok_bgr, mask_gray, result_bgr, panel_h=240) -> np.ndarray:
    """Build 4-panel debug image: OK | Mask | Result | Diff×4."""
    diff = cv2.absdiff(ok_bgr, result_bgr)
    diff_bright = cv2.convertScaleAbs(diff, alpha=4.0)
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    panels = []
    for img in [ok_bgr, mask_bgr, result_bgr, diff_bright]:
        h, w = img.shape[:2]
        pw = int(w * panel_h / h)
        panels.append(cv2.resize(img, (pw, panel_h)))
    return np.hstack(panels)


# ── Notebook Helpers ─────────────────────────────────────────────────────────

def get_circle_hough(img_bgr):
    """Fallback circle detection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20,
        param1=50, param2=30, minRadius=10, maxRadius=0
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, 0] # [x, y, r]
    return None

def get_product_contour(img_bgr):
    """Find largest outer contour."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    return max(contours, key=cv2.contourArea)

def get_mask_centroid_and_dist(mask_np, center):
    """Extract centroid, distance to product center, and polar angle."""
    M = cv2.moments(mask_np)
    if M['m00'] == 0: return None
    cX_m = int(M['m10'] / M['m00'])
    cY_m = int(M['m01'] / M['m00'])

    dx = cX_m - center[0]
    dy = center[1] - cY_m # Y is inverted in screen space
    
    dist = np.sqrt(dx**2 + (cY_m - center[1])**2)
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    return (cX_m, cY_m), dist, angle

def align_by_mask_features(good_bgr, ng_bgr, mask_ok, mask_ng):
    """Align NG to Good based on mask positions/orientations."""
    h_g, w_g = good_bgr.shape[:2]
    
    # 1. Find product geometry
    circ_g = get_circle_hough(good_bgr)
    circ_n = get_circle_hough(ng_bgr)
    
    if circ_g is None or circ_n is None:
        c_g = get_product_contour(good_bgr)
        c_n = get_product_contour(ng_bgr)
        if c_g is None or c_n is None:
            # Fallback to center
            center_g, radius_g = (w_g // 2, h_g // 2), min(w_g, h_g) // 3
            center_n, radius_n = (ng_bgr.shape[1] // 2, ng_bgr.shape[0] // 2), min(ng_bgr.shape[1], ng_bgr.shape[0]) // 3
        else:
            M_g, M_n = cv2.moments(c_g), cv2.moments(c_n)
            center_g = (int(M_g['m10']/M_g['m00']), int(M_g['m01']/M_g['m00']))
            center_n = (int(M_n['m10']/M_n['m00']), int(M_n['m01']/M_n['m00']))
            radius_g = np.sqrt(cv2.contourArea(c_g)/np.pi)
            radius_n = np.sqrt(cv2.contourArea(c_n)/np.pi)
    else:
        center_g, radius_g = (circ_g[0], circ_g[1]), circ_g[2]
        center_n, radius_n = (circ_n[0], circ_n[1]), circ_n[2]

    # 2. Extract mask features
    feat_g = get_mask_centroid_and_dist(mask_ok, center_g)
    feat_n = get_mask_centroid_and_dist(mask_ng, center_n)
    
    if not feat_g or not feat_n:
        # If masks missing, just resize ref to fit good
        return cv2.resize(ng_bgr, (w_g, h_g))

    # 3. Calculate alignment
    scale_factor = radius_g / radius_n
    rot_angle = feat_g[2] - feat_n[2]

    # 4. Apply transformations
    h_n, w_n = ng_bgr.shape[:2]
    new_size = (int(w_n * scale_factor), int(h_n * scale_factor))
    img_n_scaled = cv2.resize(ng_bgr, new_size)
    cX_n_scaled, cY_n_scaled = center_n[0] * scale_factor, center_n[1] * scale_factor

    # Rotate around product center
    M_rot = cv2.getRotationMatrix2D((cX_n_scaled, cY_n_scaled), rot_angle, 1.0)
    img_n_rot = cv2.warpAffine(img_n_scaled, M_rot, (new_size[0], new_size[1]))

    # Translate to match GOOD center
    tx = center_g[0] - cX_n_scaled
    ty = center_g[1] - cY_n_scaled
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    final_ng = cv2.warpAffine(img_n_rot, M_trans, (w_g, h_g))
    
    return final_ng


# ── CV Fallback: alpha-blend paste ───────────────────────────────────────────

def _cv_paste_defect(
    ok_bgr: np.ndarray,
    mask_gray: np.ndarray,
    ref_aligned: np.ndarray | None,
    params: dict,
) -> np.ndarray:
    """
    CV fallback using mask-aligned reference.
    """
    result = ok_bgr.copy()
    intensity = float(params.get("intensity", 0.6))

    if ref_aligned is None:
        # No reference — just darken the masked area
        mask_f = (mask_gray.astype(np.float32) / 255.0) * intensity
        mask_3ch = np.stack([mask_f] * 3, axis=-1)
        darkened = (ok_bgr.astype(np.float32) * (1.0 - mask_3ch * 0.4)).clip(0, 255).astype(np.uint8)
        return darkened

    # Gaussian blur the mask edges for smoother blending
    kernel = max(3, int(min(ok_bgr.shape[:2]) * 0.01) | 1)
    blurred_mask = cv2.GaussianBlur(mask_gray, (kernel * 2 + 1, kernel * 2 + 1), 0)
    blend_f = blurred_mask.astype(np.float32) / 255.0 * intensity
    blend_3ch = np.stack([blend_f] * 3, axis=-1)

    result = (
        ok_bgr.astype(np.float32) * (1.0 - blend_3ch)
        + ref_aligned.astype(np.float32) * blend_3ch
    ).clip(0, 255).astype(np.uint8)

    return result


# ── GenAI path ───────────────────────────────────────────────────────────────

def _try_genai(
    ok_rgb: np.ndarray,
    mask_gray: np.ndarray,
    ref_b64: str | None,
    params: dict,
) -> dict | None:
    """Try to use deep_generative for high-quality synthesis. Returns None if unavailable."""
    try:
        from .deep_generative import generate as genai_generate
    except Exception:
        return None

    # For "other" defects, we use "foreign" as a generic defect type
    # since it gives the most neutral SDXL prompt
    genai_params = dict(params)
    if ref_b64:
        genai_params["ref_image_b64"] = ref_b64
    genai_params.setdefault("intensity", 0.6)
    genai_params.setdefault("naturalness", 0.6)

    # Support custom prompt from UI
    custom_prompt = params.get("prompt")
    if custom_prompt:
        genai_params["prompts"] = [custom_prompt]

    try:
        result = genai_generate(
            base_image=ok_rgb,
            mask=mask_gray,
            defect_type="foreign",   # generic type — most neutral prompt
            material="plastic",      # generic material
            params=genai_params,
        )
        return result
    except Exception as e:
        print(f"[other_engine] GenAI failed, falling back to CV: {e}")
        traceback.print_exc()
        return None


# ── Main Entry Point ─────────────────────────────────────────────────────────

def generate(
    base_image_b64: str,
    mask_b64: str | None,
    ref_image_b64: str | None = None,
    ref_mask_b64: str | None = None,
    params: dict | None = None,
) -> dict:
    """
    Generate one "other" defect image using reference-based approach with Alignment.

    Parameters
    ----------
    base_image_b64 : base64 PNG of the OK product image
    mask_b64       : base64 PNG grayscale mask on OK image
    ref_image_b64  : base64 PNG of the NG reference image
    ref_mask_b64   : base64 PNG grayscale mask on NG image (for alignment)
    params         : {seed, intensity, use_ai (bool)}

    Returns
    -------
    dict: {
        result_image : base64 PNG,
        mask_b64     : base64 PNG,
        engine       : "genai" | "cv_paste",
        debug_panel  : base64 PNG (4-panel),
        metadata     : {...},
    }
    """
    if params is None:
        params = {}

    # Decode OK image
    ok_bgr = _b64_to_bgr(base_image_b64)
    h, w = ok_bgr.shape[:2]
    ok_rgb = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2RGB)

    # Decode masks
    if mask_b64:
        mask_ok = decode_b64_gray(mask_b64)
        if mask_ok.shape[:2] != (h, w):
            mask_ok = cv2.resize(mask_ok, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_ok = np.zeros((h, w), dtype=np.uint8)
        cy, cx = h // 2, w // 2
        rh, rw = h // 5, w // 5
        mask_ok[cy - rh : cy + rh, cx - rw : cx + rw] = 255

    mask_ng = decode_b64_gray(ref_mask_b64) if ref_mask_b64 else None

    # Decode NG reference
    ref_bgr = _b64_to_bgr(ref_image_b64) if ref_image_b64 else None
    
    # ── ALIGNMENT STEP ────────────────────────────────────────────────────────
    ref_aligned = None
    if ref_bgr is not None and mask_ng is not None:
        ref_aligned = align_by_mask_features(ok_bgr, ref_bgr, mask_ok, mask_ng)
    elif ref_bgr is not None:
        # Fallback to simple resize if NG mask is missing
        ref_aligned = cv2.resize(ref_bgr, (w, h))

    # Try GenAI first
    use_ai = params.get("use_ai", params.get("use_genai", True))
    engine_used = "cv_paste"
    result_bgr = None

    if use_ai:
        genai_result = _try_genai(ok_rgb, mask_ok, ref_image_b64, params)
        if genai_result and "result_image" in genai_result:
            result_bgr = _b64_to_bgr(genai_result["result_image"])
            engine_used = "genai"

    # Fallback / Use CV
    if result_bgr is None:
        result_bgr = _cv_paste_defect(ok_bgr, mask_ok, ref_aligned, params)
        engine_used = "cv_paste"

    # Build debug panel
    debug_panel = _make_debug_panel(ok_bgr, mask_ok, result_bgr)

    return {
        "result_image": _bgr_to_b64(result_bgr),
        "mask_b64":     _bgr_to_b64(cv2.cvtColor(mask_ok, cv2.COLOR_GRAY2BGR)),
        "engine":       engine_used,
        "debug_panel":  _bgr_to_b64(debug_panel),
        "metadata": {
            "engine":    engine_used,
            "intensity": params.get("intensity", 0.6),
            "seed":      params.get("seed"),
            "has_ref":   ref_image_b64 is not None,
            "aligned":   ref_aligned is not None,
        },
    }
