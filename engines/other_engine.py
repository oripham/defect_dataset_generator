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

from .utils import encode_b64, decode_b64, decode_b64_gray


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


# ── CV Fallback: alpha-blend paste ───────────────────────────────────────────

def _cv_paste_defect(
    ok_bgr: np.ndarray,
    mask_gray: np.ndarray,
    ref_bgr: np.ndarray | None,
    params: dict,
) -> np.ndarray:
    """
    Simple CV fallback when GenAI is not available.
    Extracts the masked region from the NG reference and blends it
    onto the OK image at the same position.
    """
    result = ok_bgr.copy()
    intensity = float(params.get("intensity", 0.6))

    if ref_bgr is None:
        # No reference — just darken the masked area to simulate a generic defect
        mask_f = (mask_gray.astype(np.float32) / 255.0) * intensity
        mask_3ch = np.stack([mask_f] * 3, axis=-1)
        darkened = (ok_bgr.astype(np.float32) * (1.0 - mask_3ch * 0.4)).clip(0, 255).astype(np.uint8)
        return darkened

    # Resize ref to match OK
    ref_resized = cv2.resize(ref_bgr, (ok_bgr.shape[1], ok_bgr.shape[0]))

    # Alpha blend: result = ok * (1 - alpha) + ref * alpha, only in masked region
    mask_f = (mask_gray.astype(np.float32) / 255.0) * intensity
    mask_3ch = np.stack([mask_f] * 3, axis=-1)

    blended = (
        ok_bgr.astype(np.float32) * (1.0 - mask_3ch)
        + ref_resized.astype(np.float32) * mask_3ch
    ).clip(0, 255).astype(np.uint8)

    # Gaussian blur the edges for smoother blending
    kernel = max(3, int(min(ok_bgr.shape[:2]) * 0.01) | 1)
    blurred_mask = cv2.GaussianBlur(mask_gray, (kernel, kernel), 0)
    blend_f = blurred_mask.astype(np.float32) / 255.0 * intensity
    blend_3ch = np.stack([blend_f] * 3, axis=-1)

    result = (
        ok_bgr.astype(np.float32) * (1.0 - blend_3ch)
        + ref_resized.astype(np.float32) * blend_3ch
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
    params: dict | None = None,
) -> dict:
    """
    Generate one "other" defect image using reference-based approach.

    Parameters
    ----------
    base_image_b64 : base64 PNG of the OK product image
    mask_b64       : base64 PNG grayscale mask (white = defect region).
                     If None, auto-generates a center mask.
    ref_image_b64  : base64 PNG of the NG reference image (style source).
                     If None, uses a generic darkening effect.
    params         : {seed, intensity (0-1), use_genai (bool, default True)}

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

    # Decode or auto-generate mask
    if mask_b64:
        mask_gray = decode_b64_gray(mask_b64)
        if mask_gray.shape[:2] != (h, w):
            mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        # Auto-generate center mask (20% of image)
        mask_gray = np.zeros((h, w), dtype=np.uint8)
        cy, cx = h // 2, w // 2
        rh, rw = h // 5, w // 5
        mask_gray[cy - rh : cy + rh, cx - rw : cx + rw] = 255

    # Decode NG reference
    ref_bgr = _b64_to_bgr(ref_image_b64) if ref_image_b64 else None

    # Try GenAI first
    use_genai = params.get("use_genai", True)
    engine_used = "cv_paste"
    result_bgr = None

    if use_genai:
        genai_result = _try_genai(ok_rgb, mask_gray, ref_image_b64, params)
        if genai_result and "result_image" in genai_result:
            # GenAI succeeded
            result_bgr = _b64_to_bgr(genai_result["result_image"])
            engine_used = "genai"

    # Fallback to CV paste
    if result_bgr is None:
        result_bgr = _cv_paste_defect(ok_bgr, mask_gray, ref_bgr, params)
        engine_used = "cv_paste"

    # Build debug panel
    debug_panel = _make_debug_panel(ok_bgr, mask_gray, result_bgr)

    return {
        "result_image": _bgr_to_b64(result_bgr),
        "mask_b64":     _bgr_to_b64(cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)),
        "engine":       engine_used,
        "debug_panel":  _bgr_to_b64(debug_panel),
        "metadata": {
            "engine":    engine_used,
            "intensity": params.get("intensity", 0.6),
            "seed":      params.get("seed"),
            "has_ref":   ref_image_b64 is not None,
        },
    }
