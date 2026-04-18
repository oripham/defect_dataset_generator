"""
engines/capsule_engine.py — Pharma Defect CV Engine
====================================================

Wraps capsule_experiments.py synthesis functions into the unified
generate() interface used by router_engine.py.

Supported defect_type values:
  "hollow"    → synth_rong()        (Elongated Capsule — empty body)
  "underfill" → synth_thieu()       (Elongated Capsule — partial fill)
  "crack"     → synth_nut_tron()    (Round Tablet — chip/break)
  "dent"      → synth_lom_tron()    (Round Tablet — surface dent)  [stub]

Auto-mask: detect_tablet_mask() via Otsu — no user-drawn mask required.
The `mask` param from router is IGNORED for pharma (auto-detected internally).
"""

from __future__ import annotations

import os
import base64
import cv2
import numpy as np

# ── Data root (pharma images: ok/, ref/) ─────────────────────────────────────
_DATA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "defect_samples")
)
if not os.path.isdir(_DATA_ROOT):
    _DATA_ROOT = r"V:\defect_samples"

try:
    from ..synthesis import capsule_experiments as _ce
    _HAS_CE = True
except ImportError as _e:
    _HAS_CE = False
    _CE_ERR = str(_e)

try:
    from ..synthesis.thuoc_tron_generate_dataset import synth_dent as _synth_dent_tron, synth_chip as _synth_chip_tron
    _HAS_DENT_TRON = True
except Exception:
    _HAS_DENT_TRON = False

from ..utils import encode_b64, decode_b64


# ── Public: auto-mask ─────────────────────────────────────────────────────────

def auto_mask(base_image_b64: str) -> dict:
    """
    Detect tablet/capsule mask from OK image via Otsu threshold.
    Returns mask as base64 PNG + bbox.

    Called by POST /api/pharma/auto-mask
    """
    if not _HAS_CE:
        return {"error": f"capsule_experiments not available: {_CE_ERR}"}

    img_rgb = decode_b64(base_image_b64)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    mask, bbox = _ce.detect_capsule_mask(img_bgr)

    # Encode mask as b64 PNG (grayscale)
    _, buf = cv2.imencode(".png", mask)
    mask_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "mask_b64":  mask_b64,
        "bbox":      list(bbox),          # [x, y, w, h]
        "mask_area": int((mask > 127).sum()),
    }


# ── Internal synthesis dispatchers ───────────────────────────────────────────

def _run_hollow(img_bgr: np.ndarray, mask: np.ndarray,
                bbox: tuple, params: dict) -> np.ndarray:
    """Elongated Capsule — Hollow defect."""
    rng_seed     = int(params.get("seed", 42))
    intensity    = float(params.get("intensity", 0.7))
    fixed_region = bool(params.get("fixed_region", False))
    ref_b64      = params.get("ref_image_b64")
    
    if ref_b64 and _HAS_CE:
        ng_rgb = decode_b64(ref_b64)
        ng_bgr = cv2.cvtColor(ng_rgb, cv2.COLOR_RGB2BGR)
        refine_ai = bool(params.get("refine_ai", False))
        
        # New Hybrid CV replacement logic (referenced-based)
        result, _ = _ce.synth_hybrid_cv_replacement(
            img_bgr, ng_bgr, 
            seed=rng_seed, 
            intensity=intensity,
            refine_ai=refine_ai
        )
        return result

    result = _ce.synth_rong(
        img_bgr, mask, bbox,
        seed=rng_seed,
        intensity=intensity,
        fixed_region=fixed_region,
    )
    return result


def _run_underfill(img_bgr: np.ndarray, mask: np.ndarray,
                   bbox: tuple, params: dict) -> np.ndarray:
    """Elongated Capsule — Underfill defect."""
    rng_seed     = int(params.get("seed", 42))
    intensity    = float(params.get("intensity", 0.6))
    fixed_region = bool(params.get("fixed_region", False))
    ref_b64      = params.get("ref_image_b64")

    if ref_b64 and _HAS_CE:
        # For underfill, we use the same hybrid replacement as hollow for high fidelity.
        # Variations can be handled by adjusting intensity or providing specific NG samples.
        return _run_hollow(img_bgr, mask, bbox, params)

    fn = getattr(_ce, "synth_thieu", None) or getattr(_ce, "synth_thieu_ham_luong", None)
    if fn is None:
        params_copy = dict(params, intensity=intensity * 0.5)
        return _run_hollow(img_bgr, mask, bbox, params_copy)
    return fn(img_bgr, mask, bbox, seed=rng_seed, intensity=intensity, fixed_region=fixed_region)


def _run_crack(img_bgr: np.ndarray, mask: np.ndarray,
               bbox: tuple, params: dict):
    """Round Tablet — Crack via capsule_experiments.synth_nut_tron (plane-cut, multi-type).
    Uses detect_tablet_mask (dual-polarity) already run in generate().
    Returns result_bgr (no tuple — defect mask derived from diff in generate()).
    """
    seed       = int(params.get("seed", 42))
    intensity  = float(params.get("intensity", 0.8))
    cut_depth  = params.get("depth")
    cut_angle  = params.get("angle")
    break_type = str(params.get("break_type", "straight"))

    cut_depth = float(cut_depth) if cut_depth is not None else None
    cut_angle = float(cut_angle) if cut_angle is not None else None

    result = _ce.synth_nut_tron(
        img_bgr, mask, bbox,
        seed=seed,
        intensity=intensity,
        cut_depth=cut_depth,
        cut_angle=cut_angle,
        break_type=break_type,
    )
    return result


def _run_dent(img_bgr: np.ndarray, mask: np.ndarray,
              bbox: tuple, params: dict):
    """Round Tablet — Dent via thuoc_tron_generate_dataset.synth_dent (Hough-aware).
    Workflow identical to run_all_generators.py → thuoc_tron_generate_dataset.py:
    just pass the full OK image, synth_dent handles detect + placement internally.
    Returns (result_bgr, defect_mask) tuple.
    """
    seed         = int(params.get("seed", 42))
    intensity    = float(params.get("intensity", 0.7))
    dent_strength = float(params.get("dent_strength", 1.0))
    dent_size     = float(params.get("dent_size", 0.08))
    if _HAS_DENT_TRON:
        # Pass full img_bgr — synth_dent calls detect_pill_mask internally
        result, dmask = _synth_dent_tron(img_bgr, seed=seed, intensity=intensity,
                                          dent_strength=dent_strength, dent_size=dent_size)
        # If no change detected (detect_pill_mask failed), result == original
        # Try with flipped threshold (light pill on dark bg)
        if np.abs(result.astype(np.float32) - img_bgr.astype(np.float32)).mean() < 0.5:
            img_inv = cv2.bitwise_not(img_bgr)
            result_inv, dmask_inv = _synth_dent_tron(img_inv, seed=seed, intensity=intensity,
                                                      dent_strength=dent_strength, dent_size=dent_size)
            if np.abs(result_inv.astype(np.float32) - img_inv.astype(np.float32)).mean() > 0.5:
                delta = result_inv.astype(np.int16) - img_inv.astype(np.int16)
                result = np.clip(img_bgr.astype(np.int16) - delta, 0, 255).astype(np.uint8)
                dmask = dmask_inv
        return result, dmask
    # fallback
    return _ce.synth_tron_lom(
        img_bgr, mask, bbox,
        seed=seed, intensity=intensity, return_mask=True,
    )


_DISPATCH = {
    "hollow":    _run_hollow,
    "underfill": _run_underfill,
    "crack":     _run_crack,
    "dent":      _run_dent,
}


# ── QC check ─────────────────────────────────────────────────────────────────

def _qc_check(result_bgr: np.ndarray, mask: np.ndarray,
               defect_type: str) -> dict:
    """Run auto-QC and return verdict + metrics."""
    try:
        if defect_type == "hollow":
            return _ce.auto_check_rong(result_bgr, mask)
        # Generic QC: measure diff vs... (no OK available here, skip)
        return {"verdict": "SKIP", "reason": "no QC for this defect type"}
    except Exception as e:
        return {"verdict": "ERROR", "reason": str(e)}


# ── Public: generate() — unified interface ────────────────────────────────────

def generate(
    base_image_b64: str,
    mask_b64:       str | None,
    defect_type:    str,
    params:         dict,
    ref_image_b64:  str | None = None,
) -> dict:
    """
    Generate one pharma defect image.

    Parameters
    ----------
    base_image_b64 : base64 PNG — OK image (RGB)
    mask_b64       : base64 PNG grayscale — pre-computed mask, or None → auto-detect
    defect_type    : "hollow" | "underfill" | "crack" | "dent"
    params         : dict with intensity, seed, and defect-specific keys

    Returns
    -------
    dict:
        result_image : base64 PNG
        mask_b64     : base64 PNG (mask used)
        engine       : "cv"
        qc           : dict {verdict, ...metrics}
        metadata     : dict
    """
    if not _HAS_CE:
        return {"error": f"capsule_experiments not available: {_CE_ERR}"}

    fn = _DISPATCH.get(defect_type)
    if fn is None:
        return {"error": f"Unknown defect_type: {defect_type!r}"}

    # Update params with ref if provided as kwarg (from FastAPI)
    if ref_image_b64 and "ref_image_b64" not in params:
        params["ref_image_b64"] = ref_image_b64

    # Decode input
    img_rgb = decode_b64(base_image_b64)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    print(f"[capsule_engine] Generating '{defect_type}'... size={img_rgb.shape[:2]}")

    H, W = img_bgr.shape[:2]

    # dent: synth_dent handles detection internally
    # crack: round tablet → detect_tablet_mask (Otsu dual-polarity)
    # hollow / underfill: elongated capsule → detect_capsule_mask (threshold=60)
    if defect_type == "dent":
        mask = np.ones((H, W), dtype=np.uint8) * 255
        bbox = (0, 0, W, H)
    elif defect_type == "crack":
        mask, bbox = _ce.detect_tablet_mask(img_bgr)
        if (mask > 127).sum() < 50:
            return {"error": "Tablet mask detection failed — tablet not found in image"}
    else:
        mask, bbox = _ce.detect_capsule_mask(img_bgr)
        if (mask > 127).sum() < 50:
            mask = np.ones((H, W), dtype=np.uint8) * 255
            bbox = (0, 0, W, H)

    _, buf = cv2.imencode(".png", mask)
    mask_b64_out = base64.b64encode(buf).decode("utf-8")

    # Run synthesis
    try:
        print(f"[capsule_engine] Dispatching {fn.__name__}...")
        raw = fn(img_bgr, mask, bbox, params)
        print(f"[capsule_engine] Synthesis done. Result type: {type(raw)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Synthesis error: {e}"}

    # Some fns return (result, defect_mask) tuple
    if isinstance(raw, tuple):
        result_bgr, defect_mask = raw
        _, buf = cv2.imencode(".png", defect_mask)
        mask_b64_out = base64.b64encode(buf).decode("utf-8")
    else:
        result_bgr = raw

    # Ensure same size as input
    if result_bgr.shape != img_bgr.shape:
        result_bgr = cv2.resize(result_bgr, (img_bgr.shape[1], img_bgr.shape[0]))

    # --- SDXL Texture Refinement (Global toggle for Capsule) ---
    if params.get("sdxl", False) and hasattr(_ce, "_apply_sdxl_refine_to_bgr"):
        try:
            # Force enable the singleton flag if requested by API
            _ce.SDXL_REFINE_ENABLED = True
            refined = _ce._apply_sdxl_refine_to_bgr(result_bgr)
            if refined is not None:
                result_bgr = refined
        except Exception as e:
            print(f"[capsule_engine] Background SDXL ignored: {e}")

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # QC
    qc = _qc_check(result_bgr, mask, defect_type)

    return {
        "result_image": encode_b64(result_rgb),
        "mask_b64":     mask_b64_out,
        "engine":       "cv",
        "qc":           qc,
        "metadata": {
            "defect_type": defect_type,
            "bbox":        list(bbox),
            "mask_area":   int((mask > 127).sum()),
            "params":      params,
        },
    }
