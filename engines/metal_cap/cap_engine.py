"""
engines/cap_engine.py — CV Engine for MKA Circular Cap Products
===============================================================
Wraps synthesis functions from cap_experiments.py (polar-based)
and experiments.py (mask-based) into the unified generate() interface.

Defect types and their synthesis strategy:
  "scratch"      → experiments.synth_scratch_lines + synth_plastic_scuff  (mask-based)
  "mc_deform"    → cap_experiments.synth_mc_deform                        (polar-based)
  "ring_fracture"→ cap_experiments.synth_ring_fracture                    (polar-based)
  "dent"         → experiments.gen_dent_patch + inject/fast_physics        (mask-based)
  "plastic_flow" → experiments.gen_nhựa_patch + inject/fast_physics        (mask-based)
  "dark_spots"   → experiments.synth_dark_spots                            (mask-based)
  "thread"       → experiments.synth_thread                                (mask-based)
"""
from __future__ import annotations

import os
import base64
import glob
import math
import random as _random

import cv2
import numpy as np

# ── Data root (MKA images: ok/, ref/, mask/) ─────────────────────────────────
# Looks for defect_samples/ next to repo root; falls back to V:\defect_samples
_DATA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "defect_samples")
)
if not os.path.isdir(_DATA_ROOT):
    _DATA_ROOT = r"V:\defect_samples"
_MKA_ROOT = os.path.join(_DATA_ROOT, "MKA")

try:
    from ..synthesis import cap_experiments as _cap
    _HAS_CAP = True
except ImportError as _e:
    _HAS_CAP = False
    _CAP_ERR = str(_e)

try:
    from ..synthesis import experiments as _exp
    _HAS_EXP = True
except ImportError as _e2:
    _HAS_EXP = False
    _EXP_ERR = str(_e2)

from ..utils import encode_b64, decode_b64

POLAR_H = 720
POLAR_W = 512


# ── Polar helpers ─────────────────────────────────────────────────────────────

def _prep_polar(img_bgr: np.ndarray):
    """Hough circle → polar space. Returns (center, max_r, polar, polar_illum, r_col)."""
    gray       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cx, cy, r  = _cap.detect_circle(gray)
    max_r      = int(r * 1.3)
    center     = (cx, cy)
    polar      = _cap.to_polar(img_bgr, center, max_r, (POLAR_H, POLAR_W))
    pg         = cv2.cvtColor(polar, cv2.COLOR_BGR2GRAY).astype(np.float32)
    polar_illum = cv2.GaussianBlur(pg, (15, 15), 5) / 255.0
    r_col      = _cap.find_rim_col(pg)
    return center, max_r, polar, polar_illum, r_col


def _inverse_polar(polar_out, mask_p, center, max_r, orig_bgr):
    w, h = orig_bgr.shape[1], orig_bgr.shape[0]
    result   = _cap.from_polar(polar_out, center, max_r, (w, h))
    if mask_p.ndim == 3:
        mask_p = mask_p[:, :, 0]
    mask_out = _cap.from_polar(mask_p, center, max_r, (w, h))
    return result, mask_out


# ── Mask helpers (mask-based defects) ────────────────────────────────────────

def _pick_mask(mask_dir: str) -> np.ndarray | None:
    """Pick a random mask from folder."""
    files = (glob.glob(os.path.join(mask_dir, "*.png")) +
             glob.glob(os.path.join(mask_dir, "*.jpg")))
    if not files:
        return None
    path = _random.choice(files)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return m


def _auto_place_mask(mask_src: np.ndarray, img_bgr: np.ndarray,
                      seed: int) -> np.ndarray:
    """Re-place mask randomly inside product bbox."""
    rng2 = np.random.default_rng(seed)
    bbox = _exp.detect_product_bbox(img_bgr)
    shape = _exp.extract_defect_shape(mask_src)
    if shape is None:
        return mask_src
    return _exp.place_mask_random(shape, bbox, img_bgr.shape[:2], rng2, rotate=True)


# ── Synthesis dispatchers ─────────────────────────────────────────────────────

def _run_scratch(img_bgr: np.ndarray, mask_dir: str, params: dict):
    """Scratch lines + plastic scuff on mask region."""
    if not _HAS_EXP:
        return None, None, "experiments not available"
    seed  = int(params.get("seed", 42))
    alpha_mult  = float(params.get("alpha_mult", 1.35))
    whiten_add  = float(params.get("whiten_add", 120))
    mode        = str(params.get("mode", "auto"))
    size        = float(params.get("scratch_size", 1.0))

    # Prefer user-drawn mask from params, fall back to mask_dir
    mask_b64_user = params.get("mask_b64") or params.get("user_mask_b64")
    if mask_b64_user:
        data = base64.b64decode(mask_b64_user)
        arr  = np.frombuffer(data, np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None, "Could not decode user mask"
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = _pick_mask(mask_dir)
        if mask is None:
            return None, None, "No mask found — please draw a mask on the image"
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = _auto_place_mask(mask, img_bgr, seed)

    result = _exp.apply_plastic_matte(img_bgr, mask, seed=seed, strength=0.6)
    result, defect_mask = _exp.synth_plastic_scuff(result, mask, seed=seed,
                                       alpha_mult=alpha_mult,
                                       whiten_add=whiten_add,
                                       mode=mode, size=size)
    return result, defect_mask, None


def _run_mc_deform(img_bgr: np.ndarray, _mask_dir: str, params: dict):
    """Polar-space rim deformation."""
    if not _HAS_CAP:
        return None, None, f"cap_experiments not available: {_CAP_ERR}"
    center, max_r, polar, polar_illum, r_col = _prep_polar(img_bgr)
    polar_out, mask_p = _cap.synth_mc_deform(
        polar, polar_illum, r_col, max_r,
        theta_center=float(params.get("theta_center", math.pi)),
        theta_span=float(params.get("theta_span", math.pi / 8)),
        seed=int(params.get("seed", 42)),
        deform_strength=float(params.get("deform_strength", 15.0)),
        warp_factor=float(params.get("warp_factor", 1.0)),
    )
    result, mask_out = _inverse_polar(polar_out, mask_p, center, max_r, img_bgr)
    return result, mask_out, None


def _run_ring_fracture(img_bgr: np.ndarray, _mask_dir: str, params: dict):
    """Polar-space ring fracture."""
    if not _HAS_CAP:
        return None, None, f"cap_experiments not available: {_CAP_ERR}"
    center, max_r, polar, _, r_col = _prep_polar(img_bgr)
    polar_out, mask_p = _cap.synth_ring_fracture(
        polar, r_col,
        seed=int(params.get("seed", 42)),
        jitter_amplitude=float(params.get("jitter_amplitude", 6.0)),
        falloff_power=float(params.get("falloff_power", 1.0)),
    )
    result, mask_out = _inverse_polar(polar_out, mask_p, center, max_r, img_bgr)
    return result, mask_out, None


def _run_dark_spots(img_bgr: np.ndarray, _mask_dir: str, params: dict):
    """Dark spots (Dị_vật_đen) — reads n_spots and r from params."""
    if not _HAS_EXP:
        return None, None, "experiments not available"
    bbox = _exp.detect_product_bbox(img_bgr)
    n_min = int(params.get("n_spots_min", 1))
    n_max = int(params.get("n_spots_max", 3))
    r_min = int(params.get("r_min", 5))
    r_max = int(params.get("r_max", 14))
    result, mask_out = _exp.synth_dark_spots(
        img_bgr, bbox,
        seed=int(params.get("seed", 42)),
        n_spots_range=(n_min, n_max),
        r_range=(r_min, r_max),
        bump=True,
    )
    return result, mask_out, None


def _run_thread(img_bgr: np.ndarray, mask_dir: str, params: dict):
    """Thread foreign object (Dị_vật_chỉ)."""
    if not _HAS_EXP:
        return None, None, "experiments not available"
    seed = int(params.get("seed", 42))

    mask_b64_user = params.get("mask_b64") or params.get("user_mask_b64")
    if mask_b64_user:
        data = base64.b64decode(mask_b64_user)
        arr  = np.frombuffer(data, np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
        else:
            mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = _pick_mask(mask_dir)
        if mask is None:
            mask = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
        else:
            mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            mask = _auto_place_mask(mask, img_bgr, seed)

    result, defect_mask = _exp.synth_thread(img_bgr, mask, seed=seed)
    return result, defect_mask, None


def _run_dent(img_bgr: np.ndarray, _mask_dir: str, params: dict):
    """Dent (Lõm) — gen_dent_patch + place_mask_random → fast_physics shaded_warp.
    Matches experiments.py exactly: calls fast_physics.generate with defect_type='dent'.
    """
    if not _HAS_EXP:
        return None, None, "experiments not available"

    seed      = int(params.get("seed", 42))
    intensity = float(params.get("intensity", 0.7))

    rng_shape = np.random.default_rng(seed)
    rng_place = np.random.default_rng(seed + 1)
    bbox  = _exp.detect_product_bbox(img_bgr)
    patch = _exp.gen_dent_patch(rng_shape, bbox)
    mask  = _exp.place_mask_random(patch, bbox, img_bgr.shape[:2], rng_place, rotate=True)

    # Call fast_physics the same way experiments.py does
    try:
        from ..core import fast_physics as _fp
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        fp_params = {
            "intensity":         intensity,
            "naturalness":       0.6,
            "position_jitter":   0.0,
            "seed":              seed,
            "sdxl_refine":       False,
            "skip_struct_adapt": True,
        }
        res        = _fp.generate(img_rgb, mask, "dent", "metal", fp_params)
        result_rgb = decode_b64(res["result_pre_refine"])
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None, None, f"fast_physics dent failed: {e}"

    return result_bgr, mask, None


def _run_plastic_flow(img_bgr: np.ndarray, _mask_dir: str, params: dict):
    """Plastic flow (Nhựa_chảy) — gen_nhựa_patch + synth_nhựa_chảy.
    Mirrors experiments.py exactly: gen_nhựa_patch + place_mask_random + synth_nhựa_chảy.
    """
    if not _HAS_EXP:
        return None, None, "experiments not available"
    seed      = int(params.get("seed", 42))
    intensity = float(params.get("intensity", 0.6))

    rng_shape = np.random.default_rng(seed)
    rng_place = np.random.default_rng(seed + 1)
    bbox  = _exp.detect_product_bbox(img_bgr)
    patch = _exp.gen_nhựa_patch(rng_shape, bbox)
    mask  = _exp.place_mask_random(patch, bbox, img_bgr.shape[:2], rng_place, rotate=True)

    result = _exp.synth_nhựa_chảy(img_bgr, mask, seed=seed, intensity=intensity)
    return result, mask, None


_DISPATCH = {
    "scratch":       (_run_scratch,       "Xước"),
    "mc_deform":     (_run_mc_deform,     "Cấn_miệng"),
    "ring_fracture": (_run_ring_fracture, "Cấn_miệng"),
    "dark_spots":    (_run_dark_spots,    "Dị_vật_đen"),
    "thread":        (_run_thread,        "Dị_vật_chỉ"),
    "dent":          (_run_dent,          "Lõm"),
    "plastic_flow":  (_run_plastic_flow,  "Nhựa_chảy"),
}


# ── Public: generate() ────────────────────────────────────────────────────────

def generate(
    base_image_b64: str,
    defect_type:    str,
    params:         dict,
    data_root:      str | None = None,   # defaults to _MKA_ROOT
) -> dict:
    """
    Generate one MKA cap defect image.

    Parameters
    ----------
    base_image_b64 : base64 PNG/JPG — OK image (RGB)
    defect_type    : "scratch"|"mc_deform"|"ring_fracture"|"dark_spots"|"thread"|"dent"|"plastic_flow"
    params         : dict with seed and defect-specific keys
    data_root      : path to MKA data root (optional; for mask lookup)

    Returns
    -------
    dict: result_image, mask_b64, engine, qc, metadata
    """
    entry = _DISPATCH.get(defect_type)
    if entry is None:
        return {"error": f"Unknown defect_type: {defect_type!r}"}

    fn, defect_folder = entry

    img_rgb = decode_b64(base_image_b64)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Determine mask_dir (only used by mask-based defects)
    root = data_root or _MKA_ROOT
    mask_dir = os.path.join(root, defect_folder, "mask")

    try:
        result_bgr, mask_out, err = fn(img_bgr, mask_dir, params)
    except Exception as e:
        return {"error": f"Synthesis error: {e}"}

    if err:
        return {"error": err}

    if result_bgr is None:
        return {"error": "Synthesis returned None"}

    if result_bgr.shape[:2] != img_bgr.shape[:2]:
        result_bgr = cv2.resize(result_bgr, (img_bgr.shape[1], img_bgr.shape[0]))

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    mask_b64 = ""
    if mask_out is not None:
        if mask_out.ndim == 3:
            mask_out = mask_out[:, :, 0]
        _, mask_buf = cv2.imencode(".png", mask_out)
        mask_b64 = base64.b64encode(mask_buf).decode("utf-8")

    return {
        "result_image": encode_b64(result_rgb),
        "mask_b64":     mask_b64,
        "engine":       "cv",
        "qc":           {"verdict": "SKIP"},
        "metadata": {
            "defect_type": defect_type,
            "params":      params,
        },
    }


# ── Public: detect_circle_info() ─────────────────────────────────────────────

def detect_circle_info(base_image_b64: str) -> dict:
    """Detect Hough Circle. Returns {cx, cy, r, max_r, img_w, img_h}."""
    if not _HAS_CAP:
        return {"error": f"cap_experiments not available: {_CAP_ERR}"}
    img_rgb = decode_b64(base_image_b64)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cx, cy, r = _cap.detect_circle(gray)
    return {"cx": cx, "cy": cy, "r": r, "max_r": int(r * 1.3),
            "img_w": img_bgr.shape[1], "img_h": img_bgr.shape[0]}
