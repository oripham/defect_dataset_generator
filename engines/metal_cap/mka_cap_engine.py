"""
engines/mka_cap_engine.py — CV Engine for MKA Cap (plastic bottle cap)
=======================================================================
Synthesis functions from experiments.py (mask-based, no polar transform).

Supported defect types:
  "scratch"      → apply_plastic_matte + synth_plastic_scuff
  "dent"         → gen_dent_patch + place_mask_random → fast_physics shaded_warp
  "plastic_flow" → gen_nhựa_patch + place_mask_random → synth_nhựa_chảy
  "thread"       → synth_thread (procedural organic curve)
  "dark_spots"   → synth_dark_spots (1 spot, r=(1,2))

API endpoint: POST /api/cap/preview
"""
from __future__ import annotations

import os
import base64
import glob
import random as _random

import cv2
import numpy as np

# ── Data root (MKA images: ok/, ref/, mask/) ─────────────────────────────────
_DATA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "defect_samples")
)
if not os.path.isdir(_DATA_ROOT):
    _DATA_ROOT = r"V:\defect_samples"
_MKA_ROOT = os.path.join(_DATA_ROOT, "MKA")

try:
    from ..synthesis import experiments as _exp
    _HAS_EXP = True
except ImportError as _e:
    _HAS_EXP = False
    _EXP_ERR = str(_e)

from ..utils import encode_b64, decode_b64
from .plastic_flow_engine import generate as _plastic_flow_generate
from ..synthesis.mka_can_mieng import synth_can_mieng as _synth_can_mieng


# ── Mask helpers ──────────────────────────────────────────────────────────────

def _pick_mask(mask_dir: str) -> np.ndarray | None:
    files = (glob.glob(os.path.join(mask_dir, "*.png")) +
             glob.glob(os.path.join(mask_dir, "*.jpg")))
    if not files:
        return None
    return cv2.imread(_random.choice(files), cv2.IMREAD_GRAYSCALE)


def _auto_place_mask(mask_src: np.ndarray, img_bgr: np.ndarray, seed: int) -> np.ndarray:
    rng2  = np.random.default_rng(seed)
    bbox  = _exp.detect_product_bbox(img_bgr)
    shape = _exp.extract_defect_shape(mask_src)
    if shape is None:
        return mask_src
    return _exp.place_mask_random(shape, bbox, img_bgr.shape[:2], rng2, rotate=True)


def _resolve_mask(params: dict, img_bgr: np.ndarray, mask_dir: str, seed: int,
                  incoming_mask_b64: str | None = None) -> np.ndarray | None:
    """Decode user mask from params/args, or pick from mask_dir and auto-place."""
    mask_b64 = incoming_mask_b64 or params.get("mask_b64") or params.get("user_mask_b64")

    if mask_b64:
        # data:image/png;base64, ...
        if "," in mask_b64: mask_b64 = mask_b64.split(",")[1]
        arr  = np.frombuffer(base64.b64decode(mask_b64), np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        return cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = _pick_mask(mask_dir)
    if mask is None:
        return None
    mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return _auto_place_mask(mask, img_bgr, seed)


# ── Synthesis functions ───────────────────────────────────────────────────────

def _run_scratch(img_bgr: np.ndarray, mask_dir: str, params: dict, mask_b64: str | None = None):
    if not _HAS_EXP:
        return None, None, "experiments not available"
    seed       = int(params.get("seed", 42))
    alpha_mult = float(params.get("alpha_mult", 2.5))
    whiten_add = float(params.get("whiten_add", 120))
    mode       = str(params.get("mode", "auto"))
    size       = float(params.get("scratch_size", 4.0))

    mask = _resolve_mask(params, img_bgr, mask_dir, seed, incoming_mask_b64=mask_b64)
    if mask is None:
        return None, None, "No mask found — please draw a mask on the image"

    result, defect_mask = _exp.synth_plastic_scuff(img_bgr, mask, seed=seed,
                                      alpha_mult=alpha_mult,
                                      whiten_add=whiten_add,
                                      mode=mode, size=size,
                                      n_hairlines_range=(10, 20),
                                      thick_range=(2, 4),
                                      matte_strength=0.0)
    return result, defect_mask, None


def _run_dent(img_bgr: np.ndarray, _mask_dir: str, params: dict):
    """Matches experiments.py: gen_dent_patch → fast_physics shaded_warp."""
    if not _HAS_EXP:
        return None, None, "experiments not available"

    seed      = int(params.get("seed", 42))
    intensity = float(params.get("intensity", 0.7))

    bbox  = _exp.detect_product_bbox(img_bgr)
    patch = _exp.gen_dent_patch(np.random.default_rng(seed), bbox)
    mask  = _exp.place_mask_random(patch, bbox, img_bgr.shape[:2],
                                   np.random.default_rng(seed + 1), rotate=True)
    try:
        from ..core import fast_physics as _fp
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res       = _fp.generate(img_rgb, mask, "dent", "metal", {
            "intensity": intensity, "naturalness": 0.6,
            "position_jitter": 0.0, "seed": seed,
            "sdxl_refine": False, "skip_struct_adapt": True,
        })
        result_bgr = cv2.cvtColor(decode_b64(res["result_pre_refine"]), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None, None, f"fast_physics dent failed: {e}"

    return result_bgr, mask, None


def _run_plastic_flow(img_bgr: np.ndarray, _mask_dir: str, params: dict):
    """
    Plastic flow generation engine.
    Default: CV-only.
    Optional: SDXL refine when params["sdxl_refine"]=True.
    """
    # Delegate to dedicated engine (keeps API consistent with other engines)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    payload_b64 = encode_b64(img_rgb)
    out = _plastic_flow_generate(payload_b64, params)
    if "error" in out:
        return None, None, out["error"]

    result_rgb = decode_b64(out["result_image"])
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    # mask_b64 returned by engine is PNG grayscale
    mask_b64 = out.get("mask_b64", "")
    if mask_b64:
        mask_arr = np.frombuffer(base64.b64decode(mask_b64), np.uint8)
        mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape[:2] != img_bgr.shape[:2]:
            mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = None

    engine = out.get("engine", "cv")
    return result_bgr, mask, None, engine


def _run_thread(img_bgr: np.ndarray, mask_dir: str, params: dict, mask_b64: str | None = None):
    if not _HAS_EXP:
        return None, None, "experiments not available"

    seed = int(params.get("seed", 42))
    mask = _resolve_mask(params, img_bgr, mask_dir, seed, incoming_mask_b64=mask_b64)
    if mask is None:
        mask = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255

    result, defect_mask = _exp.synth_thread(img_bgr, mask, seed=seed)
    return result, defect_mask, None


def _run_dark_spots(img_bgr: np.ndarray, _mask_dir: str, params: dict):
    """Dark spots — reads n_spots and r from params."""
    if not _HAS_EXP:
        return None, None, "experiments not available"

    bbox = _exp.detect_product_bbox(img_bgr)
    n_min = int(params.get("n_spots_min", 1))
    n_max = int(params.get("n_spots_max", 3))
    r_min = int(params.get("r_min", 2))
    r_max = int(params.get("r_max", 5))
    result, mask_out = _exp.synth_dark_spots(
        img_bgr, bbox,
        seed=int(params.get("seed", 42)),
        n_spots_range=(n_min, n_max),
        r_range=(r_min, r_max),
        bump=True,
    )
    return result, mask_out, None


def _run_can_mieng(img_bgr: np.ndarray, _mask_dir: str, params: dict):
    """Rim Crush — auto-detects inner rim, no external mask needed."""
    seed          = int(params.get("seed", 42))
    intensity     = float(params.get("intensity", 0.7))
    warp_strength = float(params.get("warp_strength", 1.0))
    streak_length = float(params.get("streak_length", 1.0))
    thickness     = float(params.get("thickness", 1.0))
    result, dmask = _synth_can_mieng(img_bgr, seed=seed,
                                     intensity=intensity,
                                     warp_strength=warp_strength,
                                     streak_length=streak_length,
                                     thickness=thickness)
    return result, dmask, None


# ── Dispatch table ────────────────────────────────────────────────────────────

_DISPATCH = {
    "scratch":      (_run_scratch,      "Xước"),
    "dent":         (_run_dent,         "Lõm"),
    "plastic_flow": (_run_plastic_flow, "Nhựa_chảy"),
    "thread":       (_run_thread,       "Dị_vật_chỉ"),
    "dark_spots":   (_run_dark_spots,   "Dị_vật_đen"),
    "can_mieng":    (_run_can_mieng,    u"Cấn_miệng"),
}


# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    base_image_b64: str,
    defect_type:    str,
    params:         dict,
    data_root:      str | None = None,
    mask_b64:       str | None = None,
) -> dict:
    """
    Generate one MKA Cap defect image.

    Parameters
    ----------
    base_image_b64 : base64 PNG/JPG — OK image (RGB)
    defect_type    : "scratch" | "dent" | "plastic_flow" | "thread" | "dark_spots"
    params         : dict with seed and defect-specific keys
    data_root      : path to MKA data folder (optional)

    Returns
    -------
    dict: result_image, mask_b64, engine, metadata
    """
    entry = _DISPATCH.get(defect_type)
    if entry is None:
        return {"error": f"Unknown defect_type for MKA Cap: {defect_type!r}. "
                         f"Valid: {list(_DISPATCH)}"}

    fn, defect_folder = entry
    img_rgb = decode_b64(base_image_b64)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    root     = data_root or _MKA_ROOT
    import unicodedata
    normalized_folder = unicodedata.normalize('NFC', defect_folder)
    mask_dir = os.path.join(root, normalized_folder, "mask")

    engine_name = "cv"
    try:
        if mask_b64:
            params["mask_b64"] = mask_b64
        if fn in (_run_scratch, _run_thread):
            result_bgr, mask_out, err = fn(img_bgr, mask_dir, params, mask_b64=mask_b64)
        else:
            res = fn(img_bgr, mask_dir, params)
            if len(res) == 4:
                result_bgr, mask_out, err, engine_name = res
            else:
                result_bgr, mask_out, err = res
    except Exception as e:
        return {"error": f"Synthesis error ({defect_type}): {e}"}

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
        _, buf = cv2.imencode(".png", mask_out)
        mask_b64 = base64.b64encode(buf).decode()

    return {
        "result_image": encode_b64(result_rgb),
        "mask_b64":     mask_b64,
        "engine":       engine_name,
        "qc":           {"verdict": "SKIP"},
        "metadata":     {"defect_type": defect_type, "params": params},
    }
