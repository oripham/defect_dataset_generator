"""
engines/fast_physics.py — CV Defect Engine (Vuong)
===================================================

Unified interface wrapper around generator_classical.py.
Router calls: generate(base_image, mask, defect_type, material, params) -> dict

Defect type → method mapping
-----------------------------
  scratch, crack, foreign  →  signal_injection  (needs ref_image, fallback: shaded_warp)
  dent                     →  shaded_warp        (warp_mode="dent")
  bulge                    →  shaded_warp        (warp_mode="bulge")
  chip, rust               →  ref_paste          (needs ref_image, fallback: shaded_warp)

Params → config mapping
-----------------------
  intensity        [0-1]  →  intensity_min/max, warp_strength, blend_strength
  naturalness      [0-1]  →  alpha_blur, alpha_dilate
  position_jitter  [0-1]  →  random mask offset (pixels)
  seed             int    →  numpy seed for reproducibility
  ref_image_b64    str    →  base64 PNG NG reference (optional)
"""

import sys
import os
import random
import numpy as np
import cv2
from PIL import Image

# Allow import from scripts/ regardless of where this file is called from
_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCRIPTS))

from generator_classical import (
    generate_signal_injection,
    generate_ref_paste,
    generate_shaded_warp,
    generate_elastic_warp,
)
from ..utils import encode_b64, decode_b64
from ..shared.structure_adapt import structure_adapt as _structure_adapt

# ── SDXL Refiner singleton (lazy-loaded, optional) ────────────────────────────
_sdxl_refiner = None

def _get_sdxl_refiner():
    global _sdxl_refiner
    if _sdxl_refiner is not None:
        return _sdxl_refiner
    try:
        from sdxl_refiner import SDXLRefiner
        refine_cfg = {
            "enabled":        True,
            "strength":       0.14,
            "guidance_scale": 5.0,
            "steps":          20,
            "model":          "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        }
        _sdxl_refiner = SDXLRefiner(refine_cfg, device="cuda")
        print("[fast_physics] SDXLRefiner loaded.")
    except Exception as e:
        print(f"[fast_physics] SDXLRefiner unavailable: {e}")
        _sdxl_refiner = None
    return _sdxl_refiner


# ── Defect type → method ─────────────────────────────────────────────────────

_NEEDS_REF = {"scratch", "foreign"}
_WARP_TYPES = {"dent", "bulge", "chip"}   # chip: edge deformation, no ref needed

def _resolve_method(defect_type: str, has_ref: bool) -> str:
    if defect_type in _WARP_TYPES:
        return "shaded_warp"
    if defect_type == "rust" and has_ref:
        return "ref_paste"
    if defect_type in _NEEDS_REF and has_ref:
        return "signal_injection"
    return "shaded_warp"


# ── Params → gen_cfg ─────────────────────────────────────────────────────────

def _build_gen_cfg(defect_type: str, method: str, params: dict, light_dir=None) -> dict:
    intensity   = float(params.get("intensity",   0.6))   # 0–1
    naturalness = float(params.get("naturalness", 0.6))   # 0–1

    # Clamp to avoid edge cases
    intensity   = max(0.05, min(intensity,   1.0))
    naturalness = max(0.05, min(naturalness, 1.0))

    if method == "signal_injection":
        # crack = duong nut mong (line-shaped), dung poisson + tight alpha
        # scratch = vung nho bi xuoc (area), KHONG dung poisson
        #   Ly do: Poisson tren vung toi nho keo toan bo interior ve boundary (~50)
        #          → defect bien mat. Dung additive blend thay the.
        is_line = defect_type in ("crack",)
        return {
            "method":        "signal_injection",
            "blur_kernel":   51,
            "intensity_min": 1.5 + intensity * 1.5,   # tang len: 1.5 – 3.0
            "intensity_max": 2.0 + intensity * 2.0,   # tang len: 2.0 – 4.0
            # Area scratch: alpha_dilate=0 — KHONG mo rong ngoai mask
            # Dilate > 0 → alpha lan ra ngoai mask → inject signal duong → vien trang
            "alpha_dilate":    3 if is_line else 0,
            "alpha_blur":      max(3, int(naturalness * 7)) if is_line else max(11, int(naturalness * 25)),
            "thin_mask":       is_line,
            "use_poisson":     is_line,
            "radial_falloff":  not is_line,   # area scratch: dam o tam, nhat ra bien
        }

    if method == "ref_paste":
        return {
            "method":         "ref_paste",
            "blend_strength": 0.5 + intensity * 0.45,  # 0.5 – 0.95
            "alpha_dilate":   max(3, int(naturalness * 10)),
            "alpha_blur":     max(7, int(naturalness * 40)),
        }

    if method in ("shaded_warp", "elastic_warp"):
        warp_mode = "bulge" if defect_type == "bulge" else "dent"
        # chip: stronger amplitude + sharper normal to simulate broken edge
        if defect_type == "chip":
            warp_mode = "dent"  # same displacement field, edge geometry handles the rest
        cfg = {
            "method":        method,
            "warp_mode":     warp_mode,
            "warp_strength": 2.0 + intensity * 8.0,    # 2 – 10 px
            "warp_blur":     max(31, int(naturalness * 70)),
            "shading_gain":  50.0 + intensity * 70.0,  # 50 – 120  (giảm để không quá tối)
            "amplitude":     1.0 + intensity * 2.0,    # 1 – 3     (giảm độ sâu)
            "normal_scale":  25.0 + intensity * 25.0,  # 25 – 50
            "shininess":     12.0 + intensity * 36.0,  # 12 – 48
            "diffuse_w":     0.7,
            "specular_w":    0.3,
            "alpha_dilate":  max(3, int(naturalness * 12)),
            "alpha_blur":    max(15, int(naturalness * 40)),
        }
        if light_dir is not None:
            cfg["light_dir"] = light_dir  # radial normal overrides Sobel global
        return cfg

    return {}


# ── Ref image cleanup ────────────────────────────────────────────────────────

_REF_BRIGHT_THRESHOLD = 200   # pixel sang nhat trong cot/hang > nay → viền sáng → trim
_REF_BRIGHT_MIN_FRAC  = 0.15  # it nhat 15% so pixel trong cot phai sang → trim
                               # (tranh trim do 1-2 pixel nhieu)

def _trim_bright_edges(ref_pil: Image.Image) -> Image.Image:
    """
    Crop ref image: loai bo cac hang/cot bia co qua nhieu pixel sang (rim/highlight).
    Dung max + fraction check de tranh bi danh lua boi 1-2 pixel nhieu.
    Giu nguyen neu sau trim anh qua nho (<20px).
    """
    arr  = np.array(ref_pil.convert("L"))   # grayscale
    h, w = arr.shape

    def is_bright_edge(vec):
        """True neu cot/hang nay la vien sang can trim."""
        frac = np.mean(vec > _REF_BRIGHT_THRESHOLD)
        return frac >= _REF_BRIGHT_MIN_FRAC

    def find_bounds(arr2d, axis):
        # axis=0: duyet cot (W); axis=1: duyet hang (H)
        n = arr2d.shape[1] if axis == 0 else arr2d.shape[0]
        lo, hi = 0, n - 1
        while lo < n and is_bright_edge(arr2d[:, lo] if axis == 0 else arr2d[lo, :]):
            lo += 1
        while hi >= 0 and is_bright_edge(arr2d[:, hi] if axis == 0 else arr2d[hi, :]):
            hi -= 1
        return lo, hi

    x0, x1 = find_bounds(arr, axis=0)   # trim cot trai/phai
    y0, y1 = find_bounds(arr, axis=1)   # trim hang tren/duoi

    if x1 - x0 < 20 or y1 - y0 < 20:
        return ref_pil   # qua nho sau trim → giu nguyen

    trimmed = ref_pil.crop((x0, y0, x1 + 1, y1 + 1))
    if trimmed.size != ref_pil.size:
        ow, oh = ref_pil.size
        nw, nh = trimmed.size
        print(f"[REF TRIM] {ow}x{oh} → {nw}x{nh}  (removed bright edges)")
    return trimmed


# ── Position jitter ───────────────────────────────────────────────────────────

def _jitter_mask(mask: np.ndarray, jitter: float) -> np.ndarray:
    """Randomly shift mask by up to jitter * 50 pixels."""
    if jitter <= 0.01:
        return mask
    h, w = mask.shape
    max_shift = int(jitter * 50)
    dy = random.randint(-max_shift, max_shift)
    dx = random.randint(-max_shift, max_shift)
    M  = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ── Main entry point ──────────────────────────────────────────────────────────

def generate(
    base_image:  np.ndarray,
    mask:        np.ndarray,
    defect_type: str,
    material:    str,
    params:      dict,
) -> dict:
    """
    Generate one defect image using the CV physics engine.

    Parameters
    ----------
    base_image  : uint8 RGB (H, W, 3)
    mask        : uint8 grayscale (H, W), white = defect region
    defect_type : str — see interface_spec.md for valid values
    material    : "metal" | "plastic" | "pharma"
    params      : dict with keys: intensity, naturalness, position_jitter,
                  seed (optional), ref_image_b64 (optional)

    Returns
    -------
    dict with keys: result_image (base64 PNG), engine ("cv"), metadata (dict)
    """
    seed = params.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Apply position jitter to mask
    jitter = float(params.get("position_jitter", 0.0))
    if jitter > 0.01:
        mask = _jitter_mask(mask, jitter)

    # Structure-aware placement: snap mask to ring, compute radial light_dir
    # skip_struct_adapt=True → giữ nguyên mask vị trí random, chỉ lấy light_dir
    if params.get("skip_struct_adapt", False):
        _light_dir = None
    else:
        mask, _light_dir, _ = _structure_adapt(base_image, mask, defect_type)

    # Decode ref image if provided, then trim bright edges (rim artifact)
    ref_b64 = params.get("ref_image_b64")
    ref_image = None
    if ref_b64:
        try:
            ref_image = _trim_bright_edges(Image.fromarray(decode_b64(ref_b64)))
        except Exception as e:
            print(f"[WARN] fast_physics: could not decode ref_image_b64: {e}")

    # Resolve method
    method  = _resolve_method(defect_type, ref_image is not None)
    gen_cfg = _build_gen_cfg(defect_type, method, params, light_dir=_light_dir)

    base_pil = Image.fromarray(base_image)

    # Dispatch
    if method == "signal_injection":
        result_pil = generate_signal_injection(base_pil, mask, ref_image, gen_cfg)
    elif method == "ref_paste":
        result_pil = generate_ref_paste(base_pil, mask, ref_image, gen_cfg)
    elif method in ("shaded_warp", "elastic_warp"):
        result_pil = generate_shaded_warp(base_pil, mask, None, gen_cfg)
    else:
        result_pil = base_pil  # no-op fallback

    # Snapshot CV result truoc SDXL (de so sanh tren panel)
    pre_refine_arr = np.array(result_pil).copy()

    # SDXL texture refinement — runs on surroundings, defect pixels protected
    sdxl_ran = False
    do_refine = params.get("use_sdxl")
    if do_refine is None: do_refine = params.get("sdxl_refine", True)
    
    if do_refine:
        refiner = _get_sdxl_refiner()
        if refiner is not None:
            try:
                pre_sdxl   = np.array(result_pil)
                result_pil = refiner.refine_with_sdxl(result_pil)
                # Restore defect pixels — SDXL chỉ được chạm surroundings
                post_arr             = np.array(result_pil)
                post_arr[mask > 127] = pre_sdxl[mask > 127]
                result_pil           = Image.fromarray(post_arr)
                sdxl_ran = True
                print("[fast_physics] SDXL done — defect pixels protected")
            except Exception as e:
                print(f"[fast_physics] SDXL refine failed, using CV result: {e}")

    # Restore background pixels — prevent signal bleeding into dark background
    result_arr = np.array(result_pil)
    base_gray  = cv2.cvtColor(base_image, cv2.COLOR_RGB2GRAY)
    bg_region  = base_gray < 15  # pure black background only (inner gray surface ~40-80)
    result_arr[bg_region] = base_image[bg_region]
    pre_refine_arr[bg_region] = base_image[bg_region]

    return {
        "result_image":         encode_b64(result_arr),
        "result_pre_refine":    encode_b64(pre_refine_arr),   # CV only, truoc SDXL
        "engine":               "cv",
        "metadata": {
            "method":       method,
            "defect_type":  defect_type,
            "material":     material,
            "sdxl_refined": sdxl_ran,
        },
    }
