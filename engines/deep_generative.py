"""
engines/deep_generative.py — GenAI Defect Engine (Oanh)
========================================================

Wraps the two existing SDXL pipelines under one unified interface:

  Appearance defects (scratch / crack / chip / rust / burn / foreign)
      → generator_poisson_depth.DefectGenerator
        (Poisson seamless clone + IP-Adapter + ControlNet Depth + SDXL)

  Shape defects (dent / bulge)
      → generator_controlnet_depth.ControlNetDepthGenerator
        (Synthesized depth map + ControlNet Depth + SDXL, no IP-Adapter)

Router calls: generate(base_image, mask, defect_type, material, params) -> dict
See engines/interface_spec.md for full parameter documentation.

params → pipeline mapping
--------------------------
  intensity        [0-1]  →  strength / ip_scale / depth_amplitude
  naturalness      [0-1]  →  guidance_scale (lower = more natural blending)
  ref_image_b64    str    →  IP-Adapter ref image (appearance path only)
  seed             int    →  torch.manual_seed
"""

import os
import sys
import threading
import numpy as np
from PIL import Image

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCRIPTS))

from .utils import encode_b64, decode_b64
from .structure_adapt import structure_adapt as _structure_adapt


import cv2 as _cv2

def _polygonize_mask(mask: np.ndarray, epsilon_factor: float = 0.03) -> np.ndarray:
    """Convert smooth mask blob to polygon approximation for angular chip edges."""
    _, binary = _cv2.threshold(mask, 127, 255, _cv2.THRESH_BINARY)
    contours, _ = _cv2.findContours(binary, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    poly_mask = np.zeros_like(mask)
    for cnt in contours:
        if _cv2.contourArea(cnt) < 50:
            continue
        epsilon = epsilon_factor * _cv2.arcLength(cnt, True)
        approx = _cv2.approxPolyDP(cnt, epsilon, True)
        _cv2.fillPoly(poly_mask, [approx], 255)
    return poly_mask



# ── Compatibility patch ───────────────────────────────────────────────────────
# diffusers <0.26 does not include IPAdapterMixin in
# StableDiffusionXLControlNetInpaintPipeline, so load_ip_adapter() is missing.
# We dynamically inject the mixin so generator_poisson_depth.py works as-is.

try:
    from diffusers import StableDiffusionXLControlNetInpaintPipeline as _CtrlPipe
    from diffusers.loaders import IPAdapterMixin as _IPA
    if not hasattr(_CtrlPipe, "load_ip_adapter"):
        _CtrlPipe.__bases__ = (_IPA,) + _CtrlPipe.__bases__
        print("[deep_generative] IPAdapterMixin injected into "
              "StableDiffusionXLControlNetInpaintPipeline (diffusers compat patch)")
except Exception as _patch_err:
    print(f"[deep_generative] IPAdapterMixin patch skipped: {_patch_err}")


# ── Routing ──────────────────────────────────────────────────────────────────

_SHAPE_TYPES      = {"dent", "bulge"}           # → ControlNetDepthGenerator
_APPEARANCE_TYPES = {"scratch", "crack", "chip",
                     "rust", "burn", "micro_crack", "foreign",
                     "plastic_flow"}  # → DefectGenerator


# ── Lazy model cache (load once, reuse) ──────────────────────────────────────
# _GEN_LOCK: serializes all GenAI requests — GPU VRAM cannot run 2 inferences
# simultaneously (19.7 GB), and model swap must not be interrupted mid-inference.

_GEN_LOCK       = threading.Lock()
_appearance_gen = None
_shape_gen      = None

# ── Generation progress (read by /api/generate/progress endpoint) ─────────────
_gen_progress: dict = {
    "status":      "idle",   # "idle" | "queued" | "generating"
    "queued":      0,        # number of requests waiting for lock
    "defect_type": "",
    "step":        0,
    "total_steps": 0,
}
def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

_MODEL_CFG      = {
    "model": {
        "device":       _auto_device(),
        "base_model":   "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "enable_cpu_offload": True,   # RTX 4050 6GB: offload layers CPU↔GPU to avoid VRAM overflow
    }
}


def _free_gpu():
    """Release VRAM from unused generator before loading another."""
    try:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _get_appearance_gen():
    global _appearance_gen, _shape_gen
    if _appearance_gen is None:
        # Unload shape gen to free VRAM
        if _shape_gen is not None:
            del _shape_gen
            _shape_gen = None
            _free_gpu()
        from generator_poisson_depth import DefectGenerator
        _appearance_gen = DefectGenerator(_MODEL_CFG)
    return _appearance_gen


def _get_shape_gen():
    global _shape_gen, _appearance_gen
    if _shape_gen is None:
        # Unload appearance gen to free VRAM
        if _appearance_gen is not None:
            del _appearance_gen
            _appearance_gen = None
            _free_gpu()
        from generator_controlnet_depth import ControlNetDepthGenerator
        _shape_gen = ControlNetDepthGenerator(_MODEL_CFG)
    return _shape_gen


# ── Material → hard-coded prompt presets ─────────────────────────────────────

_PROMPTS = {
    "metal": (
        "industrial metal surface defect, realistic {defect}, "
        "professional quality inspection photograph, sharp focus, metallic surface",
        "cartoon, painting, blurry, low quality, plastic, text",
    ),
    "plastic": (
        "plastic component surface defect, realistic {defect}, "
        "quality control photography, sharp focus, matte plastic surface",
        "cartoon, painting, blurry, low quality, metal, shiny, text",
    ),
    "pharma": (
        "pharmaceutical tablet surface defect, realistic {defect}, "
        "quality inspection close-up, sharp focus, uniform background",
        "cartoon, painting, blurry, low quality, text",
    ),
}

_DEFECT_WORDS = {
    "scratch":    "surface scratch mark",
    "crack":      "surface crack line",
    "dent":       "physical dent depression with shadow shading, surface deformation, concentric ring distortion, directional light shadow",
    "bulge":      "surface bulge protrusion",
    "chip":       "chipped edge defect",
    "rust":       "rust corrosion spot",
    "burn":       "burn mark discoloration",
    "micro_crack": "spider-web micro crack pattern",
    "foreign":    "foreign particle contamination",
    "plastic_flow": "molten plastic flow smear, glossy streak, flow mark haze",
}


def _build_prompt(defect_type: str, material: str,
                  prompts: list = None, negative_prompt: str = None,
                  clock_str: str = None) -> tuple:
    """Build prompt. If prompts list provided from frontend, use it; else hardcode."""
    if prompts:
        pos = ", ".join(p for p in prompts if p)
    else:
        template_pos, _ = _PROMPTS.get(material, _PROMPTS["metal"])
        word = _DEFECT_WORDS.get(defect_type, defect_type.replace("_", " "))
        pos = template_pos.format(defect=word)
    if clock_str:
        pos = pos + f", at {clock_str} position on rim"
    _, default_neg = _PROMPTS.get(material, _PROMPTS["metal"])
    neg = negative_prompt if negative_prompt else default_neg
    return pos, neg


# ── Params → gen_cfg per pipeline ────────────────────────────────────────────

def _cfg_appearance(params: dict, prompt: str, negative: str) -> dict:
    intensity   = float(params.get("intensity",   0.6))
    naturalness = float(params.get("naturalness", 0.6))
    seed        = params.get("seed")
    
    # Calculate formulaic defaults
    def_strength = round(0.10 + intensity * 0.50, 3)    # 0.10 – 0.60
    def_guidance = round(10.0 - naturalness * 5.0, 1)  # 10.0 – 5.0
    def_ip       = round(0.6 + intensity * 0.4,  2)    # 0.60 – 1.00
    def_cn       = round(0.25 + intensity * 0.2, 2)    # 0.25 – 0.45
    def_alpha    = round(0.85 + intensity * 0.10, 2)   # 0.85 – 0.95

    # Manual overrides from Advanced Parameters (if provided in params)
    return {
        "prompt":            prompt,
        "negative_prompt":   negative,
        "strength":          float(params.get("strength",         def_strength)),
        "guidance_scale":    float(params.get("guidance_scale",   def_guidance)),
        "ip_scale":          float(params.get("ip_scale",         def_ip)),
        "controlnet_scale":  float(params.get("controlnet_scale",  def_cn)),
        "inject_alpha":      float(params.get("inject_alpha",      def_alpha)),
        "steps":             int(params.get("steps", 15)),
        "seed":              seed,
    }



def _cfg_shape(params: dict, defect_type: str, prompt: str, negative: str) -> dict:
    intensity   = float(params.get("intensity",   0.6))
    naturalness = float(params.get("naturalness", 0.6))
    seed        = params.get("seed")
    depth_mode  = "bulge" if defect_type == "bulge" else "dent"
    
    # Calculate formulaic defaults
    def_strength = round(0.80 + intensity * 0.15, 3)    # 0.80 – 0.95
    def_guidance = round(12.0 - naturalness * 4.5, 1)  # 12.0 – 7.5
    def_cn       = round(0.75 + intensity * 0.20, 2)   # 0.75 – 0.95

    return {
        "prompt":            prompt,
        "negative_prompt":   negative,
        "depth_mode":        depth_mode,
        "depth_amplitude":   round(0.30 + intensity * 0.35, 3),  # 0.30 – 0.65
        "depth_sigma_factor": 1.2,
        "controlnet_scale":  float(params.get("controlnet_scale",  def_cn)),
        "strength":          float(params.get("strength",          def_strength)),
        "guidance_scale":    float(params.get("guidance_scale",    def_guidance)),
        "steps":             int(params.get("steps", 20)),
        "seed":              seed,
    }



# ── Main entry point ──────────────────────────────────────────────────────────

def generate(
    base_image:  np.ndarray,
    mask:        np.ndarray,
    defect_type: str,
    material:    str,
    params:      dict,
) -> dict:
    """
    Generate one defect image using Poisson + ControlNet Depth + SDXL.

    Parameters
    ----------
    base_image  : uint8 RGB  (H, W, 3)
    mask        : uint8 gray (H, W)   — white = defect region
    defect_type : str  — "scratch" | "dent" | "chip" | "rust" | "burn" | ...
    material    : "metal" | "plastic" | "pharma"
    params      : intensity, naturalness, position_jitter, seed, ref_image_b64

    Returns
    -------
    dict: { result_image: base64 PNG, engine: "genai", metadata: {...} }
    """
    # Polygonize mask for chip, dent, bulge — angular polygon edges look more realistic on industrial parts
    if defect_type in ["chip", "dent", "bulge"]:
        eps_factor = params.get("epsilon_factor", 0.03)
        mask = _polygonize_mask(mask, epsilon_factor=eps_factor)

    # Position Jitter (Random Rotation)
    jitter = float(params.get("position_jitter", 0.0))
    if jitter > 0:
        import random
        angle = random.uniform(-jitter * 180.0, jitter * 180.0)
        h, w = mask.shape
        M = _cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        mask = _cv2.warpAffine(mask, M, (w, h), flags=_cv2.INTER_NEAREST, borderMode=_cv2.BORDER_CONSTANT, borderValue=0)

    # Structure-aware placement: snap mask to ring + clock position for prompt
    mask, _, _clock_str = _structure_adapt(base_image, mask, defect_type)

    prompt, negative = _build_prompt(
        defect_type, material,
        prompts=params.get("prompts"),
        negative_prompt=params.get("negative_prompt"),
        clock_str=_clock_str,
    )



    base_pil = Image.fromarray(base_image)
    mask_pil = Image.fromarray(mask)

    # Resize mask to match base image if canvas size differs (e.g. 800×600 canvas vs 1600×1200 image)
    if mask_pil.size != base_pil.size:
        mask_pil = mask_pil.resize(base_pil.size, Image.NEAREST)

    # Decode IP-Adapter reference image (appearance path only)
    ref_pil = None
    ref_b64 = params.get("ref_image_b64")
    if ref_b64:
        try:
            ref_pil = Image.fromarray(decode_b64(ref_b64))
        except Exception as e:
            print(f"[WARN] deep_generative: cannot decode ref_image_b64: {e}")

    # ── Route to correct pipeline (serialized — 1 inference at a time) ─────────
    global _gen_progress
    _gen_progress["queued"] += 1
    _gen_progress["status"] = "queued"
    print(f"[GenAI] Waiting for GPU lock (defect={defect_type}, queued={_gen_progress['queued']})", flush=True)
    with _GEN_LOCK:
        _gen_progress["queued"]      -= 1
        _gen_progress["status"]       = "generating"
        _gen_progress["defect_type"]  = defect_type
        _gen_progress["step"]         = 0
        _gen_progress["total_steps"]  = 0
        print(f"[GenAI] GPU lock acquired (defect={defect_type})", flush=True)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        if defect_type in _SHAPE_TYPES:
            gen_cfg     = _cfg_shape(params, defect_type, prompt, negative)
            _gen_progress["total_steps"] = gen_cfg.get("steps", 20)
            generator   = _get_shape_gen()
            result_pil  = generator.generate(base_pil, mask_pil, ref_pil, gen_cfg)
            pipeline_id = "controlnet_depth"

        else:
            gen_cfg     = _cfg_appearance(params, prompt, negative)
            _gen_progress["total_steps"] = gen_cfg.get("steps", 15)
            generator   = _get_appearance_gen()
            result_pil  = generator.generate(base_pil, mask_pil, ref_pil, gen_cfg)
            pipeline_id = "poisson_ipadapter"

        result_arr = np.array(result_pil.convert("RGB"))

    _gen_progress["status"] = "idle"
    _gen_progress["step"]   = 0
    print(f"[GenAI] GPU lock released (defect={defect_type})", flush=True)

    return {
        "result_image": encode_b64(result_arr),
        "engine":       "genai",
        "metadata": {
            "defect_type":  defect_type,
            "material":     material,
            "pipeline":     pipeline_id,
            "prompt":       prompt,
            "strength":     gen_cfg.get("strength"),
            "guidance":     gen_cfg.get("guidance_scale"),
        },
    }
