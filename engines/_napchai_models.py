"""
engines/_napchai_models.py — Shared lazy model loader for Napchai Metal Cap engines
=====================================================================================
All 3 Napchai pipelines (mc_deform, ring_fracture, scratch) use identical models:
  - SDXL Inpainting: diffusers/stable-diffusion-xl-1.0-inpainting-0.1
  - ControlNet Depth: diffusers/controlnet-depth-sdxl-1.0
  - IP-Adapter Plus:  h94/IP-Adapter / sdxl_models / ip-adapter-plus_sdxl_vit-h.safetensors
  - Depth Estimator:  transformers depth-estimation

Loaded once, shared across all 3 engines via module-level globals.
GPU lock shared with deep_generative._GEN_LOCK to prevent concurrent inference.
"""
from __future__ import annotations

import threading

_pipe      = None   # StableDiffusionXLControlNetInpaintPipeline
_depth_est = None   # transformers depth-estimation pipeline
_lock      = None   # threading.Lock — shared with deep_generative when possible


def get_lock() -> threading.Lock:
    global _lock
    if _lock is None:
        try:
            from .deep_generative import _GEN_LOCK
            _lock = _GEN_LOCK
        except Exception:
            _lock = threading.Lock()
    return _lock


def load_models():
    """
    Lazy-load SDXL + ControlNet Depth + IP-Adapter Plus.
    Safe to call multiple times — only loads once.
    Mirrors model loading in all 3 notebook cell-2 / cell-6.
    """
    global _pipe, _depth_est
    if _pipe is not None:
        return

    import torch
    from transformers import CLIPVisionModelWithProjection
    from transformers import pipeline as _tfpipe
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    # IPAdapterMixin compat patch (diffusers < 0.26)
    try:
        from diffusers.loaders import IPAdapterMixin as _IPA
        if not hasattr(StableDiffusionXLControlNetInpaintPipeline, "load_ip_adapter"):
            StableDiffusionXLControlNetInpaintPipeline.__bases__ = (
                (_IPA,) + StableDiffusionXLControlNetInpaintPipeline.__bases__)
            print("[napchai_models] IPAdapterMixin injected")
    except Exception:
        pass

    print("[napchai_models] Loading ControlNet Depth...")
    depth_model = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=dtype
    ).to(device)

    print("[napchai_models] Loading CLIP image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=dtype
    ).to(device)

    print("[napchai_models] Loading SDXL Inpainting pipeline...")
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet=depth_model,
        image_encoder=image_encoder,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    )

    print("[napchai_models] Loading IP-Adapter Plus weights...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        image_encoder_folder=None,
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[napchai_models] xformers enabled")
    except Exception:
        pass

    pipe = pipe.to(device)
    pipe.enable_vae_slicing()

    print("[napchai_models] Loading depth estimator...")
    _depth_est = _tfpipe(
        "depth-estimation",
        device=0 if device == "cuda" else -1,
    )

    _pipe = pipe
    print("[napchai_models] ALL MODELS LOADED")


def get_pipe():
    load_models()
    return _pipe


def get_depth_est():
    load_models()
    return _depth_est
