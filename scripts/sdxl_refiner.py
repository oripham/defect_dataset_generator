"""
sdxl_refiner.py — SDXL img2img appearance refinement pass.

Adds photorealistic micro-texture to CV2-generated defect images.
Geometry and defect placement are preserved via low strength (≤ 0.20).

Design:
  - CV2 pipeline handles all geometry (warp, shading, signal injection)
  - SDXL img2img runs at strength ≤ 0.20 → only ~4 denoising steps from the
    final diffusion trajectory, nudging texture/noise without rebuilding structure
  - The resulting image is resized back to the original resolution if SDXL
    output size differs (prevents shape mismatch)

Config (config.yaml):
    sdxl_refine:
      enabled:        true
      strength:       0.14      # keep ≤ 0.20 — above this geometry drifts
      guidance_scale: 5.0       # low guidance → conservative texture edit
      steps:          20        # total scheduler steps (only ~3 actually run)
      model:          stabilityai/stable-diffusion-xl-base-1.0
                      # or a local path, e.g.:
                      # /models/hub/models--stabilityai--stable-diffusion-xl-base-1.0/...
"""

import gc
import os

import numpy as np
import torch
from PIL import Image


# ── Industrial appearance prompt ───────────────────────────────────────────────
# Biased toward real inspection photography characteristics:
#   - uneven studio lighting (not perfect diffuse)
#   - machining marks / concentric grooves
#   - micro noise / grain (camera sensor)
# Negative prompt suppresses SDXL's default "beautifying" tendency.

_PROMPT = (
    "industrial metal surface, machined aluminum disc, "
    "factory inspection photograph, grayscale, "
    "concentric machining marks, metallic luster, natural surface grain, "
    "realistic inspection image, high detail, photorealistic"
)
_NEGATIVE = (
    "smooth, polished, glossy, CGI, 3d render, cartoon, "
    "illustration, oversaturated, text, watermark, blurry, "
    "low quality, airbrushed, painted"
)

_DEFAULT_MODEL = (
    "/models/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1"
    "/snapshots/115134f363124c53c7d878647567d04daf26e41e"
)

_IP_ADAPTER_MODEL = (
    "/models/models--h94--IP-Adapter"
    "/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729"
)


class SDXLRefiner:
    """
    Lazy-loaded SDXL appearance refiner.

    Load once, call refine_with_sdxl() per generated image.
    strength ≤ 0.20 ensures only surface texture is nudged — geometry intact.

    Implementation note:
        Uses StableDiffusionXLInpaintPipeline with an all-white mask, which is
        mathematically equivalent to img2img (no region is masked out, so the
        full image is refined). This allows reuse of the inpainting model
        already cached in the Docker environment without downloading a separate
        base model.
    """

    def __init__(self, refine_cfg: dict, device: str = "cuda"):
        self.strength       = float(refine_cfg.get("strength",       0.14))
        self.guidance_scale = float(refine_cfg.get("guidance_scale", 5.0))
        self.steps          = int(  refine_cfg.get("steps",          20))
        self.device         = device
        self.prompt         = refine_cfg.get("prompt")          or _PROMPT
        self.negative       = refine_cfg.get("negative_prompt") or _NEGATIVE

        model_path = refine_cfg.get("model", _DEFAULT_MODEL)

        dtype_kw = refine_cfg.get("torch_dtype")
        if dtype_kw is not None:
            torch_dtype = dtype_kw
        elif device == "cpu":
            torch_dtype = torch.float32
        elif getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        print(f"[REFINE] Loading SDXL inpaint pipeline from '{model_path}' …")
        from diffusers import StableDiffusionXLInpaintPipeline

        # Cache HuggingFace thường chỉ có *.fp16.safetensors — cần variant="fp16"
        load_kw = {
            "torch_dtype": torch_dtype,
            "use_safetensors": True,
        }
        if os.path.isdir(str(model_path)):
            te_fp16 = os.path.join(model_path, "text_encoder", "model.fp16.safetensors")
            te_std = os.path.join(model_path, "text_encoder", "model.safetensors")
            if os.path.isfile(te_fp16) and not os.path.isfile(te_std):
                load_kw["variant"] = "fp16"
                print("[REFINE] using variant=fp16 (model.fp16.safetensors in cache)")

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_path,
            **load_kw,
        )
        # IP-Adapter chỉ cần cho inpaint_defect khi dùng ip_scale>0.
        # refine_with_sdxl (mask trắng, strength thấp) không cần IP — tắt mặc định qua load_ip_adapter=False
        # để tránh log lỗi đường Docker /models/... khi chạy local.
        load_ip = refine_cfg.get("load_ip_adapter")
        if load_ip is None:
            load_ip = True
        self._ip_loaded = False
        if load_ip:
            ip_model = refine_cfg.get("ip_adapter_model", _IP_ADAPTER_MODEL)
            try:
                self.pipe.load_ip_adapter(
                    ip_model,
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl.bin",
                )
                self.pipe.set_ip_adapter_scale(0.0)
                self._ip_loaded = True
                print(f"[REFINE] IP-Adapter loaded from '{ip_model}'")
            except Exception:
                print(
                    "[REFINE] IP-Adapter không load được (offline hoặc thiếu weight). "
                    "refine_with_sdxl vẫn chạy; inpaint+IP cần tải h94/IP-Adapter hoặc chỉ định ip_adapter_model."
                )
        # load_ip_adapter=False: không gọi load_ip_adapter — không cảnh báo, không cần file IP

        use_offload = bool(refine_cfg.get("enable_model_cpu_offload")) and self.device == "cuda"
        if use_offload:
            self.pipe.enable_model_cpu_offload()
            print("[REFINE] enable_model_cpu_offload (low VRAM / ~6GB)")
        else:
            self.pipe = self.pipe.to(self.device)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[REFINE] xformers enabled")
        except Exception:
            pass

        print(
            f"[REFINE] Ready — strength={self.strength} "
            f"guidance={self.guidance_scale} steps={self.steps}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def inpaint_defect(self, image: Image.Image, mask_np: np.ndarray,
                       inpaint_cfg: dict,
                       ref_image: Image.Image = None) -> Image.Image:
        """
        High-strength inpainting on the defect mask region.

        Unlike refine_with_sdxl (strength≤0.20, all-white mask), this method
        uses the actual defect mask + high strength to let SDXL redraw the
        masked area guided by a defect-specific prompt.

        If ref_image is provided, IP-Adapter is used to condition generation
        on the visual appearance of the NG reference image.

        Args:
            image:       PIL Image RGB — CV2 output (geometry already warped)
            mask_np:     numpy uint8 array (H×W), 255=inpaint, 0=keep
            inpaint_cfg: dict with prompt, negative_prompt, strength, etc.
            ref_image:   PIL Image RGB — NG reference (optional, for IP-Adapter)
        """
        import cv2 as _cv2
        W, H = image.size

        # Dilate mask for better context blending at boundaries
        dilate_px = int(inpaint_cfg.get("mask_dilate", 20))
        kern = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE,
                                          (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask_dilated = _cv2.dilate(mask_np, kern)
        mask_pil = Image.fromarray(mask_dilated)

        prompt   = inpaint_cfg.get("prompt",   _PROMPT)
        negative = inpaint_cfg.get("negative_prompt", _NEGATIVE)
        strength = float(inpaint_cfg.get("strength",       0.80))
        guidance = float(inpaint_cfg.get("guidance_scale", 9.0))
        steps    = int(  inpaint_cfg.get("steps",          30))
        ip_scale = float(inpaint_cfg.get("ip_scale",       0.0))

        use_ip = ref_image is not None and ip_scale > 0.0
        if use_ip:
            self.pipe.set_ip_adapter_scale(ip_scale)
            print(f"[INPAINT] IP-Adapter ON — ip_scale={ip_scale}")
        else:
            self.pipe.set_ip_adapter_scale(0.0)

        print(f"[INPAINT] strength={strength} guidance={guidance} steps={steps}")

        pipe_kwargs = dict(
            prompt              = prompt,
            negative_prompt     = negative,
            image               = image,
            mask_image          = mask_pil,
            strength            = strength,
            num_inference_steps = steps,
            guidance_scale      = guidance,
        )
        if use_ip:
            pipe_kwargs["ip_adapter_image"] = ref_image
        elif hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
            # IP-Adapter loaded but not used — must still pass dummy image (UNet config is permanent)
            pipe_kwargs["ip_adapter_image"] = Image.new("RGB", (224, 224), (0, 0, 0))

        with torch.inference_mode():
            result = self.pipe(**pipe_kwargs).images[0]

        if result.size != (W, H):
            result = result.resize((W, H), Image.LANCZOS)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        return result

    def refine_with_sdxl(self, image: Image.Image) -> Image.Image:
        """
        Run SDXL img2img at low strength over the full image.

        Only surface micro-texture is affected.
        Geometry, defect position, and shape are preserved because:
          - strength=0.14 → ≈ 3 active denoising steps
          - SDXL starts from a heavily-noised version of the input,
            but 3 steps from a good starting latent cannot rebuild large structure
          - guidance_scale=5 (moderate) → avoids over-saturating the prompt

        Args:
            image: PIL.Image RGB — output of CV2 classical pipeline

        Returns:
            PIL.Image RGB — texture-refined image, same resolution as input
        """
        W, H = image.size
        # All-white mask → inpainting model behaves identically to img2img
        white_mask = Image.fromarray(np.ones((H, W), dtype=np.uint8) * 255)

        pipe_kwargs = dict(
            prompt              = self.prompt,
            negative_prompt     = self.negative,
            image               = image,
            mask_image          = white_mask,
            strength            = self.strength,
            num_inference_steps = self.steps,
            guidance_scale      = self.guidance_scale,
        )
        # IP-Adapter changes UNet config permanently → must always pass ip_adapter_image
        # Use a black dummy image with scale=0 so it has zero effect on output
        if hasattr(self.pipe, "image_encoder") and self.pipe.image_encoder is not None:
            self.pipe.set_ip_adapter_scale(0.0)
            pipe_kwargs["ip_adapter_image"] = Image.new("RGB", (224, 224), (0, 0, 0))

        with torch.inference_mode():
            result = self.pipe(**pipe_kwargs).images[0]

        # SDXL may output at its native resolution (e.g. 1024×1024).
        # Always resize back to preserve the exact defect geometry coordinates.
        if result.size != (W, H):
            result = result.resize((W, H), Image.LANCZOS)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        return result
