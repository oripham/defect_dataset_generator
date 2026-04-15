"""
generator.py  v10.0 — SDXL + ControlNet Canny + IP-Adapter
============================================================

Pipeline (per image):
  1. BBox + adaptive padding (>= 0.5× mask extent, min 40 px)
  2. Crop base image and mask
  3. Extract Canny edges from crop  →  ControlNet conditioning image
  4. Resize crop + canny + mask → INPAINT_SIZE (512)
  5. SDXL ControlNetInpaintPipeline
       · ControlNet Canny  (scale 0.6) → preserves ring geometry/edges
       · IP-Adapter                     → defect appearance from NG ref
       · strength up to 0.95 (safe because ControlNet holds structure)
  6. Resize result back to crop size
  7. Laplacian Pyramid Blend paste → seamless seam
"""

import gc
import numpy as np
import torch
import cv2
from PIL import Image

from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
)


INPAINT_SIZE  = 512    # square inpaint res; 512 = good balance VRAM / quality
CANNY_LOW     = 80     # Canny lower threshold (metallic surfaces: keep mid-freq edges)
CANNY_HIGH    = 180    # Canny upper threshold
BLEND_LEVELS  = 4      # Laplacian pyramid levels


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_canny(image_pil: Image.Image, low: int, high: int) -> Image.Image:
    """
    Extract Canny edges for ControlNet conditioning.
    Returns RGB PIL image (3-channel edge map, white edges on black).
    """
    gray  = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


def _laplacian_blend(
    base_arr:   np.ndarray,   # uint8 (H, W, 3)  full-image crop original
    result_arr: np.ndarray,   # uint8 (H, W, 3)  inpainted crop
    mask_np:    np.ndarray,   # uint8 (H, W)     binary mask (crop coords)
    levels:     int = BLEND_LEVELS,
) -> np.ndarray:
    """
    Laplacian pyramid blending of result into base using mask.
    Produces seamless seam — eliminates the halo from simple Gaussian blend.
    """
    def gauss_pyr(img, n):
        g = [img.astype(np.float32)]
        for _ in range(n):
            img = cv2.pyrDown(img)
            g.append(img.astype(np.float32))
        return g

    def lap_pyr(gp):
        lp = []
        for i in range(len(gp) - 1):
            up = cv2.pyrUp(gp[i + 1],
                           dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1])
        return lp

    # Smooth mask for pyramid (dilate then blur for feathered edge)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated  = cv2.dilate(mask_np, kernel).astype(np.float32) / 255.0
    mask_f   = cv2.GaussianBlur(dilated, (21, 21), 0)
    mask_3   = np.stack([mask_f] * 3, axis=-1)

    gp_base   = gauss_pyr(base_arr,   levels)
    gp_result = gauss_pyr(result_arr, levels)
    gp_mask   = gauss_pyr(mask_3,     levels)

    lp_base   = lap_pyr(gp_base)
    lp_result = lap_pyr(gp_result)

    blended = []
    for lb, lr, gm in zip(lp_base, lp_result, gp_mask):
        blended.append(lb * (1.0 - gm) + lr * gm)

    # Reconstruct from coarsest to finest
    out = blended[-1]
    for i in range(len(blended) - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(blended[i].shape[1], blended[i].shape[0]))
        out = out + blended[i]

    return np.clip(out, 0, 255).astype(np.uint8)


# ── Generator ──────────────────────────────────────────────────────────────────

class DefectGenerator:

    def __init__(self, cfg: dict):
        model_cfg  = cfg.get("model", {})
        device     = model_cfg.get("device", "cuda")
        self.device = device

        base_model = model_cfg.get(
            "base_model",
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        )
        controlnet_model = model_cfg.get(
            "controlnet_model",
            "diffusers/controlnet-canny-sdxl-1.0",
        )

        print(f"[INFO] Loading ControlNet Canny SDXL from '{controlnet_model}' ...")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir="/models",
        )

        print(f"[INFO] Loading SDXL ControlNetInpaintPipeline from '{base_model}' ...")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(device)
        self.pipe.enable_xformers_memory_efficient_attention()

        # IP-Adapter (optional)
        ip_model     = model_cfg.get("ip_adapter_model",     "h94/IP-Adapter")
        ip_weight    = model_cfg.get("ip_adapter_weight",    "ip-adapter_sdxl.bin")
        ip_subfolder = model_cfg.get("ip_adapter_subfolder", "sdxl_models")
        try:
            self.pipe.load_ip_adapter(
                ip_model, subfolder=ip_subfolder, weight_name=ip_weight
            )
            self._has_ip = True
            print("[INFO] IP-Adapter loaded OK")
        except Exception as e:
            print(f"[WARN] IP-Adapter not loaded ({e}) — proceeding without")
            self._has_ip = False

        print("[INFO] Generator v10.0 — SDXL + ControlNet Canny ready")

    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        base_image,         # PIL.Image RGB, already at image_size
        mask: np.ndarray,   # (H, W) uint8  binary mask
        ref_image,          # PIL.Image RGB  defect reference (IP-Adapter)
        gen_cfg: dict,
    ) -> Image.Image:

        base_arr = np.array(base_image)
        H, W     = mask.shape[:2]

        # ── 1. Bounding box + adaptive padding ──────────────────────────────
        ys, xs = np.where(mask > 127)
        if len(ys) == 0:
            print("[WARN] Empty mask — returning base image unchanged")
            return base_image

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        mh = y_max - y_min + 1
        mw = x_max - x_min + 1

        pad  = max(int(max(mh, mw) * 0.5), 40)
        cy_c = (y_min + y_max) // 2
        cx_c = (x_min + x_max) // 2
        half = max(mh, mw) // 2 + pad

        y1 = max(0, cy_c - half);  y2 = min(H, cy_c + half)
        x1 = max(0, cx_c - half);  x2 = min(W, cx_c + half)
        h_c = y2 - y1;             w_c = x2 - x1
        mask_pct = np.sum(mask > 127) / (h_c * w_c) * 100
        print(f"[INFO] Crop {w_c}×{h_c}  mask {mask_pct:.1f}%")

        # ── 2. Crop ──────────────────────────────────────────────────────────
        crop_img     = Image.fromarray(base_arr[y1:y2, x1:x2])
        crop_mask_np = mask[y1:y2, x1:x2]
        crop_mask    = Image.fromarray(crop_mask_np)

        # ── 3. Canny edges → ControlNet conditioning image ───────────────────
        canny_low  = int(gen_cfg.get("canny_low",  CANNY_LOW))
        canny_high = int(gen_cfg.get("canny_high", CANNY_HIGH))
        control_image = _extract_canny(crop_img, canny_low, canny_high)
        print(f"[INFO] Canny edges extracted (low={canny_low}, high={canny_high})")

        # ── 4. Resize to INPAINT_SIZE ────────────────────────────────────────
        sz = INPAINT_SIZE
        crop_img_r    = crop_img.resize((sz, sz), Image.LANCZOS)
        crop_mask_r   = crop_mask.resize((sz, sz), Image.NEAREST)
        control_img_r = control_image.resize((sz, sz), Image.NEAREST)

        # ── 5. SDXL ControlNet Inpainting ────────────────────────────────────
        if self._has_ip:
            self.pipe.set_ip_adapter_scale(float(gen_cfg.get("ip_scale", 0.6)))

        prompt          = gen_cfg.get("prompt", "surface defect on metallic part, realistic")
        negative_prompt = gen_cfg.get(
            "negative_prompt",
            "blurry, low quality, unrealistic, cartoon, distorted geometry",
        )
        strength               = float(gen_cfg.get("strength",               0.90))
        guidance_scale         = float(gen_cfg.get("guidance_scale",         7.5))
        steps                  = int(  gen_cfg.get("steps",                  30))
        controlnet_scale       = float(gen_cfg.get("controlnet_scale",       0.6))

        pipe_kwargs = dict(
            prompt                       = prompt,
            negative_prompt              = negative_prompt,
            image                        = crop_img_r,
            mask_image                   = crop_mask_r,
            control_image                = control_img_r,
            controlnet_conditioning_scale= controlnet_scale,
            guidance_scale               = guidance_scale,
            num_inference_steps          = steps,
            strength                     = strength,
            width                        = sz,
            height                       = sz,
        )
        if self._has_ip and ref_image is not None:
            pipe_kwargs["ip_adapter_image"] = ref_image

        print(f"[INFO] Inpainting: strength={strength} guidance={guidance_scale} "
              f"controlnet_scale={controlnet_scale} steps={steps}")

        with torch.inference_mode():
            result_r = self.pipe(**pipe_kwargs).images[0]

        # ── 6. Resize result back to crop size ───────────────────────────────
        result_crop = result_r.resize((w_c, h_c), Image.LANCZOS)

        # ── 7. Laplacian Pyramid Blend paste ─────────────────────────────────
        result_arr  = np.array(result_crop)
        orig_region = base_arr[y1:y2, x1:x2]

        blended = _laplacian_blend(orig_region, result_arr, crop_mask_np)

        output = base_arr.copy()
        output[y1:y2, x1:x2] = blended

        # Free GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        return Image.fromarray(output)
