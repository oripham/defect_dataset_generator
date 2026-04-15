"""
generator_controlnet_depth.py — ControlNet Depth + SDXL Inpainting for Shape Defects
=====================================================================================

Shape branch pipeline (整形変形 dent / MC変形 rim protrusion):

  1. Synthesize depth map from mask:
     - Base depth: smoothed grayscale of good image (rough depth proxy)
     - Perturbation: Gaussian bump at mask location
       · dent  → darken (push surface away from camera)
       · bulge → brighten (pull surface toward camera)
  2. Crop good image, mask, depth around defect bbox
  3. Resize to INPAINT_SIZE (512)
  4. SDXL ControlNetInpaintPipeline
     · ControlNet Depth (scale 0.8) — conditions on 3D geometry
     · NO IP-Adapter (supervisor: weak for localized defects)
     · strength 0.85 — add realistic shading to depth hint
  5. Resize back → Laplacian pyramid blend → output

Key difference vs v10 (ControlNet Canny):
  - Canny only preserves edges — zero info about convex/concave
  - Depth map encodes full 3D height field — dent vs bulge is explicit
"""

import gc
import numpy as np
import torch
import cv2
from PIL import Image

from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
    DDIMScheduler,
)


INPAINT_SIZE = 512
BLEND_LEVELS = 4


# ── Depth map synthesis ────────────────────────────────────────────────────────

def synthesize_depth_map(
    base_image: Image.Image,
    mask: np.ndarray,
    depth_mode: str = "dent",
    depth_amplitude: float = 0.25,
    depth_sigma_factor: float = 1.2,
) -> Image.Image:
    """
    Create a depth conditioning image for ControlNet Depth.

    Base depth: smoothed grayscale of good image.
    Perturbation: Gaussian bump at mask location.
      - dent  → mask area darker in depth (surface depressed / farther)
      - bulge → mask area brighter in depth (surface raised / closer)

    Args:
        base_image: PIL RGB good image
        mask: uint8 (H,W) binary mask, white=defect
        depth_mode: "dent" or "bulge"
        depth_amplitude: perturbation magnitude [0, 1]
        depth_sigma_factor: Gaussian sigma relative to sqrt(mask area)

    Returns:
        PIL RGB depth image (3-channel grayscale)
    """
    gray = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # Base depth: smooth grayscale → removes texture, keeps gross geometry
    depth = cv2.GaussianBlur(gray, (51, 51), 15) / 255.0

    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        d_u8 = (depth * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(np.stack([d_u8] * 3, axis=-1))

    # Gaussian-feathered perturbation centered on mask
    mask_f = (mask > 127).astype(np.float32)
    area = mask_f.sum()
    sigma = max(area ** 0.5 * depth_sigma_factor, 8)
    k = int(sigma * 4) | 1

    perturbation = cv2.GaussianBlur(mask_f, (k, k), sigma)
    pmax = perturbation.max()
    if pmax > 0:
        perturbation /= pmax

    if depth_mode == "dent":
        depth = depth - depth_amplitude * perturbation
    else:  # bulge
        depth = depth + depth_amplitude * perturbation

    depth = np.clip(depth, 0, 1)
    d_u8 = (depth * 255).astype(np.uint8)
    return Image.fromarray(np.stack([d_u8] * 3, axis=-1))


# ── Laplacian blend (same as generator.py v10) ────────────────────────────────

def _laplacian_blend(
    base_arr: np.ndarray,
    result_arr: np.ndarray,
    mask_np: np.ndarray,
    levels: int = BLEND_LEVELS,
) -> np.ndarray:
    def gauss_pyr(img, n):
        g = [img.astype(np.float32)]
        for _ in range(n):
            img = cv2.pyrDown(img)
            g.append(img.astype(np.float32))
        return g

    def lap_pyr(gp):
        lp = []
        for i in range(len(gp) - 1):
            up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1])
        return lp

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(mask_np, kernel).astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(dilated, (21, 21), 0)
    mask_3 = np.stack([mask_f] * 3, axis=-1)

    gp_base = gauss_pyr(base_arr, levels)
    gp_result = gauss_pyr(result_arr, levels)
    gp_mask = gauss_pyr(mask_3, levels)

    lp_base = lap_pyr(gp_base)
    lp_result = lap_pyr(gp_result)

    blended = []
    for lb, lr, gm in zip(lp_base, lp_result, gp_mask):
        blended.append(lb * (1.0 - gm) + lr * gm)

    out = blended[-1]
    for i in range(len(blended) - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(blended[i].shape[1], blended[i].shape[0]))
        out = out + blended[i]

    return np.clip(out, 0, 255).astype(np.uint8)


# ── Generator ──────────────────────────────────────────────────────────────────

class ControlNetDepthGenerator:

    def __init__(self, cfg: dict):
        model_cfg = cfg.get("model", {})
        device = model_cfg.get("device", "cuda")
        self.device = device

        base_model = model_cfg.get(
            "base_model",
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        )
        depth_model = model_cfg.get(
            "controlnet_depth_model",
            "diffusers/controlnet-depth-sdxl-1.0",
        )

        print(f"[INFO] Loading ControlNet Depth SDXL from '{depth_model}' ...")
        controlnet = ControlNetModel.from_pretrained(
            depth_model,
            torch_dtype=torch.float16,
            variant="fp16",
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
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not available, use default attention

        # Switch to DDIM scheduler — EulerDiscreteScheduler has an off-by-one
        # in sigmas array (index out of bounds at last step) in diffusers <0.27.
        # DDIM is stable for inpainting and avoids this bug entirely.
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # No IP-Adapter — supervisor says it's weak for localized defects
        self._has_ip = False
        print("[INFO] ControlNetDepthGenerator ready (no IP-Adapter, DDIM scheduler)")

    def generate(
        self,
        base_image,
        mask: np.ndarray,
        ref_image,          # unused — no IP-Adapter
        gen_cfg: dict,
    ) -> Image.Image:

        base_arr = np.array(base_image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask.convert("L"))
        H, W = mask.shape[:2]

        # ── 1. BBox + adaptive padding ───────────────────────────────────────
        ys, xs = np.where(mask > 127)
        if len(ys) == 0:
            print("[WARN] Empty mask — returning base image unchanged")
            return base_image

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        mh = y_max - y_min + 1
        mw = x_max - x_min + 1

        pad = max(int(max(mh, mw) * 0.5), 40)
        cy_c = (y_min + y_max) // 2
        cx_c = (x_min + x_max) // 2
        half = max(mh, mw) // 2 + pad

        y1 = max(0, cy_c - half);  y2 = min(H, cy_c + half)
        x1 = max(0, cx_c - half);  x2 = min(W, cx_c + half)
        h_c = y2 - y1;             w_c = x2 - x1
        mask_pct = np.sum(mask > 127) / (h_c * w_c) * 100
        print(f"[DEPTH] Crop {w_c}×{h_c}  mask {mask_pct:.1f}%")

        # ── 2. Synthesize depth map ──────────────────────────────────────────
        depth_mode = gen_cfg.get("depth_mode", "dent")
        depth_amplitude = float(gen_cfg.get("depth_amplitude", 0.25))
        depth_sigma = float(gen_cfg.get("depth_sigma_factor", 1.2))

        depth_full = synthesize_depth_map(
            base_image, mask,
            depth_mode=depth_mode,
            depth_amplitude=depth_amplitude,
            depth_sigma_factor=depth_sigma,
        )
        print(f"[DEPTH] Depth map: mode={depth_mode} amp={depth_amplitude}")

        # ── 3. Crop ──────────────────────────────────────────────────────────
        crop_img = Image.fromarray(base_arr[y1:y2, x1:x2])
        crop_mask_np = mask[y1:y2, x1:x2]
        crop_mask = Image.fromarray(crop_mask_np)
        crop_depth = depth_full.crop((x1, y1, x2, y2))

        # ── 4. Resize to INPAINT_SIZE ────────────────────────────────────────
        sz = INPAINT_SIZE
        crop_img_r = crop_img.resize((sz, sz), Image.LANCZOS)
        crop_mask_r = crop_mask.resize((sz, sz), Image.NEAREST)
        crop_depth_r = crop_depth.resize((sz, sz), Image.LANCZOS)

        # ── 5. SDXL ControlNet Depth Inpainting ─────────────────────────────
        prompt = gen_cfg.get("prompt", "a metallic surface with a physical deformation defect, realistic lighting and shading")
        negative_prompt = gen_cfg.get(
            "negative_prompt",
            "blurry, low quality, unrealistic, cartoon, distorted geometry, text, watermark",
        )
        strength = float(gen_cfg.get("strength", 0.85))
        guidance_scale = float(gen_cfg.get("guidance_scale", 7.5))
        steps = int(gen_cfg.get("steps", 30))
        controlnet_scale = float(gen_cfg.get("controlnet_scale", 0.8))

        print(f"[DEPTH] Inpainting: strength={strength} guidance={guidance_scale} "
              f"controlnet_scale={controlnet_scale} steps={steps}")

        with torch.inference_mode():
            result_r = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=crop_img_r,
                mask_image=crop_mask_r,
                control_image=crop_depth_r,
                controlnet_conditioning_scale=controlnet_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                strength=strength,
                width=sz,
                height=sz,
            ).images[0]

        # ── 6. Resize back ───────────────────────────────────────────────────
        result_crop = result_r.resize((w_c, h_c), Image.LANCZOS)

        # ── 7. Laplacian Pyramid Blend ───────────────────────────────────────
        result_arr = np.array(result_crop)
        orig_region = base_arr[y1:y2, x1:x2]
        blended = _laplacian_blend(orig_region, result_arr, crop_mask_np)

        output = base_arr.copy()
        output[y1:y2, x1:x2] = blended

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        return Image.fromarray(output)
