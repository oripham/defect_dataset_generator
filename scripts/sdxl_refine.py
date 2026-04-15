#!/usr/bin/env python3
"""
SDXL Refine — post-processing pass for classical-generated defect images.

Pipeline:
  classical_gen → defect_image
      ↓
  SDXL img2img (full image, no mask, strength=0.15)
      ↓
  Option C: blend defect region back from classical image (preserve geometry)
      ↓
  refined_image

Usage:
  python sdxl_refine.py \\
    --run_dir  /workspace/jobs/c730ee1d/output/run_20260319_055355 \\
    --mask_root /workspace/jobs/c730ee1d/mask_root \\
    --strength  0.15 \\
    --blend_alpha 0.6   # 0=full SDXL, 1=full classical; 0.6 = keep 60% classical geometry
"""
import argparse, os, glob, sys, random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch


# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--run_dir",     required=True, help="Run output directory")
parser.add_argument("--mask_root",   required=True, help="mask_root of the job")
parser.add_argument("--strength",    type=float, default=0.15,
                    help="SDXL img2img strength (0.10–0.20 recommended)")
parser.add_argument("--blend_alpha", type=float, default=0.60,
                    help="In defect region: alpha * classical + (1-alpha) * SDXL. 0.6 preserves geometry.")
parser.add_argument("--steps",       type=int, default=30)
parser.add_argument("--seed",        type=int, default=-1)
parser.add_argument("--out_suffix",  default="_refined",
                    help="Suffix added to output filenames (e.g. 000003_refined.jpg)")
args = parser.parse_args()

run_dir   = Path(args.run_dir)
img_dir   = run_dir / "images"
out_dir   = run_dir / "images_refined"
out_dir.mkdir(exist_ok=True)

# ── prompts (industrial imperfection vocabulary) ──────────────────────────────
PROMPT = (
    "industrial metal surface, machined metal disc, grayscale, "
    "factory inspection photograph, uneven lighting, subtle surface roughness, "
    "micro scratches, slight noise, realistic manufacturing defects, "
    "non-uniform texture, metallic sheen, machining marks"
)
NEGATIVE = (
    "perfect, clean, polished, CGI, 3d render, smooth plastic, "
    "cartoon, illustration, oversaturated, text, watermark, blurry"
)

# ── load SDXL inpaint model (used as img2img with all-white mask) ─────────────
MODEL_SNAPSHOT = (
    "/models/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1"
    "/snapshots/115134f363124c53c7d878647567d04daf26e41e"
)
if not Path(MODEL_SNAPSHOT).exists():
    sys.exit(f"[ERROR] Model not found at {MODEL_SNAPSHOT}")

print("[SDXL] Loading pipeline…")
from diffusers import StableDiffusionXLInpaintPipeline

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    MODEL_SNAPSHOT,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.enable_model_cpu_offload()
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("[SDXL] xformers enabled")
except Exception:
    pass

# ── helper: find defect mask for an image index ───────────────────────────────
def find_mask(image_idx: int, mask_root: Path) -> np.ndarray | None:
    """Return combined binary mask (H,W uint8) for image_idx, or None."""
    combined = None
    for class_dir in sorted(mask_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for subfolder in class_dir.iterdir():
            for mf in sorted(subfolder.glob("mask_*.png")):
                m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if combined is None:
                    combined = m.astype(np.float32)
                else:
                    combined = np.maximum(combined, m.astype(np.float32))
                break  # one mask file per subfolder per class
    if combined is None:
        return None
    # soft feather for blend
    kernel = np.ones((31, 31), np.uint8)
    dilated = cv2.dilate((combined > 0).astype(np.uint8) * 255, kernel, iterations=2)
    soft = cv2.GaussianBlur(dilated.astype(np.float32), (61, 61), 20)
    soft = soft / (soft.max() + 1e-8)
    return soft

mask_root = Path(args.mask_root)

# ── process each generated image ─────────────────────────────────────────────
images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
print(f"[SDXL] Found {len(images)} images in {img_dir}")

for img_path in images:
    idx = int(img_path.stem)
    print(f"\n[SDXL] Processing {img_path.name}…")

    # Load classical-generated image (geometry correct)
    classical_pil = Image.open(img_path).convert("RGB")
    W, H = classical_pil.size
    classical_arr = np.array(classical_pil, dtype=np.float32)

    # For full-image img2img: use all-white mask (refine entire image)
    white_mask = Image.fromarray(np.ones((H, W), dtype=np.uint8) * 255)

    # Run SDXL img2img (inpaint with all-white mask ≡ img2img)
    gen = None
    if args.seed >= 0:
        gen = torch.Generator("cuda").manual_seed(args.seed + idx)
    else:
        gen = torch.Generator("cuda").manual_seed(random.randint(0, 2**31))

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=classical_pil,
        mask_image=white_mask,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=5.0,       # moderate guidance for texture realism
        generator=gen,
    ).images[0]

    # SDXL outputs at its native resolution — resize back to original image size
    if result.size != (W, H):
        result = result.resize((W, H), Image.LANCZOS)
    sdxl_arr = np.array(result, dtype=np.float32)

    # ── Option C: blend defect region back to preserve geometry ──────────────
    soft_mask = find_mask(idx, mask_root)   # (H,W) float in [0,1], None if not found
    if soft_mask is not None and soft_mask.shape == (H, W):
        sm = soft_mask[:, :, np.newaxis]  # (H,W,1)
        # In defect region: keep blend_alpha of classical (geometry) + (1-alpha) of SDXL (texture)
        # Outside defect region: use SDXL result
        blend_alpha = args.blend_alpha
        final_arr = (
            sm       * (blend_alpha * classical_arr + (1 - blend_alpha) * sdxl_arr) +
            (1 - sm) * sdxl_arr
        )
        print(f"  [BLEND] Defect region: {blend_alpha:.0%} classical + {1-blend_alpha:.0%} SDXL")
    else:
        final_arr = sdxl_arr
        print("  [BLEND] No mask found — using full SDXL output")

    # Save
    out_path = out_dir / img_path.name
    final_u8 = np.clip(final_arr, 0, 255).astype(np.uint8)
    Image.fromarray(final_u8).save(str(out_path), quality=95)
    print(f"  → Saved: {out_path}")

print(f"\n✅ SDXL refine complete. Output: {out_dir}")
