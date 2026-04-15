# scripts/generator.py
import importlib.util
import math
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
)
from transformers import CLIPVisionModelWithProjection
from transformers import pipeline as transformers_pipeline


# ──────────────────────────────────────────────────────────
# Pipeline Analytics 
# ──────────────────────────────────────────────────────────

def estimate_bg_orientation(bg_img, mask_img):
    mask_np = np.array(mask_img)
    # Distance Transform to find the thickest center
    dist = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    cx, cy = max_loc
    
    gray = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2GRAY)
    # Focal Tensor (60px radius)
    r = 60
    y1, x1 = max(0, cy - r), max(0, cx - r)
    y2, x2 = min(gray.shape[0], cy + r), min(gray.shape[1], cx + r)
    
    gx, gy = cv2.Sobel(gray[y1:y2, x1:x2], cv2.CV_32F, 1, 0, ksize=5), cv2.Sobel(gray[y1:y2, x1:x2], cv2.CV_32F, 0, 1, ksize=5)
    local_mask = mask_np[y1:y2, x1:x2]
    M_xx = np.sum((gx**2)[local_mask>50])
    M_yy = np.sum((gy**2)[local_mask>50])
    M_xy = np.sum((gx*gy)[local_mask>50])
    
    angle = math.degrees(0.5 * math.atan2(2 * M_xy, M_xx - M_yy)) + 90.0
    return angle, (cx, cy)

def estimate_defect_orientation(crop_img):
    gray = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sig = cv2.add(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k))
    _, thresh = cv2.threshold(sig, 40, 255, cv2.THRESH_BINARY)  
    m = cv2.moments(thresh)
    cx, cy = crop_img.width // 2, crop_img.height // 2
    if m['m00'] > 0:
        angle = math.degrees(0.5 * math.atan2(2 * m['mu11'], m['mu20'] - m['mu02']))
        return angle, (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    return 0.0, (cx, cy)

def align_and_inject_alpha(bg_img, crop_img, mask_img, opt_rot, src_center, tgt_center, alpha_paste=0.95):
    mask_np = np.array(mask_img)
    y, x = np.where(mask_np > 50)
    if len(y) == 0:
        return np.array(bg_img), (0,0,bg_img.width,bg_img.height)

    pad = 40
    x1, y1 = max(0, x.min() - pad), max(0, y.min() - pad)
    x2, y2 = min(mask_np.shape[1], x.max() + pad), min(mask_np.shape[0], y.max() + pad)
    bw, bh = x2 - x1, y2 - y1
    
    tgt_cx, tgt_cy = tgt_center[0] - x1, tgt_center[1] - y1
    M = cv2.getRotationMatrix2D((float(src_center[0]), float(src_center[1])), float(-opt_rot), 1.0)
    M[0, 2] += (tgt_cx - src_center[0])
    M[1, 2] += (tgt_cy - src_center[1])
    
    crop_aligned = cv2.warpAffine(np.array(crop_img).astype(np.float32), M, (bw, bh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    
    bg_np = np.array(bg_img)
    bg_c = bg_np[y1:y1+bh, x1:x1+bw].astype(np.float32)
    mask_crop = mask_np[y1:y1+bh, x1:x1+bw]
    
    # 1. Cân bằng ánh sáng (để Poisson lấy đúng tone nền ban đầu)
    mean_bg = np.mean(bg_c)
    mean_crop = np.mean(crop_aligned)
    ratio = (mean_bg + 5.0) / (mean_crop + 1e-5)
    crop_matched = np.clip(crop_aligned * ratio, 0, 255).astype(np.uint8)
    
    # 2. OpenCV Poisson Blending (Seamless Clone)
    obj_mask = np.where(mask_crop > 50, 255, 0).astype(np.uint8)
    center = (x1 + bw // 2, y1 + bh // 2)
    
    try:
        # cv2.NORMAL_CLONE giữ kết cấu của crop nhưng đồng nhất sáng/tối với vùng nền
        bg_np = cv2.seamlessClone(crop_matched, bg_np, obj_mask, center, cv2.NORMAL_CLONE)
    except Exception as e:
        print("[WARN] Poisson blend failed:", e)
        # Fallback về cách trộn cũ (Alpha Blend) nếu lỗi
        mask_blur = gaussian_filter(mask_crop.astype(np.float32) / 255.0, sigma=4.0)[:, :, None]
        bg_np_f = bg_np.astype(np.float32)
        bg_np_f[y1:y1+bh, x1:x1+bw] = bg_c * (1.0 - mask_blur * alpha_paste) + crop_matched.astype(np.float32) * (mask_blur * alpha_paste)
        bg_np = np.clip(bg_np_f, 0, 255).astype(np.uint8)
        
    return bg_np, (x1, y1, bw, bh)


def apply_geometry_dent(inj_np, alpha_mask, bb, warp_strength=18.0):
    res_np = inj_np.copy().astype(np.float32)
    x1, y1, bw, bh = bb
    h = gaussian_filter(alpha_mask, sigma=1.5)
    h_norm = h / np.max(h) if np.max(h) > 0 else h
    
    gy, gx = np.gradient(h_norm * 25.0) 
    YY, XX = np.mgrid[0:bh, 0:bw]
    map_x, map_y = (XX + gx * warp_strength).astype(np.float32), (YY + gy * warp_strength).astype(np.float32)
    warped_crop = cv2.remap(res_np[y1:y1+bh, x1:x1+bw], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    normals = np.dstack((-gx, -gy, np.ones_like(h_norm)))
    normals /= (np.linalg.norm(normals, axis=2, keepdims=True) + 1e-5)
    L = np.array([-1.0, -1.0, 1.5])
    L /= np.linalg.norm(L)
    diffuse = np.clip(np.sum(normals * L, axis=2), 0.3, 1.0)
    H_vec = L + np.array([0, 0, 1.0])
    H_vec /= np.linalg.norm(H_vec)
    specular = np.clip(np.sum(normals * H_vec, axis=2), 0, 1) ** 30
    
    flat_shd = (np.sum(np.array([0,0,1])*L) + np.sum(np.array([0,0,1])*H_vec)**30 * 0.4)
    shading = (diffuse + specular * 0.5) / (flat_shd + 1e-5)
    shaded_crop = np.clip(warped_crop * shading[:, :, None], 0, 255)
    
    # Giảm sigma từ 3.0 xuống 1.0 để giữ độ gắt của viền lỗi
    blend_fade = gaussian_filter(alpha_mask, sigma=1.0)[:, :, None]
    res_np[y1:y1+bh, x1:x1+bw] = res_np[y1:y1+bh, x1:x1+bw] * (1 - blend_fade) + shaded_crop * blend_fade
    return Image.fromarray(np.clip(res_np, 0, 255).astype(np.uint8)).convert("RGB")


# ──────────────────────────────────────────────────────────
# Main generator class
# ──────────────────────────────────────────────────────────

class DefectGenerator:
    """SDXL ControlNet-Depth + IP-Adapter pipeline."""

    def __init__(self, cfg):
        device = cfg["model"]["device"]
        use_fp16 = str(device).startswith("cuda")
        dtype = torch.bfloat16 if use_fp16 else torch.float32

        print("[INFO] Loading ViT-H image encoder for IP-Adapter Plus...")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=dtype,
        ).to(device)

        print("[INFO] Loading ControlNet Depth...")
        depth_model = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=dtype,
        ).to(device)

        print("[INFO] Loading depth estimator...")
        self.depth_estimator = transformers_pipeline("depth-estimation")

        print("[INFO] Loading SDXL ControlNet Inpaint pipeline...")
        load_kwargs = {"torch_dtype": dtype}
        if False:  # bfloat16 has no fp16 variant files
            load_kwargs["variant"] = "fp16"

        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            cfg["model"]["base_model"],
            controlnet=depth_model,
            image_encoder=image_encoder,
            **load_kwargs,
        ).to(device)

        if importlib.util.find_spec("xformers") is not None:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[INFO] xformers attention enabled")
            except Exception as e:
                print(f"[WARN] xformers unavailable: {e}")

        if False:  # disabled: conflicts with .to(device) causing device mismatch
            self.pipe.enable_model_cpu_offload()  # noqa

        print("[INFO] Loading IP-Adapter Plus...")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.bin",
        )

        print("[INFO] Pipeline ready ✓")

    def get_depth(self, image: Image.Image) -> Image.Image:
        depth = self.depth_estimator(image)["depth"]
        return depth.convert("RGB").resize(image.size)
        
    def force_rgb(self, img_input):
        if not isinstance(img_input, Image.Image):
            img_input = Image.fromarray(np.array(img_input).astype(np.uint8))
        return img_input.convert("RGB")

    def generate(self, base_image, mask, ref_image, gen_cfg):
        """
        Parameters
        ----------
        base_image : PIL.Image   – good product image (RGB)
        mask       : PIL.Image   – binary mask (white = inpaint area, RGB)
        ref_image  : PIL.Image   – defect reference for IP-Adapter
        gen_cfg    : dict        – generation hyper-params
        """
        inject_alpha = float(gen_cfg.get("inject_alpha", 0.95))
        controlnet_scale = float(gen_cfg.get("controlnet_scale", 0.35))
        ip_scale = float(gen_cfg.get("ip_scale", 0.6))
        
        # In UI, 'strength' is often denoise between 0.1-1.0
        denoise = float(gen_cfg.get("strength", 0.28))
        
        # Prepare components
        base_rgb = self.force_rgb(base_image)
        mask_l = mask.convert("L")
        crop_rgb = self.force_rgb(ref_image[0] if isinstance(ref_image, list) else ref_image)

        # 1. Measure and Align
        bg_angle, bg_center = estimate_bg_orientation(base_rgb, mask_l)
        defect_angle, def_center = estimate_defect_orientation(crop_rgb)
        opt_angle = bg_angle - defect_angle

        # 2. Inject Affine + Alpha
        injected_np, bb = align_and_inject_alpha(
            base_rgb, crop_rgb, mask_l, 
            opt_rot=opt_angle, 
            src_center=def_center, 
            tgt_center=bg_center, 
            alpha_paste=inject_alpha
        )

        if injected_np is None:
            return base_rgb # Fallback if empty mask
            
        # 3. Chuyển đổi sang ảnh RGB (Đã loại bỏ hẳn hàm apply_geometry_dent để y hệt code mẫu)
        geometric_image = Image.fromarray(injected_np).convert("RGB")

        # 4. SDXL Rendering — Crop→Inpaint→Paste back
        INPAINT_SIZE = int(gen_cfg.get('inpaint_size', 512))

        # 4a. Crop tight region around mask + padding
        mask_np2 = np.array(mask_l)
        ys, xs = np.where(mask_np2 > 50)
        if len(ys) == 0:
            return geometric_image  # empty mask fallback
        y1m, y2m = int(ys.min()), int(ys.max())
        x1m, x2m = int(xs.min()), int(xs.max())
        bh2, bw2 = y2m - y1m, x2m - x1m
        pad = max(int(max(bh2, bw2) * 0.5), 32)
        W, H = geometric_image.size
        x1c = max(0, x1m - pad); y1c = max(0, y1m - pad)
        x2c = min(W, x2m + pad); y2c = min(H, y2m + pad)

        img_crop  = geometric_image.crop((x1c, y1c, x2c, y2c)).resize((INPAINT_SIZE, INPAINT_SIZE), Image.LANCZOS)
        mask_crop = mask_l.crop((x1c, y1c, x2c, y2c)).resize((INPAINT_SIZE, INPAINT_SIZE), Image.NEAREST)
        print(f'[GEN] crop=({x1c},{y1c},{x2c},{y2c}) -> {INPAINT_SIZE}px', flush=True)

        import torch; torch.cuda.empty_cache()
        self.pipe.set_ip_adapter_scale(ip_scale)
        depth_img = self.force_rgb(self.get_depth(img_crop))

        seed = gen_cfg.get('seed') or random.randint(0, 999999)
        generator = torch.manual_seed(seed)

        prompt = gen_cfg['prompt']
        negative = gen_cfg['negative_prompt']

        pipe_kw = {
            'prompt': prompt,
            'negative_prompt': negative,
            'image': [self.force_rgb(img_crop)],
            'control_image': [depth_img],
            'controlnet_conditioning_scale': controlnet_scale,
            'ip_adapter_image': [crop_rgb],
            'num_inference_steps': int(gen_cfg.get('steps', 15)),
            'guidance_scale': float(gen_cfg.get('guidance_scale', 6.5)),
            'strength': denoise,
            'generator': generator
        }

        if 'Inpaint' in str(self.pipe.__class__):
            mask_for_sdxl = mask_crop.convert('RGB').filter(ImageFilter.GaussianBlur(6))
            pipe_kw['mask_image'] = [mask_for_sdxl]

        def _progress(pipe, i, t, kwargs):
            total = int(pipe_kw.get('num_inference_steps', 15))
            print(f'[GEN] step {i+1}/{total}', flush=True)
            return kwargs

        try:
            result_crop = self.pipe(**pipe_kw, callback_on_step_end=_progress).images[0]
        except ValueError:
            pipe_kw = {k: (v[0] if isinstance(v, list) else v) for k, v in pipe_kw.items()}
            result_crop = self.pipe(**pipe_kw).images[0]

        # 4b. Paste back onto full image
        result_full = geometric_image.copy()
        crop_orig_size = (x2c - x1c, y2c - y1c)
        result_paste = result_crop.resize(crop_orig_size, Image.LANCZOS)
        mask_paste = mask_l.crop((x1c, y1c, x2c, y2c)).filter(ImageFilter.GaussianBlur(8))
        result_full.paste(result_paste, (x1c, y1c), mask_paste)
        print(f'[GEN] pasted back -> {result_full.size}', flush=True)

        return result_full
