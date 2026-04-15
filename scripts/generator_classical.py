"""
generator_classical.py — Defect Signal Injection Pipeline
==========================================================

Architecture:
    METHOD_REGISTRY maps method name → function
    ClassicalDefectGenerator dispatches by config["method"]

Config per class:
    generation:
      method: signal_injection
      blur_kernel: 51          # background removal kernel (must be odd)
      intensity_min: 0.8       # random intensity scale range
      intensity_max: 1.5
      alpha_blur: 15           # soft mask blur kernel (must be odd)
      alpha_dilate: 7          # mask dilation before blur
"""

import random
import cv2
import numpy as np
from PIL import Image


# ── Core signal functions ──────────────────────────────────────────────────────

def extract_signal(image: np.ndarray, blur_kernel: int = 51) -> np.ndarray:
    """
    Extract high-frequency defect signal from NG crop.
    Removes background lighting/texture — keeps only anomaly signal.

    Args:
        image: float32 (H, W, 3)
        blur_kernel: must be odd; larger = removes more background

    Returns:
        signal: float32 (H, W, 3), zero-mean high-frequency component
    """
    k = blur_kernel | 1  # ensure odd
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (k, k), 0)
    signal_2d  = gray - background                      # high-freq, ~zero-mean
    signal_3d  = np.stack([signal_2d] * 3, axis=-1)    # (H, W, 3)
    return signal_3d


def normalize_signal(signal: np.ndarray, intensity_min: float, intensity_max: float) -> np.ndarray:
    """
    Normalize signal to unit std then scale by random intensity.
    Strictly enforce zero-mean so the injected region does not shift in brightness.

    Args:
        signal: float32 (H, W, 3)
        intensity_min/max: random scale range

    Returns:
        signal_scaled: float32 (H, W, 3)
    """
    std = signal.std() + 1e-6
    signal_norm = signal / std                   # unit std (do NOT subtract mean yet)
    signal_norm = signal_norm - signal_norm.mean()  # strict zero-mean
    scale = random.uniform(intensity_min, intensity_max)
    return signal_norm * scale * 12.0            # ±12px shift at scale=1.0 (was 25.0)


def thin_mask(mask: np.ndarray, target_width_px: int = 8) -> np.ndarray:
    """
    Thin mask via erosion — reduce wide scratch/crack mask toward target_width_px.
    target_width_px: desired final width of the scratch region in pixels.
    Falls back to original mask if erosion removes everything.
    """
    h, w = mask.shape[:2]
    short = min(h, w)
    # Erode just enough to reach target width (not too aggressive)
    erosion_px = max(1, (short - target_width_px) // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (erosion_px * 2 + 1, erosion_px * 2 + 1))
    thinned = cv2.erode(mask, kernel)
    if cv2.countNonZero(thinned) == 0:
        return mask  # fallback: mask too small to thin
    return thinned


def create_soft_mask(mask: np.ndarray, dilate_size: int = 7, blur_size: int = 15,
                     radial_falloff: bool = False) -> np.ndarray:
    """
    Build smooth alpha from binary mask.

    radial_falloff=True  → alpha = 1 tai tam, giam dan ra bien (tu nhien hon)
    radial_falloff=False → Gaussian blur cua binary mask (default cu)
    dilate_size=0        → skip dilation (no white halo)

    Returns: alpha float32 (H, W) in [0, 1]
    """
    if radial_falloff:
        ys, xs = np.where(mask > 127)
        if len(ys) == 0:
            return mask.astype(np.float32) / 255.0
        cy, cx  = float(ys.mean()), float(xs.mean())
        H, W    = mask.shape
        yy, xx  = np.mgrid[0:H, 0:W].astype(np.float32)
        dist    = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # Ban kinh hieu qua = khoang cach tu tam den bien xa nhat cua mask
        r_eff   = float(np.sqrt(((xs.max()-xs.min())/2)**2 + ((ys.max()-ys.min())/2)**2))
        r_eff   = max(r_eff, 1.0)
        # Radial alpha: 1 tai tam, 0 tai r_eff, smooth falloff (cosine curve)
        t       = np.clip(dist / r_eff, 0.0, 1.0)
        radial  = ((np.cos(t * np.pi) + 1.0) / 2.0).astype(np.float32)  # cosine falloff
        radial[mask <= 127] = 0.0   # zero ngoai mask boundary
        # Smooth nhe bien ngoai
        k = max(3, blur_size // 3) | 1
        return cv2.GaussianBlur(radial, (k, k), 0)

    # Gaussian blur cua binary mask (fallback / line-shaped defects)
    if dilate_size > 0:
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        dilated = cv2.dilate(mask, kernel)
    else:
        dilated = mask
    blurred = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0,
                               (blur_size | 1, blur_size | 1), 0)
    return blurred


def apply_signal_injection(
    good_image: np.ndarray,     # float32 (H, W, 3)
    signal: np.ndarray,         # float32 (fit_h, fit_w, 3)
    mask_crop: np.ndarray,      # uint8 (bbox_h, bbox_w)
    alpha: np.ndarray,          # float32 (bbox_h, bbox_w)
    bbox: tuple,                # (y_min, y_max, x_min, x_max)
) -> np.ndarray:
    """
    Inject signal into good_image at bbox region using soft alpha.
    Additive blend: result = region + signal * alpha
    Only modifies pixels inside mask — no foreign texture pasted.
    """
    y_min, y_max, x_min, x_max = bbox
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1

    # Resize signal to fill bbox (fill mode: scale to cover, center-crop)
    # Avoids empty canvas zones that reduce visible defect signal
    sh, sw = signal.shape[:2]
    scale  = max(bbox_w / sw, bbox_h / sh)
    fit_w  = max(int(sw * scale), bbox_w)
    fit_h  = max(int(sh * scale), bbox_h)
    signal_fit = cv2.resize(signal, (fit_w, fit_h), interpolation=cv2.INTER_LANCZOS4)
    # Center-crop to canvas size
    cy_off = (fit_h - bbox_h) // 2
    cx_off = (fit_w - bbox_w) // 2
    canvas = signal_fit[cy_off:cy_off+bbox_h, cx_off:cx_off+bbox_w].copy()

    # Additive injection — ring texture preserved
    a3          = alpha[:, :, np.newaxis]
    orig_region = good_image[y_min:y_max+1, x_min:x_max+1]
    result      = orig_region + canvas * a3

    output = good_image.copy()
    output[y_min:y_max+1, x_min:x_max+1] = np.clip(result, 0, 255)
    return output


# ── Method implementations ─────────────────────────────────────────────────────

def generate_ref_paste(
    base_image,
    mask: np.ndarray,
    ref_image,
    gen_cfg: dict,
) -> Image.Image:
    """
    Reference paste pipeline for appearance-based defects.
    Directly blends NG reference appearance into good image at mask region.
    Applies background brightness matching to reduce seam visibility.

    Use for: bright metallic protrusions, discoloration blobs, visible material defects.

    Config:
        blend_strength: float [0..1]  how strongly ref replaces base (default 0.85)
        alpha_blur:     int           soft edge feather (default 21)
        alpha_dilate:   int           mask dilation before blur (default 3)
    """
    base_arr = np.array(base_image).astype(np.float32)
    ref_arr  = np.array(ref_image).astype(np.float32)
    h, w = base_arr.shape[:2]

    # Ensure mask matches base image dimensions
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        print("[WARN] Empty mask — returning base image unchanged")
        return base_image

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    print(f"[PASTE] Mask bbox {bbox_w}×{bbox_h} at ({x_min},{y_min})")

    # Resize ref to fit bbox preserving aspect ratio, center in canvas
    sh, sw = ref_arr.shape[:2]
    scale  = min(bbox_w / sw, bbox_h / sh)
    fit_w  = max(int(sw * scale), 1)
    fit_h  = max(int(sh * scale), 1)
    ref_fit = cv2.resize(ref_arr, (fit_w, fit_h), interpolation=cv2.INTER_LANCZOS4)

    # Place ref in canvas backed by base region (so corners outside ref keep base)
    base_region = base_arr[y_min:y_max+1, x_min:x_max+1].copy()
    canvas = base_region.copy()
    py = (bbox_h - fit_h) // 2
    px = (bbox_w - fit_w) // 2
    canvas[py:py+fit_h, px:px+fit_w] = ref_fit

    # Background brightness match: adjust canvas so non-mask area matches base
    # This anchors brightness without flattening the defect itself
    mask_crop     = mask[y_min:y_max+1, x_min:x_max+1]
    non_mask_bool = mask_crop < 128
    if non_mask_bool.any():
        base_bg_mean   = base_region[non_mask_bool].mean()
        canvas_bg_mean = canvas[non_mask_bool].mean()
        delta = base_bg_mean - canvas_bg_mean
        canvas = canvas + delta
        print(f"[PASTE] Brightness delta={delta:+.1f}")

    # Try Poisson MIXED_CLONE (seamless, gradient-domain) — same as original pipeline
    base_uint8   = np.clip(base_arr, 0, 255).astype(np.uint8)
    canvas_uint8 = np.clip(canvas, 0, 255).astype(np.uint8)

    # Build full-image src: paste canvas back into base
    src = base_uint8.copy()
    src[y_min:y_max+1, x_min:x_max+1] = canvas_uint8

    # Blend mask: dilate + blur for soft irregular boundary
    alpha_dilate = int(gen_cfg.get("alpha_dilate", 5))
    ks = alpha_dilate * 2 + 1
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    blend_mask_crop = cv2.dilate(mask_crop, k, iterations=1)
    blend_mask_crop = cv2.GaussianBlur(blend_mask_crop, (7, 7), 0)
    _, blend_mask_crop = cv2.threshold(blend_mask_crop, 15, 255, cv2.THRESH_BINARY)

    blend_mask_full = np.zeros((h, w), dtype=np.uint8)
    blend_mask_full[y_min:y_max+1, x_min:x_max+1] = blend_mask_crop
    border = 5
    blend_mask_full[:border, :] = blend_mask_full[-border:, :] = 0
    blend_mask_full[:, :border] = blend_mask_full[:, -border:] = 0

    ys_m, xs_m = np.where(blend_mask_full > 0)
    if len(ys_m) > 0:
        center = (int(xs_m.mean()), int(ys_m.mean()))
        try:
            result = cv2.seamlessClone(src, base_uint8, blend_mask_full, center, cv2.MIXED_CLONE)
            print(f"[PASTE] Poisson MIXED_CLONE OK center={center}")
            return Image.fromarray(result)
        except Exception as e:
            print(f"[PASTE] Poisson failed ({e}), fallback to alpha blend")

    # Fallback: soft alpha blend
    blend_strength = float(gen_cfg.get("blend_strength", 0.85))
    alpha_blur     = int(gen_cfg.get("alpha_blur", 21))
    alpha = create_soft_mask(mask_crop, alpha_dilate, alpha_blur) * blend_strength
    a3    = alpha[:, :, np.newaxis]
    result_region = canvas * a3 + base_arr[y_min:y_max+1, x_min:x_max+1] * (1.0 - a3)
    output = base_arr.copy()
    output[y_min:y_max+1, x_min:x_max+1] = np.clip(result_region, 0, 255)
    return Image.fromarray(output.astype(np.uint8))


def generate_elastic_warp(
    base_image,
    mask: np.ndarray,
    ref_image,       # unused for geometric warp
    gen_cfg: dict,
) -> Image.Image:
    """
    Elastic warp pipeline for shape deformation defects.
    Displaces pixels inside the mask region toward/away from the mask centroid,
    simulating dent (inward) or bulge (outward) deformation.
    """
    base_arr = np.array(base_image).astype(np.float32)
    h, w = base_arr.shape[:2]

    # Ensure mask matches base image dimensions
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        print("[WARN] Empty mask — returning base image unchanged")
        return base_image

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    cy = (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0
    print(f"[WARP] centroid ({cx:.0f},{cy:.0f}) bbox {x_max-x_min+1}×{y_max-y_min+1}")

    warp_strength = float(gen_cfg.get("warp_strength", 8.0))
    warp_mode     = gen_cfg.get("warp_mode", "dent")   # "dent" | "bulge"
    warp_blur     = int(gen_cfg.get("warp_blur", 31))
    print(f"[WARP] mode={warp_mode} strength={warp_strength}")

    # Displacement direction: unit vector FROM centroid
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy_dir  = yy - cy
    dx_dir  = xx - cx
    dist    = np.sqrt(dy_dir**2 + dx_dir**2) + 1e-6
    dy_norm = dy_dir / dist
    dx_norm = dx_dir / dist

    # Smooth mask weight so displacement feathers at edges
    k      = warp_blur | 1
    weight = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), 0)

    scale = warp_strength * random.uniform(0.7, 1.3)
    sign  = -1.0 if warp_mode == "dent" else 1.0   # dent = pull toward center

    map_x = xx + sign * dx_norm * weight * scale
    map_y = yy + sign * dy_norm * weight * scale

    warped = cv2.remap(
        base_arr.astype(np.uint8), map_x, map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Blend warped region back using soft alpha
    alpha_dilate = int(gen_cfg.get("alpha_dilate", 5))
    alpha_blur   = int(gen_cfg.get("alpha_blur", 31))
    alpha = create_soft_mask(mask, alpha_dilate, alpha_blur)
    a3    = alpha[:, :, np.newaxis]

    result = warped.astype(np.float32) * a3 + base_arr * (1.0 - a3)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def generate_shaded_warp(
    base_image,
    mask: np.ndarray,
    ref_image,       # unused
    gen_cfg: dict,
) -> Image.Image:
    """
    Shaded warp: elastic_warp (geometry) + physics-based shading (shadow/highlight).

    Pipeline:
      [1] Elastic warp — pixel displacement (dent inward / bulge outward)
      [2] Height field — Gaussian bump from smoothed mask
      [3] Surface normals — gradient of height field
      [4] Light estimation — dominant gradient of low-freq image envelope
      [5] Lambert diffuse + Blinn-Phong specular shading delta
      [6] Apply shading delta additively with feathered mask

    Config:
        warp_strength:  float  pixel displacement magnitude (default 18.0)
        warp_mode:      str    "dent" | "bulge" (default "dent")
        warp_blur:      int    displacement blur kernel (default 25)
        shading_gain:   float  overall shading intensity (default 150.0)
        amplitude:      float  height field peak magnitude (default 2.0)
        normal_scale:   float  surface normal sensitivity (default 40.0)
        shininess:      float  specular exponent (default 32.0)
        diffuse_w:      float  Lambert weight (default 0.7)
        specular_w:     float  Blinn-Phong weight (default 0.3)
        alpha_blur:     int    mask feather blur (default 25)
        alpha_dilate:   int    mask dilation (default 7)
    """
    base_arr = np.array(base_image).astype(np.float32)
    h, w = base_arr.shape[:2]

    # Ensure mask matches base image dimensions
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        print("[WARN] Empty mask — returning base image unchanged")
        return base_image

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    cy = (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0
    print(f"[SHWARP] centroid ({cx:.0f},{cy:.0f}) bbox {x_max-x_min+1}×{y_max-y_min+1}")

    warp_strength = float(gen_cfg.get("warp_strength", 18.0))
    warp_mode     = gen_cfg.get("warp_mode", "dent")
    warp_blur     = int(gen_cfg.get("warp_blur", 25))

    # ── [1] Elastic warp ────────────────────────────────────────────────────
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy_dir = yy - cy
    dx_dir = xx - cx
    dist   = np.sqrt(dy_dir**2 + dx_dir**2) + 1e-6
    dy_norm = dy_dir / dist
    dx_norm = dx_dir / dist

    k_w    = warp_blur | 1
    weight = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k_w, k_w), 0)

    warp_scale = warp_strength * random.uniform(0.7, 1.3)
    sign       = -1.0 if warp_mode == "dent" else 1.0

    map_x = xx + sign * dx_norm * weight * warp_scale
    map_y = yy + sign * dy_norm * weight * warp_scale

    warped = cv2.remap(
        base_arr.astype(np.uint8), map_x, map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32)

    # Hot-pixel removal: isolated bright pixels from displaced specular reflections
    # Replace pixels that are abnormally bright vs their 5×5 neighborhood median
    warped_u8  = np.clip(warped, 0, 255).astype(np.uint8)
    median5    = cv2.medianBlur(warped_u8, 5).astype(np.float32)
    hot        = (warped - median5) > 35          # isolated highlight (any channel)
    hot_any    = hot.any(axis=2, keepdims=True)
    warped     = np.where(hot_any, median5, warped)
    print(f"[SHWARP] Warp done: mode={warp_mode} strength={warp_scale:.1f}")

    # Blend warp into base
    alpha_dilate = int(gen_cfg.get("alpha_dilate", 7))
    alpha_blur_k = int(gen_cfg.get("alpha_blur", 25))
    warp_alpha   = create_soft_mask(mask, alpha_dilate, alpha_blur_k)
    wa3          = warp_alpha[:, :, np.newaxis]
    base_warped  = warped * wa3 + base_arr * (1.0 - wa3)

    # ── [2] Height field from mask ──────────────────────────────────────────
    amplitude    = float(gen_cfg.get("amplitude", 2.0))
    normal_scale = float(gen_cfg.get("normal_scale", 40.0))

    # Smooth mask → height field (Gaussian shape, centered on mask)
    mask_f = mask.astype(np.float32) / 255.0
    hf_sigma = max(int(np.sqrt(mask_f.sum()) * 0.8), 5)
    hf_k = hf_sigma * 4 | 1
    height_field = cv2.GaussianBlur(mask_f, (hf_k, hf_k), hf_sigma)
    hf_max = height_field.max()
    if hf_max > 0:
        height_field = height_field / hf_max  # normalize to [0, 1]

    # dent = depression (negative), bulge = protrusion (positive)
    if warp_mode == "dent":
        height_field = -amplitude * height_field
    else:
        height_field = amplitude * height_field

    # Slight smooth noise for organic texture (very subtle — gain amplifies it)
    noise = np.random.normal(0, 0.02, height_field.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (31, 31), 8)
    height_field = height_field * (1.0 + noise)

    # ── [3] Surface normals ─────────────────────────────────────────────────
    dh_dx = cv2.Sobel(height_field, cv2.CV_32F, 1, 0, ksize=5)
    dh_dy = cv2.Sobel(height_field, cv2.CV_32F, 0, 1, ksize=5)

    nx = -dh_dx * normal_scale
    ny = -dh_dy * normal_scale
    nz = np.ones_like(nx)
    mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx /= mag
    ny /= mag
    nz /= mag

    # ── [4] Estimate light direction ────────────────────────────────────────
    # illum always needed for shading delta at line ~501
    gray  = cv2.cvtColor(base_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    illum = cv2.GaussianBlur(gray, (0, 0), 60)  # low-freq illumination envelope

    _ext_ld = gen_cfg.get("light_dir")
    if _ext_ld is not None:
        # Override: radial normal from structure_adapt (more accurate for ring parts)
        lx, ly, lz = float(_ext_ld[0]), float(_ext_ld[1]), float(_ext_ld[2])
        l_mag = np.sqrt(lx**2 + ly**2 + lz**2) + 1e-8
        lx /= l_mag; ly /= l_mag; lz /= l_mag
        print(f"[SHWARP] Light direction (override): ({lx:.2f}, {ly:.2f}, {lz:.2f})")
    else:

        grad_x = cv2.Sobel(illum, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(illum, cv2.CV_32F, 0, 1, ksize=5)

        # Mean gradient in mask region → dominant light direction
        mask_bool = mask > 127
        if mask_bool.any():
            lx = float(grad_x[mask_bool].mean())
            ly = float(grad_y[mask_bool].mean())
        else:
            lx = float(grad_x.mean())
            ly = float(grad_y.mean())

        lmag = np.sqrt(lx**2 + ly**2) + 1e-8
        lx /= lmag
        ly /= lmag
        lz = 0.6  # light ~53° elevation from surface
        l_mag = np.sqrt(lx**2 + ly**2 + lz**2)
        lx /= l_mag
        ly /= l_mag
        lz /= l_mag
        print(f"[SHWARP] Light direction: ({lx:.2f}, {ly:.2f}, {lz:.2f})")

    # ── [5] Lambert + Blinn-Phong shading delta ────────────────────────────
    shininess = float(gen_cfg.get("shininess", 32.0))
    diffuse_w = float(gen_cfg.get("diffuse_w", 0.7))
    specular_w = float(gen_cfg.get("specular_w", 0.3))

    # Lambert diffuse: n·L
    n_dot_l = nx * lx + ny * ly + nz * lz
    # Delta = how much this differs from flat surface (nz=1)
    flat_dot_l = lz  # flat surface normal = (0,0,1)
    lambert_delta = n_dot_l - flat_dot_l

    # Blinn-Phong specular: (n·H)^shininess
    # H = normalize(L + V), V = (0, 0, 1) (camera straight above)
    hx = lx
    hy = ly
    hz = lz + 1.0
    h_mag = np.sqrt(hx**2 + hy**2 + hz**2) + 1e-8
    hx /= h_mag
    hy /= h_mag
    hz /= h_mag

    n_dot_h = np.clip(nx * hx + ny * hy + nz * hz, 0, 1)
    flat_dot_h = hz  # flat: (0,0,1)·H
    spec_delta = np.power(n_dot_h, shininess) - np.power(flat_dot_h, shininess)

    # Combined shading delta, modulated by local illumination
    illum_norm = illum / (illum.max() + 1e-6)  # [0, 1]
    delta = (lambert_delta * diffuse_w + spec_delta * specular_w) * np.sqrt(illum_norm + 0.1)

    # ── [6] Apply shading delta ─────────────────────────────────────────────
    shading_gain = float(gen_cfg.get("shading_gain", 150.0))
    shading_gain *= random.uniform(0.8, 1.2)

    # Feathered mask for shading — use same soft mask as warp to avoid background bleed
    shading_mask = create_soft_mask(mask, alpha_dilate, alpha_blur_k)

    # Exclude background pixels from shading: auto-detect BG brightness from outside mask
    # Works for both dark and bright backgrounds (uses distance from BG mean)
    gray_base = cv2.cvtColor(base_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg_pixels = gray_base[mask < 127]
    if len(bg_pixels) > 100:
        bg_mean = float(bg_pixels.mean())
        bg_std  = float(bg_pixels.std())
        # Distance from BG mean → high for product pixels (different from BG), 0 for BG
        diff_from_bg = np.abs(gray_base - bg_mean)
        soft_thresh  = max(2.0 * bg_std, 8.0)   # at least 8 gray levels
        product_region = np.clip(diff_from_bg / soft_thresh, 0, 1)
        shading_mask = shading_mask * product_region

    delta_3d = np.stack([delta] * 3, axis=-1) * shading_gain * shading_mask[:, :, np.newaxis]

    result = np.clip(base_warped + delta_3d, 0, 255)
    print(f"[SHWARP] Shading applied: gain={shading_gain:.0f} delta_range=[{delta.min():.3f}, {delta.max():.3f}]")

    return Image.fromarray(result.astype(np.uint8))


def generate_contrast_reduction(
    base_image,
    mask: np.ndarray,
    ref_image,
    gen_cfg: dict,
) -> Image.Image:
    """
    Contrast Reduction cuc bo — mo phong vung bi xuoc mat do bong.

    Calibrate muc giam contrast tu ref image:
      - Do std(center ref) / std(full ref) → contrast_factor thuc te
      - Apply: result = local_mean + (base - local_mean) × contrast_factor
      - Blend voi soft alpha mask

    Uu diem: khong co artifact vien, khong can HF extraction.
    """
    base_arr = np.array(base_image).astype(np.float32)
    h, w = base_arr.shape[:2]

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        return base_image

    # ── Calibrate contrast_factor tu ref ────────────────────────────────────
    if ref_image is not None:
        ref_gray = cv2.cvtColor(
            np.array(ref_image).astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32)
        rh, rw = ref_gray.shape
        # Center 50% cua ref = vung defect chinh
        cy0, cy1 = rh // 4, 3 * rh // 4
        cx0, cx1 = rw // 4, 3 * rw // 4
        std_center = ref_gray[cy0:cy1, cx0:cx1].std() + 1e-6
        std_full   = ref_gray.std() + 1e-6
        # factor < 1: defect center mo hon surroundings → giam contrast
        raw_factor = float(std_center / std_full)
        # Clamp: khong de qua phang (0.05) hay qua it thay doi (0.85)
        contrast_factor = float(np.clip(raw_factor, 0.05, 0.85))
        print(f"[CR] Ref calibrated: std_center={std_center:.1f} std_full={std_full:.1f}"
              f" → factor={contrast_factor:.3f}")
    else:
        contrast_factor = float(gen_cfg.get("contrast_factor", 0.35))
        print(f"[CR] No ref — using fixed factor={contrast_factor:.3f}")

    # Intensity param scale factor xuong them neu muon defect ro hon
    intensity = float(gen_cfg.get("intensity", 1.0))
    contrast_factor = contrast_factor ** intensity   # intensity>1 → factor nho hon → mo hon

    # ── Build soft alpha (no dilation to avoid halo) ─────────────────────────
    alpha_blur = int(gen_cfg.get("alpha_blur", 15))
    alpha      = create_soft_mask(mask, dilate_size=0, blur_size=alpha_blur)

    # ── Apply contrast reduction ─────────────────────────────────────────────
    # Local mean trong vung mask (dung de giu brightness, chi giam contrast)
    mask_bool   = mask > 127
    local_mean  = float(base_arr[mask_bool].mean())

    reduced     = local_mean + (base_arr - local_mean) * contrast_factor
    a3          = alpha[:, :, np.newaxis]
    result_arr  = base_arr * (1.0 - a3) + reduced * a3

    print(f"[CR] local_mean={local_mean:.1f}  factor={contrast_factor:.3f}"
          f"  bbox={xs.max()-xs.min()+1}x{ys.max()-ys.min()+1}")
    return Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8))


def generate_signal_injection(
    base_image,
    mask: np.ndarray,
    ref_image,
    gen_cfg: dict,
) -> Image.Image:
    """
    Signal injection pipeline.
    Transfers only the HIGH-FREQUENCY defect anomaly from ref_image.
    Background and texture of ref_image are NOT copied.
    """
    base_arr = np.array(base_image).astype(np.float32)
    ref_arr  = np.array(ref_image).astype(np.float32)
    h, w = base_arr.shape[:2]

    # Ensure mask matches base image dimensions
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        print("[WARN] Empty mask — returning base image unchanged")
        return base_image

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    print(f"[SIG] Mask bbox {x_max-x_min+1}×{y_max-y_min+1} at ({x_min},{y_min})")

    # Extract signal
    # Auto-scale blur_kernel to ref image size: kernel must be << image for meaningful HF signal
    ref_short     = min(ref_arr.shape[:2])
    blur_kernel   = int(gen_cfg.get("blur_kernel", 51))
    blur_kernel   = max(5, min(blur_kernel, ref_short // 8)) | 1   # odd, ≤ ref_short/8 → finer HF signal
    signal        = extract_signal(ref_arr, blur_kernel)
    print(f"[SIG] blur_kernel={blur_kernel} (ref {ref_arr.shape[1]}×{ref_arr.shape[0]})")

    # Normalize + random intensity
    intensity_min = float(gen_cfg.get("intensity_min", 0.8))
    intensity_max = float(gen_cfg.get("intensity_max", 1.5))
    signal        = normalize_signal(signal, intensity_min, intensity_max)
    print(f"[SIG] Signal extracted, intensity∈[{intensity_min},{intensity_max}]")

    # Soft alpha mask
    mask_crop    = mask[y_min:y_max+1, x_min:x_max+1]
    if gen_cfg.get("thin_mask", False):
        mask_crop = thin_mask(mask_crop)
    alpha_dilate   = int(gen_cfg.get("alpha_dilate", 7))
    alpha_blur     = int(gen_cfg.get("alpha_blur", 15))
    radial_falloff = bool(gen_cfg.get("radial_falloff", False))
    alpha          = create_soft_mask(mask_crop, alpha_dilate, alpha_blur, radial_falloff)

    # Inject
    output_arr = apply_signal_injection(
        base_arr, signal, mask_crop, alpha,
        (y_min, y_max, x_min, x_max)
    )

    # Poisson seamless clone — removes rectangular bbox artifact at edges
    # Hybrid: Poisson fixes boundary seam, alpha-blend signal preserved in core
    if gen_cfg.get("use_poisson", False):
        bbox_h = y_max - y_min + 1
        bbox_w = x_max - x_min + 1
        alpha_blend_arr = output_arr.copy()  # preserve full-strength alpha blend

        src_crop  = np.clip(output_arr[y_min:y_max+1, x_min:x_max+1], 0, 255).astype(np.uint8)
        dst_img   = np.clip(base_arr, 0, 255).astype(np.uint8)
        k_poi     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clone_msk = cv2.dilate((mask_crop > 64).astype(np.uint8) * 255, k_poi)
        center_pt = (x_min + bbox_w // 2, y_min + bbox_h // 2)
        try:
            poisson_arr = cv2.seamlessClone(
                src_crop, dst_img, clone_msk, center_pt, cv2.MIXED_CLONE
            ).astype(np.float32)

            # Hybrid: use Poisson at boundary, alpha blend in core
            # Erosion kernel = 3px (small so thin masks keep a core)
            k_core   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            core_msk = cv2.erode((mask_crop > 64).astype(np.uint8) * 255, k_core)
            core_msk_full = np.zeros_like(output_arr[:, :, 0])
            core_msk_full[y_min:y_max+1, x_min:x_max+1] = core_msk.astype(np.float32) / 255.0
            w3 = core_msk_full[:, :, np.newaxis]

            output_arr = alpha_blend_arr * w3 + poisson_arr * (1.0 - w3)
            print("[SIG] Poisson hybrid applied (core=alpha, boundary=Poisson)")
        except Exception as e:
            print(f"[WARN] Poisson blend failed, using alpha blend: {e}")

    return Image.fromarray(np.clip(output_arr, 0, 255).astype(np.uint8))


# ── Method registry ────────────────────────────────────────────────────────────

METHOD_REGISTRY = {
    "signal_injection": generate_signal_injection,   # texture: scratch, stain
    "elastic_warp":     generate_elastic_warp,        # geometry: dent, bulge
    "shaded_warp":      generate_shaded_warp,         # geometry + physics shading
    "ref_paste":        generate_ref_paste,           # appearance: paste NG ref into mask
}


# ── Generator class ────────────────────────────────────────────────────────────

class ClassicalDefectGenerator:

    def __init__(self, cfg: dict):
        print(f"[INFO] ClassicalDefectGenerator — methods: {list(METHOD_REGISTRY.keys())}")

    def generate(
        self,
        base_image,
        mask: np.ndarray,
        ref_image,
        gen_cfg: dict,
    ) -> Image.Image:

        method = gen_cfg.get("method", "signal_injection")
        fn     = METHOD_REGISTRY.get(method)

        if fn is None:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available: {list(METHOD_REGISTRY.keys())}"
            )

        print(f"[INFO] Dispatching → {method}")
        return fn(base_image, mask, ref_image, gen_cfg)
