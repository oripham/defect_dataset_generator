"""
engines/scratch_napchai_engine.py — Scratch Napchai (Xước vành polar)
=======================================================================
Port of pipeline_scratch.ipynb.

Pipeline (exactly as notebook):
  1. Hough Circle → center, max_radius = radius * 1.3
  2. Polar transform 720×1024
  3. r_ring_col = last col where profile > min + (max-min)*0.2  ← outer boundary
     (pipeline_scratch cell-6 — different from mc_deform/ring_fracture)
  4. User mask (base64) → to_polar, or auto band around outer rim
  5. synthesize_scratch_realistic: sinh vết xước realistic trong polar space
  6. Inverse polar → cv_res, mask_res
  7. SDXL ControlNet-Depth + IP-Adapter Plus refine
     - TARGET_SIZE = (1024, 1024)
     - ip_image at (224, 224), ip_scale=0.8 (0.5 if no ref)
     - strength=0.45, guidance=8.0, steps=35
  8. ai_res_low.resize(original_size, LANCZOS)  ← explicit LANCZOS

API endpoint: POST /api/metal_cap/preview  {defect_type: "scratch"}

Tham số người dùng (params dict):
  seed         int    — random seed (default 42)
  severity     str    — "light" | "medium" | "heavy"  (default "medium")
  count        int    — số vết xước chính 1–5  (default 2)
  sdxl_refine  bool   — bật SDXL refine  (default False)
  ref_image_b64 str   — base64 ảnh NG mẫu cho IP-Adapter
  mask_b64     str    — user-drawn mask (auto-band nếu không có)
"""
from __future__ import annotations

import math
import random as _random
import base64 as _b64

import cv2
import numpy as np
from PIL import Image as _PIL

from ..utils import encode_b64, decode_b64
from ..models._napchai_models import get_pipe, get_depth_est, get_lock

# ── Constants ─────────────────────────────────────────────────────────────────
POLAR_H = 720
POLAR_W = 1024
_TARGET = (1024, 1024)

_PROMPT = (
    "realistic metal scratch, deep industrial gouge, torn raw steel, "
    "irregular jagged edges, harsh specular glints, metallic burrs, "
    "highly detailed metallic texture, industrial damage, 8k, harsh lighting"
)
_NEG      = "paint, drawing, plastic, blur, soft edges, uniform texture, artificial, flat, cartoon"
_STRENGTH = 0.45
_GUIDANCE = 8.0
_STEPS    = 35

# ── FLUX disabled (model too large for local dev) ────────────────────────────

# ── Severity presets ───────────────────────────────────────────────────────────
# Mỗi preset định nghĩa: (shadow_strength, ridge_strength, width_range, length_range, companion_prob)
_SEVERITY_PRESETS = {
    #                shadow  ridge  width      length (px polar)  companion
    "light":  dict(shadow=0.28, ridge=0.55, width=(1, 1), length=(300,  600), companion_prob=0.35),
    "medium": dict(shadow=0.38, ridge=0.75, width=(1, 1), length=(450,  850), companion_prob=0.55),
    "heavy":  dict(shadow=0.50, ridge=0.95, width=(1, 2), length=(650, 1100), companion_prob=0.72),
}


# ── Polar helpers ─────────────────────────────────────────────────────────────

def _detect_circle(gray):
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 1.2, 100,
        param1=80, param2=40,
        minRadius=int(min(gray.shape) * 0.2),
        maxRadius=int(min(gray.shape) * 0.48),
    )
    if circles is None:
        return (gray.shape[1] // 2, gray.shape[0] // 2, min(gray.shape) // 3)
    c = circles[0][0]
    return (int(c[0]), int(c[1]), int(c[2]))


def _to_polar(img, center, max_radius):
    return cv2.warpPolar(img, (POLAR_W, POLAR_H), center, max_radius,
                         cv2.WARP_POLAR_LINEAR + cv2.INTER_LANCZOS4)


def _from_polar(polar, center, max_radius, osize):
    return cv2.warpPolar(polar, osize, center, max_radius,
                         cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LANCZOS4)


def _find_rim_col_outer(polar_gray: np.ndarray) -> int:
    """
    pipeline_scratch cell-6: last col where mean > min + (max-min)*0.2.
    """
    profile   = polar_gray.mean(axis=0)
    threshold = profile.min() + (profile.max() - profile.min()) * 0.2
    indices   = np.where(profile > threshold)[0]
    if len(indices) > 0:
        return int(indices[-1])
    return int(POLAR_W * 0.8)


# ── Scratch primitive helpers ─────────────────────────────────────────────────

def _draw_scratch_stroke(
    trench: np.ndarray,
    glint: np.ndarray,
    sx: int, sy: int,
    angle: float,
    length: int,
    width: int,
    intensity: float,
    H: int, W: int,
):
    """
    Vẽ 1 vết xước đơn vào trench + glint layer.
    - Độ đậm fade in/out ở 2 đầu (taper).
    - Đường cong nhẹ bằng cách xoay angle từ từ.
    - Glint là vệt sáng mỏng song song cạnh trench.
    """
    taper_zone = max(length // 6, 8)  # vùng fade ở mỗi đầu

    # Độ cong nhỏ: angle drift tích lũy theo chiều dài
    angle_drift = _random.uniform(-0.004, 0.004)  # rad/step
    curr_angle  = angle

    cx, cy = float(sx), float(sy)
    prev_x, prev_y = cx, cy  # dùng để vẽ line segment liên tục

    for step in range(length):
        # Taper: fade in đầu, fade out cuối
        if step < taper_zone:
            alpha = step / taper_zone
        elif step > length - taper_zone:
            alpha = (length - step) / taper_zone
        else:
            alpha = 1.0
        alpha = alpha ** 0.7

        curr_angle += angle_drift
        # Jitter rất nhỏ — chỉ để có rung nhẹ, không đủ tạo gap
        cx += math.cos(curr_angle) + _random.gauss(0, 0.04)
        cy += math.sin(curr_angle) + _random.gauss(0, 0.04)

        tx, ty = int(cx), int(cy)
        px, py = int(prev_x), int(prev_y)

        if not (0 <= tx < W and 0 <= ty < H):
            break

        # Line segment nối prev→curr: không bao giờ có gap
        cv2.line(trench, (px, py), (tx, ty), float(intensity * alpha), width)
        prev_x, prev_y = cx, cy

        # Glint: line ngắn song song, chỉ ở đoạn giữa
        if alpha > 0.5 and _random.random() > 0.55:
            perp_x = int(cx - math.sin(curr_angle) * 1.5)
            perp_y = int(cy + math.cos(curr_angle) * 1.5)
            if 0 <= perp_x < W and 0 <= perp_y < H:
                glen = _random.randint(4, 9)
                gx2  = max(0, min(W - 1, perp_x + int(glen * math.cos(curr_angle))))
                gy2  = max(0, min(H - 1, perp_y + int(glen * math.sin(curr_angle))))
                cv2.line(glint, (perp_x, perp_y), (gx2, gy2),
                         float(alpha * _random.uniform(0.6, 1.0)), 1)


def _draw_companion_scratches(
    trench: np.ndarray,
    glint: np.ndarray,
    sx: int, sy: int,
    angle: float,
    base_length: int,
    H: int, W: int,
    n: int = 2,
):
    """
    Vẽ 2–4 vết xước mờ song song sát nhau (hiệu ứng bó xước thực tế).
    """
    for _ in range(n):
        off_angle = angle + _random.uniform(-0.08, 0.08)
        offset    = _random.randint(2, 8)
        perp_off_x = int(-math.sin(angle) * offset * _random.choice([-1, 1]))
        perp_off_y = int( math.cos(angle) * offset * _random.choice([-1, 1]))
        osx = sx + perp_off_x + _random.randint(-3, 3)
        osy = sy + perp_off_y + _random.randint(-3, 3)
        olen   = int(base_length * _random.uniform(0.3, 0.75))
        owidth = max(1, _random.randint(1, 2) - 1)  # companion mỏng hơn
        _draw_scratch_stroke(
            trench, glint,
            osx, osy, off_angle, olen, owidth,
            intensity=_random.uniform(0.25, 0.50),
            H=H, W=W,
        )


# ── Main synthesis ────────────────────────────────────────────────────────────

def _synthesize_scratch_realistic(
    polar_img: np.ndarray,
    polar_mask: np.ndarray,
    rim_col: int,
    severity: str = "medium",
    count: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sinh vết xước realistic trên polar image.

    Cải tiến so với phiên bản cũ:
    - Hướng xước bias theo tangential (horizontal trong polar ≈ theo vành)
    - Taper (fade in/out) ở 2 đầu
    - Độ cong nhẹ tự nhiên
    - Glint là vệt sáng, không phải điểm đơn
    - Companion scratches (bó xước) song song
    - Ridge tính bằng Sobel gradient magnitude (đúng hướng)
    - Shadow/ridge strength theo severity preset
    - Bỏ hoàn toàn "pitting" (không có trong ảnh thực tế nắp chai)
    """
    preset = _SEVERITY_PRESETS.get(severity, _SEVERITY_PRESETS["medium"])
    H, W   = polar_img.shape[:2]

    # Zero out beyond rim
    clean_mask = polar_mask.copy()
    clean_mask[:, rim_col:] = 0
    mask_f = clean_mask.astype(np.float32) / 255.0
    if polar_img.ndim == 3:
        mask_f3 = mask_f[:, :, np.newaxis]
    else:
        mask_f3 = mask_f

    base_f       = polar_img.astype(np.float32) / 255.0
    trench_layer = np.zeros((H, W), dtype=np.float32)
    glint_layer  = np.zeros((H, W), dtype=np.float32)

    # Lấy vị trí hợp lệ trong mask
    y_coords, x_coords = np.where(clean_mask > 32)
    if len(x_coords) == 0:
        return polar_img, clean_mask

    count = max(1, min(count, 5))

    for _ in range(count):
        idx = _random.randint(0, len(x_coords) - 1)
        sx, sy = int(x_coords[idx]), int(y_coords[idx])

        # ── Hướng: bias mạnh theo tangential (horizontal = 0 rad)
        # Phân bố Gaussian quanh 0, std ~0.3 rad (~17°)
        # Đôi khi lệch nhiều hơn để có vết chéo tự nhiên (~15% xác suất)
        if _random.random() < 0.15:
            base_angle = _random.gauss(0, 0.55)  # chéo hơn
        else:
            base_angle = _random.gauss(0, 0.28)  # gần như nằm ngang

        length = _random.randint(*preset["length"])
        width  = _random.randint(*preset["width"])

        # Vết xước chính
        _draw_scratch_stroke(
            trench_layer, glint_layer,
            sx, sy, base_angle, length, width,
            intensity=_random.uniform(0.75, 1.0),
            H=H, W=W,
        )

        # Companion scratches (bó vết song song)
        if _random.random() < preset["companion_prob"]:
            n_comp = _random.randint(1, 3)
            _draw_companion_scratches(
                trench_layer, glint_layer,
                sx, sy, base_angle, length,
                H=H, W=W, n=n_comp,
            )

    # ── Ridge: Sobel của trench → viền sáng 2 bên rãnh ──────────────────────
    gx = cv2.Sobel(trench_layer, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(trench_layer, cv2.CV_32F, 0, 1, ksize=3)
    ridge_raw   = np.sqrt(gx ** 2 + gy ** 2)
    ridge_layer = np.clip(ridge_raw / (ridge_raw.max() + 1e-6), 0, 1)

    # Glint: blur cực nhẹ
    glint_smooth = cv2.GaussianBlur(glint_layer, (3, 3), 0.5)

    # ── Detect độ sáng bề mặt trong vùng mask ────────────────────────────────
    if polar_img.ndim == 3:
        gray_f = 0.299 * base_f[:,:,0] + 0.587 * base_f[:,:,1] + 0.114 * base_f[:,:,2]
    else:
        gray_f = base_f
    mask_pixels        = gray_f[clean_mask > 32]
    surface_brightness = float(mask_pixels.mean()) if len(mask_pixels) > 0 else 0.5
    is_dark_surface    = surface_brightness < 0.35

    # Expand về 3 channels nếu cần
    def _to3(arr):
        return arr[:, :, np.newaxis] if polar_img.ndim == 3 else arr

    ridge_3  = _to3(ridge_layer)
    glint_3  = _to3(glint_smooth)
    trench_3 = _to3(trench_layer)

    r = preset["ridge"]

    # ── Blend chung cho cả 2 loại bề mặt ─────────────────────────────────────
    # Ý tưởng: vết xước = vùng kim loại bị cào → màu nền bị "khuếch đại"
    # về phía sáng hơn, KHÔNG tối hơn — dùng Screen blend hoàn toàn.
    #
    # Screen(a, layer) = 1 - (1-a)(1-layer)
    # → Kết quả luôn >= a (không bao giờ tối hơn nền)
    # → Màu kết quả kế thừa tone màu của nền (đen blend ra xám, bạc blend ra trắng)

    # 1. Rãnh chính (trench): screen nhẹ — lộ màu kim loại bên dưới
    #    Trên nền tối: ra vệt xám bạc nhạt
    #    Trên nền sáng: ra vệt trắng bạc
    trench_strength = 0.18 if is_dark_surface else 0.12
    res_f = 1.0 - (1.0 - base_f) * (1.0 - trench_3 * trench_strength * mask_f3)
    res_f = np.clip(res_f, 0, 1)

    # 2. Ridge (gờ kim loại đùn 2 bên): screen mạnh hơn trench
    ridge_strength = r * (1.4 if is_dark_surface else 1.0)
    res_f = 1.0 - (1.0 - res_f) * (1.0 - ridge_3 * ridge_strength * 0.5 * mask_f3)
    res_f = np.clip(res_f, 0, 1)

    # 3. Glint (vệt lấp lánh): screen mạnh nhất, điểm sáng nhất
    glint_strength = 0.80 if is_dark_surface else 0.65
    res_f = 1.0 - (1.0 - res_f) * (1.0 - glint_3 * glint_strength * mask_f3)
    res_f = np.clip(res_f, 0, 1)

    return (res_f * 255.0).astype(np.uint8), clean_mask


# ── CV step ───────────────────────────────────────────────────────────────────

def _cv_step(img_rgb: np.ndarray, params: dict):
    seed = int(params.get("seed", 42))
    _random.seed(seed)
    np.random.seed(seed)

    # Nếu không set từ UI → random tự động
    # severity: 60% medium, 25% heavy, 15% light  (phân bố thực tế nhà máy)
    severity = str(params.get("severity", "")).lower()
    if severity not in _SEVERITY_PRESETS:
        severity = _random.choices(
            ["light", "medium", "heavy"],
            weights=[15, 60, 25],
        )[0]

    # count: 1–3 vết (hay gặp hơn 4–5)
    count_raw = params.get("count")
    count = int(count_raw) if count_raw is not None else _random.choices(
        [1, 2, 3, 4, 5],
        weights=[20, 35, 25, 12, 8],
    )[0]

    print(f"[scratch_napchai] seed={seed}  severity={severity}  count={count}")
    params["_resolved_severity"] = severity
    params["_resolved_count"]    = count
    params["_resolved_seed"]     = seed   # lưu seed thực tế dù caller không set

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cx, cy, radius = _detect_circle(gray)
    center     = (cx, cy)
    max_radius = int(radius * 1.3)

    polar_img  = _to_polar(img_rgb, center, max_radius)
    polar_gray = cv2.cvtColor(polar_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r_ring_col = _find_rim_col_outer(polar_gray)

    # Polar mask: user-drawn (base64) hoặc auto-band hẹp quanh rim
    user_mask_b64 = params.get("mask_b64") or params.get("user_mask_b64")
    polar_mask_raw = None

    if user_mask_b64:
        arr       = np.frombuffer(_b64.b64decode(user_mask_b64), np.uint8)
        user_mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if user_mask is not None:
            user_mask      = cv2.resize(user_mask, (img_rgb.shape[1], img_rgb.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            polar_mask_raw = _to_polar(user_mask, center, max_radius)

    if polar_mask_raw is None:
        # Auto-band hẹp (sigma ~4%) xung quanh outer rim
        col_idx        = np.arange(POLAR_W, dtype=np.float32)
        band_sigma     = max(POLAR_W * 0.04, 5.0)   # hẹp hơn phiên bản cũ (0.08→0.04)
        band           = np.exp(-0.5 * ((col_idx - r_ring_col) / band_sigma) ** 2)
        polar_mask_raw = (band[None, :] * 255).astype(np.uint8).repeat(POLAR_H, axis=0)

    if polar_mask_raw.ndim == 3:
        polar_mask_raw = polar_mask_raw[:, :, 0]

    polar_scratched, polar_mask_out = _synthesize_scratch_realistic(
        polar_img, polar_mask_raw, r_ring_col,
        severity=severity,
        count=count,
    )

    osize       = (img_rgb.shape[1], img_rgb.shape[0])
    cv_res      = _from_polar(polar_scratched, center, max_radius, osize)
    mask_result = _from_polar(polar_mask_out,  center, max_radius, osize)
    if mask_result.ndim == 3:
        mask_result = mask_result[:, :, 0]

    return cv_res, mask_result


# ── SDXL refine step — pipeline_scratch cell-10 ──────────────────────────────

def _sdxl_step(cv_res_rgb, mask_gray, ref_rgb, seed, prompt=None, negative_prompt=None):
    """
    Returns final_rgb (same HW as cv_res_rgb).
    ip_scale: 0.8 if ref provided, 0.5 if not.
    """
    import torch
    import gc

    orig_hw = (cv_res_rgb.shape[1], cv_res_rgb.shape[0])

    with get_lock():
        pipe      = get_pipe()
        depth_est = get_depth_est()

        gc.collect()
        torch.cuda.empty_cache()

        cv_low    = _PIL.fromarray(cv_res_rgb).resize(_TARGET)
        mask_low  = _PIL.fromarray(mask_gray).resize(_TARGET)
        depth_map = depth_est(cv_low)["depth"].convert("RGB").resize(_TARGET)

        if ref_rgb is not None:
            ip_image = _PIL.fromarray(ref_rgb).convert("RGB").resize((224, 224))
            ip_scale = 0.8
        else:
            ip_image = cv_low
            ip_scale = 0.5

        pipe.set_ip_adapter_scale(ip_scale)

        with torch.inference_mode():
            pipe.to("cuda" if __import__("torch").cuda.is_available() else "cpu")
            ai_res_low = pipe(
                prompt=prompt or _PROMPT,
                negative_prompt=negative_prompt or _NEG,
                image=cv_low,
                mask_image=mask_low,
                control_image=depth_map,
                ip_adapter_image=ip_image,
                strength=_STRENGTH,
                num_inference_steps=_STEPS,
                guidance_scale=_GUIDANCE,
            ).images[0]

        final = ai_res_low.resize(orig_hw, _PIL.LANCZOS)

        torch.cuda.empty_cache()
        gc.collect()

    return np.array(final.convert("RGB"))


# ── FLUX refine step (disabled) ──────────────────────────────────────────────

def _flux_step(cv_res_rgb, mask_gray, seed, prompt=None):
    """FLUX disabled — returns CV result unchanged."""
    print("[scratch_napchai] FLUX disabled, returning CV result.")
    return cv_res_rgb


# ── Public generate() ─────────────────────────────────────────────────────────

def generate(base_image_b64: str, params: dict, mask_b64: str | None = None) -> dict:
    """
    Generate one Scratch Napchai defect image.

    params keys:
      seed             int        (default 42)
      severity         str        "light" | "medium" | "heavy"  (default "medium")
      count            int        số vết xước chính 1–5  (default 2)
      sdxl_refine      bool       (default False)
      ref_image_b64    str        — base64 NG crop cho IP-Adapter
      mask_b64         str        — user-drawn mask (auto-band nếu không có)
    """
    if mask_b64:
        params["mask_b64"] = mask_b64
    img_rgb = decode_b64(base_image_b64)

    try:
        cv_res, mask_res = _cv_step(img_rgb, params)
    except Exception as e:
        return {"error": f"Scratch CV error: {e}"}

    _, buf  = cv2.imencode(".png", cv2.cvtColor(cv_res, cv2.COLOR_RGB2BGR))
    pre_b64 = _b64.b64encode(buf).decode()
    _, mbuf = cv2.imencode(".png", mask_res)
    mask_b64_out = _b64.b64encode(mbuf).decode()

    do_refine = params.get("use_sdxl") or params.get("sdxl_refine") or False
    use_flux  = params.get("use_flux", False)
    seed      = int(params.get("seed", 42))
    ref_b64   = params.get("ref_image_b64")

    if use_flux:
        try:
            print(f"[scratch_napchai] Using FLUX.1-dev refinement (seed={seed})")
            final_rgb  = _flux_step(cv_res, mask_res, seed, prompt=params.get("prompt"))
            result_b64 = encode_b64(final_rgb)
            engine     = "cv+flux"
        except Exception as e:
            print(f"[scratch_napchai] FLUX failed: {e} — returning CV result")
            result_b64 = encode_b64(cv_res)
            engine     = "cv"
    elif do_refine:
        ref_rgb = decode_b64(ref_b64) if ref_b64 else None
        try:
            final_rgb  = _sdxl_step(cv_res, mask_res, ref_rgb, seed,
                                     prompt=params.get("prompt"),
                                     negative_prompt=params.get("negative_prompt"))
            result_b64 = encode_b64(final_rgb)
            engine     = "cv+sdxl"
        except Exception as e:
            print(f"[scratch_napchai] SDXL failed: {e} — returning CV result")
            result_b64 = encode_b64(cv_res)
            engine     = "cv"
    else:
        result_b64 = encode_b64(cv_res)
        engine     = "cv"

    return {
        "result_image":      result_b64,
        "result_pre_refine": pre_b64,
        "mask_b64":          mask_b64_out,
        "engine":            engine,
        "metadata": {
            "defect_type": "scratch",
            "sdxl_refine": do_refine,
            "params": params,
            "resolved": {
                "seed":     params.get("_resolved_seed"),
                "severity": params.get("_resolved_severity"),
                "count":    params.get("_resolved_count"),
            },
        },
    }