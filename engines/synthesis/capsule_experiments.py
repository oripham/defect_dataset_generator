"""
Capsule defect synthesis — Thuốc_dài + Thuốc_tròn
==================================================
Chạy: python V:/defect_samples/capsule_experiments.py

Thuốc_dài: theo SYNTHESIS_PLAN.md (Rỗng, Thiếu_hàm_lượng) — detect mask + procedural synth.
Thuốc_tròn: chưa có mask/ref pairs — tổng hợp thủ công nhẹ trên vùng capsule (Lõm, Nứt).

Output: V:/defect_samples/results/<timestamp>/<Product>/<Defect>/...

ROI từ thumb: --rois V:/dataHondatPlus/rois --product none

Mặc định: --synth-style ng_residual khi có ref/ (1 ảnh NG vẫn dùng được).
  • Blend OK→NG mềm trong thân + feature injection (band/haze/…) để đa dạng.
  • Nếu có thư mục mask/ trong từng defect, ưu tiên mask đó (resize về kích thước OK).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime

import cv2
import random
import numpy as np
from PIL import Image

# ── Optional: import signal injection helpers từ HondaPlus scripts ─────────────
_HONDA_SCRIPTS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "HondaPlus",
                 "defect_dataset_generator", "scripts")
)
if os.path.isdir(_HONDA_SCRIPTS) and _HONDA_SCRIPTS not in sys.path:
    sys.path.insert(0, _HONDA_SCRIPTS)

try:
    from generator_classical import (
        extract_signal,
        normalize_signal,
        create_soft_mask,
        apply_signal_injection,
    )
    _HAS_SIGNAL_INJECT = True
    print("[capsule] signal_injection helpers loaded from HondaPlus scripts.")
except ImportError:
    _HAS_SIGNAL_INJECT = False
    print("[capsule] WARNING: generator_classical not found — --signal mode unavailable.")

ROOT = "V:/defect_samples"
_OVR = os.environ.get("DEFECT_SAMPLES_RESULT_DIR", "").strip()
RESULT_DIR = _OVR if _OVR else os.path.join(ROOT, "results", datetime.now().strftime("%Y%m%d_%H%M%S"))

# Hai root capsule (cùng một entry point)
CAPSULE_ROOTS = {
    "Thuốc_dài": os.path.join(ROOT, "Thuốc_dài"),
    "Thuốc_tròn": os.path.join(ROOT, "Thuốc_tròn"),
}

INTENSITIES = [0.5, 0.8]
N_SHAPES = 4
# Tối thiểu pixel trong mask/ (vùng lỗi) để tin dùng làm hint; nhỏ hơn → coi như không có, synth cả viên.
MIN_DEFECT_HINT_AREA = 40
# Nếu mask/ giao viên chiếm > tỉ lệ này của silhouette → coi là mask cả viên (legacy), không dùng làm “chỉ vùng lỗi”.
MAX_DEFECT_HINT_COVERAGE = 0.88
TH_PANEL = 320
PAD_CTX = 80

# Optional SDXL texture pass sau synth (SDXLRefiner từ HondaPlus defect_dataset_generator/scripts)
_capsule_sdxl_refiner: object | None = None  # None | False | SDXLRefiner
SDXL_REFINE_ENABLED = False
SDXL_REFINE_CFG: dict = {}
SDXL_REFINE_DEVICE = "cuda"

_CAPSULE_SDXL_PROMPT = (
    "pharmaceutical capsule on dark studio background, industrial inspection photograph, grayscale, "
    "natural surface texture, subtle noise grain, photorealistic"
)
_CAPSULE_SDXL_NEGATIVE = (
    "metal disc, machining grooves, cartoon, 3d render, CGI, oversharpened, text, watermark, "
    "airbrushed, oversaturated, blurry, low quality, "
    "salt and pepper noise, speckled highlights, harsh grain in bright areas, digital noise blobs"
)


def _highlight_despeckle_bgr(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 0.55,
    percentile: float = 89.0,
    floor_thr: float = 196.0,
    bilateral_d: int = 7,
    sigma_color: float = 58.0,
    sigma_space: float = 5.0,
) -> np.ndarray:
    """
    Giảm nhiễu hạt trong vùng cháy sáng (specular) trên viên: blend bilateral chỉ nơi gray cao trong mask.
    ``strength`` 0=tắt; ~0.5–0.7 thường đủ, quá cao có thể làm bóng quá “dẻo”.
    """
    st = float(np.clip(strength, 0.0, 1.0))
    if st <= 1e-6:
        return bgr
    m = mask > 127
    if int(np.count_nonzero(m)) < 40:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = float(np.percentile(gray[m], percentile))
    thr = max(thr, floor_thr)
    hot = (gray.astype(np.float32) >= thr) * m.astype(np.float32)
    if float(hot.max()) < 1e-3:
        return bgr
    hot = cv2.GaussianBlur(hot, (0, 0), 3.2)
    hot = np.clip(hot * st, 0.0, 1.0)
    d = int(bilateral_d) | 1  # must be odd
    if d < 3:
        d = 3
    smooth = cv2.bilateralFilter(bgr, d, float(sigma_color), float(sigma_space))
    hot3 = hot[:, :, np.newaxis]
    out = bgr.astype(np.float32) * (1.0 - hot3) + smooth.astype(np.float32) * hot3
    return np.clip(out, 0, 255).astype(np.uint8)


def _resolve_honda_dg_scripts() -> str | None:
    env = os.environ.get("HONDAPLUS_DG_SCRIPTS", "").strip()
    if env and os.path.isfile(os.path.join(env, "sdxl_refiner.py")):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    c1 = os.path.join(parent, "HondaPlus", "defect_dataset_generator", "scripts")
    if os.path.isfile(os.path.join(c1, "sdxl_refiner.py")):
        return c1
    c2 = os.path.normpath(os.path.join(here, "..", "defect_dataset_generator", "scripts"))
    if os.path.isfile(os.path.join(c2, "sdxl_refiner.py")):
        return c2
    return None


def _get_sdxl_refiner_singleton():
    global _capsule_sdxl_refiner
    if _capsule_sdxl_refiner is False:
        return None
    if _capsule_sdxl_refiner is not None:
        return _capsule_sdxl_refiner
    if not bool(globals().get("SDXL_REFINE_ENABLED", False)):
        return None
    scripts = _resolve_honda_dg_scripts()
    if not scripts:
        print(
            "[sdxl_refine] Không tìm thấy sdxl_refiner.py — đặt HONDAPLUS_DG_SCRIPTS "
            "hoặc đặt defect_samples cạnh V:\\HondaPlus. Bỏ qua refine."
        )
        _capsule_sdxl_refiner = False
        return None
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        from sdxl_refiner import SDXLRefiner

        cfg = dict(globals().get("SDXL_REFINE_CFG") or {})
        device = str(globals().get("SDXL_REFINE_DEVICE", "cuda"))
        ref = SDXLRefiner(cfg, device=device)
        _capsule_sdxl_refiner = ref
        print("[sdxl_refine] SDXLRefiner sẵn sàng (pass texture nhẹ sau synth).")
        return ref
    except Exception as e:
        print(f"[sdxl_refine] Không khởi tạo SDXLRefiner: {e}")
        _capsule_sdxl_refiner = False
        return None


def _apply_sdxl_refine_to_bgr(img_bgr: np.ndarray) -> np.ndarray | None:
    ref = _get_sdxl_refiner_singleton()
    if ref is None:
        return None
    try:
        from PIL import Image

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_in = Image.fromarray(rgb)
        pil_out = ref.refine_with_sdxl(pil_in)
        out = cv2.cvtColor(np.asarray(pil_out.convert("RGB")), cv2.COLOR_RGB2BGR)
        if out.shape[:2] != img_bgr.shape[:2]:
            out = cv2.resize(out, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        return out.astype(np.uint8)
    except Exception as e:
        print(f"[sdxl_refine] refine_with_sdxl lỗi: {e}")
        return None


def _apply_feature_injection_rong(
    base_bgr: np.ndarray,
    ok_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
    *,
    bands: bool,
    haze: bool,
    halo: bool,
    grain: bool,
    specular: bool,
    strength: float,
) -> np.ndarray:
    """
    Tăng đa dạng feature cho class Rỗng nhưng vẫn giữ hình học:
    - bands: band dọc theo trục capsule (X)
    - haze: blur nội vùng để tạo cảm giác translucent shell
    - halo: loang nhẹ quanh vùng band trong mask
    - grain: micro-grain/mottling nhẹ trong vùng lỗi
    """
    if strength <= 0:
        return base_bgr
    rng = np.random.default_rng(seed)
    out = base_bgr.astype(np.float32)
    ok = ok_bgr.astype(np.float32)
    m = (mask > 127).astype(np.float32)
    x, y, w, h = bbox
    H, W = out.shape[:2]

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = np.clip((xx - float(x)) / max(float(w), 1.0), 0.0, 1.0)
    yn = np.clip((yy - float(y)) / max(float(h), 1.0), 0.0, 1.0)

    # Keep micro-texture from OK so injected regions aren't flat.
    ok_g = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = ok_g - cv2.GaussianBlur(ok_g, (0, 0), 1.2)
    den = max(float(np.percentile(np.abs(hf[m > 0.5]), 95)), 1.0)
    hf = np.clip(hf / den, -1.0, 1.0)

    feat = np.zeros((H, W), np.float32)

    if bands:
        n_b = int(rng.integers(2, 5))
        for _ in range(n_b):
            cxn = float(rng.uniform(0.18, 0.88))
            sx = float(rng.uniform(0.028, 0.070))
            amp = float(rng.uniform(0.50, 1.00))
            feat += amp * np.exp(-0.5 * ((xn - cxn) / sx) ** 2)
        # make bands stronger near capsule centerline, weaker near top/bottom edges
        y_soft = np.exp(-0.5 * ((yn - 0.50) / float(rng.uniform(0.22, 0.34))) ** 2)
        feat *= (0.65 + 0.35 * y_soft)

    if halo:
        feat = 0.72 * feat + 0.50 * cv2.GaussianBlur(feat, (0, 0), float(rng.uniform(4.0, 7.0)))

    if haze:
        blur_in = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(2.0, 3.6)))
        haze_w = (0.12 + 0.10 * intensity) * np.clip(feat, 0.0, 1.0) * m
        haze_w = cv2.GaussianBlur(haze_w, (0, 0), 2.0)
        out = out * (1.0 - haze_w[:, :, None]) + blur_in * haze_w[:, :, None]

    # Normalize feature map inside mask
    feat = np.clip(feat, 0.0, None) * m
    fmax = float(feat.max())
    if fmax > 1e-6:
        feat = feat / fmax

    # Darken along feature map (Rỗng look)
    dark = float(rng.uniform(0.05, 0.14)) * (0.78 + 0.32 * intensity) * float(np.clip(strength, 0.0, 2.5))
    out *= (1.0 - dark * np.power(feat, 0.80))[:, :, None]

    # Giảm HF/grain trên pixel OK đã sáng → tránh nhiễu hạt chồng lên vùng chá sáng.
    ok_bright = np.clip((ok_g.astype(np.float32) - 152.0) / 98.0, 0.0, 1.0)
    damp_hi = (1.0 - 0.82 * ok_bright)[:, :, np.newaxis]

    # Micro texture preservation
    out += (
        (3.5 + 2.5 * intensity)
        * hf[:, :, None]
        * np.power(feat, 0.92)[:, :, None]
        * damp_hi
    )

    if grain:
        gsig = float(rng.uniform(1.2, 3.2)) * (0.85 + 0.30 * intensity) * float(np.clip(strength, 0.0, 2.5))
        g = rng.normal(0.0, gsig, (H, W, 3)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.55)
        out += g * np.power(feat, 0.85)[:, :, None] * damp_hi

    # Avoid pure black
    floor_dn = float(rng.uniform(16.0, 34.0))
    out = np.maximum(out, floor_dn * feat[:, :, None] * m[:, :, None])

    # Specular variation: jitter highlight band so the shell doesn't look identically glossy.
    if specular:
        ok_g2 = ok_g.copy()
        # highlight candidates: top bright pixels inside capsule
        hi_thr = float(np.percentile(ok_g2[m > 0.5], 88.0 + float(rng.uniform(0, 6.0))))
        hi = (ok_g2 >= hi_thr).astype(np.float32) * m
        hi = cv2.GaussianBlur(hi, (0, 0), float(rng.uniform(1.0, 2.2)))
        # random shift highlight (gốc sáng) + gain (độ bóng)
        dy = int(rng.integers(-8, 9))
        dx = int(rng.integers(-14, 15))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        hi_s = cv2.warpAffine(hi, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        hi_s = cv2.GaussianBlur(hi_s, (0, 0), float(rng.uniform(0.9, 2.2)))
        amp = float(rng.uniform(-26.0, 42.0)) * float(np.clip(strength, 0.0, 2.5))
        # apply only where feature exists a bit to avoid changing whole capsule equally
        w_hi = np.power(np.clip(0.20 + 0.80 * feat, 0.0, 1.0), 0.65) * m
        out += amp * hi_s[:, :, None] * w_hi[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8)


def _randomize_shell_lighting(
    bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Đa dạng hoá **gốc sáng** (gradient diffuse theo hướng ngẫu nhiên) và **độ bóng**
    (1–3 lobe Gaussian specular, độ rộng/ biên độ ngẫu nhiên) + gamma nhẹ trên kênh L (LAB),
    chỉ trong ``mask`` (viên), nền giữ nguyên.

    ``strength`` ∈ [0,1] tắt dần hiệu ứng (0 = trả ảnh gốc).
    """
    if strength <= 1e-6:
        return bgr
    st = float(np.clip(strength, 0.0, 2.0))
    rng = np.random.default_rng(int(seed) % (2**31))
    H, W = bgr.shape[:2]
    m = (mask > 127).astype(np.float32)
    if float(m.sum()) < 30:
        return bgr

    x, y, bw, bh = bbox
    cx = float(x + 0.5 * bw)
    cy = float(y + 0.5 * bh)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = (xx - cx) / max(float(bw), 1.0)
    v = (yy - cy) / max(float(bh), 1.0)

    # --- Gamma / exposure on L (inside mask only) ---
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    gamma = float(rng.uniform(0.88, 1.14))
    gain_l = float(rng.uniform(0.92, 1.10))
    l_new = np.clip((l_ch / 255.0) ** gamma * gain_l * 255.0, 0, 255)
    blend_l = m * st
    l_out = l_ch * (1.0 - blend_l) + l_new * blend_l
    lab2 = lab.copy()
    lab2[:, :, 0] = np.clip(l_out, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR).astype(np.float32)

    # --- Directional diffuse (2D light angle) ---
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    lx, ly = float(np.cos(ang)), float(np.sin(ang))
    dot = u * lx + v * ly
    sel = m > 0.5
    if np.count_nonzero(sel) > 20:
        dvals = dot[sel]
        d0 = float(np.percentile(dvals, 4.0))
        d1 = float(np.percentile(dvals, 96.0))
        shade01 = (dot - d0) / max(d1 - d0, 1e-6)
        shade01 = np.clip(shade01, 0.0, 1.0)
    else:
        shade01 = np.ones((H, W), np.float32)
    diff_lo = float(rng.uniform(0.78, 0.94))
    diff_hi = float(rng.uniform(1.04, 1.22))
    shading = diff_lo + (diff_hi - diff_lo) * shade01
    shading = np.clip(shading, 0.65, 1.35)
    sh3 = (1.0 + (shading - 1.0) * (m * st))[:, :, np.newaxis]
    img = np.clip(img * sh3, 0, 255)

    # --- Specular lobes (gloss): sharp vs soft ---
    nlobs = int(rng.integers(1, 4))
    spec = np.zeros((H, W), np.float32)
    gloss = float(rng.uniform(0.45, 1.0)) * st
    for _ in range(nlobs):
        px = float(rng.uniform(x + 0.06 * bw, x + 0.94 * bw))
        py = float(rng.uniform(y + 0.12 * bh, y + 0.88 * bh))
        sig_x = float(rng.uniform(1.1, 5.5))
        sig_y = float(rng.uniform(0.9, 4.2))
        amp = float(rng.uniform(10.0, 52.0)) * gloss * (0.7 + 0.6 * st)
        spec += amp * np.exp(
            -0.5 * (((xx - px) / max(sig_x, 0.5)) ** 2 + ((yy - py) / max(sig_y, 0.5)) ** 2)
        )
    spec = cv2.GaussianBlur(spec, (0, 0), float(rng.uniform(0.35, 1.35)))
    spec *= m * st
    img = np.clip(img + spec[:, :, np.newaxis], 0, 255)

    return img.astype(np.uint8)


def load_ok(defect_path: str) -> np.ndarray | None:
    ok_dir = os.path.join(defect_path, "ok")
    imgs = sorted(glob.glob(os.path.join(ok_dir, "*")))
    imgs = [p for p in imgs if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if not imgs:
        return None
    return cv2.imread(imgs[0])


def load_first_ref(defect_path: str) -> np.ndarray | None:
    """Ảnh NG / tham chiếu trong defect_path/ref/ (bỏ file tên mask)."""
    ref_dir = os.path.join(defect_path, "ref")
    imgs = sorted(glob.glob(os.path.join(ref_dir, "*")))
    imgs = [
        p
        for p in imgs
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        and "mask" not in os.path.basename(p).lower()
    ]
    if not imgs:
        return None
    return cv2.imread(imgs[0])


def load_optional_disk_mask(defect_path: str, hw: tuple[int, int]) -> np.ndarray | None:
    """
    ``mask/*.png|jpg`` đầu tiên — resize về (H,W) của OK.

    Semantics: vùng **chỉ báo vị trí lỗi** trên sản phẩm (không phải silhouette cả viên).
    Giao với mask sản phẩm tự động rồi mới dùng để sinh lỗi.
    """
    mdir = os.path.join(defect_path, "mask")
    if not os.path.isdir(mdir):
        return None
    imgs = sorted(glob.glob(os.path.join(mdir, "*")))
    imgs = [p for p in imgs if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if not imgs:
        return None
    m = cv2.imread(imgs[0], cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    h, w = hw
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def detect_capsule_mask(ok_bgr: np.ndarray, fallback_xywh: tuple[int, int, int, int] | None = None):
    """
    Threshold → largest contour → filled mask + bbox (x, y, w, h).
    """
    gray = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        if fallback_xywh is None:
            h, w = gray.shape
            return np.zeros_like(gray), (0, 0, w, h)
        x, y, w, h = fallback_xywh
        mask = np.zeros_like(gray)
        mask[y : y + h, x : x + w] = 255
        return mask, (x, y, w, h)
    cnt = max(cnts, key=cv2.contourArea)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(cnt)
    return mask, (x, y, w, h)


def detect_capsule_mask_robust(ok_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Giống detect_capsule_mask; nếu mask quá nhỏ thử Otsu (ảnh ROI thumb có thể khác ngưỡng 60)."""
    m, bb = detect_capsule_mask(ok_bgr, fallback_xywh=None)
    if int((m > 127).sum()) > 80:
        return m, bb
    gray = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m, bb
    cnt = max(cnts, key=cv2.contourArea)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(cnt)
    return mask, (x, y, w, h)


def detect_capsule_mask_hough(
    ok_bgr: np.ndarray,
    fallback_xywh: tuple[int, int, int, int] | None = None,
    *,
    thresh_val: int = 60,
    canny_lo: int = 35,
    canny_hi: int = 120,
    hough_thresh: int = 22,
    min_line_frac: float = 0.12,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Canny (trong vùng dilate từ threshold) + HoughLinesP: tìm cặp cạnh gần song song
    (thuốc dài nằm ngang). Giao với threshold để có mask giống viên, không chỉ HCN.

    Fallback: ``detect_capsule_mask`` nếu không đủ đường thẳng ngang.
    """
    gray = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    _, th = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return detect_capsule_mask(ok_bgr, fallback_xywh=fallback_xywh)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_lo, canny_hi)
    roi = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=1)
    edges = cv2.bitwise_and(edges, roi)

    min_len = max(25, int(min(H, W) * min_line_frac))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_len,
        maxLineGap=14,
    )
    if lines is None or len(lines) < 4:
        return detect_capsule_mask(ok_bgr, fallback_xywh=fallback_xywh)

    horiz: list[tuple[float, float, float, float]] = []
    for seg in lines[:, 0]:
        x1, y1, x2, y2 = [float(v) for v in seg]
        dx, dy = x2 - x1, y2 - y1
        ln = (dx * dx + dy * dy) ** 0.5
        if ln < 1e-3:
            continue
        ang = np.degrees(np.arctan2(abs(dy), abs(dx)))
        if ang <= 22.0:
            horiz.append((x1, y1, x2, y2))

    if len(horiz) < 4:
        return detect_capsule_mask(ok_bgr, fallback_xywh=fallback_xywh)

    mids_y = [0.5 * (y1 + y2) for x1, y1, x2, y2 in horiz]
    y_med = float(np.median(mids_y))
    top = [h for h, my in zip(horiz, mids_y) if my < y_med]
    bot = [h for h, my in zip(horiz, mids_y) if my >= y_med]
    if len(top) < 2 or len(bot) < 2:
        return detect_capsule_mask(ok_bgr, fallback_xywh=fallback_xywh)

    def _xspan(segs: list[tuple[float, float, float, float]]) -> tuple[float, float]:
        xs: list[float] = []
        for x1, y1, x2, y2 in segs:
            xs.extend([x1, x2])
        return float(min(xs)), float(max(xs))

    y_top = float(np.median([0.5 * (y1 + y2) for x1, y1, x2, y2 in top]))
    y_bot = float(np.median([0.5 * (y1 + y2) for x1, y1, x2, y2 in bot]))
    x0t, x1t = _xspan(top)
    x0b, x1b = _xspan(bot)
    x0 = min(x0t, x0b)
    x1 = max(x1t, x1b)

    pad_x = max(3, int(0.02 * W))
    pad_y = max(2, int(0.02 * H))
    if y_top >= y_bot:
        return detect_capsule_mask(ok_bgr, fallback_xywh=fallback_xywh)

    strip = np.zeros((H, W), np.uint8)
    xi0 = int(np.clip(x0 - pad_x, 0, W - 1))
    xi1 = int(np.clip(x1 + pad_x, 0, W - 1))
    yt0 = int(np.clip(y_top - pad_y, 0, H - 1))
    yb1 = int(np.clip(y_bot + pad_y, 0, H - 1))
    cv2.rectangle(strip, (xi0, yt0), (xi1, yb1), 255, -1)

    mask = cv2.bitwise_and(th, strip)
    if int((mask > 127).sum()) < 80:
        return detect_capsule_mask(ok_bgr, fallback_xywh=fallback_xywh)

    cnts2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts2:
        return detect_capsule_mask(ok_bgr, fallback_xywh=fallback_xywh)
    cnt2 = max(cnts2, key=cv2.contourArea)
    mask_fill = np.zeros((H, W), np.uint8)
    cv2.drawContours(mask_fill, [cnt2], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(cnt2)
    return mask_fill, (x, y, w, h)


def detect_capsule_mask_min_area_rect(
    ok_bgr: np.ndarray,
    fallback_xywh: tuple[int, int, int, int] | None = None,
    thresh_val: int = 60,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Threshold → contour lớn nhất → ``minAreaRect`` → fill đa giác xoay.
    Góc cạnh nhọn hơn silhouette thật; hữu ích khi cần bbox ổn định theo biên.
    """
    gray = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return detect_capsule_mask(ok_bgr, fallback_xywh=fallback_xywh)
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int32(np.round(box))
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [box], 255)
    x, y, w, h = cv2.boundingRect(box)
    return mask, (x, y, w, h)


def auto_product_mask_and_synth_mask(
    ok_bgr: np.ndarray,
    defect_path: str,
    mode: str,
    fallback_xywh: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], bool]:
    """
    Trả về (mask_sản_phẩm, mask_dùng_để_sinh_lỗi, bbox, có_dùng_hint_từ_mask/).

    - mask sản phẩm: threshold / min_rect / hough / robust theo ``mode``.
    - ``mask/`` trên đĩa = vùng lỗi → giao với sản phẩm, dilate nhẹ → chỉ trong đó mới blend lỗi.
    - Không có ``mask/`` hoặc quá nhỏ → sinh lỗi trên toàn vùng sản phẩm (hành vi cũ trên viên).
    """
    if mode == "min_rect":
        mask_p, bb = detect_capsule_mask_min_area_rect(ok_bgr, fallback_xywh=fallback_xywh)
    elif mode == "hough":
        mask_p, bb = detect_capsule_mask_hough(ok_bgr, fallback_xywh=fallback_xywh)
    elif mode == "robust":
        mask_p, bb = detect_capsule_mask_robust(ok_bgr)
    else:
        mask_p, bb = detect_capsule_mask(ok_bgr, fallback_xywh=fallback_xywh)

    h, w = ok_bgr.shape[:2]
    disk = load_optional_disk_mask(defect_path, (h, w))
    used_hint = False
    mask_s = mask_p.copy()
    if disk is not None:
        dh = cv2.bitwise_and(disk, mask_p)
        da = int((dh > 127).sum())
        pa = max(int((mask_p > 127).sum()), 1)
        frac = da / float(pa)
        if da >= MIN_DEFECT_HINT_AREA and frac <= MAX_DEFECT_HINT_COVERAGE:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask_s = cv2.dilate(dh, k, iterations=1)
            mask_s = cv2.bitwise_and(mask_s, mask_p)
            used_hint = True
        # frac > MAX_*: mask/ gần như cả viên → chỉ dùng ``mask_p`` để sinh lỗi (đúng nghĩa sản phẩm).
    return mask_p, mask_s, bb, used_hint


def _intensity_scale(intensity: float, low: float, high: float) -> float:
    return float(low + (high - low) * (intensity - 0.5) / 0.3)


def synth_rong(
    ok_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
    fixed_region: bool = False,
) -> np.ndarray:
    """
    Rỗng v2 — Optical model dựa trên vật lý thật:
      - Vỏ gelatin TRONG SUỐT → ánh sáng xuyên qua thấy background đen bên trong
      - Interior: UNIFORM DARK (gần đen, không gradient đều)
      - Rim trên/dưới: SPECULAR HIGHLIGHT sáng (vỏ phản chiếu ánh sáng)
      - 2 đầu tròn capsule: bảo vệ sáng như OK (cap region)
      - Sharp edge giữa rim sáng và interior tối

    Ref thật:  [sáng rim] [sharp edge] [tối gần đen interior] [sharp edge] [sáng rim]
    """
    rng = np.random.default_rng(seed)
    lit = float(globals().get("LIGHTING_AUG_STRENGTH", 1.0))
    ok_u = (
        _randomize_shell_lighting(ok_bgr, mask, bbox, int(seed) + 185_241, strength=lit)
        if lit > 1e-6
        else ok_bgr
    )
    x, y, w, h = bbox
    H, W = ok_bgr.shape[:2]
    ok  = ok_u.astype(np.float32)
    m   = (mask > 127).astype(np.float32)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = np.clip((xx - float(x)) / max(float(w), 1.0), 0.0, 1.0)
    yn = np.clip((yy - float(y)) / max(float(h), 1.0), 0.0, 1.0)

    # ── 1. Distance transform → normalized dist từ biên mask ─────────────────
    m_u8  = (m > 0.5).astype(np.uint8)
    dist  = cv2.distanceTransform(m_u8, cv2.DIST_L2, 3).astype(np.float32)
    dmax  = float(dist.max())
    dist_n = (dist / dmax) if dmax > 1e-6 else m.copy()  # 0=rim, 1=center

    # ── 2. Cap protection: 2 đầu tròn giữ sáng như OK ────────────────────────
    # sin(π·xn) = 0 ở 2 đầu, 1 ở giữa → axial dùng để fade darkening tại cap
    cap_protect = np.power(np.clip(np.sin(np.pi * xn), 0.0, 1.0), 1.2)

    # ── 3. Full capsule darkening — toàn bộ mask bị tối (kể cả outer rim) ───────
    # Chỉ bảo vệ 2 đầu tròn (cap) theo cap_protect
    # interior = toàn bộ vùng mask * cap_protect (không có rim_thresh bảo vệ outer edge)
    interior = m * cap_protect   # 1.0 = toàn bộ body viên, 0.0 = cap + ngoài mask

    # ── 4. Noise thấp tần: tránh flat digital
    tex = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), 3.0)
    tex = (tex - tex.min()) / max(float(tex.max() - tex.min()), 1e-6)
    interior_noisy = interior * (0.92 + 0.16 * tex)

    # ── 5. Darkening toàn capsule — gần đen ──────────────────────────────────
    floor_mul = float(rng.uniform(0.02, 0.05)) if intensity >= 0.65 else float(rng.uniform(0.04, 0.10))
    depth     = float(rng.uniform(0.95, 1.00))
    mult      = 1.0 - (1.0 - floor_mul) * interior_noisy * depth
    mult      = np.clip(mult, floor_mul, 1.0)
    out       = ok * mult[:, :, np.newaxis]

    # ── 6. Specular highlight — đường sáng bóng bên trong viền capsule ────────
    # Đặt specular tại dist_n ~ 0.10-0.20 (gần outer edge, bên trong capsule)
    spec_pos      = float(rng.uniform(0.10, 0.20))
    spec_sigma    = float(rng.uniform(0.04, 0.07))
    specular_zone = np.exp(-0.5 * ((dist_n - spec_pos) / spec_sigma) ** 2)
    specular_zone = specular_zone * cap_protect * m
    ok_gray       = cv2.cvtColor(ok_u, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_bright  = ok_gray / max(float(ok_gray[m > 0.5].mean()), 1.0)
    spec_strength = float(rng.uniform(30.0, 55.0)) * (0.8 + 0.4 * intensity)
    out += spec_strength * specular_zone[:, :, None] * local_bright[:, :, None]

    # ── 7. Grain trong void ───────────────────────────────────────────────────
    g_sig = float(rng.uniform(0.8, 2.0))
    grain = rng.normal(0.0, g_sig, (H, W, 3)).astype(np.float32)
    out  += grain * interior[:, :, None]

    out = np.clip(out, 0, 255).astype(np.uint8)

    # Blur nhẹ để tránh aliasing tại sharp edge
    bsig = float(rng.uniform(0.4, 0.7))
    out  = cv2.merge([cv2.GaussianBlur(out[:, :, c], (0, 0), bsig) for c in range(3)])
    return out


def auto_check_rong(ok_bgr: np.ndarray, result_bgr: np.ndarray,
                    mask: np.ndarray, bbox: tuple | None = None) -> dict:
    """
    Tự động kiểm tra chất lượng output Rỗng dựa trên optical model.

    Zones (normalized dist từ capsule edge):
      specular_rim : dist_n 0.25-0.45  → vùng sáng bóng gelatin (specular highlight)
      dark_interior: dist_n > 0.55     → interior tối (void)
      cap_exclude  : axial_fade < 0.4  → 2 đầu tròn không tính (phải sáng như OK)

    Metrics:
      interior_dark  : res_int / ok_int        target < 0.40 (tối hơn OK)
      rim_bright     : res_rim / ok_rim        target > 0.85 (giữ sáng)
      contrast_ratio : res_rim / res_int       target > 2.0
      specular_boost : res_rim / ok_rim - 1    target > 0.05 (rim sáng hơn OK do specular)
      verdict        : PASS / WARN / FAIL
    """
    m    = (mask > 127).astype(np.uint8)
    dist = cv2.distanceTransform(m.copy(), cv2.DIST_L2, 3).astype(np.float32)
    dmax = float(dist.max()) + 1e-6
    dist_n = dist / dmax

    H, W = mask.shape[:2]
    ok_g  = cv2.cvtColor(ok_bgr,     cv2.COLOR_BGR2GRAY).astype(np.float32)
    res_g = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Axial fade để exclude cap region (2 đầu tròn)
    if bbox is not None:
        x, y, w, h = bbox
        xx = np.clip((np.arange(W).reshape(1, W) - float(x)) / max(float(w), 1), 0, 1)
        axial = np.power(np.clip(np.sin(np.pi * xx), 0, 1), 1.2)
        axial = np.broadcast_to(axial, (H, W))
        cap_ok = (axial > 0.4).astype(np.float32)   # exclude 2 đầu tròn
    else:
        cap_ok = np.ones((H, W), np.float32)

    # Zones — khớp với rim_thresh mới (0.18-0.28)
    # spec_rim: outer rim zone (dist_n 0.05-0.22) — vùng gelatin sáng bóng
    # dark_int: interior (dist_n > 0.35) — void bên trong
    spec_rim_mask = ((dist_n > 0.05) & (dist_n < 0.22) & (m > 0)).astype(np.float32) * cap_ok
    dark_int_mask = ((dist_n > 0.35) & (m > 0)).astype(np.float32) * cap_ok

    def mean_m(img, msk):
        s = float(msk.sum())
        return float((img * msk).sum() / s) if s > 1 else float(img[m > 0].mean())

    ok_rim   = mean_m(ok_g,  spec_rim_mask)
    res_rim  = mean_m(res_g, spec_rim_mask)
    ok_int   = mean_m(ok_g,  dark_int_mask)
    res_int  = mean_m(res_g, dark_int_mask)

    interior_dark  = res_int  / max(ok_int,  1.0)   # target < 0.40
    rim_bright     = res_rim  / max(ok_rim,  1.0)   # target > 0.85
    contrast_ratio = res_rim  / max(res_int, 1.0)   # target > 2.0
    specular_boost = rim_bright - 1.0               # target > 0.05

    if interior_dark < 0.25 and rim_bright > 0.80 and contrast_ratio > 3.0:
        verdict = "PASS ✅"
    elif interior_dark < 0.40 and contrast_ratio > 2.0:
        verdict = "WARN ⚠️"
    else:
        verdict = "FAIL ❌"

    return {
        "interior_dark":  round(interior_dark,  3),
        "rim_bright":     round(rim_bright,     3),
        "contrast_ratio": round(contrast_ratio, 3),
        "specular_boost": round(specular_boost, 3),
        "ok_rim":  round(ok_rim,  1),
        "res_rim": round(res_rim, 1),
        "ok_int":  round(ok_int,  1),
        "res_int": round(res_int, 1),
        "verdict": verdict,
    }


def synth_thieu_ham_luong(
    ok_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
    fixed_region: bool = False,
) -> np.ndarray:
    """
    Thiếu hàm lượng — biểu hiện chủ yếu là các pixel hơi tối, rải cục bộ trong thân viên,
    không làm tối cả viên thuốc.

    fixed_region=True: giữ phân bố không gian tương đối ổn định; seed vẫn đổi pixel-level detail.
    """
    rng = np.random.default_rng(seed)
    lit = float(globals().get("LIGHTING_AUG_STRENGTH", 1.0))
    ok_u = (
        _randomize_shell_lighting(ok_bgr, mask, bbox, int(seed) + 293_771, strength=lit)
        if lit > 1e-6
        else ok_bgr
    )
    x, y, w, h = bbox
    H, W = ok_bgr.shape[:2]
    m = (mask > 127).astype(np.float32)
    out = ok_u.astype(np.float32)

    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return ok_u.copy()
    xnf = (xs.astype(np.float32) - float(x)) / max(float(w), 1.0)
    xnf = np.clip(xnf, 0.0, 1.0)

    # Chỉ tập trung về một đầu viên (trái hoặc phải), không phủ cả hai đầu.
    end_right = True if fixed_region else bool(rng.random() > 0.5)
    if end_right:
        sel = np.where(xnf >= 0.62)[0]
        if len(sel) < 30:
            sel = np.where(xnf >= 0.55)[0]
    else:
        sel = np.where(xnf <= 0.38)[0]
        if len(sel) < 30:
            sel = np.where(xnf <= 0.45)[0]
    if len(sel) < 20:
        sel = np.arange(len(xs))

    # Region map: ít cụm hơn, hình ngẫu nhiên (không tròn), hơi theo dạng gân nhẹ.
    vein = np.zeros((H, W), np.float32)
    n_regions = int(3 + 2 * intensity) if fixed_region else int(rng.integers(3, 5) + 2 * intensity)
    min_start_dist = max(11.0, 0.16 * float(w))
    starts: list[tuple[float, float]] = []

    made = 0
    tries = 0
    max_tries = max(120, n_regions * 35)
    while made < n_regions and tries < max_tries:
        tries += 1
        idx = int(sel[int(rng.integers(0, len(sel)))])
        cx = float(xs[idx])
        cy = float(ys[idx])
        if any((cx - sx0) * (cx - sx0) + (cy - sy0) * (cy - sy0) < (min_start_dist * min_start_dist) for sx0, sy0 in starts):
            continue
        starts.append((cx, cy))

        # Tạo shape méo bằng polygon trong hệ trục ellipse kéo dài nhẹ.
        major = float(rng.uniform(6.0, 10.5))
        minor = float(rng.uniform(3.2, 5.4))
        ang = float(rng.uniform(0.0, np.pi))
        c, s = np.cos(ang), np.sin(ang)
        amp = float(rng.uniform(0.85, 1.20))
        n_poly = int(rng.integers(9, 14))
        poly = []
        phi0 = float(rng.uniform(0.0, 2.0 * np.pi))
        for k in range(n_poly):
            t = phi0 + (2.0 * np.pi * k / n_poly) + float(rng.uniform(-0.18, 0.18))
            wav = 1.0 + 0.25 * np.sin(2.0 * t + float(rng.uniform(-0.5, 0.5)))
            jitter = float(rng.uniform(0.78, 1.24))
            r_major = major * wav * jitter
            r_minor = minor * (0.82 + 0.30 * jitter)
            ex = r_major * np.cos(t)
            ey = r_minor * np.sin(t)
            px = cx + ex * c - ey * s
            py = cy + ex * s + ey * c
            px = float(np.clip(px, x + 1, x + w - 2))
            py = float(np.clip(py, y + 1, y + h - 2))
            poly.append((int(px), int(py)))
        if len(poly) >= 3:
            cv2.fillPoly(vein, [np.array(poly, dtype=np.int32)], amp)

        # Nhánh ngắn để có cảm giác gân nhẹ.
        if rng.random() < 0.30:
            tx = int(np.clip(cx + major * c * float(rng.uniform(0.25, 0.55)), x + 1, x + w - 2))
            ty = int(np.clip(cy + major * s * float(rng.uniform(0.25, 0.55)), y + 1, y + h - 2))
            nx = int(np.clip(tx + float(rng.uniform(3.5, 7.0)) * c, x + 1, x + w - 2))
            ny = int(np.clip(ty + float(rng.uniform(3.0, 6.5)) * s, y + 1, y + h - 2))
            cv2.line(vein, (tx, ty), (nx, ny), amp * 0.72, 3, cv2.LINE_AA)
        made += 1

    vein *= m
    vein = cv2.GaussianBlur(vein, (0, 0), 1.35 if intensity < 0.65 else 1.60)
    # Tạo độ đậm không đều theo vùng.
    tex = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), 2.2)
    tex = (tex - tex.min()) / max(float(tex.max() - tex.min()), 1e-6)
    vein *= (0.76 + 0.52 * tex) * m
    vmax = float(vein.max())
    if vmax > 1e-6:
        vein /= vmax

    # Tạo halo lan cận vùng lỗi (vẫn nằm trong mask) và chỉ ưu tiên một đầu viên.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = np.clip((xx - float(x)) / max(float(w), 1.0), 0.0, 1.0)
    if end_right:
        side_gate = np.clip((xn - 0.50) / 0.24, 0.0, 1.0)
    else:
        side_gate = np.clip((0.50 - xn) / 0.24, 0.0, 1.0)
    side_gate = cv2.GaussianBlur(side_gate * m, (0, 0), 2.2)

    halo = cv2.GaussianBlur(vein, (0, 0), 3.8 if intensity < 0.65 else 4.6) * side_gate
    defect_map = np.clip(0.78 * vein + 0.62 * halo, 0.0, 1.0) * m
    dmax = float(defect_map.max())
    if dmax > 1e-6:
        defect_map /= dmax

    # Giữ feature/texture của ảnh OK trong vùng xám: điều biến theo high-frequency.
    ok_g = cv2.cvtColor(ok_u, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ok_blur = cv2.GaussianBlur(ok_g, (0, 0), 1.2)
    hf = ok_g - ok_blur
    hf_den = max(float(np.percentile(np.abs(hf[m > 0.5]), 95)), 1.0)
    hf = np.clip(hf / hf_den, -1.0, 1.0)

    # Chỉ tối theo các vùng, phần giữa vùng đậm hơn và có loang ra lân cận.
    core_emphasis = np.power(defect_map, 0.72)
    delta = float(rng.uniform(0.10, 0.16)) if intensity < 0.65 else float(rng.uniform(0.15, 0.24))
    scale = 1.0 - delta * core_emphasis
    scale = np.clip(scale, 0.72, 1.0)
    # Chỗ có texture cao giữ lại sáng/tối vi mô để không thành mảng xám phẳng.
    feat_mod = 1.0 + (0.14 * hf * np.power(defect_map, 0.88))
    feat_mod = np.clip(feat_mod, 0.88, 1.16)
    out *= scale[:, :, None]
    out *= feat_mod[:, :, None]

    # Grain rất nhẹ bám quanh gân.
    g_sig = float(rng.uniform(1.2, 2.8)) if intensity < 0.65 else float(rng.uniform(1.8, 3.8))
    grain = rng.normal(0.0, g_sig, (H, W, 3)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.5)
    out += grain * np.power(defect_map, 0.92)[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8)


def capsule_body_mask(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    x_margin_lo: float,
    x_margin_hi: float,
) -> np.ndarray:
    """Vùng thân viên (bỏ hai đầu cong theo trục X), giao với mask capsule."""
    x, y, w, h = bbox
    H, W = mask.shape[:2]
    body = np.zeros((H, W), dtype=np.uint8)
    x0 = int(np.clip(x + w * x_margin_lo, 0, W - 1))
    x1 = int(np.clip(x + w * x_margin_hi, x0 + 1, W))
    y0 = int(np.clip(y, 0, H - 1))
    y1 = int(np.clip(y + h, y0 + 1, H))
    body[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(body, mask)


def _warp_ng_ecc_to_ok(
    ok_bgr: np.ndarray,
    ng_rs: np.ndarray,
    mask_u8: np.ndarray,
) -> np.ndarray:
    """
    Căn NG → OK bằng ECC (rotation + translation), chỉ tin khi correlation đủ cao và góc nhỏ.
    """
    H, W = ok_bgr.shape[:2]
    ok_g = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ng_g = cv2.cvtColor(ng_rs, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
    max_deg = float(globals().get("NG_ALIGN_ECC_MAX_DEG", 8.0))
    min_corr = float(globals().get("NG_ALIGN_ECC_MIN_CORR", 0.2))
    try:
        r = cv2.findTransformECC(
            ok_g,
            ng_g,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            mask_u8,
            5,
        )
    except cv2.error:
        return ng_rs
    if isinstance(r, tuple):
        corr = float(r[0])
    elif r is None:
        return ng_rs
    else:
        corr = float(r)
    if not np.isfinite(corr) or corr < min_corr:
        return ng_rs
    if not np.all(np.isfinite(warp)):
        return ng_rs
    a = float(np.arctan2(warp[1, 0], warp[0, 0]))
    if abs(np.degrees(a)) > max_deg:
        return ng_rs
    return cv2.warpAffine(
        ng_rs,
        warp,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _warp_ng_shift_to_ok(
    ok_bgr: np.ndarray,
    ng_rs: np.ndarray,
    mask_u8: np.ndarray,
) -> np.ndarray:
    """
    Căn NG → OK chỉ bằng tịnh tiến (dx,dy), không xoay.

    Dùng ECC MOTION_TRANSLATION trong vùng mask. Ổn khi camera/part không xoay,
    chỉ lệch vị trí nhẹ (crop/ROI khác).
    """
    H, W = ok_bgr.shape[:2]
    ok_g = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ng_g = cv2.cvtColor(ng_rs, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 90, 1e-6)
    min_corr = float(globals().get("NG_ALIGN_SHIFT_MIN_CORR", 0.15))
    max_shift = float(globals().get("NG_ALIGN_SHIFT_MAX_PX", 35.0))
    try:
        r = cv2.findTransformECC(
            ok_g,
            ng_g,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            mask_u8,
            5,
        )
    except cv2.error:
        return ng_rs
    if isinstance(r, tuple):
        corr = float(r[0])
    elif r is None:
        return ng_rs
    else:
        corr = float(r)
    if not np.isfinite(corr) or corr < min_corr:
        return ng_rs
    if not np.all(np.isfinite(warp)):
        return ng_rs
    dx = float(warp[0, 2])
    dy = float(warp[1, 2])
    if abs(dx) > max_shift or abs(dy) > max_shift:
        return ng_rs
    return cv2.warpAffine(
        ng_rs,
        warp,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _warp_ng_rotation_grid_to_ok(
    ok_bgr: np.ndarray,
    ng_rs: np.ndarray,
    mask_u8: np.ndarray,
) -> np.ndarray:
    """
    Thử nhiều góc xoay quanh tâm khối mask; chọn góc có MSE(OK−NG) nhỏ nhất trong mask.
    Hữu ích khi ECC kém ổn định (sáng tốt khác nhiều).
    """
    m = mask_u8 > 127
    npx = int(np.count_nonzero(m))
    if npx < 120:
        return ng_rs
    ok_g = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ng_g = cv2.cvtColor(ng_rs, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ys, xs = np.where(m)
    cx = float(xs.mean())
    cy = float(ys.mean())
    H, W = ok_g.shape
    max_deg = float(globals().get("NG_ALIGN_GRID_MAX_DEG", 3.0))
    steps = int(globals().get("NG_ALIGN_GRID_STEPS", 31))
    steps = max(3, steps)
    angles = np.linspace(-max_deg, max_deg, steps)
    best_score = 1e30
    best_M: np.ndarray | None = None
    for deg in angles:
        M = cv2.getRotationMatrix2D((cx, cy), float(deg), 1.0)
        ngw = cv2.warpAffine(
            ng_g,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        diff = ok_g.astype(np.float32) - ngw.astype(np.float32)
        score = float(np.mean(diff[m] ** 2))
        if score < best_score:
            best_score = score
            best_M = M
    if best_M is None:
        return ng_rs
    return cv2.warpAffine(
        ng_rs,
        best_M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )



def get_product_contour(img_bgr: np.ndarray) -> np.ndarray:
    """Helper to detect largest product contour for alignment."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Dual-polarity Otsu or simple threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Check if inverted works better (dark pill on light bg) - usually pill is lighter
    if (th > 127).sum() > (th.size * 0.9):
        th = cv2.bitwise_not(th)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Fallback to simple rectangle
        H, W = gray.shape
        return np.array([[0,0], [W-1,0], [W-1,H-1], [0,H-1]])
    return max(cnts, key=cv2.contourArea)


def sort_pts(pts: np.ndarray) -> np.ndarray:
    """Sorts 4 points (top-left, top-right, bottom-right, bottom-left)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def align_ng_to_good_pca(
    ok_bgr: np.ndarray,
    ng_bgr: np.ndarray,
) -> np.ndarray:
    """
    High-fidelity alignment using PCA for orientation and Perspective Transform for bounds.
    Ported from capsule_pipeline_viz.ipynb.
    """
    H, W = ok_bgr.shape[:2]
    c_good = get_product_contour(ok_bgr)
    c_ng = get_product_contour(ng_bgr)

    # 1. PCA orientation for Good
    mu, eigen = cv2.PCACompute(c_good.reshape(-1, 2).astype(np.float32), mean=None)
    angle_good = np.degrees(np.arctan2(eigen[0, 1], eigen[0, 0]))
    
    # 2. PCA orientation for NG
    center_ng, eigen_ng = cv2.PCACompute(c_ng.reshape(-1, 2).astype(np.float32), mean=None)
    angle_ng = np.degrees(np.arctan2(eigen_ng[0, 1], eigen_ng[0, 0]))

    # Rotate NG to align long-axis with Good
    rot_diff = angle_good - angle_ng
    M_rot = cv2.getRotationMatrix2D(tuple(center_ng[0]), rot_diff, 1.0)
    ng_rotated = cv2.warpAffine(ng_bgr, M_rot, (ng_bgr.shape[1], ng_bgr.shape[0]), 
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 3. Perspective Transform for fine-tuning
    c_ng_rot = get_product_contour(ng_rotated)
    rect_good = cv2.minAreaRect(c_good)
    rect_ng_rot = cv2.minAreaRect(c_ng_rot)

    M_warp = cv2.getPerspectiveTransform(
        sort_pts(cv2.boxPoints(rect_ng_rot)), 
        sort_pts(cv2.boxPoints(rect_good))
    )
    warped_ng = cv2.warpPerspective(ng_rotated, M_warp, (W, H), 
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return warped_ng


def get_white_powder_mask(img_bgr: np.ndarray, threshold: int = 180) -> np.ndarray:
    """Detects 'white powder' area (body of transparent capsule) for replacement."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def synth_hybrid_cv_replacement(
    ok_bgr: np.ndarray,
    ng_bgr: np.ndarray,
    seed: int,
    intensity: float = 0.5,
    *,
    white_threshold: int = 180,
    alpha_blend: float = 0.95,
    refine_ai: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pharma Hybrid CV-AI Synthesis:
    1. Align NG reference to Good (PCA + Perspective)
    2. Detect target mask (white powder area)
    3. Alpha-blend (feathered) aligned NG texture into Good
    4. Optional: SDXL Refinement
    """
    # 1. Alignment
    warped_ng = align_ng_to_good_pca(ok_bgr, ng_bgr)
    
    # 2. Masking
    target_mask = get_white_powder_mask(ok_bgr, threshold=white_threshold)
    
    # Feathering
    mask_blur = cv2.GaussianBlur(target_mask, (21, 21), 0).astype(np.float32) / 255.0
    mask_blur_3c = cv2.merge([mask_blur, mask_blur, mask_blur])
    
    # 3. Blending
    ok_f = ok_bgr.astype(np.float32)
    ng_f = warped_ng.astype(np.float32)
    
    # Apply intensity to alpha_blend
    blend = mask_blur_3c * alpha_blend * (0.7 + 0.3 * intensity)
    result_f = ok_f * (1.0 - blend) + ng_f * blend
    result_bgr = np.clip(result_f, 0, 255).astype(np.uint8)
    
    # 4. Optional AI Refine
    if refine_ai:
        ai_res = _apply_sdxl_refine_to_bgr(result_bgr)
        if ai_res is not None:
            # Low strength blend back to preserve geometry if needed, 
            # though sdxl_refiner usually handles strength internally.
            result_bgr = ai_res

    # Returns (result, target_mask)
    return result_bgr, target_mask


def _align_ng_to_ok(
    ok_bgr: np.ndarray,
    ng_bgr: np.ndarray,
    align_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Đưa NG cùng kích thước OK; tùy ``NG_ALIGN_MODE`` có thể thêm rotation (+ translation ECC).

    - ``resize`` (mặc định): chỉ resize — hành vi cũ.
    - ``ecc``: OpenCV findTransformECC, EUCLIDEAN (xoay + tịnh tiến nhẹ) trong vùng ``align_mask``.
    - ``grid``: quét góc quanh tâm mask, chọn MSE nhỏ nhất (ổn khi sáng lệch nhưng hình học chỉ lệch góc).
    """
    oh, ow = ok_bgr.shape[:2]
    ng_rs = (
        ng_bgr
        if ng_bgr.shape[:2] == (oh, ow)
        else cv2.resize(ng_bgr, (ow, oh), interpolation=cv2.INTER_AREA)
    )
    mode = str(globals().get("NG_ALIGN_MODE", "resize")).lower().strip()
    if mode in ("", "none", "resize"):
        return ng_rs
    if align_mask is None:
        return ng_rs
    m_u8 = np.where(align_mask > 127, 255, 0).astype(np.uint8)
    if int(np.count_nonzero(m_u8 > 127)) < 120:
        return ng_rs
    if mode == "ecc":
        return _warp_ng_ecc_to_ok(ok_bgr, ng_rs, m_u8)
    if mode in ("shift", "translate", "translation"):
        return _warp_ng_shift_to_ok(ok_bgr, ng_rs, m_u8)
    if mode in ("grid", "rotate", "angles"):
        return _warp_ng_rotation_grid_to_ok(ok_bgr, ng_rs, m_u8)
    return ng_rs


def _harmonize_ng_illumination_to_ok(
    ok_bgr: np.ndarray,
    ng_bgr: np.ndarray,
    illum_mask: np.ndarray,
    *,
    sigma_scale: float = 0.30,
    scale_clip: tuple[float, float] = (0.82, 1.18),
) -> np.ndarray:
    """
    Kéo thành phần luminance tần thấp của NG về gần OK trong vùng ``illum_mask``,
    giảm residual do bóng / hướng sáng toàn cục của ảnh NG (diff map bớt dính highlight NG).
    """
    hw = float(globals().get("NG_ILLUM_HARMONIZE", 1.0))
    if hw <= 1e-6:
        return ng_bgr
    ng = ng_bgr.astype(np.float32)
    H, W = ok_bgr.shape[:2]
    m = (illum_mask > 127).astype(np.float32)
    if float(m.sum()) < 80:
        return ng_bgr
    sig = float(max(5.0, sigma_scale * float(min(H, W))))
    ok_g = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ng_g = cv2.cvtColor(ng_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ok_lf = cv2.GaussianBlur(ok_g, (0, 0), sig)
    ng_lf = cv2.GaussianBlur(ng_g, (0, 0), sig)
    lo, hi = scale_clip
    scale = ok_lf / np.clip(ng_lf, 5.0, 255.0)
    scale = np.clip(scale, lo, hi)
    mw = cv2.GaussianBlur(m, (0, 0), max(2.5, sig * 0.12))
    scale_eff = mw * scale + (1.0 - mw)
    adj = np.clip(ng * scale_eff[:, :, np.newaxis], 0, 255)
    out = np.clip((1.0 - hw) * ng + hw * adj, 0, 255)
    return out.astype(np.uint8)


def _attenuate_ng_specular_hotspots(
    ok_bgr: np.ndarray,
    ng_bgr: np.ndarray,
    illum_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Làm dịu **vệt bóng / highlight cục bộ** trên NG: pixel NG sáng hơn OK đáng kể
    được trộn về phía OK (chỉ trong mask), giảm vệt trắng trong residual & diff.

    Cường độ: global ``NG_SPECULAR_ATTEN`` (0 = tắt).
    """
    st = float(globals().get("NG_SPECULAR_ATTEN", 0.75))
    if st <= 1e-6:
        return ng_bgr
    rng = np.random.default_rng(int(seed) % (2**31))
    H, W = ng_bgr.shape[:2]
    m = illum_mask > 127
    npx = int(np.count_nonzero(m))
    if npx < 40:
        return ng_bgr

    ng = ng_bgr.astype(np.float32)
    ok_f = ok_bgr.astype(np.float32)
    gray = cv2.cvtColor(ng_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ok_g = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    p_hi = float(rng.uniform(84.0, 93.0))
    thr_n = float(np.percentile(gray[m], p_hi))
    margin = float(rng.uniform(14.0, 32.0))
    # NG sáng hơn OK tại cùng pixel → nghi specular NG
    rel_hot = np.maximum(0.0, gray - ok_g - margin) / 45.0
    rel_hot = np.clip(rel_hot, 0.0, 1.0) * m.astype(np.float32)
    abs_hot = np.clip((gray - thr_n) / max(255.0 - thr_n, 15.0), 0.0, 1.0) * m.astype(np.float32)
    wmap = np.clip(0.9 * rel_hot + 0.55 * abs_hot, 0.0, 1.0)
    wmap = cv2.GaussianBlur(wmap, (0, 0), float(rng.uniform(1.6, 3.4)))
    wmap = np.clip(wmap * st, 0.0, 1.0)
    w3 = wmap[:, :, np.newaxis]
    out = ng * (1.0 - w3) + ok_f * w3
    return np.clip(out, 0, 255).astype(np.uint8)


def _prepare_ng_for_residual_blend(
    ok_bgr: np.ndarray,
    ng_aligned: np.ndarray,
    illum_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Chuỗi: harmonize LF → attenuate specular → (tuỳ chọn) harmonize lần 2 σ lớn hơn.
    """
    ng = _harmonize_ng_illumination_to_ok(ok_bgr, ng_aligned, illum_mask)
    ng = _attenuate_ng_specular_hotspots(ok_bgr, ng, illum_mask, int(seed) + 51_011)
    h2 = float(globals().get("NG_SECOND_HARMONIZE", 0.42))
    if h2 > 1e-6:
        ng2 = _harmonize_ng_illumination_to_ok(
            ok_bgr,
            ng,
            illum_mask,
            sigma_scale=0.52,
            scale_clip=(0.86, 1.16),
        )
        ng = np.clip(
            (1.0 - h2) * ng.astype(np.float32) + h2 * ng2.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
    return ng


def _hf_residual_delta(
    ok_f: np.ndarray,
    ng_f: np.ndarray,
    *,
    sigma_scale: float = 0.018,
) -> np.ndarray:
    """
    Lấy residual tần cao: (NG - blur(NG)) - (OK - blur(OK)).
    Mục tiêu: loại bỏ vệt bóng/gradient tần thấp còn sót trong NG residual.
    """
    H, W = ok_f.shape[:2]
    sig = float(max(3.0, sigma_scale * float(min(H, W))))
    ok_lf = cv2.GaussianBlur(ok_f, (0, 0), sig)
    ng_lf = cv2.GaussianBlur(ng_f, (0, 0), sig)
    return (ng_f - ng_lf) - (ok_f - ok_lf)


def synth_thieu_ng_residual(
    ok_bgr: np.ndarray,
    ng_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
    illum_ref_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Thiếu hàm lượng: vùng thân — blend residual (NG−OK) với α mềm; giữ texture NG.

    Trả về ``(result, ok_u)`` — ``ok_u`` dùng cho debug DIFF đúng baseline đã blend.
    ``illum_ref_mask``: mask silhouette viên để cân sáng NG→OK (mặc định = ``mask``).
    """
    rng = np.random.default_rng(seed)
    lit = float(globals().get("LIGHTING_AUG_STRENGTH", 1.0))
    ls = int(seed) + 401_227
    m_illum = illum_ref_mask if illum_ref_mask is not None else mask
    ng_a = _align_ng_to_ok(ok_bgr, ng_bgr, m_illum)
    ng_a = _prepare_ng_for_residual_blend(ok_bgr, ng_a, m_illum, seed)
    ok_u = _randomize_shell_lighting(ok_bgr, mask, bbox, ls, strength=lit) if lit > 1e-6 else ok_bgr
    ng_u = _randomize_shell_lighting(ng_a, mask, bbox, ls, strength=lit) if lit > 1e-6 else ng_a
    ok = ok_u.astype(np.float32)
    ng = ng_u.astype(np.float32)
    delta = ng - ok
    hf_w = float(globals().get("NG_HF_RESIDUAL", 0.0))
    if hf_w > 1e-6:
        delta_hf = _hf_residual_delta(ok, ng)
        delta = (1.0 - hf_w) * delta + hf_w * delta_hf
    x, y, w, h = bbox
    H, W = ok.shape[:2]

    m_lo = float(rng.uniform(0.08, 0.14))
    m_hi = float(rng.uniform(0.86, 0.92))
    body = capsule_body_mask(mask, bbox, m_lo, m_hi)
    ys, xs = np.where(body > 127)
    if len(ys) < 12:
        body = mask.copy()
        ys, xs = np.where(body > 127)

    w_lo = max(4, int(w * (0.10 if intensity < 0.65 else 0.14)))
    w_hi = min(max(w_lo + 2, int(w * (0.38 if intensity < 0.65 else 0.48))), max(w_lo + 1, w - 2))
    h_lo = max(4, int(h * (0.18 if intensity < 0.65 else 0.24)))
    h_hi = min(max(h_lo + 2, int(h * (0.55 if intensity < 0.65 else 0.72))), max(h_lo + 1, h - 2))
    w_hi = max(w_hi, w_lo)
    h_hi = max(h_hi, h_lo)
    rw = int(rng.integers(w_lo, w_hi + 1))
    rh = int(rng.integers(h_lo, h_hi + 1))
    rw = min(rw, W - 1)
    rh = min(rh, H - 1)

    hard = np.zeros((H, W), dtype=np.uint8)
    placed = False
    for _ in range(50):
        idx = int(rng.integers(0, len(ys)))
        cx, cy = int(xs[idx]), int(ys[idx])
        x0 = int(np.clip(cx - rw // 2, 0, W - rw))
        y0 = int(np.clip(cy - rh // 2, 0, H - rh))
        patch_b = body[y0 : y0 + rh, x0 : x0 + rw] > 127
        if patch_b.size and np.count_nonzero(patch_b) >= 0.45 * patch_b.size:
            hard[y0 : y0 + rh, x0 : x0 + rw] = 255
            placed = True
            break
    if not placed:
        cx, cy = x + w // 2, y + h // 2
        x0 = int(np.clip(cx - rw // 2, 0, W - rw))
        y0 = int(np.clip(cy - rh // 2, 0, H - rh))
        hard[y0 : y0 + rh, x0 : x0 + rw] = 255

    hard = cv2.bitwise_and(hard, body)
    sigma = float(rng.uniform(2.2, 5.0))
    alpha = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), sigma) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0) * (mask.astype(np.float32) / 255.0)
    gain = float(rng.uniform(0.55, 1.0)) * (0.82 + 0.18 * intensity)
    n_sig = float(rng.uniform(0.8, 2.2))
    noise = rng.normal(0.0, n_sig, (H, W, 3)).astype(np.float32)
    a3 = alpha[:, :, np.newaxis]
    out = ok + a3 * delta * gain + a3 * noise
    return np.clip(out, 0, 255).astype(np.uint8), ok_u


def synth_rong_ng_residual(
    ok_bgr: np.ndarray,
    ng_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
    *,
    feat_bands: bool = True,
    feat_haze: bool = True,
    feat_halo: bool = True,
    feat_grain: bool = True,
    feat_specular: bool = True,
    feat_strength: float = 1.0,
    illum_ref_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rỗng (NG residual + new feature): giữ nền NG bằng OK+alpha*(NG-OK),
    sau đó thêm feature nhẹ để đa dạng (band/rỗng loang) nhưng vẫn bám texture ảnh OK.

    Trả về ``(result, ok_u)`` cho debug DIFF. ``illum_ref_mask``: silhouette viên để cân sáng NG→OK.
    """
    rng = np.random.default_rng(seed)
    lit = float(globals().get("LIGHTING_AUG_STRENGTH", 1.0))
    ls = int(seed) + 185_241
    m_illum = illum_ref_mask if illum_ref_mask is not None else mask
    ng_a = _align_ng_to_ok(ok_bgr, ng_bgr, m_illum)
    ng_a = _prepare_ng_for_residual_blend(ok_bgr, ng_a, m_illum, seed)
    ok_u = (
        _randomize_shell_lighting(ok_bgr, mask, bbox, ls, strength=lit) if lit > 1e-6 else ok_bgr
    )
    ng_u = _randomize_shell_lighting(ng_a, mask, bbox, ls, strength=lit) if lit > 1e-6 else ng_a
    ok = ok_u.astype(np.float32)
    ng = ng_u.astype(np.float32)
    x, y, w, h = bbox
    H, W = ok.shape[:2]
    m = (mask > 127).astype(np.float32)
    ng_soft = cv2.GaussianBlur(ng, (0, 0), float(rng.uniform(0.8, 1.6)))
    hf_w = float(globals().get("NG_HF_RESIDUAL", 0.0))
    if hf_w > 1e-6:
        # Dùng OK làm nền LF, chỉ lấy HF residual của NG để tránh “vệt bóng” dài.
        ng_hf = ok + _hf_residual_delta(ok, ng)
        ng_soft = (1.0 - hf_w) * ng_soft + hf_w * ng_hf

    full = bool(globals().get("RONG_FULL_CAPSULE", False))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = np.clip((xx - float(x)) / max(float(w), 1.0), 0.0, 1.0)
    yn = np.clip((yy - float(y)) / max(float(h), 1.0), 0.0, 1.0)

    if full:
        # Rỗng phủ cả viên: alpha gần như toàn mask, chỉ làm mềm biên để không gắt.
        alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), float(rng.uniform(2.8, 4.6)))
        mx = float(alpha.max()) if alpha.max() > 1e-6 else 1.0
        alpha = (alpha / mx) * m
        # Khi phủ cả viên, giảm cường độ dark/HF để tránh "đen bệt" và diff quá dày cục bộ.
        full_scale = 0.72
    else:
        # Rỗng theo dải (legacy)
        full_scale = 1.0
        cx = float(x + w * 0.5 + rng.uniform(-0.03, 0.03) * max(w, 1))
        cy = float(y + h * 0.5 + rng.uniform(-0.08, 0.12) * max(h, 1))
        band_h = float(rng.uniform(0.32, 0.52)) * h
        sigma_y = max(3.5, band_h * float(rng.uniform(0.42, 0.58)))
        gauss_y = np.exp(-0.5 * ((yy - cy) / sigma_y) ** 2)

        half_w = w * 0.5
        end_fade = float(rng.uniform(0.40, 0.58)) * half_w
        dist_x = np.abs(xx - cx)
        end_mask = np.clip(
            1.0 - np.maximum(0.0, dist_x - (half_w - end_fade)) / max(end_fade, 1.0),
            0.0,
            1.0,
        )
        end_mask = np.power(end_mask, 0.55)

        alpha = gauss_y * end_mask * m
        s_pre = float(rng.uniform(3.0, 6.5))
        alpha = cv2.GaussianBlur(alpha, (0, 0), s_pre)
        alpha = np.clip(alpha, 0.0, None)
        mx = float(alpha.max()) if alpha.max() > 1e-6 else 1.0
        alpha = alpha / mx
        gamma = float(rng.uniform(0.40, 0.68))
        alpha = np.power(alpha, gamma)
        s_post = float(rng.uniform(2.0, 4.2))
        alpha = cv2.GaussianBlur(alpha, (0, 0), s_post)
        alpha = alpha * m

    # Base composite:
    # - full=True: mặc định tránh cảm giác "paste NG" → OK anchor + residual inject
    #             có thể bật lại blend NG trực tiếp bằng RONG_FULL_USE_NG_BLEND
    # - full=False: legacy blend OK↔NG mềm trong vùng lỗi
    base_min = float(globals().get("RONG_BASE_MIN", 0.16))
    base_min = float(np.clip(base_min, 0.0, 0.95))
    alpha_base = np.clip(base_min + (1.0 - base_min) * alpha, 0.0, 1.0) * m
    a3 = alpha[:, :, np.newaxis]
    ab3 = alpha_base[:, :, np.newaxis]
    if full:
        if bool(globals().get("RONG_FULL_USE_NG_BLEND", False)):
            out = ok * (1.0 - ab3) + ng_soft * ab3
        else:
            # Không yêu cầu NG khớp pixel: chỉ lấy feature/texture từ NG và bơm vào OK.
            # - mode=hf: dùng delta (NG-OK) đã HF-biased (nhạy với alignment hơn)
            # - mode=profile_x: lấy profile 1D theo trục X từ NG (banding/texture), ổn khi NG lệch hình học
            mode = str(globals().get("RONG_NG_TEXTURE_MODE", "profile_x")).lower().strip()
            k = float(globals().get("RONG_FULL_DELTA_K", 0.55))
            k = float(np.clip(k, 0.0, 1.5))
            if mode in ("hf", "delta", "residual"):
                delta = (ng_soft - ok)
                out = ok + (k * ab3) * delta
            else:
                # Profile-X texture transfer (alignment-robust):
                # 1) compute HF of NG inside mask
                ng_g = cv2.cvtColor(ng_u, cv2.COLOR_BGR2GRAY).astype(np.float32)
                ng_hf = ng_g - cv2.GaussianBlur(ng_g, (0, 0), 1.4)
                # 2) estimate per-column mean HF within mask → 1D band profile
                m01 = (m > 0.5).astype(np.float32)
                col_w = np.sum(m01, axis=0) + 1e-6
                prof = np.sum(ng_hf * m01, axis=0) / col_w
                prof = cv2.GaussianBlur(prof[np.newaxis, :], (0, 0), 3.2).reshape(-1)
                # robust normalize
                p95 = float(np.percentile(np.abs(prof), 95.0))
                if p95 < 1e-3:
                    prof_n = np.zeros_like(prof, dtype=np.float32)
                else:
                    prof_n = np.clip(prof / p95, -1.0, 1.0).astype(np.float32)
                # 3) lift to 2D map and fade near capsule ends
                prof2 = prof_n[np.newaxis, :].repeat(H, axis=0)
                end = np.power(np.clip(np.sin(np.pi * xn), 0.0, 1.0), 0.55)
                tmap = prof2 * end * m
                # 4) apply as subtle luminance modulation on OK (avoid turning into NG)
                amp = (10.0 + 6.0 * intensity) * k
                out = ok + (ab3 * amp) * tmap[:, :, None]
    else:
        out = ok * (1.0 - ab3) + ng_soft * ab3

    # CV "empty" feel: lõi tối + viền sáng mỏng (shell) + gradient nhẹ trong lòng viên.
    empty_cv = float(globals().get("RONG_EMPTY_CV", 0.0))
    empty_cv = float(np.clip(empty_cv, 0.0, 1.5))
    if empty_cv > 1e-6:
        m_u8 = (m > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(m_u8, cv2.DIST_L2, 3).astype(np.float32)
        dmax = float(dist.max())
        if dmax > 1e-6:
            dn = np.clip(dist / dmax, 0.0, 1.0)
            # core: tối mạnh nhất ở giữa; rim: sáng mỏng gần biên (tạo cảm giác vỏ)
            core = np.power(dn, 0.48)
            rim = np.clip(1.0 - dn, 0.0, 1.0)
            rim = np.power(rim, 0.35)
            rim = cv2.GaussianBlur(rim, (0, 0), 1.4)
            # inner rim: highlight mỏng phía "bên trong" (tạo cảm giác vỏ rỗng rõ hơn)
            inner = np.exp(-0.5 * ((dn - 0.62) / 0.17) ** 2).astype(np.float32) * m
            inner = cv2.GaussianBlur(inner, (0, 0), 1.1)

            # gradient nhẹ trong lòng theo trục Y để tránh phẳng (giống "rỗng" loang)
            gy = np.clip((yn - 0.5) / 0.45, -1.0, 1.0)
            gmap = 0.5 + 0.5 * np.cos(np.pi * gy)
            gmap = cv2.GaussianBlur((gmap * m).astype(np.float32), (0, 0), 2.2)

            w = empty_cv * full_scale * np.power(alpha, 0.85) * m
            w3 = w[:, :, np.newaxis]
            # darken core
            core_dn = (0.20 + 0.14 * intensity) * np.power(core, 1.08) * gmap
            out = out * (1.0 - w3 * core_dn[:, :, None])
            # brighten rim a bit (thin shell reflection)
            rim_up = (2.2 + 2.2 * intensity) * rim * gmap
            inner_up = (8.0 + 6.0 * intensity) * inner * gmap
            out = out + w3 * (rim_up + inner_up)[:, :, None]
            # De-highlight: giảm bớt vùng cháy trắng của OK để nhìn "empty" hơn (ít dính highlight cũ).
            ok_lum2 = np.max(ok, axis=2).astype(np.float32)
            ok_bright = np.clip((ok_lum2 - 178.0) / 62.0, 0.0, 1.0)
            dehi_k = float(globals().get("RONG_DEHIGHLIGHT_K", 1.0))
            dehi_k = float(np.clip(dehi_k, 0.0, 3.0))
            dehi = dehi_k * (0.28 + 0.10 * intensity) * ok_bright * core * gmap
            out = out * (1.0 - w3 * dehi[:, :, None])
            out = np.clip(out, 0, 255)

            # Black-core mode: đen gần như toàn viên, chỉ chừa viền mỏng.
            black_k = float(globals().get("RONG_BLACK_CORE_K", 0.0))
            black_k = float(np.clip(black_k, 0.0, 1.0))
            if black_k > 1e-6:
                rim_frac = float(globals().get("RONG_RIM_FRAC", 0.16))
                rim_frac = float(np.clip(rim_frac, 0.04, 0.45))
                # dn=0 ở biên, dn=1 ở core. core_mask=1 ở lõi, 0 ở viền.
                core_mask = np.clip((dn - rim_frac) / max(1.0 - rim_frac, 1e-6), 0.0, 1.0)
                core_mask = cv2.GaussianBlur(core_mask, (0, 0), 1.4)
                # target rất tối nhưng không "dead black" để còn texture
                floor_dn = float(globals().get("RONG_BLACK_FLOOR", 10.0))
                floor_dn = float(np.clip(floor_dn, 0.0, 35.0))
                tgt = (floor_dn + (6.0 + 6.0 * intensity) * (1.0 - core_mask)) * m
                tgt3 = tgt[:, :, None]
                cm3 = (black_k * core_mask * m)[:, :, None]
                out = out * (1.0 - cm3) + tgt3 * cm3
                out = np.clip(out, 0, 255)

            # Plastic-like reflection texture (streaks/bands) to avoid "flat black".
            refl_k = float(globals().get("RONG_REFLECT_K", 0.0))
            refl_k = float(np.clip(refl_k, 0.0, 2.0))
            if refl_k > 1e-6:
                # 1D band profile along capsule axis (x), then anisotropic blur -> reflection streaks
                n_st = int(globals().get("RONG_REFLECT_STREAKS", 5))
                n_st = int(np.clip(n_st, 1, 14))
                rng2 = np.random.default_rng(int(seed) + 91_337)
                band = np.zeros((W,), np.float32)
                for _ in range(n_st):
                    cxn = float(rng2.uniform(0.10, 0.90))
                    sx = float(rng2.uniform(0.025, 0.080))
                    amp = float(rng2.uniform(0.6, 1.0))
                    band += amp * np.exp(-0.5 * ((np.linspace(0.0, 1.0, W) - cxn) / sx) ** 2)
                band = (band - band.min()) / max(float(band.max() - band.min()), 1e-6)
                streak = band[np.newaxis, :].repeat(H, axis=0)
                streak = cv2.GaussianBlur(streak, (0, 0), 2.2)
                streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=6.0, sigmaY=1.2)

                # Weight reflections: stronger near rim/inner shell, weaker in deep core.
                w_ref = (0.25 + 0.75 * (1.0 - core)) * m * np.power(alpha, 0.55)
                w_ref = cv2.GaussianBlur(w_ref, (0, 0), 1.6)
                w3 = (refl_k * w_ref)[:, :, None]

                # Add small highlight lift + subtle modulation (looks like plastic reflection).
                lift = (10.0 + 10.0 * intensity) * streak
                mod = (1.0 + 0.10 * (streak - 0.5))
                out = out * (1.0 - w3) + (out * mod[:, :, None] + lift[:, :, None]) * w3
                out = np.clip(out, 0, 255)

    # Pixel augmentation trên NG: noise nhẹ; tắt dần ở pixel sáng (OK) để khỏi hạt ở vùng chá.
    ok_lum = np.max(ok, axis=2).astype(np.float32)
    damp_n = 1.0 - 0.88 * np.clip((ok_lum - 150.0) / 102.0, 0.0, 1.0)
    n_sig = float(rng.uniform(1.6, 3.8))
    noise = rng.normal(0.0, n_sig, (H, W, 3)).astype(np.float32)
    out += noise * np.power(ab3, 0.55) * damp_n[:, :, np.newaxis]

    # New feature layer: band + loang nhẹ, vẫn neo theo alpha residual.
    f = np.zeros((H, W), np.float32)
    n_feat = int(rng.integers(2, 4))
    for _ in range(n_feat):
        cxn = float(rng.uniform(0.22, 0.86))
        sx = float(rng.uniform(0.035, 0.075))
        amp = float(rng.uniform(0.55, 1.00))
        f += amp * np.exp(-0.5 * ((xn - cxn) / sx) ** 2)
    y_soft = np.exp(-0.5 * ((yn - (0.50 + float(rng.uniform(-0.05, 0.05)))) / float(rng.uniform(0.20, 0.30))) ** 2)
    f = f * (0.65 + 0.35 * y_soft) * np.power(alpha, 0.84)
    f = cv2.GaussianBlur(f, (0, 0), float(rng.uniform(2.2, 3.8)))

    # Loang lân cận trong mask (halo) để đỡ cảm giác chỉ dải cứng.
    halo = cv2.GaussianBlur(f, (0, 0), float(rng.uniform(4.2, 6.0))) * m
    feat_map = np.clip(0.72 * f + 0.38 * halo, 0.0, 1.0)
    fm = float(feat_map.max()) if feat_map.max() > 1e-6 else 1.0
    feat_map = feat_map / fm

    # Blend feature: thêm tối (rỗng) + giữ micro texture của NG.
    dark_k = float(globals().get("RONG_DARK_K", 1.0))
    dark_k = float(np.clip(dark_k, 0.0, 3.5))
    heavy = float(globals().get("RONG_HEAVY_MULT", 1.0))
    heavy = float(np.clip(heavy, 0.5, 2.5))
    dark_k *= full_scale
    heavy *= full_scale
    extra_dark = (
        float(rng.uniform(0.11, 0.24))
        * (0.88 + 0.30 * intensity)
        * dark_k
        * heavy
    )
    out *= (1.0 - extra_dark * np.power(feat_map, 0.72))[:, :, np.newaxis]
    ng_g = cv2.cvtColor(ng_u, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if ng_g.shape[:2] != (H, W):
        ng_g = cv2.resize(ng_g, (W, H), interpolation=cv2.INTER_AREA)
    hf = ng_g - cv2.GaussianBlur(ng_g, (0, 0), 1.2)
    den = max(float(np.percentile(np.abs(hf[m > 0.5]), 95)), 1.0)
    hf = np.clip(hf / den, -1.0, 1.0)
    damp_hf = 1.0 - 0.85 * np.clip((ok_lum - 148.0) / 100.0, 0.0, 1.0)
    out += (
        (6.2 + 2.4 * intensity)
        * heavy
        * hf[:, :, np.newaxis]
        * np.power(feat_map, 0.88)[:, :, np.newaxis]
        * damp_hf[:, :, np.newaxis]
    )

    # Không để vùng lỗi chạm sát 0 (giống ảnh thật: than đen + grain)
    floor_dn = float(rng.uniform(10.0, 26.0))
    out = np.maximum(out, floor_dn * alpha[:, :, np.newaxis] * m[:, :, np.newaxis])
    out = np.clip(out, 0, 255).astype(np.uint8)
    bsig = float(rng.uniform(0.45, 0.95))
    out = cv2.merge([cv2.GaussianBlur(out[:, :, c], (0, 0), bsig) for c in range(3)])

    # Optional feature injection for diversity (keeps geometry)
    if feat_strength > 0 and (
        feat_bands or feat_haze or feat_halo or feat_grain or feat_specular
    ):
        out = _apply_feature_injection_rong(
            out,
            ok_u,
            mask,
            bbox,
            seed=seed + 1337,
            intensity=intensity,
            bands=feat_bands,
            haze=feat_haze,
            halo=feat_halo,
            grain=feat_grain,
            specular=feat_specular,
            strength=feat_strength,
        )

    return out, ok_u


def get_thuoc_dai_defects(style: str, *, skip_rong: bool = False) -> dict:
    """Chỉ procedural tại đây; ng_residual xử lý riêng trong run_dai (cần cặp OK+NG)."""
    if style != "procedural":
        raise ValueError("get_thuoc_dai_defects chỉ hỗ trợ style='procedural'")
    out = {"Thiếu_hàm_lượng": (synth_thieu_ham_luong, "thieu")}
    if not skip_rong:
        out = {"Rỗng": (synth_rong, "rong"), **out}
    return out


def synth_tron_lom(
    ok_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
    return_mask: bool = False,
):
    """Dent on round tablet: dark ellipse offset to one side.
    If return_mask=True, returns (result, defect_mask_uint8).
    """
    rng = np.random.default_rng(seed)
    x, y, w, h = bbox
    H, W = ok_bgr.shape[:2]
    out = ok_bgr.astype(np.float32)
    m = (mask > 127).astype(np.float32)

    cx = x + w * 0.5 + float(rng.uniform(-0.08, 0.08)) * w
    cy = y + h * 0.5 + float(rng.uniform(-0.12, 0.12)) * h
    ax = max(3.0, w * float(rng.uniform(0.18, 0.32)))
    ay = max(3.0, h * float(rng.uniform(0.22, 0.38)))
    angle = float(rng.uniform(0, 180))
    amp = float(rng.uniform(-55.0, -35.0)) if intensity < 0.65 else float(rng.uniform(-85.0, -55.0))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rad = np.radians(angle)
    xr = (xx - cx) * np.cos(rad) + (yy - cy) * np.sin(rad)
    yr = -(xx - cx) * np.sin(rad) + (yy - cy) * np.cos(rad)
    ell = np.exp(-0.5 * ((xr / ax) ** 2 + ((yr + ay * 0.25) / ay) ** 2))
    ell = cv2.GaussianBlur(ell.astype(np.float32), (0, 0), sigmaX=2.5, sigmaY=2.5)
    delta = amp * ell * m
    for c in range(3):
        out[:, :, c] = np.clip(out[:, :, c] + delta, 0, 255)
    result = out.astype(np.uint8)

    if return_mask:
        # Threshold at 0.95 → ~4-5% of tablet area (core dent region only)
        ell_raw = np.exp(-0.5 * ((xr / ax) ** 2 + ((yr + ay * 0.25) / ay) ** 2))
        defect_mask = ((ell_raw * m) > 0.95).astype(np.uint8) * 255
        return result, defect_mask

    return result


def synth_tron_nut(
    ok_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
) -> np.ndarray:
    """Nứt: đường tối mảnh gấp khúc nhẹ trên capsule."""
    rng = np.random.default_rng(seed)
    x, y, w, h = bbox
    H, W = ok_bgr.shape[:2]
    out = ok_bgr.astype(np.float32).copy()
    m = (mask > 127).astype(np.uint8) * 255

    n_pts = int(rng.integers(4, 8))
    cx, cy = x + w // 2, y + h // 2
    pts = []
    for i in range(n_pts):
        px = int(np.clip(rng.normal(cx, w * 0.22), x + 2, x + w - 2))
        py = int(np.clip(rng.normal(cy, h * 0.22), y + 2, y + h - 2))
        pts.append((px, py))

    crack = np.zeros((H, W), np.uint8)
    thick = 1 if intensity < 0.65 else 2
    cv2.polylines(crack, [np.array(pts, np.int32)], False, 255, thick, cv2.LINE_AA)
    crack = cv2.dilate(crack, np.ones((3, 3), np.uint8), iterations=1)
    crack = cv2.bitwise_and(crack, m)
    crack_f = cv2.GaussianBlur(crack.astype(np.float32), (0, 0), 1.1) / 255.0

    dv = float(rng.uniform(-60.0, -40.0)) if intensity < 0.65 else float(rng.uniform(-95.0, -60.0))
    for c in range(3):
        out[:, :, c] = np.clip(out[:, :, c] + dv * crack_f, 0, 255)
    return out.astype(np.uint8)


def save_capsule_result(
    product: str,
    defect: str,
    name: str,
    ok: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray,
    ok_for_diff: np.ndarray | None = None,
    bbox: tuple[int, int, int, int] | None = None,
):
    if bool(globals().get("SDXL_REFINE_ENABLED", False)):
        refined = _apply_sdxl_refine_to_bgr(result)
        if refined is not None:
            result = refined

    hds = float(globals().get("HIGHLIGHT_DESPECKLE", 0.52))
    if hds > 1e-6:
        result = _highlight_despeckle_bgr(result, mask, strength=hds)

    out_dir = os.path.join(RESULT_DIR, product, defect)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, result)
    print(f"  → {path}")

    ys, xs = np.where(mask > 127)
    if not len(ys):
        return
    oh, ow = ok.shape[:2]
    x0 = max(0, int(xs.min()) - PAD_CTX)
    x1 = min(ow, int(xs.max()) + PAD_CTX)
    y0 = max(0, int(ys.min()) - PAD_CTX)
    y1 = min(oh, int(ys.max()) + PAD_CTX)

    ok_ref = ok_for_diff if ok_for_diff is not None else ok

    def rh(p: np.ndarray, h: int = TH_PANEL) -> np.ndarray:
        ih, iw = p.shape[:2]
        return cv2.resize(p, (max(1, int(iw * h / ih)), h))

    def lbl(p: np.ndarray, txt: str, color=(0, 255, 255)) -> np.ndarray:
        p = p.copy()
        cv2.putText(p, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(p, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return p

    crop_ok = ok_ref[y0:y1, x0:x1]
    mc = mask[y0:y1, x0:x1]
    ov = crop_ok.copy()
    tmp = ov.copy()
    tmp[mc > 127] = (0, 0, 220)
    ov = cv2.addWeighted(tmp, 0.45, ov, 0.55, 0)

    diff = cv2.absdiff(ok_ref, result)
    diff_g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Thiếu hàm lượng: chỉnh sáng nhẹ → absdiff max thường ~8–35 → panel DIFF x1 trông "xám nhạt".
    # Chuẩn hóa theo vùng crop (chỉ để debug), không đổi ảnh RESULT.
    dc = diff_g[y0:y1, x0:x1].astype(np.float32)
    p99 = float(np.percentile(dc, 99.0))
    p92 = float(np.percentile(dc, 92.0))
    anchor = max(p99, p92 * 1.2, 4.0)
    if anchor < 22.0:
        scale = min(255.0 / anchor, 32.0)
        diff_vis = np.clip(diff_g.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    else:
        diff_vis = diff_g

    panels = [
        lbl(rh(crop_ok), "OK"),
        lbl(rh(ov), "MASK"),
        lbl(rh(result[y0:y1, x0:x1]), "RESULT"),
    ]

    # Panel CROP — hiển thị vùng capsule mà SD 1.5 thực sự nhìn thấy (với bbox padding 8px)
    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        pad = 8
        bcx1 = max(0, bx1 - pad); bcy1 = max(0, by1 - pad)
        bcx2 = min(ok.shape[1], bx2 + pad); bcy2 = min(ok.shape[0], by2 + pad)
        crop_sd = ok[bcy1:bcy2, bcx1:bcx2]
        crop_res = result[bcy1:bcy2, bcx1:bcx2]
        # Vẽ mask lên crop để thấy vùng inpaint
        crop_mask_roi = mask[bcy1:bcy2, bcx1:bcx2]
        crop_ov2 = crop_sd.copy()
        tmp2 = crop_ov2.copy()
        tmp2[crop_mask_roi > 127] = (0, 200, 0)
        crop_ov2 = cv2.addWeighted(tmp2, 0.4, crop_ov2, 0.6, 0)
        panels.insert(2, lbl(rh(crop_ov2),  "CROP-IN",  color=(0, 255, 0)))
        panels.insert(3, lbl(rh(crop_res),  "CROP-OUT", color=(0, 255, 0)))
    star = "*" if anchor < 22.0 else ""
    for mult in [1.0, 2.0, 4.0, 8.0, 16.0]:
        d = np.clip(diff_vis.astype(np.float32) * mult, 0, 255).astype(np.uint8)
        d_bgr = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
        panels.append(lbl(rh(d_bgr), f"DIFF x{int(mult)}{star}"))

    debug = np.hstack(panels)
    dbg_path = os.path.join(out_dir, "debug_" + name)
    cv2.imwrite(dbg_path, debug)
    print(f"  → {dbg_path}")


THUOC_TRON_DEFECTS = {
    "Lõm": synth_tron_lom,
    "Nứt": synth_tron_nut,
}

FALLBACK_DAI = (36, 38, 167, 85)


def run_rois(
    roi_dir: str,
    product: str = "thumb_long",
    synth_style: str = "procedural",
    fixed_region: bool = False,
):
    """
    Sinh Rỗng + Thiếu_hàm_lượng từ từng file roi_XX.jpg trong roi_dir.
    """
    if synth_style == "ng_residual":
        print("[--rois] ng_residual cần cặp OK+NG trong Thuốc_dài/ref — ROI chỉ dùng procedural")
    defects = get_thuoc_dai_defects("procedural")
    paths = sorted(
        glob.glob(os.path.join(roi_dir, "roi_*.jpg"))
        + glob.glob(os.path.join(roi_dir, "roi_*.png"))
    )
    if not paths:
        print(f"[SKIP --rois] Không thấy roi_*.jpg/png trong {roi_dir}")
        return
    for path in paths:
        ok = cv2.imread(path)
        if ok is None:
            print(f"[SKIP] Không đọc được {path}")
            continue
        base = os.path.basename(path)
        m = re.match(r"roi_(\d+)", os.path.splitext(base)[0], re.I)
        ri = m.group(1) if m else "0"
        ri_num = int(ri) if ri.isdigit() else hash(base) % 1000
        mask, bbox = detect_capsule_mask_robust(ok)
        area = int((mask > 127).sum())
        if area < 50:
            print(f"[WARN] {base}: mask nhỏ ({area}px), bỏ qua")
            continue
        print(f"[{product}/roi_{ri}] bbox={bbox} mask_area={area}")
        for defect_name, (_, slug0) in defects.items():
            slug = slug0 + ("_fx" if fixed_region else "")
            for si in range(N_SHAPES):
                for vi, intensity in enumerate(INTENSITIES):
                    seed = ri_num * 7919 + si * 1000 + vi * 17 + hash(defect_name) % 997
                    if defect_name == "Thiếu_hàm_lượng":
                        res = synth_thieu_ham_luong(
                            ok, mask, bbox, seed, intensity, fixed_region=fixed_region
                        )
                    else:
                        res = synth_rong(ok, mask, bbox, seed, intensity, fixed_region=fixed_region)
                    tag = f"roi{ri}_{slug}_s{si}_i{int(intensity * 10)}.jpg"
                    save_capsule_result(product, defect_name, tag, ok, mask, res)


def run_dai(synth_style: str = "procedural", fixed_region: bool = False, *, skip_rong: bool = False):
    product = "Thuốc_dài"
    root = CAPSULE_ROOTS[product]
    procedural = get_thuoc_dai_defects("procedural", skip_rong=skip_rong)

    for defect_name in procedural:
        defect_path = os.path.join(root, defect_name)
        ok = load_ok(defect_path)
        if ok is None:
            print(f"[SKIP] No OK image: {defect_path}")
            continue

        pm = globals().get("PRODUCT_MASK_MODE", "threshold")
        mask_product, mask_synth, bbox, used_defect_hint = auto_product_mask_and_synth_mask(
            ok, defect_path, str(pm), FALLBACK_DAI
        )
        ng = load_first_ref(defect_path)
        use_residual = synth_style == "ng_residual" and ng is not None
        if synth_style == "ng_residual" and ng is None:
            print(f"[WARN] {defect_name}: không có ref/ — fallback procedural")
        ng_r = ng if use_residual else None

        mode = "ng_residual" if use_residual else "procedural"
        print(
            f"[{product}/{defect_name}] {mode} product_mask={pm} bbox={bbox} "
            f"defect_hint_mask={'on' if used_defect_hint else 'off'} "
            f"area_synth={(mask_synth > 127).sum()} area_product={(mask_product > 127).sum()}"
        )

        # Feature injection knobs (only used for Rong in ng_residual mode)
        feat_bands = bool(globals().get("FEAT_BANDS", True))
        feat_haze = bool(globals().get("FEAT_HAZE", True))
        feat_halo = bool(globals().get("FEAT_HALO", True))
        feat_grain = bool(globals().get("FEAT_GRAIN", True))
        feat_specular = bool(globals().get("FEAT_SPECULAR", True))
        feat_strength = float(globals().get("FEAT_STRENGTH", 0.0))

        for si in range(N_SHAPES):
            for vi, intensity in enumerate(INTENSITIES):
                seed = si * 1000 + vi * 17 + hash(defect_name) % 997
                ok_d: np.ndarray | None = None
                if use_residual and ng_r is not None:
                    if defect_name == "Thiếu_hàm_lượng":
                        res, ok_d = synth_thieu_ng_residual(
                            ok,
                            ng_r,
                            mask_synth,
                            bbox,
                            seed,
                            intensity,
                            illum_ref_mask=mask_product,
                        )
                        slug = "thieu_nr"
                    else:
                        res, ok_d = synth_rong_ng_residual(
                            ok,
                            ng_r,
                            mask_synth,
                            bbox,
                            seed,
                            intensity,
                            feat_bands=feat_bands,
                            feat_haze=feat_haze,
                            feat_halo=feat_halo,
                            feat_grain=feat_grain,
                            feat_specular=feat_specular,
                            feat_strength=feat_strength,
                            illum_ref_mask=mask_product,
                        )
                        slug = "rong_nr"
                else:
                    _, slug0 = procedural[defect_name]
                    slug = slug0 + ("_fx" if fixed_region else "")
                    if defect_name == "Thiếu_hàm_lượng":
                        res = synth_thieu_ham_luong(
                            ok, mask_synth, bbox, seed, intensity, fixed_region=fixed_region
                        )
                    else:
                        res = synth_rong(ok, mask_synth, bbox, seed, intensity, fixed_region=fixed_region)
                tag = f"{slug}_s{si}_i{int(intensity * 10)}.jpg"
                save_capsule_result(
                    product,
                    defect_name,
                    tag,
                    ok,
                    mask_product,
                    res,
                    ok_for_diff=ok_d,
                )


def run_tron():
    product = "Thuốc_tròn"
    root = CAPSULE_ROOTS[product]
    for defect_name, synth_fn in THUOC_TRON_DEFECTS.items():
        defect_path = os.path.join(root, defect_name)
        ok = load_ok(defect_path)
        if ok is None:
            print(f"[SKIP] No OK image: {defect_path}")
            continue
        mask, bbox = detect_capsule_mask(ok, fallback_xywh=None)
        area = int((mask > 127).sum())
        if area < 50:
            print(f"[WARN] {product}/{defect_name}: mask nhỏ ({area}px), kiểm tra OK/segmentation")
        print(f"[{product}/{defect_name}] bbox={bbox} mask_area={area}")
        for si in range(N_SHAPES):
            for vi, intensity in enumerate(INTENSITIES):
                seed = si * 1000 + vi * 19 + hash(defect_name) % 997
                res = synth_fn(ok, mask, bbox, seed=seed, intensity=intensity)
                short = "lom" if defect_name == "Lõm" else "nut"
                tag = f"{short}_s{si}_i{int(intensity * 10)}.jpg"
                save_capsule_result(product, defect_name, tag, ok, mask, res)


def main():
    ap = argparse.ArgumentParser(description="Capsule synthesis: Thuốc_dài + Thuốc_tròn + ROI thumb")
    ap.add_argument(
        "--product",
        choices=["all", "Thuốc_dài", "Thuốc_tròn", "none"],
        default="all",
        help="'none' = không chạy Thuốc_dài/tròn (chỉ dùng với --rois)",
    )
    ap.add_argument(
        "--rois",
        default="",
        help="Thư mục chứa roi_00.jpg, roi_01.jpg, … — sinh Rỗng + Thiếu_hàm_lượng trên từng ROI",
    )
    ap.add_argument(
        "--rois-product",
        default="thumb_long",
        help="Tên thư mục product trong results khi dùng --rois",
    )
    ap.add_argument(
        "--synth-style",
        choices=["procedural", "ng_residual"],
        default="ng_residual",
        help="ng_residual: OK+NG blend từ ref/ (khuyến nghị khi có 1 ảnh NG); procedural: không cần ref",
    )
    ap.add_argument(
        "--fixed-region",
        action="store_true",
        help="Cố định hình học vùng lỗi (procedural); chỉ random cường độ/noise — tên file *_fx",
    )
    ap.add_argument("--skip-rong", action="store_true", help="Bỏ chạy class Rỗng (dùng cho workflow NG→NEW NG / debug)")
    ap.add_argument(
        "--feat-strength",
        type=float,
        default=0.0,
        help="(ng_residual/Rỗng) strength chồng feature (band/haze/halo/grain) để tăng diversity. 0=off",
    )
    ap.add_argument("--feat-bands", action="store_true", help="(ng_residual/Rỗng) bật band feature")
    ap.add_argument("--feat-haze", action="store_true", help="(ng_residual/Rỗng) bật haze feature")
    ap.add_argument("--feat-halo", action="store_true", help="(ng_residual/Rỗng) bật halo feature")
    ap.add_argument("--feat-grain", action="store_true", help="(ng_residual/Rỗng) bật grain feature")
    ap.add_argument("--feat-specular", action="store_true", help="(ng_residual/Rỗng) bật biến thiên highlight/bóng vỏ")
    ap.add_argument(
        "--product-mask",
        choices=["threshold", "min_rect", "hough", "robust"],
        default="threshold",
        help="Mask silhouette viên (tự động). Thư mục mask/ = vùng lỗi (hint), giao với viên rồi mới sinh lỗi",
    )
    ap.add_argument(
        "--product-mask-dilate",
        type=int,
        default=0,
        help="Nở mask sản phẩm thêm N px (giúp phủ kín viền khi edge bị blur/chá). 0=tắt",
    )
    ap.add_argument(
        "--lighting-aug",
        type=float,
        default=1.0,
        help="0=tắt; 1=random gốc sáng+độ bóng (LAB+specular) trên viên; >1 tăng mạnh",
    )
    ap.add_argument(
        "--ng-illum-harmonize",
        type=float,
        default=1.0,
        help="Cân luminance tần thấp NG→OK trước blend (giảm diff dính bóng NG). 0=tắt",
    )
    ap.add_argument(
        "--ng-specular-atten",
        type=float,
        default=0.75,
        help="Kéo pixel NG quá sáng (so với OK) về phía OK trong mask — giảm vệt bóng trên diff. 0=tắt",
    )
    ap.add_argument(
        "--ng-second-harmonize",
        type=float,
        default=0.42,
        help="Trộn thêm một pass harmonize σ lớn (0=tắt; ~0.4–0.6 thường hợp)",
    )
    ap.add_argument(
        "--ng-hf-residual",
        type=float,
        default=0.0,
        help="Dùng residual tần cao (loại bỏ vệt bóng/gradient LF). 0=off; 1=full HF",
    )
    ap.add_argument(
        "--ng-align",
        choices=["resize", "shift", "ecc", "grid"],
        default="resize",
        help="Căn NG↔OK trước residual: resize (cũ); shift (chỉ tịnh tiến); ecc (xoay+tịnh tiến); grid (quét góc)",
    )
    ap.add_argument(
        "--ng-align-grid-deg",
        type=float,
        default=3.0,
        help="grid: nửa khoảng góc ±deg (mặc định 3)",
    )
    ap.add_argument(
        "--ng-align-grid-steps",
        type=int,
        default=31,
        help="grid: số góc thử (mặc định 31)",
    )
    ap.add_argument(
        "--ng-align-ecc-max-deg",
        type=float,
        default=8.0,
        help="ecc: bỏ qua nếu |góc ước lượng| vượt quá (độ)",
    )
    ap.add_argument(
        "--ng-align-ecc-min-corr",
        type=float,
        default=0.2,
        help="ecc: correlation tối thiểu; thấp hơn → giữ resize-only",
    )
    ap.add_argument(
        "--ng-align-shift-max-px",
        type=float,
        default=35.0,
        help="shift: giới hạn |dx|,|dy| tối đa (px) để tránh warp quá xa",
    )
    ap.add_argument(
        "--ng-align-shift-min-corr",
        type=float,
        default=0.15,
        help="shift: correlation tối thiểu cho ECC translation",
    )
    ap.add_argument(
        "--rong-base-min",
        type=float,
        default=0.50,
        help="(Rỗng/ng_residual) Alpha nền tối thiểu — cao hơn = rỗng/NG mạnh hơn (mặc định 0.50)",
    )
    ap.add_argument(
        "--rong-dark-k",
        type=float,
        default=2.0,
        help="(Rỗng) Hệ số độ tối feature (mặc định 2.0; giảm nếu quá đen)",
    )
    ap.add_argument(
        "--rong-heavy-mult",
        type=float,
        default=1.2,
        help="(Rỗng) Nhân thêm độ tối + HF texture (1.0=chuẩn cũ sau dark_k; mặc định 1.2)",
    )
    ap.add_argument(
        "--rong-full",
        action="store_true",
        help="Rỗng phủ cả viên (alpha gần như toàn mask) thay vì chỉ 1 dải trong thân",
    )
    ap.add_argument(
        "--rong-empty-cv",
        type=float,
        default=0.55,
        help="CV hollow feel (lõi tối + viền sáng) cho Rỗng. 0=tắt; mặc định 0.55",
    )
    ap.add_argument(
        "--rong-full-delta-k",
        type=float,
        default=0.55,
        help="(rong_full) hệ số inject residual (giảm paste NG). 0=tắt residual; mặc định 0.55",
    )
    ap.add_argument(
        "--rong-full-use-ng-blend",
        action="store_true",
        help="(rong_full) dùng blend NG trực tiếp (OK*(1-a)+NG*a) thay vì residual-inject (dễ ra cảm giác paste nếu NG khác OK)",
    )
    ap.add_argument(
        "--rong-ng-texture-mode",
        choices=["profile_x", "hf"],
        default="profile_x",
        help="(rong_full) Cách lấy texture từ NG khi không blend trực tiếp: profile_x (ổn khi lệch hình học) hoặc hf (nhạy alignment)",
    )
    ap.add_argument(
        "--rong-dehighlight-k",
        type=float,
        default=1.0,
        help="Tăng/giảm mức hạ vùng trắng OK trong mask (empty_cv). 1.0=mặc định; 2.0=giảm trắng mạnh hơn",
    )
    ap.add_argument(
        "--rong-black-core-k",
        type=float,
        default=0.0,
        help="Đen gần như toàn viên, chỉ chừa viền: 0=tắt; ~0.9 rất mạnh",
    )
    ap.add_argument(
        "--rong-rim-frac",
        type=float,
        default=0.16,
        help="Độ dày viền chừa lại (tỉ lệ theo bán kính DT): nhỏ hơn = viền mỏng hơn",
    )
    ap.add_argument(
        "--rong-black-floor",
        type=float,
        default=10.0,
        help="Mức tối sàn cho lõi (0=đen chết; 8–18 thường hợp lý)",
    )
    ap.add_argument(
        "--rong-reflect-k",
        type=float,
        default=0.0,
        help="Texture phản chiếu nhựa (streaks/bands) để diff rõ hơn. 0=tắt; 0.6–1.4 thường hợp lý",
    )
    ap.add_argument(
        "--rong-reflect-streaks",
        type=int,
        default=5,
        help="Số streaks phản chiếu theo trục capsule (mặc định 5)",
    )
    ap.add_argument(
        "--sdxl-refine",
        action="store_true",
        help="Sau synth: SDXLRefiner (texture nhẹ; cần torch+diffusers + model HF; script HondaPlus/scripts/sdxl_refiner.py)",
    )
    ap.add_argument(
        "--sdxl-refine-model",
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="HF repo hoặc thư mục local chứa SDXL inpaint",
    )
    ap.add_argument("--sdxl-refine-strength", type=float, default=0.12, help="≤0.2 để giữ hình học")
    ap.add_argument("--sdxl-refine-guidance", type=float, default=5.0)
    ap.add_argument("--sdxl-refine-steps", type=int, default=18)
    ap.add_argument("--sdxl-refine-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--sdxl-refine-low-vram",
        action="store_true",
        help="enable_model_cpu_offload (khuyến nghị ~6GB VRAM)",
    )
    ap.add_argument(
        "--highlight-despeckle",
        type=float,
        default=0.52,
        help="0=tắt. Sau synth (và SDXL nếu có): bilateral nhẹ chỉ vùng highlight trong mask — giảm nhiễu hạt ở chỗ chá sáng (mặc định 0.52)",
    )
    args = ap.parse_args()

    print(f"Output → {RESULT_DIR}\n")
    # Set globals for run_dai (small CLI plumbing without refactor)
    global FEAT_STRENGTH, FEAT_BANDS, FEAT_HAZE, FEAT_HALO, FEAT_GRAIN, FEAT_SPECULAR, PRODUCT_MASK_MODE, PRODUCT_MASK_DILATE_PX, LIGHTING_AUG_STRENGTH, NG_ILLUM_HARMONIZE, NG_SPECULAR_ATTEN, NG_SECOND_HARMONIZE, NG_HF_RESIDUAL, NG_ALIGN_MODE, NG_ALIGN_GRID_MAX_DEG, NG_ALIGN_GRID_STEPS, NG_ALIGN_ECC_MAX_DEG, NG_ALIGN_ECC_MIN_CORR, NG_ALIGN_SHIFT_MAX_PX, NG_ALIGN_SHIFT_MIN_CORR, RONG_BASE_MIN, RONG_DARK_K, RONG_HEAVY_MULT, RONG_FULL_CAPSULE, RONG_EMPTY_CV, RONG_FULL_DELTA_K, RONG_FULL_USE_NG_BLEND, RONG_NG_TEXTURE_MODE, RONG_DEHIGHLIGHT_K, RONG_BLACK_CORE_K, RONG_RIM_FRAC, RONG_BLACK_FLOOR, RONG_REFLECT_K, RONG_REFLECT_STREAKS, HIGHLIGHT_DESPECKLE, SDXL_REFINE_ENABLED, SDXL_REFINE_CFG, SDXL_REFINE_DEVICE
    FEAT_STRENGTH = float(args.feat_strength)
    PRODUCT_MASK_MODE = str(args.product_mask)
    PRODUCT_MASK_DILATE_PX = int(args.product_mask_dilate)
    LIGHTING_AUG_STRENGTH = float(args.lighting_aug)
    NG_ILLUM_HARMONIZE = float(args.ng_illum_harmonize)
    NG_SPECULAR_ATTEN = float(args.ng_specular_atten)
    NG_SECOND_HARMONIZE = float(args.ng_second_harmonize)
    NG_HF_RESIDUAL = float(args.ng_hf_residual)
    NG_ALIGN_MODE = str(args.ng_align)
    NG_ALIGN_GRID_MAX_DEG = float(args.ng_align_grid_deg)
    NG_ALIGN_GRID_STEPS = int(args.ng_align_grid_steps)
    NG_ALIGN_ECC_MAX_DEG = float(args.ng_align_ecc_max_deg)
    NG_ALIGN_ECC_MIN_CORR = float(args.ng_align_ecc_min_corr)
    NG_ALIGN_SHIFT_MAX_PX = float(args.ng_align_shift_max_px)
    NG_ALIGN_SHIFT_MIN_CORR = float(args.ng_align_shift_min_corr)
    RONG_BASE_MIN = float(args.rong_base_min)
    RONG_DARK_K = float(args.rong_dark_k)
    RONG_HEAVY_MULT = float(args.rong_heavy_mult)
    RONG_FULL_CAPSULE = bool(args.rong_full)
    RONG_EMPTY_CV = float(args.rong_empty_cv)
    RONG_FULL_DELTA_K = float(args.rong_full_delta_k)
    RONG_FULL_USE_NG_BLEND = bool(args.rong_full_use_ng_blend)
    RONG_NG_TEXTURE_MODE = str(args.rong_ng_texture_mode)
    RONG_DEHIGHLIGHT_K = float(args.rong_dehighlight_k)
    RONG_BLACK_CORE_K = float(args.rong_black_core_k)
    RONG_RIM_FRAC = float(args.rong_rim_frac)
    RONG_BLACK_FLOOR = float(args.rong_black_floor)
    RONG_REFLECT_K = float(args.rong_reflect_k)
    RONG_REFLECT_STREAKS = int(args.rong_reflect_streaks)
    HIGHLIGHT_DESPECKLE = float(args.highlight_despeckle)
    SDXL_REFINE_ENABLED = bool(args.sdxl_refine)
    SDXL_REFINE_DEVICE = str(args.sdxl_refine_device)
    SDXL_REFINE_CFG = {
        "model": str(args.sdxl_refine_model),
        "strength": float(args.sdxl_refine_strength),
        "guidance_scale": float(args.sdxl_refine_guidance),
        "steps": int(args.sdxl_refine_steps),
        "prompt": _CAPSULE_SDXL_PROMPT,
        "negative_prompt": _CAPSULE_SDXL_NEGATIVE,
        "enable_model_cpu_offload": bool(args.sdxl_refine_low_vram),
    }
    if SDXL_REFINE_ENABLED:
        # Không tải IP-Adapter: refine chỉ cần text prompt, tránh cảnh báo đường /models/ Docker.
        SDXL_REFINE_CFG["load_ip_adapter"] = False
    # if user doesn't specify any feat flags, default to all on when strength>0
    any_flag = bool(
        args.feat_bands or args.feat_haze or args.feat_halo or args.feat_grain or args.feat_specular
    )
    FEAT_BANDS = bool(args.feat_bands) if any_flag else True
    FEAT_HAZE = bool(args.feat_haze) if any_flag else True
    FEAT_HALO = bool(args.feat_halo) if any_flag else True
    FEAT_GRAIN = bool(args.feat_grain) if any_flag else True
    FEAT_SPECULAR = bool(args.feat_specular) if any_flag else True

    print(
        f"synth_style={args.synth_style}  fixed_region={args.fixed_region}  "
        f"feat_strength={FEAT_STRENGTH}  product_mask={PRODUCT_MASK_MODE}  "
        f"lighting_aug={LIGHTING_AUG_STRENGTH}  ng_illum_harmonize={NG_ILLUM_HARMONIZE}  "
        f"ng_specular_atten={NG_SPECULAR_ATTEN}  ng_second_harmonize={NG_SECOND_HARMONIZE}  "
        f"ng_hf_residual={NG_HF_RESIDUAL}  ng_align={NG_ALIGN_MODE}  "
        f"rong_base_min={RONG_BASE_MIN}  rong_dark_k={RONG_DARK_K}  "
        f"rong_heavy_mult={RONG_HEAVY_MULT}  rong_full={RONG_FULL_CAPSULE}  rong_empty_cv={RONG_EMPTY_CV}  "
        f"rong_full_delta_k={RONG_FULL_DELTA_K}  rong_full_use_ng_blend={RONG_FULL_USE_NG_BLEND}  "
        f"rong_ng_texture_mode={RONG_NG_TEXTURE_MODE}  "
        f"rong_dehighlight_k={RONG_DEHIGHLIGHT_K}  "
        f"rong_black_core_k={RONG_BLACK_CORE_K}  rong_rim_frac={RONG_RIM_FRAC}  rong_black_floor={RONG_BLACK_FLOOR}  "
        f"rong_reflect_k={RONG_REFLECT_K}  rong_reflect_streaks={RONG_REFLECT_STREAKS}  "
        f"highlight_despeckle={HIGHLIGHT_DESPECKLE}  "
        f"sdxl_refine={SDXL_REFINE_ENABLED}\n"
    )
    if args.product in ("all", "Thuốc_dài"):
        run_dai(synth_style=args.synth_style, fixed_region=args.fixed_region, skip_rong=bool(args.skip_rong))
    if args.product in ("all", "Thuốc_tròn"):
        run_tron()
    if args.rois:
        run_rois(
            os.path.normpath(args.rois),
            product=args.rois_product,
            synth_style=args.synth_style,
            fixed_region=args.fixed_region,
        )
    print(f"\nDone → {RESULT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN MỞ RỘNG — theo phản hồi sếp:
#   1. Kiểm soát kích thước lỗi (defect_size)
#   2. Biến thể cấu trúc (structural_variant) — 4 kiểu/class
#   3. GenAI path: SD 1.5 Inpainting local (6GB VRAM, CPU offload)
# ══════════════════════════════════════════════════════════════════════════════

# ── Hằng số kích thước & biến thể ────────────────────────────────────────────

DEFECT_SIZES: dict[str, float] = {
    "small":  0.20,   # 20% chiều cao/rộng viên nang
    "medium": 0.40,   # 40%
    "large":  0.65,   # 65%
}

RONG_VARIANTS   = ["band_center", "band_asymmetric", "band_double", "band_gradient"]
THIEU_VARIANTS  = ["uniform_dark", "collapse_center", "wrinkle_horizontal", "collapse_asymmetric"]

# Prompt SD 1.5 theo từng biến thể
_SD15_PROMPTS_RONG: dict[str, str] = {
    "band_center":
        "pharmaceutical capsule with dark hollow band visible through transparent shell, "
        "empty void in center, translucent gelatin, grayscale product photo, sharp focus",
    "band_asymmetric":
        "capsule partially empty on one side, dark shadow through shell, "
        "asymmetric void, underfilled pharmaceutical, grayscale, high detail",
    "band_double":
        "capsule with two separate dark air pockets visible through shell, "
        "multiple voids at different levels, pharmaceutical defect, grayscale",
    "band_gradient":
        "gelatin capsule with content settled at one end, "
        "gradual dark region from top to bottom, translucent shell, grayscale",
}

_SD15_PROMPTS_THIEU: dict[str, str] = {
    "uniform_dark":
        "underfilled pharmaceutical capsule, uniformly dark matte surface, "
        "compressed gelatin shell, grayscale product photo, sharp focus",
    "collapse_center":
        "capsule shell sagging in center due to insufficient fill, "
        "surface deformation in middle, pharmaceutical defect, grayscale",
    "wrinkle_horizontal":
        "capsule with fine horizontal wrinkle marks on surface, "
        "shell crumpled from underfill, parallel compression lines, grayscale, high detail",
    "collapse_asymmetric":
        "capsule with one end collapsed, asymmetric shell deformation, "
        "insufficient pharmaceutical content, grayscale",
}

_SD15_NEGATIVE = (
    "color, bright colors, text, watermark, blur, cartoon, illustration, "
    "metal, 3d render, CGI, oversaturated, low quality"
)

# Token ngẫu nhiên thêm vào mỗi lần để tránh kết quả lặp (mode collapse)
_SD15_DETAIL_TOKENS = [
    "sharp focus", "high detail", "clinical lighting",
    "slight shadows", "soft ambient light", "overhead lighting",
    "matte surface", "slight specularity", "clean background",
    "industrial inspection photo", "studio lighting",
]


# ── 1. Kiểm soát kích thước — tạo mask con theo tỷ lệ ───────────────────────

def make_sized_mask(
    capsule_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    defect_size: str,
    variant: str,
    seed: int,
) -> np.ndarray:
    """
    Tạo mask lỗi có kích thước kiểm soát được trong capsule mask.

    - defect_size: "small" / "medium" / "large"
    - variant: quyết định vị trí/hình dạng vùng lỗi trong capsule

    Trả về: mask cùng shape với capsule_mask (uint8, 0/255).
    """
    rng = np.random.default_rng(seed)
    ratio = DEFECT_SIZES.get(defect_size, 0.40)
    x, y, w, h = bbox
    H, W = capsule_mask.shape[:2]

    # Vùng trung tâm viên nang (tránh 2 đầu bán cầu)
    margin_x = int(w * 0.12)
    cx0 = x + margin_x
    cx1 = x + w - margin_x

    out_mask = np.zeros((H, W), dtype=np.uint8)

    if variant in RONG_VARIANTS:
        # Rỗng: ref thật cho thấy band chiếm ~60-70% chiều cao viên thuốc
        # (ánh sáng xuyên qua vỏ thấy background đen bên trong)
        # Override ratio: small→0.55, medium→0.65, large→0.75
        rong_ratio = {"small": 0.55, "medium": 0.65, "large": 0.75}.get(defect_size, 0.65)
        band_h = max(4, int(h * rong_ratio))

        if variant == "band_center":
            # Chính giữa viên nang
            by = y + h // 2 - band_h // 2

        elif variant == "band_asymmetric":
            # Lệch sang trái hoặc phải 30-50%
            side = float(rng.choice([-1.0, 1.0]))
            offset = int(w * float(rng.uniform(0.15, 0.28)))
            by = y + h // 2 - band_h // 2

        elif variant == "band_double":
            # Hai dải mỏng, cách nhau: dải 1 phía trên, dải 2 phía dưới
            band_h = max(3, band_h // 2)
            for frac in [0.30, 0.68]:
                by = y + int(h * frac) - band_h // 2
                by = int(np.clip(by, y, y + h - band_h))
                cv2.ellipse(out_mask,
                            (x + w // 2, by + band_h // 2),
                            (max(1, (cx1 - cx0) // 2), max(1, band_h // 2)),
                            0, 0, 360, 255, -1)
            out_mask = cv2.bitwise_and(out_mask, capsule_mask)
            return out_mask

        else:  # band_gradient — band chiếm nửa dưới/trên
            if bool(rng.random() > 0.5):
                by = y + h // 2          # nửa dưới
            else:
                by = y                   # nửa trên
            band_h = max(4, int(h * ratio * 0.9))

        by = int(np.clip(by, y, y + h - band_h))
        cv2.ellipse(out_mask,
                    (x + w // 2, by + band_h // 2),
                    (max(1, (cx1 - cx0) // 2), max(1, band_h // 2)),
                    0, 0, 360, 255, -1)

    else:
        # Thiếu_hàm_lượng: vùng tối theo chiều ngang (X)
        zone_w = max(4, int(w * ratio))

        if variant == "uniform_dark":
            # Toàn thân viên nang
            zone_w = int(w * min(ratio * 1.3, 0.95))
            cx = x + w // 2
            cv2.ellipse(out_mask, (cx, y + h // 2),
                        (zone_w // 2, h // 2 - 2), 0, 0, 360, 255, -1)

        elif variant == "collapse_center":
            # Vùng tối ở giữa (thân viên)
            cx = x + w // 2
            cv2.ellipse(out_mask, (cx, y + h // 2),
                        (zone_w // 2, int(h * 0.38)), 0, 0, 360, 255, -1)

        elif variant == "wrinkle_horizontal":
            # Toàn thân nhưng dạng dài theo trục X
            cv2.ellipse(out_mask,
                        (x + w // 2, y + h // 2),
                        (max(1, int(w * min(ratio * 1.2, 0.90)) // 2),
                         max(1, int(h * 0.42))),
                        0, 0, 360, 255, -1)

        else:  # collapse_asymmetric — một đầu trái hoặc phải
            if bool(rng.random() > 0.5):
                ex = x + zone_w // 2          # đầu trái
            else:
                ex = x + w - zone_w // 2      # đầu phải
            cv2.ellipse(out_mask, (ex, y + h // 2),
                        (zone_w // 2, int(h * 0.40)), 0, 0, 360, 255, -1)

    # Giới hạn trong capsule mask
    out_mask = cv2.bitwise_and(out_mask, capsule_mask)
    return out_mask


# ── 2. Wrapper synth có size + variant ───────────────────────────────────────

# Hệ số boost độ đậm theo kích thước (vùng nhỏ → tăng mạnh hơn để thấy được)
_SIZE_INTENSITY_BOOST: dict[str, float] = {
    "small":  1.55,   # +55% darkening
    "medium": 1.20,   # +20%
    "large":  1.00,   # giữ nguyên
}


def synth_rong_v2(
    ok_bgr: np.ndarray,
    capsule_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
    defect_size: str = "medium",
    variant: str = "band_center",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rỗng: capsule gelatin trong suốt → ánh sáng xuyên qua thấy background đen bên trong.
    Kết quả: dark band nằm ngang ở trung tâm viên thuốc (NOT full capsule).
    Viền trên/dưới giữ sáng (vỏ gelatin phản chiếu).
    sized_mask = band ngang trung tâm (~50-65% chiều cao capsule).
    """
    boosted = float(np.clip(intensity * _SIZE_INTENSITY_BOOST[defect_size], 0.0, 1.0))
    result = synth_rong(ok_bgr, capsule_mask, bbox, seed, boosted)
    return result, capsule_mask


def signal_inject_capsule(
    cv_result_bgr: np.ndarray,
    ng_ref_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    intensity: float = 0.7,
    seed: int = 42,
) -> np.ndarray:
    """
    Hybrid bước 2: inject high-frequency NG feature từ ref thật vào CV result.

    Flow:
      cv_result (dark capsule từ synth_rong)
        ↓  extract HF signal từ ng_ref (texture thật của hollow capsule)
        ↓  resize signal → fit bbox
        ↓  additive inject vào cv_result tại vùng mask
      → result có structure của CV + texture thật của NG

    Args:
        cv_result_bgr : output của synth_rong (uint8 BGR)
        ng_ref_bgr    : ảnh NG ref thật (bất kỳ kích thước, BGR)
        mask          : capsule mask (uint8, white=defect)
        bbox          : (x, y, w, h) bounding box capsule
        intensity     : signal strength (0-1)
        seed          : reproducibility
    """
    if not _HAS_SIGNAL_INJECT:
        print("[signal_inject] SKIP — generator_classical not available.")
        return cv_result_bgr

    random.seed(seed)
    np.random.seed(seed)

    x, y, w, h = bbox
    H, W = cv_result_bgr.shape[:2]

    # Convert BGR→RGB cho signal injection (generator_classical dùng RGB)
    base_rgb = cv2.cvtColor(cv_result_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    ref_rgb  = cv2.cvtColor(ng_ref_bgr,    cv2.COLOR_BGR2RGB).astype(np.float32)

    # Resize ref về cùng kích thước OK để tránh scale mismatch
    ref_rgb = cv2.resize(ref_rgb, (W, H), interpolation=cv2.INTER_LANCZOS4)

    # ── Extract + normalize HF signal từ NG ref ──────────────────────────────
    ref_short   = min(ref_rgb.shape[:2])
    blur_kernel = max(5, ref_short // 8) | 1
    signal = extract_signal(ref_rgb, blur_kernel)

    intensity_scale = 0.5 + intensity * 1.0   # 0.5–1.5
    signal = normalize_signal(signal, intensity_scale * 0.8, intensity_scale * 1.2)

    # ── Soft alpha mask trong bbox ────────────────────────────────────────────
    ys, xs = np.where(mask > 127)
    if not len(ys):
        return cv_result_bgr
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    mask_crop   = mask[y_min:y_max+1, x_min:x_max+1]
    alpha       = create_soft_mask(mask_crop, dilate_size=0, blur_size=7, radial_falloff=True)

    # ── Inject ────────────────────────────────────────────────────────────────
    result_rgb = apply_signal_injection(
        base_rgb, signal, mask_crop, alpha,
        (y_min, y_max, x_min, x_max),
    )
    result_bgr = cv2.cvtColor(np.clip(result_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return result_bgr


def synth_thieu_v2(
    ok_bgr: np.ndarray,
    capsule_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
    defect_size: str = "medium",
    variant: str = "uniform_dark",
) -> tuple[np.ndarray, np.ndarray]:
    """
    synth_thieu_ham_luong() mở rộng: có kiểm soát kích thước và biến thể cấu trúc.
    Trả về (result_bgr, sized_mask).
    """
    sized_mask = make_sized_mask(capsule_mask, bbox, defect_size, variant, seed + 22)
    if not (sized_mask > 127).any():
        sized_mask = capsule_mask
    boosted = float(np.clip(intensity * _SIZE_INTENSITY_BOOST[defect_size], 0.0, 1.0))
    # Blur mask để tránh artifact cạnh sắc từ polygon
    sized_mask_soft = cv2.GaussianBlur(sized_mask, (0, 0), 3.0)
    _, sized_mask_soft = cv2.threshold(sized_mask_soft, 60, 255, cv2.THRESH_BINARY)
    result = synth_thieu_ham_luong(ok_bgr, sized_mask_soft, bbox, seed, boosted)
    return result, sized_mask_soft


# ── 3. SD 1.5 Inpainting — GenAI path (local 6GB VRAM) ──────────────────────

_sd15_pipe = None   # lazy load singleton


def _get_sd15_pipe():
    """Khởi tạo SD 1.5 Inpainting pipeline (chỉ tải 1 lần)."""
    global _sd15_pipe
    if _sd15_pipe is not None:
        return _sd15_pipe
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        print("[SD15] Đang tải runwayml/stable-diffusion-inpainting ...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,   # tránh torch.load CVE với torch < 2.6
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.enable_model_cpu_offload()   # ~3.5 GB VRAM peak — OK với 6GB
        pipe.enable_attention_slicing()   # giảm thêm ~20% VRAM
        _sd15_pipe = pipe
        print("[SD15] Pipeline sẵn sàng.")
        return pipe
    except Exception as e:
        print(f"[SD15] Không khởi tạo được: {e}")
        return None


def _sd15_inpaint(
    ok_bgr: np.ndarray,
    inpaint_mask: np.ndarray,
    prompt: str,
    seed: int,
    strength: float = 0.82,
    guidance: float = 8.0,
    steps: int = 28,
    bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """
    Gọi SD 1.5 inpainting trên vùng mask.
    Nếu có bbox: crop tight quanh capsule, inpaint, paste lại (Crop-Inpaint-Paste).
    Trả về ảnh BGR cùng kích thước với ok_bgr, hoặc None nếu lỗi.
    """
    pipe = _get_sd15_pipe()
    if pipe is None:
        return None

    try:
        import torch
        from PIL import Image

        oh, ow = ok_bgr.shape[:2]

        # --- Crop tight quanh capsule (bbox) với padding nhỏ ---
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            pad = 8
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(ow, x2 + pad)
            cy2 = min(oh, y2 + pad)
            crop_bgr  = ok_bgr[cy1:cy2, cx1:cx2]
            crop_mask = inpaint_mask[cy1:cy2, cx1:cx2]
        else:
            cx1, cy1 = 0, 0
            crop_bgr  = ok_bgr
            crop_mask = inpaint_mask

        ch, cw = crop_bgr.shape[:2]

        # Resize crop về 512×256 (tỷ lệ 2:1 cho viên nang nằm ngang)
        inpaint_w, inpaint_h = 512, 256

        ok_pil   = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).resize(
                       (inpaint_w, inpaint_h), Image.LANCZOS)
        mask_pil = Image.fromarray(crop_mask).resize(
                       (inpaint_w, inpaint_h), Image.NEAREST)

        # Thêm token ngẫu nhiên tránh mode collapse
        rng_tok = np.random.default_rng(seed + 9999)
        extras  = rng_tok.choice(_SD15_DETAIL_TOKENS,
                                  size=int(rng_tok.integers(2, 4)), replace=False)
        full_prompt = prompt + ", " + ", ".join(extras)

        generator = torch.Generator().manual_seed(int(seed))
        result_pil = pipe(
            prompt          = full_prompt,
            negative_prompt = _SD15_NEGATIVE,
            image           = ok_pil,
            mask_image      = mask_pil,
            num_inference_steps = steps,
            guidance_scale  = guidance,
            strength        = strength,
            generator       = generator,
        ).images[0]

        # Chuyển về grayscale rồi back BGR (camera sản xuất = grayscale)
        result_gray = result_pil.convert("L").convert("RGB")
        result_crop = cv2.cvtColor(np.array(result_gray), cv2.COLOR_RGB2BGR)

        # Resize crop result về kích thước crop gốc
        result_crop = cv2.resize(result_crop, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

        # Paste lại vào full image
        result_bgr = ok_bgr.copy()
        result_bgr[cy1:cy1+ch, cx1:cx1+cw] = result_crop
        return result_bgr

    except Exception as e:
        print(f"[SD15] inpaint lỗi: {e}")
        return None


def synth_rong_sd15(
    ok_bgr: np.ndarray,
    capsule_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
    defect_size: str = "medium",
    variant: str = "band_center",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rỗng qua SD 1.5 Inpainting — mask = 100% capsule.
    Fallback sang synth_rong_v2() nếu SD15 không khả dụng.
    """
    # Rỗng: toàn bộ viên thuốc là vùng inpaint
    prompt   = _SD15_PROMPTS_RONG.get(variant, _SD15_PROMPTS_RONG["band_center"])
    strength = 0.75 + intensity * 0.10

    result = _sd15_inpaint(ok_bgr, capsule_mask, prompt, seed,
                           strength=strength, guidance=8.0, steps=28, bbox=bbox)
    if result is None:
        print(f"  [SD15 fallback → CV] {variant}")
        return synth_rong_v2(ok_bgr, capsule_mask, bbox, seed, intensity,
                              defect_size=defect_size, variant=variant)
    return result, capsule_mask   # full capsule mask


def synth_thieu_sd15(
    ok_bgr: np.ndarray,
    capsule_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int,
    intensity: float,
    defect_size: str = "medium",
    variant: str = "uniform_dark",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Thiếu_hàm_lượng qua SD 1.5 Inpainting. Trả về (result_bgr, sized_mask).
    Fallback sang synth_thieu_v2() nếu SD15 không khả dụng.
    """
    sized_mask = make_sized_mask(capsule_mask, bbox, defect_size, variant, seed + 44)
    if not (sized_mask > 127).any():
        sized_mask = capsule_mask

    prompt   = _SD15_PROMPTS_THIEU.get(variant, _SD15_PROMPTS_THIEU["uniform_dark"])
    strength = 0.78 + intensity * 0.08

    result = _sd15_inpaint(ok_bgr, sized_mask, prompt, seed,
                           strength=strength, guidance=8.5, steps=28, bbox=bbox)
    if result is None:
        print(f"  [SD15 fallback → CV] {variant}")
        return synth_thieu_v2(ok_bgr, capsule_mask, bbox, seed, intensity,
                               defect_size=defect_size, variant=variant)
    return result, sized_mask


# ── Hybrid: CV → SD img2img refinement ───────────────────────────────────────

_sdxl_inpaint_pipe = None

def _get_sdxl_inpaint_pipe():
    """SDXL Inpainting pipeline (singleton, cpu_offload)."""
    global _sdxl_inpaint_pipe
    if _sdxl_inpaint_pipe is not None:
        return _sdxl_inpaint_pipe
    try:
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline
        print("[SDXL] Tải SDXL Inpainting pipeline ...")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _dtype  = torch.float16 if _device == "cuda" else torch.float32
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=_dtype,
            variant="fp16",
            use_safetensors=True,
        )
        if _device == "cuda":
            pipe = pipe.to("cuda")
            print(f"[SDXL] Chạy trên GPU ({torch.cuda.get_device_name(0)})")
        else:
            pipe.enable_model_cpu_offload()
            print("[SDXL] Không có GPU — fallback CPU offload (chậm)")
        pipe.enable_attention_slicing()
        _sdxl_inpaint_pipe = pipe
        print("[SDXL] Pipeline sẵn sàng.")
        return pipe
    except Exception as e:
        print(f"[SDXL] Không tải được pipeline: {e}")
        return None


# ── Prompts SDXL cho từng loại lỗi ──────────────────────────────────────────
# SDXL hỗ trợ prompt dài hơn và chi tiết hơn SD 1.5
_SDXL_PROMPTS = {
    "rong": (
        # Insight từ ref thật:
        # - Vỏ gelatin TRONG SUỐT → ánh sáng xuyên qua
        # - Bên trong rỗng = background đen nhìn thấy được
        # - Viền ngoài (rim) CÒN SÁNG + bóng loáng vì vỏ phản chiếu ánh sáng
        # - Trung tâm viên TỐI GẦN ĐEN (interior void)
        # - Gradient sáng → tối → sáng từ rim ngoài → center → rim ngoài
        # - Texture bề mặt bóng mịn của gelatin vẫn còn ở viền
        "pharmaceutical hard gelatin capsule completely empty inside, "
        "transparent gelatin shell with bright glossy specular highlight on outer rim, "
        "dark black void visible through transparent shell in center, "
        "strong contrast between bright shiny rim and dark hollow interior, "
        "concentric gradient: bright glossy edge transitioning to deep dark center, "
        "smooth glossy gelatin surface texture on capsule wall, "
        "macro industrial inspection photo, grayscale, photorealistic, sharp focus",
        # Negative — tránh model sinh ra filled capsule hoặc mất texture bóng
        "color, powder filled interior, bright center, uniform brightness, "
        "matte flat surface, no rim highlight, opaque shell, "
        "watermark, text, blur, cartoon, deformed shape",
    ),
    "thieu": (
        # Thiếu hàm lượng: bề mặt hơi tối + lõm nhẹ do không đủ content bên trong
        "pharmaceutical capsule slightly underfilled with content, "
        "surface marginally darker and subtly compressed, "
        "minor surface deformation visible, matte gray, "
        "industrial quality inspection photo, grayscale, "
        "photorealistic, macro photography, sharp focus, no background",
        # Negative
        "color, completely hollow, severely deformed, "
        "perfect filled capsule, bright shiny, watermark, text, blur, cartoon",
    ),
}


def _refine_with_sdxl(
    cv_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    capsule_mask: np.ndarray,      # full capsule mask (255 = capsule region)
    defect_slug: str,              # "rong" hoặc "thieu"
    seed: int,
    strength: float = 0.55,       # Rỗng: 0.50-0.65 (thay đổi toàn bộ capsule)
    guidance: float = 7.5,        # Thiếu: 0.30-0.45 (subtle)
    steps: int = 30,
) -> np.ndarray:
    """
    SDXL Inpainting trên crop capsule.

    Rỗng:  mask = 100% viên thuốc → SDXL tái tạo toàn bộ bề mặt capsule
           theo prompt "hollow inside" → ánh sáng và texture thay đổi cả viên
    Thiếu: mask = vùng defect (sized_mask) → chỉ thay đổi vùng thiếu hàm lượng

    strength cao hơn SD 1.5 vì:
    - SDXL prompt following tốt hơn → không drift khỏi domain
    - Có mask explicit → không lo thay đổi vùng ngoài mask
    """
    pipe = _get_sdxl_inpaint_pipe()
    if pipe is None:
        return cv_bgr

    try:
        import torch
        from PIL import Image

        oh, ow = cv_bgr.shape[:2]
        bx1, by1, bx2, by2 = bbox
        pad = 8
        cx1 = max(0, bx1 - pad);  cy1 = max(0, by1 - pad)
        cx2 = min(ow, bx2 + pad); cy2 = min(oh, by2 + pad)

        crop_img  = cv_bgr[cy1:cy2, cx1:cx2]
        # Rỗng: dùng toàn bộ capsule mask; Thiếu: dùng mask được truyền vào
        crop_mask = capsule_mask[cy1:cy2, cx1:cx2]
        ch, cw    = crop_img.shape[:2]

        # SDXL native: 1024×1024, nhưng hỗ trợ 1024×512 cho landscape
        # Viên thuốc nằm ngang → dùng 1024×512
        iw, ih = 1024, 512

        img_pil  = Image.fromarray(
            cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        ).resize((iw, ih), Image.LANCZOS)
        mask_pil = Image.fromarray(crop_mask).resize((iw, ih), Image.NEAREST)

        pos_p, neg_p = _SDXL_PROMPTS.get(defect_slug, _SDXL_PROMPTS["rong"])
        gen = torch.Generator().manual_seed(seed)

        out_pil = pipe(
            prompt              = pos_p,
            negative_prompt     = neg_p,
            image               = img_pil,
            mask_image          = mask_pil,
            strength            = strength,
            guidance_scale      = guidance,
            num_inference_steps = steps,
            generator           = gen,
        ).images[0]

        # Camera công nghiệp = grayscale → convert
        out_gray = out_pil.convert("L").convert("RGB")
        out_bgr  = cv2.cvtColor(np.array(out_gray), cv2.COLOR_RGB2BGR)
        out_bgr  = cv2.resize(out_bgr, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

        # Paste lại vào full image
        result = cv_bgr.copy()
        result[cy1:cy1+ch, cx1:cx1+cw] = out_bgr
        return result

    except Exception as e:
        print(f"[SDXL] inpaint lỗi: {e}")
        return cv_bgr


# ── 4. run_dai_v2 — chạy toàn bộ ma trận đa dạng ────────────────────────────

def run_dai_v2(use_genai: bool = False, refine: bool = False,
               refine_seeds: list[int] | None = None,
               refine_strength: float = 0.28,
               signal: bool = False,
               signal_intensity: float = 0.7,
               n: int = 0):
    """
    Chạy sinh ảnh Thuốc_dài theo kế hoạch mới.

    Modes:
      use_genai=False, refine=False, signal=False → CV thuần (24 ảnh/class)
      use_genai=True               → SD 1.5 Inpainting (4 ảnh preview)
      refine=True                  → Hybrid: CV → SDXL img2img refinement
      signal=True                  → Hybrid: CV → Signal Injection từ NG ref
                                     Inject HF texture thật từ NG ref vào CV dark base
                                     Không cần GPU, chạy nhanh hơn SDXL refine
    """
    product       = "Thuốc_dài"
    root          = CAPSULE_ROOTS[product]
    refine_seeds  = refine_seeds or [42, 137, 256]

    defect_configs = {
        "Rỗng":             (RONG_VARIANTS,  synth_rong_sd15  if use_genai else synth_rong_v2,  "rong"),
        "Thiếu_hàm_lượng": (THIEU_VARIANTS, synth_thieu_sd15 if use_genai else synth_thieu_v2, "thieu"),
    }

    if signal:
        mode_tag = "signal"
    elif refine:
        mode_tag = "hybrid"
    elif use_genai:
        mode_tag = "sd15"
    else:
        mode_tag = "cv"

    for defect_name, (variants, synth_fn, slug) in defect_configs.items():
        defect_path = os.path.join(root, defect_name)
        ok = load_ok(defect_path)
        if ok is None:
            print(f"[SKIP] Không có ảnh OK: {defect_path}")
            continue

        mask_product, _, bbox, _ = auto_product_mask_and_synth_mask(
            ok, defect_path, "threshold", FALLBACK_DAI
        )

        # Load NG ref cho signal injection
        ng_ref_bgr = None
        if signal:
            ref_dir = os.path.join(defect_path, "ref")
            ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.png")) +
                               glob.glob(os.path.join(ref_dir, "*.jpg")))
            if ref_files:
                ng_ref_bgr = cv2.imread(ref_files[0])
                print(f"  [SIGNAL] NG ref: {os.path.basename(ref_files[0])} {ng_ref_bgr.shape if ng_ref_bgr is not None else 'FAIL'}")
            else:
                print(f"  [SIGNAL] WARN: no ref images in {ref_dir} — signal inject skipped")

        print(f"\n[{product}/{defect_name}] mode={mode_tag} bbox={bbox}")

        # Khi GenAI-only: preview nhỏ
        variants_run = variants[:1]       if use_genai else variants
        sizes_run    = ["medium", "large"] if use_genai else list(DEFECT_SIZES.keys())
        _max_images  = n   # 0 = không giới hạn

        _n_gen = 0
        for vi, variant in enumerate(variants_run):
            for si, size_name in enumerate(sizes_run):
                for ii, intensity in enumerate(INTENSITIES):
                    if _max_images and _n_gen >= _max_images:
                        break
                    cv_seed = vi * 10000 + si * 1000 + ii * 100 + hash(defect_name) % 97
                    _n_gen += 1

                    # ── Bước 1: CV synthesis ──────────────────────────────
                    cv_result, sized_mask = synth_fn(
                        ok, mask_product, bbox, cv_seed, intensity,
                        defect_size=size_name, variant=variant,
                    )

                    if signal and ng_ref_bgr is not None:
                        # ── Hybrid Signal Injection: CV dark base + NG HF texture ──
                        # Lưu CV gốc
                        tag_cv = f"{slug}_cv_{variant}_{size_name}_i{int(intensity*10)}.jpg"
                        save_capsule_result(product, defect_name, tag_cv,
                                            ok, sized_mask, cv_result, bbox=bbox)
                        # Signal inject với nhiều seeds để tăng diversity texture
                        for si_idx, sig_seed in enumerate([cv_seed, cv_seed+1, cv_seed+2]):
                            injected = signal_inject_capsule(
                                cv_result, ng_ref_bgr, mask_product, bbox,
                                intensity=signal_intensity * intensity,
                                seed=sig_seed,
                            )
                            tag_sig = (f"{slug}_signal_{variant}_{size_name}"
                                       f"_i{int(intensity*10)}_s{si_idx}.jpg")
                            save_capsule_result(product, defect_name, tag_sig,
                                                ok, sized_mask, injected, bbox=bbox)
                            print(f"    [SIGNAL seed={sig_seed}] → {tag_sig}")
                            # QC cho Rỗng
                            if slug == "rong":
                                metrics = auto_check_rong(ok, injected, mask_product, bbox)
                                print(f"    [QC] {metrics['verdict']} | "
                                      f"interior={metrics['res_int']:.0f}(ok={metrics['ok_int']:.0f}) "
                                      f"dark={metrics['interior_dark']} | "
                                      f"contrast={metrics['contrast_ratio']}")

                    elif not refine:
                        # CV thuần hoặc GenAI-only → lưu trực tiếp
                        tag = f"{slug}_{mode_tag}_{variant}_{size_name}_i{int(intensity*10)}.jpg"
                        save_capsule_result(product, defect_name, tag,
                                            ok, sized_mask, cv_result, bbox=bbox)
                        # Auto-check chất lượng Rỗng
                        if slug == "rong":
                            metrics = auto_check_rong(ok, cv_result, mask_product, bbox)
                            print(f"    [QC] {metrics['verdict']} | "
                                  f"interior={metrics['res_int']:.0f}(ok={metrics['ok_int']:.0f}) "
                                  f"dark={metrics['interior_dark']} | "
                                  f"rim={metrics['res_rim']:.0f}(ok={metrics['ok_rim']:.0f}) "
                                  f"bright={metrics['rim_bright']} | "
                                  f"contrast={metrics['contrast_ratio']} spec_boost={metrics['specular_boost']}")
                    else:
                        # ── Bước 2: Hybrid — refine CV result với SD img2img ──
                        # Lưu CV gốc để so sánh
                        tag_cv = f"{slug}_cv_{variant}_{size_name}_i{int(intensity*10)}.jpg"
                        save_capsule_result(product, defect_name, tag_cv,
                                            ok, sized_mask, cv_result, bbox=bbox)

                        # Refine với SDXL — mask và strength khác nhau theo loại lỗi
                        # Rỗng:  mask = 100% viên thuốc, strength cao hơn (0.55)
                        #        vì toàn bộ capsule phải trông khác (rỗng bên trong)
                        # Thiếu: mask = vùng defect (sized_mask), strength thấp hơn (0.40)
                        #        vì chỉ một phần capsule bị ảnh hưởng
                        if slug == "rong":
                            refine_mask   = mask_product   # 100% capsule
                            sdxl_strength = 0.55
                        else:
                            refine_mask   = sized_mask     # vùng thiếu
                            sdxl_strength = 0.40

                        for ri, rseed in enumerate(refine_seeds):
                            refined = _refine_with_sdxl(
                                cv_result, bbox,
                                capsule_mask = refine_mask,
                                defect_slug  = slug,
                                seed         = rseed,
                                strength     = sdxl_strength,
                            )
                            tag_r = (f"{slug}_hybrid_{variant}_{size_name}"
                                     f"_i{int(intensity*10)}_r{ri}.jpg")
                            save_capsule_result(product, defect_name, tag_r,
                                                ok, sized_mask, refined, bbox=bbox)
                            print(f"    [SDXL refine seed={rseed} strength={sdxl_strength}] → {tag_r}")

    total_cv  = len(variants_run) * len(sizes_run) * len(INTENSITIES)
    total_out = total_cv * (1 + len(refine_seeds)) if refine else total_cv
    print(f"\nDone → {RESULT_DIR}")
    print(f"  CV base: {total_cv} ảnh/class")
    if refine:
        print(f"  Hybrid output: {total_out} ảnh/class "
              f"(1 CV + {len(refine_seeds)} refined per config)")


# ── Entry point CLI mới ───────────────────────────────────────────────────────

def main_v2():
    """
    Chạy: python capsule_experiments.py --v2
          python capsule_experiments.py --v2 --genai      (cần diffusers)
    """
    import argparse
    p = argparse.ArgumentParser(description="Capsule synthesis v2 (size + variant + SD15 + Hybrid)")
    p.add_argument("--v2",              action="store_true", help="Dùng pipeline v2 mới")
    p.add_argument("--genai",           action="store_true", help="SD 1.5 Inpainting (GenAI path)")
    p.add_argument("--refine",          action="store_true", help="Hybrid: CV → SD img2img refinement")
    p.add_argument("--refine_seeds",    type=int, nargs="+", default=[42, 137, 256],
                   help="Seeds cho refinement (mỗi seed = 1 texture variation)")
    p.add_argument("--refine_strength", type=float, default=0.28,
                   help="SD img2img strength (0.20-0.40, thấp hơn = giữ CV nhiều hơn)")
    p.add_argument("--signal",           action="store_true",
                   help="Hybrid: CV dark base → Signal Injection từ NG ref (không cần GPU)")
    p.add_argument("--signal_intensity", type=float, default=0.7,
                   help="Signal injection strength (0.3-1.0, cao hơn = NG texture mạnh hơn)")
    p.add_argument("--n", type=int, default=0,
                   help="Giới hạn số ảnh gen mỗi class (0 = không giới hạn, dùng khi debug)")
    p.add_argument("--tron_nut", action="store_true",
                   help="Gen Nứt vỡ cho Thuốc_tròn")
    p.add_argument("--tron_nut_sdxl", action="store_true",
                   help="Gen Nứt + SDXL crack texture tại broken edge")
    args, _ = p.parse_known_args()

    if args.tron_nut_sdxl:
        run_tron_nut(n=args.n, sdxl_crack=True)
    elif args.tron_nut:
        run_tron_nut(n=args.n)
    elif args.v2:
        run_dai_v2(
            use_genai        = args.genai,
            refine           = args.refine,
            refine_seeds     = args.refine_seeds,
            refine_strength  = args.refine_strength,
            signal           = args.signal,
            signal_intensity = args.signal_intensity,
            n                = args.n,
        )
    else:
        main()   # pipeline cũ


# ═══════════════════════════════════════════════════════════════════════════════
# THUỐC TRÒN — NỨT (chip/break defect)
# Pipeline: CV mask → random cut line → remove chunk → blend edge
# ═══════════════════════════════════════════════════════════════════════════════

def detect_tablet_mask(ok_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    """
    Detect tablet mask bằng Otsu threshold.
    Returns: (mask uint8, bbox (x,y,w,h))
    """
    gray = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    H, W    = ok_bgr.shape[:2]

    def _best_contour(thresh_img):
        t = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, k_close)
        t = cv2.morphologyEx(t,          cv2.MORPH_OPEN,  k_open)
        cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0
        c = max(cnts, key=cv2.contourArea)
        return c, cv2.contourArea(c)

    # Try INV (dark tablet, light bg) and normal (light tablet, dark bg)
    _, t_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, t_nrm = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    c_inv, a_inv = _best_contour(t_inv)
    c_nrm, a_nrm = _best_contour(t_nrm)

    # Pick the polarity that gives larger tablet contour
    # but not more than 90% of image (that would be background)
    max_area = H * W * 0.90
    if a_inv > a_nrm and a_inv < max_area:
        cnt = c_inv
    elif a_nrm > 0 and a_nrm < max_area:
        cnt = c_nrm
    elif c_inv is not None:
        cnt = c_inv
    else:
        return np.zeros((H, W), np.uint8), (0, 0, W, H)

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
    x, y, w, h = cv2.boundingRect(cnt)
    return mask, (x, y, w, h)


def _make_smooth_cut_mask(H, W, cx, cy, r, nx, ny, cut_d, rng,
                          break_type: str = "straight"):
    """
    Tạo remove_side mask theo break_type:
      straight   — đường cắt ngang (wavy)
      corner     — góc bể vát: 2 đường cắt giao nhau tại mép
      curved     — đường cong lõm vào tâm
      concave    — mảnh vỡ hình lưỡi liềm (cong ra ngoài)
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    tx, ty = -ny, nx
    px, py = cx + cut_d * nx, cy + cut_d * ny
    dot_n  = (xx - px) * nx + (yy - py) * ny
    dot_t  = (xx - px) * tx + (yy - py) * ty

    # shared wave noise (roughness)
    amp   = float(rng.uniform(2.0, 5.0))
    freq  = float(rng.uniform(0.02, 0.04))
    phase = float(rng.uniform(0, 2 * np.pi))
    wave  = amp * np.sin(freq * dot_t + phase)
    wave += (amp * 0.4) * np.sin(freq * 2.3 * dot_t + phase * 1.7)
    wave += (amp * 0.12) * np.sin(freq * 5.5 * dot_t + phase * 0.9)

    if break_type == "straight":
        eff_dist    = dot_n - wave
        remove_side = (eff_dist < 0).astype(np.uint8) * 255

    elif break_type == "corner":
        # Hai đường cắt giao nhau → bể góc tam giác
        # Đường 1: chính (nx, ny) như straight
        # Đường 2: vuông góc thêm cut_d2 ở hướng tang
        cut_d2     = float(rng.uniform(0.15, 0.40)) * r
        angle2_off = float(rng.uniform(50, 80))  # độ lệch so với cut chính
        a2         = np.radians(angle2_off)
        nx2 = nx * np.cos(a2) - ny * np.sin(a2)
        ny2 = nx * np.sin(a2) + ny * np.cos(a2)
        tx2, ty2   = -ny2, nx2
        px2, py2   = cx + cut_d2 * nx2, cy + cut_d2 * ny2
        dot_n2     = (xx - px2) * nx2 + (yy - py2) * ny2
        dot_t2     = (xx - px2) * tx2 + (yy - py2) * ty2
        wave2      = amp * 0.7 * np.sin(freq * dot_t2 + phase * 1.3)

        eff1        = dot_n  - wave
        eff2        = dot_n2 - wave2
        # Bể góc = vùng nằm ở phía cut1 VÀ cut2 cùng lúc
        remove_side = ((eff1 < 0) & (eff2 < 0)).astype(np.uint8) * 255
        eff_dist    = np.maximum(eff1, eff2)   # dist tới đường vỡ gần nhất

    elif break_type == "curved":
        # Đường cong lõm: thêm radial curvature vào wave
        dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        curvature  = float(rng.uniform(0.3, 0.7)) / (r + 1e-6)
        curve_wave = curvature * dist_from_center * dist_from_center * 0.05
        eff_dist    = dot_n - wave - curve_wave
        remove_side = (eff_dist < 0).astype(np.uint8) * 255

    elif break_type == "concave":
        # Mảnh vỡ hình lưỡi liềm: cut line cong ra ngoài (lõm từ phía center)
        dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        curve_wave = -float(rng.uniform(0.2, 0.5)) * (r - dist_from_center) * 0.08
        eff_dist    = dot_n - wave + curve_wave
        remove_side = (eff_dist < 0).astype(np.uint8) * 255

    else:
        eff_dist    = dot_n - wave
        remove_side = (eff_dist < 0).astype(np.uint8) * 255

    return remove_side, eff_dist


def _make_zigzag_cut_mask(H, W, cx, cy, r, angle_deg, depth_frac, n_bends, rng):
    """
    Đường nứt gãy khúc n_bends lần — như bẻ viên thuốc nhiều lần.
    Dùng connected-components để tìm chip region (robust với mọi hình dạng path).

    Returns:
      crack_img  : uint8 (H,W) — đường nứt rasterized (255 = crack pixel)
      eff_dist   : float32 (H,W) — khoảng cách tới crack polyline (>0)
      pts_px     : list of (x,y) waypoints
    """
    angle_rad = np.radians(angle_deg)
    cur_dir   = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)

    # Start: ngoài tablet một chút ở phía -cur_dir
    start = np.array([cx - (r + 8) * cur_dir[0],
                      cy - (r + 8) * cur_dir[1]], dtype=np.float32)
    cur   = start.copy()
    pts   = [cur.copy()]

    total_len   = 2.0 * r + 16.0                  # đủ dài để xuyên qua tablet
    seg_base    = total_len / (n_bends + 1)

    for i in range(n_bends):
        seg_len = seg_base * float(rng.uniform(0.75, 1.25))
        cur     = cur + cur_dir * seg_len
        pts.append(cur.copy())
        # Bẻ: ±25-45°
        turn = float(rng.uniform(25.0, 45.0)) * rng.choice([-1, 1])
        c_, s_ = np.cos(np.radians(turn)), np.sin(np.radians(turn))
        cur_dir = np.array([c_ * cur_dir[0] - s_ * cur_dir[1],
                            s_ * cur_dir[0] + c_ * cur_dir[1]], dtype=np.float32)
        cur_dir /= np.linalg.norm(cur_dir) + 1e-6

    # Đoạn cuối — đi qua hết tablet
    cur = cur + cur_dir * (seg_base * float(rng.uniform(0.9, 1.3)) + 10)
    pts.append(cur)

    pts_px = [(int(round(float(p[0]))), int(round(float(p[1])))) for p in pts]

    # ── Rasterize crack line với tiny pixel-level noise ───────────────────────
    crack_img = np.zeros((H, W), np.uint8)
    for i in range(len(pts_px) - 1):
        p0 = np.array(pts_px[i],   dtype=np.float32)
        p1 = np.array(pts_px[i+1], dtype=np.float32)
        seg_v    = p1 - p0
        seg_len_ = float(np.linalg.norm(seg_v)) + 1e-6
        perp_u   = np.array([-seg_v[1], seg_v[0]]) / seg_len_
        n_sub    = max(3, int(seg_len_ / 2))
        prev_pt  = p0
        for j in range(1, n_sub + 1):
            t      = j / n_sub
            mid    = p0 + t * seg_v
            noise  = float(rng.uniform(-1.0, 1.0))
            mid_n  = (int(np.clip(round(mid[0] + perp_u[0] * noise), 0, W-1)),
                      int(np.clip(round(mid[1] + perp_u[1] * noise), 0, H-1)))
            cv2.line(crack_img,
                     (int(np.clip(round(prev_pt[0]), 0, W-1)),
                      int(np.clip(round(prev_pt[1]), 0, H-1))),
                     mid_n, 255, 1)
            prev_pt = np.array(mid_n, dtype=np.float32)

    # ── eff_dist: unsigned distance từ mỗi pixel tới polyline ────────────────
    yy, xx   = np.mgrid[0:H, 0:W].astype(np.float32)
    eff_dist = np.full((H, W), 9999.0, np.float32)
    for i in range(len(pts) - 1):
        p0f = pts[i].astype(np.float32)
        p1f = pts[i + 1].astype(np.float32)
        sv  = p1f - p0f
        sl  = float(np.linalg.norm(sv)) + 1e-6
        su  = sv / sl
        dx  = xx - p0f[0];  dy = yy - p0f[1]
        t   = np.clip(dx * su[0] + dy * su[1], 0.0, sl)
        d   = np.sqrt((xx - (p0f[0] + t * su[0]))**2 +
                      (yy - (p0f[1] + t * su[1]))**2)
        eff_dist = np.minimum(eff_dist, d)

    return crack_img, eff_dist, pts_px


_NUT_PROMPT_POS = (
    "broken pharmaceutical tablet fragment, fractured pill surface, "
    "rough irregular broken edge, exposed tablet cross-section material, "
    "micro-cracks and chips along fracture line, powdery broken surface, "
    "grayscale industrial inspection photo, photorealistic, sharp focus, macro"
)
_NUT_PROMPT_NEG = (
    "color, smooth edge, perfect surface, whole tablet, "
    "blur, cartoon, watermark, text, unrealistic"
)


def sdxl_inpaint_crack_edge(
    cv_result_bgr: np.ndarray,
    remain_bool:   np.ndarray,
    chip_bool:     np.ndarray,
    tablet_mask:   np.ndarray,
    seed:          int = 42,
    edge_width:    int = 12,
    strength:      float = 0.82,
    steps:         int = 8,
    guidance:      float = 5.0,
) -> np.ndarray:
    """
    SDXL inpaint dọc theo crack edge để tạo texture vỡ photorealistic.

    Flow:
      1. Tạo crack_edge_mask = strip mỏng (edge_width px) ngay bên trong
         remaining piece, dọc theo đường vỡ (tiếp giáp với chip region)
      2. Crop tight quanh tablet → upscale 1024px cho SDXL
      3. SDXL inpaint với prompt crack texture
      4. Downscale + paste back, chỉ tại crack_edge_mask

    edge_width: độ rộng strip được inpaint (px trong ảnh gốc)
    """
    from PIL import Image as _PIL

    pipe = _get_sdxl_inpaint_pipe()
    if pipe is None:
        print("[CRACK SDXL] Pipeline không khả dụng, trả về CV result.")
        return cv_result_bgr

    import torch
    H, W = cv_result_bgr.shape[:2]

    # ── 1. Crack edge mask: dilate chip_bool vào remain_bool ─────────────────
    chip_u8  = chip_bool.astype(np.uint8) * 255
    k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width*2+1, edge_width*2+1))
    dilated  = cv2.dilate(chip_u8, k)
    # Strip = trong remain_bool VÀ gần chip (dilated) — strictly trong tablet
    crack_mask = ((dilated > 0) & remain_bool).astype(np.uint8) * 255

    if crack_mask.sum() == 0:
        return cv_result_bgr

    # ── 2. Crop bbox quanh tablet (với padding) ───────────────────────────────
    ys, xs = np.where(tablet_mask > 127)
    pad     = 20
    y0 = max(0, int(ys.min()) - pad);  y1 = min(H, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad);  x1 = min(W, int(xs.max()) + pad + 1)

    crop_img  = cv_result_bgr[y0:y1, x0:x1]
    crop_mask = crack_mask[y0:y1, x0:x1]
    ch, cw    = crop_img.shape[:2]

    # ── 3. Upscale crop → 1024×1024 cho SDXL ────────────────────────────────
    sdxl_sz = 1024
    scale_y  = sdxl_sz / ch
    scale_x  = sdxl_sz / cw
    img_up   = cv2.resize(cv2.cvtColor(crop_img,  cv2.COLOR_BGR2RGB),
                          (sdxl_sz, sdxl_sz), interpolation=cv2.INTER_LANCZOS4)
    msk_up   = cv2.resize(crop_mask, (sdxl_sz, sdxl_sz), interpolation=cv2.INTER_NEAREST)
    # Dilate mask sau upscale để SDXL có đủ context
    msk_up   = cv2.dilate(msk_up, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))

    img_pil  = _PIL.fromarray(img_up)
    msk_pil  = _PIL.fromarray(msk_up)

    # ── 4. SDXL Inpaint ───────────────────────────────────────────────────────
    gen = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        out = pipe(
            prompt          = _NUT_PROMPT_POS,
            negative_prompt = _NUT_PROMPT_NEG,
            image           = img_pil,
            mask_image      = msk_pil,
            width=sdxl_sz, height=sdxl_sz,
            strength        = strength,
            guidance_scale  = guidance,
            num_inference_steps = steps,
            generator       = gen,
        ).images[0]

    # ── 5. Downscale back + paste chỉ vùng crack_mask ────────────────────────
    out_bgr  = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    out_down = cv2.resize(out_bgr, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

    result   = cv_result_bgr.copy()
    # Feather blend tại crack mask để tránh seam
    cmask_f  = (crop_mask > 127).astype(np.float32)
    cmask_f  = cv2.GaussianBlur(cmask_f, (0,0), 2.0)
    cmask_f  = np.clip(cmask_f, 0, 1)[:, :, None]

    roi_orig = result[y0:y1, x0:x1].astype(np.float32)
    roi_new  = out_down.astype(np.float32)
    result[y0:y1, x0:x1] = np.clip(roi_new * cmask_f + roi_orig * (1 - cmask_f), 0, 255).astype(np.uint8)

    print(f"[CRACK SDXL] Done seed={seed} strength={strength}")
    return result


def _measure_tablet_edge_blur(ok_bgr: np.ndarray, mask: np.ndarray) -> float:
    """
    Đo độ mờ tự nhiên của rìa tablet trong ảnh camera.
    Dùng để match broken edge với outer edge cùng sigma.
    """
    gray = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dist = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 3)
    # Lấy gradient magnitude tại vùng rìa (dist 1-8px)
    edge_zone = (dist > 0.5) & (dist < 8.0)
    if not edge_zone.any():
        return 3.0
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    # Sigma ≈ bao nhiêu pixel để gradient giảm về 1/e
    edge_grads = grad_mag[edge_zone]
    # Ước tính: sigma ~ half-width of gradient peak tại rìa
    sigma = float(np.clip(np.percentile(dist[edge_zone], 75) / 2.0, 1.5, 6.0))
    return sigma


def synth_nut_tron(
    ok_bgr:     np.ndarray,
    mask:       np.ndarray,
    bbox:       tuple[int,int,int,int],
    seed:       int = 0,
    intensity:  float = 0.8,
    cut_depth:  float | None = None,
    cut_angle:  float | None = None,
    break_type: str = "straight",   # straight | corner | curved | concave
    sdxl_crack: bool = False,
    crack_edge_width: int = 12,
    crack_strength:   float = 0.82,
) -> np.ndarray:
    """
    Viên thuốc tròn bị vỡ/nứt — physical break simulation.

    Key insight: mảnh vỡ là physical object — toàn bộ rìa của nó
    (kể cả đường vỡ) phải có soft edge giống hệt rìa tablet tự nhiên.

    Approach: composite remaining_piece với feathered alpha lên background,
    thay vì fill + draw seam line.
    """
    rng = np.random.default_rng(seed)
    H, W = ok_bgr.shape[:2]
    x, y, w, h = bbox
    cx = float(x + w / 2); cy = float(y + h / 2)
    r  = float(min(w, h) / 2)

    angle_deg  = float(rng.uniform(0, 360)) if cut_angle is None else float(cut_angle)
    depth_frac = float(rng.uniform(0.05, 0.55)) if cut_depth is None else float(np.clip(cut_depth, 0.0, 0.8))
    angle_rad  = np.radians(angle_deg)
    nx = float(np.cos(angle_rad)); ny = float(np.sin(angle_rad))
    cut_d = depth_frac * r

    # ── 1. Cut mask ───────────────────────────────────────────────────────────
    if break_type.startswith("zigzag"):
        # Multi-bend crack: dùng connected-components để tìm chip
        _n = 3 if break_type == "zigzag3" else 2
        crack_img, eff_dist, _ = _make_zigzag_cut_mask(
            H, W, cx, cy, r, angle_deg, depth_frac, _n, rng)
        mask_split = mask.copy()
        mask_split[crack_img > 0] = 0
        n_labels, labels = cv2.connectedComponents(mask_split)
        if n_labels >= 3:
            areas = sorted(
                [(l, int((labels == l).sum())) for l in range(1, n_labels)],
                key=lambda x: x[1]
            )
            chip_label = areas[0][0]   # nhỏ nhất = mảnh vỡ
            chip_bool  = (labels == chip_label) & (mask > 127)
        else:
            chip_bool = np.zeros((H, W), dtype=bool)
        remain_bool = (mask > 127) & (~chip_bool)
    else:
        remove_side, eff_dist = _make_smooth_cut_mask(H, W, cx, cy, r, nx, ny, cut_d, rng,
                                                       break_type=break_type)
        chip_bool   = (mask > 127) & (remove_side > 127)
        remain_bool = (mask > 127) & (~chip_bool)

    # Fallback: nếu không có gì bị bể hoặc còn lại quá nhỏ → trả về OK
    if chip_bool.sum() < 50 or remain_bool.sum() < 50:
        return ok_bgr.copy()

    # ── 2. Đo edge sigma của tablet gốc để match outer edge ─────────────────
    edge_sigma = _measure_tablet_edge_blur(ok_bgr, mask)

    # ── 3. Feathered alpha — outer edge soft, crack edge hard + rough ─────────
    # Distance từ OUTER EDGE của mask (không phải remain_bool)
    # → feather chỉ áp dụng tại rìa ngoài tablet
    mask_u8     = (mask > 127).astype(np.uint8) * 255
    dist_outer  = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3).astype(np.float32)
    # Soft outer edge alpha
    alpha_outer = np.clip(dist_outer / (edge_sigma * 2.0), 0.0, 1.0)
    alpha_outer = cv2.GaussianBlur(alpha_outer, (0, 0), edge_sigma * 0.8)
    alpha_outer = np.clip(alpha_outer, 0.0, 1.0)

    # Crack boundary: vùng remain_bool giáp chip_bool
    # Dùng distance từ chip để tạo transition sắc nét tại đường vỡ
    chip_u8      = chip_bool.astype(np.uint8) * 255
    dist_to_chip = cv2.distanceTransform(
        cv2.bitwise_not(cv2.dilate(chip_u8, np.ones((3,3), np.uint8))),
        cv2.DIST_L2, 3
    ).astype(np.float32)
    # Crack edge hard (1px transition) + random roughness noise
    crack_hard = np.where(remain_bool, 1.0, 0.0).astype(np.float32)
    # Tiny Gaussian blur tại crack (0.8px) — vừa đủ để không pixel-jagged
    crack_hard = cv2.GaussianBlur(crack_hard, (0, 0), 0.8)
    crack_hard = np.clip(crack_hard, 0.0, 1.0)

    # Combine: outer feather * crack hard
    alpha = alpha_outer * crack_hard
    alpha[mask == 0] = 0.0   # hard-zero ngoài tablet

    # ── 4. Crack face texture: exposed interior material ─────────────────────
    ok_f = ok_bgr.astype(np.float32)

    # eff_dist: >0 = remain side distance từ crack line
    edge_dist = np.where(remain_bool, eff_dist, 999.0).astype(np.float32)

    # ── Zone widths (px) ──────────────────────────────────────────────────────
    interior_w = float(rng.uniform(5.0, 9.0)) * intensity   # giảm từ 10-18 → 5-9
    shadow_w   = float(rng.uniform(2.0, 4.0))               # giảm từ 3-6 → 2-4
    specular_w = float(rng.uniform(0.8, 1.5))               # giảm từ 1-2.5 → 0.8-1.5

    # ── Interior material: lighter + coarse granular texture ─────────────────
    # Viên thuốc bên trong = bột nén, sáng hơn mặt ngoài coating
    interior_blend = np.clip(1.0 - edge_dist / interior_w, 0.0, 1.0) ** 0.6
    interior_blend[~remain_bool] = 0.0

    # Tablet interior brightness: lấy mean của tablet + boost
    tablet_mean = float(np.mean(ok_bgr[mask > 127]))
    # Interior sáng hơn mặt ngoài nhưng không quá trắng: +15-20%
    interior_val = np.clip(tablet_mean * 1.15 + 15.0 * intensity, 0, 200)

    # Coarse granular noise — simulate bột nén
    noise1 = rng.standard_normal(ok_bgr.shape[:2]).astype(np.float32)
    noise2 = rng.standard_normal(ok_bgr.shape[:2]).astype(np.float32)
    grain   = cv2.GaussianBlur(noise1, (0, 0), 1.0)
    clump   = cv2.GaussianBlur(noise2, (0, 0), 3.0)
    texture = 0.6 * grain + 0.4 * clump
    texture = texture / (np.std(texture) + 1e-6)
    texture_amp = float(rng.uniform(8.0, 14.0)) * intensity   # giảm từ 18-32 → 8-14
    interior_tex = interior_val + texture * texture_amp

    # ── Specular highlight ngay mép vỡ ───────────────────────────────────────
    spec_map  = np.clip(1.0 - edge_dist / specular_w, 0.0, 1.0) ** 2.0
    spec_map[~remain_bool] = 0.0
    spec_val  = tablet_mean * 1.25 + 20.0   # tương đối so với tablet, không cứng 200-235

    # ── Shadow dọc mép vỡ (phía trong, sau specular) ─────────────────────────
    shadow_map = np.clip(1.0 - (edge_dist - specular_w) / shadow_w, 0.0, 1.0) ** 2.0
    shadow_map = shadow_map * (edge_dist > specular_w).astype(np.float32)
    shadow_map[~remain_bool] = 0.0
    shadow_mul = 1.0 - shadow_map * float(rng.uniform(0.30, 0.50)) * intensity

    # ── Composite: outer surface → interior material ─────────────────────────
    # ok_f = outer surface coating (giá trị gốc)
    # interior_tex = vật liệu bên trong
    # Blend: interior_blend điều khiển mức lộ interior
    surface_f    = ok_f * shadow_mul[:, :, None]   # outer surface + shadow
    interior_f   = np.full_like(ok_f, interior_tex[:, :, None] if interior_tex.ndim == 3
                                else interior_tex[:, :, np.newaxis])
    # spec override tại mép
    spec_f       = np.full_like(ok_f, spec_val)

    tablet_dark  = (surface_f * (1.0 - interior_blend[:, :, None])
                    + interior_f * interior_blend[:, :, None])
    # Specular override tại mép rất mỏng
    tablet_dark  = (tablet_dark * (1.0 - spec_map[:, :, None])
                    + spec_f * spec_map[:, :, None])

    # ── 5. Composite: remaining piece (soft alpha) lên background ────────────
    # bg_color = màu background thật (ngoài tablet) — KHÔNG phải ok_bgr
    bg_pixels = ok_bgr[mask == 0]
    bg_color  = np.median(bg_pixels, axis=0).astype(np.float32) if len(bg_pixels) else np.array([255., 255., 255.])
    bg_f      = np.full_like(ok_f, bg_color)

    a3     = alpha[:, :, None]
    result = tablet_dark * a3 + bg_f * (1.0 - a3)

    # Vùng ngoài mask: clean background (không dùng ok_bgr vì có drop shadow của tablet gốc)
    result[mask == 0] = bg_color

    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── SDXL crack texture (optional) ────────────────────────────────────────
    if sdxl_crack:
        result = sdxl_inpaint_crack_edge(
            result, remain_bool, chip_bool, mask,
            seed=seed, edge_width=crack_edge_width,
            strength=crack_strength,
        )

    return result


def run_tron_nut(n: int = 0, sdxl_crack: bool = False):
    """
    Gen ảnh Nứt vỡ cho Thuốc_tròn.
    n=0 → gen tất cả, n>0 → giới hạn debug.
    """
    product     = "Thuốc_tròn"
    defect_name = "Nứt"
    defect_path = os.path.join(CAPSULE_ROOTS[product], defect_name)

    ok_files = sorted(glob.glob(os.path.join(defect_path, "ok", "*.png")) +
                      glob.glob(os.path.join(defect_path, "ok", "*.jpg")))
    if not ok_files:
        print(f"[SKIP] No OK images: {defect_path}/ok/")
        return

    # break_type → (depth_range, angles)
    # straight: depth 0.30-0.55 (nứt ngang lớn)
    # corner:   depth 0.05-0.25 (góc bể nhỏ)
    # curved:   depth 0.20-0.45 (cong)
    # concave:  depth 0.15-0.35 (lưỡi liềm)
    CONFIGS = [
        ("straight", [0.10, 0.25, 0.45, 0.60, 0.70], [0, 45, 90, 135, 180, 225, 270, 315]),
        ("corner",   [0.10, 0.20, 0.35, 0.50, 0.65], [0, 60, 120, 180, 240, 300]),
        ("curved",   [0.10, 0.25, 0.40, 0.55, 0.70], [0, 45, 90, 135, 180, 225, 270, 315]),
        ("concave",  [0.10, 0.22, 0.38, 0.52, 0.65], [0, 60, 120, 180, 240, 300]),
        ("zigzag2",  [0.12, 0.28, 0.45, 0.60, 0.70], [0, 45, 90, 135, 180, 225, 270, 315]),
        ("zigzag3",  [0.15, 0.30, 0.48, 0.62, 0.70], [0, 60, 120, 180, 240, 300]),
    ]
    intensities = [0.6, 0.9]
    n_gen  = 0

    for ok_path in ok_files[:3]:
        ok_bgr = cv2.imread(ok_path)
        if ok_bgr is None:
            continue
        tablet_mask, bbox = detect_tablet_mask(ok_bgr)
        ok_name = os.path.splitext(os.path.basename(ok_path))[0]
        print(f"  [{ok_name}] mask_area={(tablet_mask>127).sum()} bbox={bbox}")

        done = False
        for btype, depths, angles in CONFIGS:
            if done: break
            for di, depth in enumerate(depths):
                if done: break
                for ai, angle in enumerate(angles):
                    if done: break
                    for ii, intensity in enumerate(intensities):
                        if n and n_gen >= n:
                            done = True
                            break
                        seed = hash((btype, di, ai, ii, ok_name)) % 9999
                        result = synth_nut_tron(
                            ok_bgr, tablet_mask, bbox,
                            seed=seed,
                            intensity=intensity,
                            cut_depth=depth,
                            cut_angle=float(angle),
                            break_type=btype,
                            sdxl_crack=sdxl_crack,
                        )
                        tag = (f"nut_{ok_name}_{btype}_d{int(depth*100)}"
                               f"_a{angle}_i{int(intensity*10)}.jpg")
                    out_dir = os.path.join(RESULT_DIR, product, defect_name)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, tag)
                    cv2.imwrite(out_path, result)

                    # Debug panel: OK | mask | result | diff
                    chip_vis = cv2.cvtColor(tablet_mask, cv2.COLOR_GRAY2BGR)
                    if result.shape == ok_bgr.shape:
                        diff = cv2.absdiff(ok_bgr, result)
                        diff_bright = cv2.convertScaleAbs(diff, alpha=4.0)
                    else:
                        diff_bright = np.zeros_like(ok_bgr)
                    panel_imgs = [ok_bgr, chip_vis, result, diff_bright]
                    th = TH_PANEL
                    panels = []
                    for img in panel_imgs:
                        ph = th
                        pw = int(img.shape[1] * ph / img.shape[0])
                        panels.append(cv2.resize(img, (pw, ph)))
                    panel = np.hstack(panels)
                    dbg_path = os.path.join(out_dir, "debug_" + tag)
                    cv2.imwrite(dbg_path, panel)

                    print(f"    → {tag}")
                    n_gen += 1

    print(f"\nDone → {RESULT_DIR}  ({n_gen} ảnh)")


if __name__ == "__main__":
    main_v2()
