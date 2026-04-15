"""
engines/plastic_flow_engine.py — Plastic Flow (Nhựa chảy) Generation Engine
============================================================================

Goal:
  Input  : OK image (base64)
  Output : OK image with synthesized plastic flow region (result + mask)

Strategy:
  - CV: generate a plausible plastic-flow mask + apply experiments.synth_nhựa_chảy()
  - Optional SDXL refine: use engines.deep_generative (poisson + depth + IP-Adapter)
    with defect_type="plastic_flow", material="plastic".

This engine is intended to be plugged into MKA cap generation path.
"""

from __future__ import annotations

import base64 as _b64
import math
import os
import sys

import cv2
import numpy as np

from .utils import encode_b64, decode_b64

# ── Import experiments.py (CV synthesis primitives) ───────────────────────────
_ENGINES_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENGINES_DIR not in sys.path:
    sys.path.insert(0, _ENGINES_DIR)

try:
    import experiments as _exp
    _HAS_EXP = True
except Exception as _e:
    _HAS_EXP = False
    _EXP_ERR = str(_e)


def _encode_gray_png_b64(mask_gray: np.ndarray) -> str:
    if mask_gray.ndim == 3:
        mask_gray = mask_gray[:, :, 0]
    _, mbuf = cv2.imencode(".png", mask_gray)
    return _b64.b64encode(mbuf).decode("utf-8")


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _bbox_to_xywh(bbox) -> tuple[int, int, int, int]:
    """
    Accept bbox from experiments.detect_product_bbox (implementation-specific).
    We handle common shapes:
      - (x, y, w, h)
      - (x1, y1, x2, y2)
    """
    if bbox is None:
        return (0, 0, 0, 0)
    if len(bbox) != 4:
        raise ValueError(f"Unexpected bbox format: {bbox}")
    x0, y0, a, b = [int(v) for v in bbox]
    # experiments.detect_product_bbox() documents (x0,y0,x1,y1)
    if a > x0 and b > y0:
        return (x0, y0, a - x0, b - y0)
    # fallback: treat as (x,y,w,h)
    return (x0, y0, a, b)


def _mask_from_small_random_region(img_shape_hw: tuple[int, int], bbox_xywh: tuple[int, int, int, int],
                                   rng: np.random.Generator, params: dict) -> np.ndarray:
    """
    Single small region mask — compact organic polygon matching plastic-flow blob shape.
    Polygon with slight radial noise (±15%) instead of plain ellipse to look natural.
    """
    H, W = img_shape_hw
    x, y, w, h = bbox_xywh
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    w = max(1, min(W - x, w))
    h = max(1, min(H - y, h))

    # radius as fraction of product bbox size
    r_min_frac = float(params.get("small_r_min", 0.015))
    r_max_frac = float(params.get("small_r_max", 0.035))
    r_min_frac = max(0.003, min(0.10, r_min_frac))
    r_max_frac = max(r_min_frac + 0.001, min(0.20, r_max_frac))
    r = float(min(w, h)) * float(rng.uniform(r_min_frac, r_max_frac))
    r = float(max(5.0, min(60.0, r)))

    cx = float(x + rng.uniform(0.20, 0.80) * w)
    cy = float(y + rng.uniform(0.20, 0.80) * h)

    # Compact organic polygon: near-circular with slight radial noise
    mask = np.zeros((H, W), np.uint8)
    n_pts = int(rng.integers(14, 22))
    angs  = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    noise = rng.uniform(0.85, 1.15, n_pts)
    # Smooth noise so shape is blobby, not spiky
    noise = np.convolve(np.tile(noise, 3), np.ones(4) / 4, mode="same")[n_pts: 2 * n_pts]
    # Slight anisotropy: one axis 0.85–1.0× the other (keeps shape near-round)
    ax = float(rng.uniform(0.85, 1.00))
    rot = float(rng.uniform(0, 2 * math.pi))
    pts = []
    for i, a in enumerate(angs):
        ra = a + rot
        px = int(round(cx + r * noise[i] * ax * math.cos(ra)))
        py = int(round(cy + r * noise[i]       * math.sin(ra)))
        pts.append([px, py])
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    # Soft Gaussian edge — larger sigma than before for more natural fade
    sig = float(max(0.8, min(8.0, r * 0.25)))
    mask = cv2.GaussianBlur(mask, (0, 0), sig)
    mask = (mask > 50).astype(np.uint8) * 255
    return mask


def _strong_warp_inside_mask(img_bgr: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                             strength: float) -> np.ndarray:
    """
    Stronger pixel displacement inside mask (extra on top of experiments.synth_nhựa_chảy()).
    """
    H, W = img_bgr.shape[:2]
    area = float((mask > 0).sum())
    if area <= 0:
        return img_bgr
    mask_r = max(4.0, math.sqrt(area / math.pi))

    # Stronger amplitude than experiments._warp_texture_inside_mask (0.18-0.30)*mask_r
    amp = mask_r * float(rng.uniform(0.55, 0.95)) * float(max(0.2, strength))
    sigma = max(2.5, mask_r * 0.35)
    raw_x = rng.normal(0, 1, (H, W)).astype(np.float32)
    raw_y = rng.normal(0, 1, (H, W)).astype(np.float32)
    dx = cv2.GaussianBlur(raw_x, (0, 0), sigma) * amp
    dy = cv2.GaussianBlur(raw_y, (0, 0), sigma) * amp

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = np.clip(xx + dx, 0, W - 1)
    map_y = np.clip(yy + dy, 0, H - 1)

    warped = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # blend only inside mask with soft edge
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), max(1.2, mask_r * 0.10))
    alpha = np.clip(alpha, 0, 1)[:, :, None]
    out = warped.astype(np.float32) * alpha + img_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(m)
    if n <= 1:
        return (m * 255).astype(np.uint8)
    best_i = 0
    best_a = 0
    for i in range(1, n):
        a = int((labels == i).sum())
        if a > best_a:
            best_a = a
            best_i = i
    return ((labels == best_i).astype(np.uint8) * 255)


def _extract_patch_and_alpha(ref_bgr: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a defect patch + soft alpha from NG reference crop.

    Mask design goals:
      - soft edge (alpha gradient)
      - irregular shape (not circular)
      - strong core + fade boundary
    Returns:
      patch_bgr (h,w,3), alpha_f (h,w) in [0,1]
    """
    ref = ref_bgr.copy()
    H, W = ref.shape[:2]
    g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Anomaly/texture map (LoG + band-pass-ish)
    sig1 = float(params.get("ref_sig1", 1.2))
    sig2 = float(params.get("ref_sig2", 4.5))
    hi = cv2.GaussianBlur(g, (0, 0), sig1)
    lo = cv2.GaussianBlur(g, (0, 0), sig2)
    bp = np.abs(hi - lo)
    log = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.5), cv2.CV_32F))
    m = (0.65 * bp + 0.35 * log)

    # Normalize and threshold by percentile → irregular binary
    # If ref image is full-frame (not cropped), low percentile will pick ring/background.
    # We auto-increase threshold until the component is small enough.
    max_area_frac = float(params.get("ref_max_area_frac", 0.08))  # component area cap (of ref frame)
    max_area_frac = max(0.01, min(0.40, max_area_frac))
    target_max_area = H * W * max_area_frac

    p0 = float(params.get("ref_thresh_pct", 96.0))
    p0 = max(70.0, min(99.5, p0))
    candidates = [p0, min(99.0, p0 + 1.5), min(99.2, p0 + 2.5), 99.4]

    bin0 = None
    for p in candidates:
        thr = np.percentile(m, p)
        b = (m >= thr).astype(np.uint8) * 255
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        b = _largest_component(b)
        a = int((b > 0).sum())
        if 50 <= a <= target_max_area:
            bin0 = b
            break
    if bin0 is None:
        # fallback to last attempt
        thr = np.percentile(m, candidates[-1])
        bin0 = (m >= thr).astype(np.uint8) * 255
        bin0 = cv2.morphologyEx(bin0, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        bin0 = _largest_component(bin0)

    ys, xs = np.where(bin0 > 0)
    if len(ys) < 20:
        # fallback: center blob if extraction fails
        r = max(6, int(min(H, W) * 0.10))
        bin0 = np.zeros((H, W), np.uint8)
        cv2.ellipse(bin0, (W // 2, H // 2), (r, int(r * 0.7)), 20, 0, 360, 255, -1)
        ys, xs = np.where(bin0 > 0)

    # Crop tight with padding
    pad = int(params.get("ref_pad_px", 14))
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad + 1)
    patch = ref[y0:y1, x0:x1].copy()
    mask = bin0[y0:y1, x0:x1].copy()

    # Build alpha with core + fade using distance transform
    mb = (mask > 0).astype(np.uint8)
    dt_in = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
    dt_out = cv2.distanceTransform(1 - mb, cv2.DIST_L2, 5)

    core_r = float(params.get("core_r_px", 8.0))
    fade_r = float(params.get("fade_r_px", 10.0))
    core_r = max(2.0, core_r)
    fade_r = max(4.0, fade_r)

    core = np.clip(dt_in / (core_r + 1e-6), 0, 1)
    fade = np.clip(1.0 - (dt_out / (fade_r + 1e-6)), 0, 1)
    alpha = np.clip(np.maximum(core, fade), 0, 1).astype(np.float32)

    # Add slight irregularity to edge (avoid "round even")
    noise = np.random.default_rng(int(params.get("seed", 42)) + 1337).normal(0, 1, alpha.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 2.0)
    alpha = np.clip(alpha + noise * 0.06 * (1.0 - core), 0, 1)

    # Final softening
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)

    # Hard constraint: alpha must be ~0 near patch borders (prevents border reflect artifacts)
    ph, pw = alpha.shape[:2]
    edge = int(params.get("alpha_edge_zero_px", 6))
    edge = max(2, min(40, edge))
    wx = np.ones((pw,), np.float32)
    wy = np.ones((ph,), np.float32)
    wx[:edge] = 0.0
    wx[-edge:] = 0.0
    wy[:edge] = 0.0
    wy[-edge:] = 0.0
    win = (wy[:, None] * wx[None, :]).astype(np.float32)
    win = cv2.GaussianBlur(win, (0, 0), max(1.0, edge * 0.35))
    alpha = np.clip(alpha * win, 0, 1)

    # Cap patch size for placement (avoid full-frame patch when ref is full NG image)
    max_dim = int(params.get("patch_max_dim", 320))
    max_dim = max(80, min(720, max_dim))
    ph, pw = patch.shape[:2]
    if max(ph, pw) > max_dim:
        scale = max_dim / float(max(ph, pw))
        new_w = max(20, int(round(pw * scale)))
        new_h = max(20, int(round(ph * scale)))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Optional extra global scaling (user-controlled size)
    size_scale = float(params.get("patch_scale", 1.0))
    size_scale = float(max(0.35, min(2.0, size_scale)))
    if abs(size_scale - 1.0) > 1e-3:
        ph, pw = patch.shape[:2]
        new_w = max(20, int(round(pw * size_scale)))
        new_h = max(20, int(round(ph * size_scale)))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return patch, alpha


def _warp_patch_and_alpha(patch_bgr: np.ndarray, alpha_f: np.ndarray,
                          rng: np.random.Generator, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp patch + alpha together (affine + smooth displacement).
    """
    ph, pw = patch_bgr.shape[:2]
    # Affine
    rot = float(params.get("warp_rot_deg", rng.uniform(-25, 25)))
    sc = float(params.get("warp_scale", rng.uniform(0.85, 1.20)))
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), rot, sc)
    patch2 = cv2.warpAffine(patch_bgr, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # Alpha must not reflect at borders
    a2 = cv2.warpAffine(alpha_f, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # Smooth displacement (elastic-ish)
    disp_strength = float(params.get("warp_disp_strength", 1.0))
    disp_strength = max(0.0, min(4.0, disp_strength))
    if disp_strength > 0:
        amp = float(min(pw, ph)) * 0.06 * disp_strength
        sig = max(3.0, float(min(pw, ph)) * 0.10)
        rx = rng.normal(0, 1, (ph, pw)).astype(np.float32)
        ry = rng.normal(0, 1, (ph, pw)).astype(np.float32)
        dx = cv2.GaussianBlur(rx, (0, 0), sig) * amp
        dy = cv2.GaussianBlur(ry, (0, 0), sig) * amp
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        map_x = np.clip(xx + dx, 0, pw - 1)
        map_y = np.clip(yy + dy, 0, ph - 1)
        patch2 = cv2.remap(patch2, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        a2 = cv2.remap(a2, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    a2 = np.clip(a2, 0, 1).astype(np.float32)
    return patch2, a2


def _blend_patch_into_ok(ok_bgr: np.ndarray, patch_bgr: np.ndarray, alpha_f: np.ndarray,
                         center_xy: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Alpha blend patch into OK at given center. Returns (result_bgr, placed_mask_u8).
    """
    H, W = ok_bgr.shape[:2]
    ph, pw = patch_bgr.shape[:2]
    cx, cy = center_xy
    x0 = int(cx - pw // 2)
    y0 = int(cy - ph // 2)
    x1 = x0 + pw
    y1 = y0 + ph

    # Clip to image bounds
    ix0 = max(0, x0); iy0 = max(0, y0)
    ix1 = min(W, x1); iy1 = min(H, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return ok_bgr, np.zeros((H, W), np.uint8)

    px0 = ix0 - x0; py0 = iy0 - y0
    px1 = px0 + (ix1 - ix0); py1 = py0 + (iy1 - iy0)

    out = ok_bgr.copy().astype(np.float32)
    roi = out[iy0:iy1, ix0:ix1]
    p = patch_bgr[py0:py1, px0:px1].astype(np.float32)
    a = alpha_f[py0:py1, px0:px1].astype(np.float32)
    a3 = a[:, :, None]

    # Simple photometric adaptation: match mean color in ROI
    roi_mean = roi.mean(axis=(0, 1), keepdims=True)
    p_mean = p.mean(axis=(0, 1), keepdims=True) + 1e-6
    p_adj = np.clip(p * (roi_mean / p_mean), 0, 255)

    roi[:] = p_adj * a3 + roi * (1.0 - a3)
    out[iy0:iy1, ix0:ix1] = roi

    placed = np.zeros((H, W), np.uint8)
    placed[iy0:iy1, ix0:ix1] = (a > 0.10).astype(np.uint8) * 255
    return np.clip(out, 0, 255).astype(np.uint8), placed


def _interior_from_bbox(img_shape_hw: tuple[int, int], bbox_xywh: tuple[int, int, int, int],
                        shrink_px: int) -> np.ndarray:
    """
    Build a conservative interior mask from product bbox.
    This is robust when Otsu picks the bright ring/background.
    """
    H, W = img_shape_hw
    x, y, w, h = bbox_xywh
    shrink_px = max(0, min(200, int(shrink_px)))
    x0 = max(0, x + shrink_px)
    y0 = max(0, y + shrink_px)
    x1 = min(W, x + w - shrink_px)
    y1 = min(H, y + h - shrink_px)
    m = np.zeros((H, W), np.uint8)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 255
    return m


def _sample_point_in_mask(mask_u8: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 50:
        return None
    idx = int(rng.integers(0, len(xs)))
    return int(xs[idx]), int(ys[idx])


def _robust_product_interior_mask(ok_bgr: np.ndarray, shrink_px: int = 30) -> np.ndarray:
    """
    Build an interior mask that prefers the actual product (often darker) over the bright ring/background.
    We try both Otsu(th) and inverted Otsu, pick the one whose largest component is more centered
    and less border-touching, then erode to interior.
    """
    H, W = ok_bgr.shape[:2]
    g = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(g, (15, 15), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cand = []
    for m in [th, cv2.bitwise_not(th)]:
        k = np.ones((20, 20), np.uint8)
        mm = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, k)
        mm = _largest_component(mm)
        ys, xs = np.where(mm > 0)
        if len(xs) < H * W * 0.02:
            continue
        cx = float(xs.mean()) / W
        cy = float(ys.mean()) / H
        # border touch penalty
        touch = 0.0
        if (mm[:, 0] > 0).any() or (mm[:, -1] > 0).any() or (mm[0, :] > 0).any() or (mm[-1, :] > 0).any():
            touch = 1.0
        # center closeness (prefer around image center)
        center_score = 1.0 - ((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        score = center_score - touch * 0.6
        cand.append((score, mm))
    if not cand:
        return np.zeros((H, W), np.uint8)
    cand.sort(key=lambda t: t[0], reverse=True)
    best = cand[0][1]
    shrink_px = max(0, min(200, int(shrink_px)))
    if shrink_px > 0:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_px * 2 + 1, shrink_px * 2 + 1))
        best = cv2.erode(best, ke, iterations=1)
    return best


def _mask_from_streaks(img_shape_hw: tuple[int, int], bbox_xywh: tuple[int, int, int, int],
                       rng: np.random.Generator, params: dict) -> np.ndarray:
    """
    Create a thin streak/smear mask inside bbox.
    Returns uint8 mask (0/255).
    """
    H, W = img_shape_hw
    x, y, w, h = bbox_xywh
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    w = max(1, min(W - x, w))
    h = max(1, min(H - y, h))

    n_streaks = int(params.get("n_streaks", 2))
    n_streaks = max(1, min(8, n_streaks))

    flow_angle = params.get("flow_angle", None)
    if flow_angle is None:
        # default: mostly downward
        flow_angle = float(rng.uniform(math.pi * 0.35, math.pi * 0.65))
    else:
        flow_angle = float(flow_angle)

    length_frac = float(params.get("streak_length", 0.30))  # fraction of min(w,h)
    length_frac = _clip01(length_frac)
    base_len = max(12.0, length_frac * float(min(w, h)))

    width_px = float(params.get("streak_width_px", 6.0))
    width_px = max(2.0, min(80.0, width_px))

    curvature = float(params.get("curvature", 0.35))  # 0..1
    curvature = _clip01(curvature)

    mask = np.zeros((H, W), np.uint8)
    for _ in range(n_streaks):
        cx = float(x + rng.uniform(0.25, 0.75) * w)
        cy = float(y + rng.uniform(0.25, 0.75) * h)

        # randomize each streak
        ang = flow_angle + float(rng.normal(0.0, 0.25))
        L = base_len * float(rng.uniform(0.7, 1.25))
        steps = int(max(16, min(64, L / 6.0)))

        pts = []
        px, py = cx, cy
        for i in range(steps):
            t = i / max(1, (steps - 1))
            # meander perpendicular to flow
            perp = ang + math.pi / 2
            jitter = (rng.normal(0, 1.0) * curvature) * (1.0 - t) * 2.0
            dx = math.cos(ang) * (L / steps) + math.cos(perp) * jitter
            dy = math.sin(ang) * (L / steps) + math.sin(perp) * jitter
            px += dx
            py += dy
            pts.append((int(round(px)), int(round(py))))

        if len(pts) < 2:
            continue

        pts_np = np.array(pts, np.int32).reshape(-1, 1, 2)
        thickness = int(max(2, round(width_px * rng.uniform(0.7, 1.2))))
        cv2.polylines(mask, [pts_np], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)

    # Feather edges slightly then binarize (keeps streak look, avoids jagged mask)
    feather = int(params.get("feather_px", 2))
    feather = max(0, min(30, feather))
    if feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = (mask > 18).astype(np.uint8) * 255
    return mask


def _mask_from_polar_band(img_shape_hw: tuple[int, int], bbox_xywh: tuple[int, int, int, int],
                          params: dict) -> np.ndarray:
    """
    Create an annulus sector-like mask (band in radius + theta gate),
    approximated from bbox geometry (no Hough dependency).
    Returns uint8 mask (0/255).
    """
    H, W = img_shape_hw
    x, y, w, h = bbox_xywh
    cx = float(x + w / 2.0)
    cy = float(y + h / 2.0)
    # Default to a relatively thin band so it doesn't create a "big blob"
    r0 = max(10.0, float(min(w, h)) * float(params.get("band_r0", 0.28)))
    r1 = max(r0 + 4.0, float(min(w, h)) * float(params.get("band_r1", 0.34)))

    theta_center = float(params.get("theta_center", math.pi))  # radians
    theta_span = float(params.get("theta_span", 0.60))         # radians
    theta_span = max(0.05, min(2 * math.pi, theta_span))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    rr = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)  # [-pi, pi]

    # shortest angular distance
    d = np.abs((ang - theta_center + math.pi) % (2 * math.pi) - math.pi)
    gate = (d <= (theta_span / 2.0))
    band = (rr >= r0) & (rr <= r1)

    m = (gate & band).astype(np.uint8) * 255

    feather = int(params.get("feather_px", 2))
    feather = max(0, min(30, feather))
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        m = (m > 18).astype(np.uint8) * 255
    return m


def _postprocess_mask(mask: np.ndarray, img_shape_hw: tuple[int, int], params: dict) -> np.ndarray:
    """
    Keep mask small and avoid harsh halo artifacts.
    - shrink_px (erosion): reduces outer-shadow ring strength in synth_nhựa_chảy()
    - max_area_frac: prevents overly large filled regions
    """
    H, W = img_shape_hw
    mask = (mask > 0).astype(np.uint8) * 255

    shrink_px = int(params.get("mask_shrink_px", 2))
    shrink_px = max(0, min(15, shrink_px))
    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_px * 2 + 1, shrink_px * 2 + 1))
        mask = cv2.erode(mask, k, iterations=1)

    # Area cap
    max_area_frac = float(params.get("max_area_frac", 0.012))  # ~1.2% of image by default
    max_area_frac = max(0.001, min(0.10, max_area_frac))
    max_area = int(H * W * max_area_frac)
    area = int((mask > 0).sum())
    if area > max_area:
        # progressively erode until within cap (or empty)
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for _ in range(8):
            mask = cv2.erode(mask, k2, iterations=1)
            area = int((mask > 0).sum())
            if area <= max_area or area == 0:
                break

    return mask


def _cv_synthesize(img_bgr: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (cv_result_bgr, mask_gray).
    """
    if not _HAS_EXP:
        raise RuntimeError(f"experiments not available: {_EXP_ERR}")

    seed = int(params.get("seed", 42))
    intensity = float(params.get("intensity", 0.6))

    bbox = _exp.detect_product_bbox(img_bgr)
    x, y, w, h = _bbox_to_xywh(bbox)
    if w <= 0 or h <= 0:
        # fallback: use whole image
        x, y, w, h = (0, 0, img_bgr.shape[1], img_bgr.shape[0])

    rng = np.random.default_rng(seed)
    # Preferred pipeline: NG ref → extract patch + alpha → warp patch+alpha → blend into OK
    ref_b64 = params.get("ref_image_b64")
    if ref_b64:
        try:
            ref_rgb = decode_b64(ref_b64)
            ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)
            patch_bgr, alpha_f = _extract_patch_and_alpha(ref_bgr, params)
            patch_bgr, alpha_f = _warp_patch_and_alpha(patch_bgr, alpha_f, rng, params)

            # Place patch strictly inside product bbox (safe margin so patch won't be clipped)
            ph, pw = patch_bgr.shape[:2]
            half_w = max(1, pw // 2)
            half_h = max(1, ph // 2)
            margin = int(params.get("place_margin_px", 10))
            margin = max(0, min(120, margin))

            # Robust interior: prefer actual product component over bright ring/background
            interior = _robust_product_interior_mask(
                img_bgr,
                shrink_px=int(params.get("product_interior_shrink_px", max(18, max(half_w, half_h) + margin))),
            )
            pt = _sample_point_in_mask(interior, rng)
            if pt is None:
                # fallback: clamp to bbox center
                cx = int(x + w / 2)
                cy = int(y + h / 2)
            else:
                cx, cy = pt
            blended_bgr, placed_mask = _blend_patch_into_ok(img_bgr, patch_bgr, alpha_f, (cx, cy))

            # Optional: additional strong warp inside placed mask to emphasize "raised plastic"
            warp_strength = float(params.get("pixel_warp_strength", 1.6))
            warp_strength = float(max(0.0, min(6.0, warp_strength)))
            if warp_strength > 0:
                blended_bgr = _strong_warp_inside_mask(blended_bgr, placed_mask, rng, strength=warp_strength)

            # Optional: run synth_nhựa_chảy on top for shading/texture (light, avoids halo)
            use_synth = bool(params.get("use_synth", True))
            if use_synth:
                # Use a much tighter "core" mask for synth_nhựa_chảy to avoid dark outer ring.
                placed_mask2 = _postprocess_mask(placed_mask, img_bgr.shape[:2], params)
                core_shrink = int(params.get("synth_core_shrink_px", 10))
                core_shrink = max(0, min(40, core_shrink))
                if core_shrink > 0:
                    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_shrink * 2 + 1, core_shrink * 2 + 1))
                    core_mask = cv2.erode(placed_mask2, ke, iterations=1)
                else:
                    core_mask = placed_mask2

                # Slightly reduce synth intensity to avoid heavy shadowing
                synth_intensity = float(params.get("synth_intensity", intensity * 0.65))
                synth_intensity = float(max(0.05, min(1.0, synth_intensity)))

                synth_bgr = _exp.synth_nhựa_chảy(blended_bgr, core_mask, seed=seed, intensity=synth_intensity)

                # Critical: synth_nhựa_chảy adds outer shadow outside mask.
                # We explicitly composite ONLY inside the (softened) core region to avoid black rings.
                core_alpha = cv2.GaussianBlur(core_mask.astype(np.float32) / 255.0, (0, 0), 3.0)
                core_alpha = np.clip(core_alpha, 0, 1)[:, :, None]
                # Option: keep only positive delta (bright/texture), drop negative delta (shadow ring).
                positive_only = bool(params.get("synth_positive_only", True))
                if positive_only:
                    delta = synth_bgr.astype(np.float32) - blended_bgr.astype(np.float32)
                    delta = np.maximum(delta, 0.0)
                    out = blended_bgr.astype(np.float32) + delta * core_alpha
                else:
                    out = synth_bgr.astype(np.float32) * core_alpha + blended_bgr.astype(np.float32) * (1.0 - core_alpha)
                out_bgr = np.clip(out, 0, 255).astype(np.uint8)

                # Return the *placed* mask (soft region) for UI/debug, not the core mask
                return out_bgr, placed_mask2
            return blended_bgr, placed_mask
        except Exception as e:
            print(f"[plastic_flow] ref patch pipeline failed: {e} — falling back to procedural mask")

    # Fallback: single small random region + strong warp + synth
    mask = _mask_from_small_random_region(img_bgr.shape[:2], (x, y, w, h), rng, params)
    mask = _postprocess_mask(mask, img_bgr.shape[:2], params)
    warp_strength = float(params.get("pixel_warp_strength", 2.2))
    warp_strength = float(max(0.0, min(6.0, warp_strength)))
    base_warped = _strong_warp_inside_mask(img_bgr, mask, rng, strength=warp_strength)
    result = _exp.synth_nhựa_chảy(base_warped, mask, seed=seed, intensity=intensity)
    return result, mask


def _sdxl_refine(base_rgb: np.ndarray, mask_gray: np.ndarray, params: dict) -> np.ndarray:
    """
    Use deep_generative's appearance pipeline for refinement.
    Returns RGB uint8 (same size).
    """
    from . import deep_generative as _dg

    # Build a minimal params dict that deep_generative understands.
    # Allow advanced overrides to pass-through if user provides them.
    dg_params = dict(params)
    dg_params.setdefault("naturalness", 0.7)

    # deep_generative reads ref_image_b64 (optional) and uses it if provided.
    out = _dg.generate(
        base_image=base_rgb,
        mask=mask_gray,
        defect_type="plastic_flow",
        material="plastic",
        params=dg_params,
    )
    if "result_image" not in out:
        raise RuntimeError("deep_generative returned no result_image")
    return decode_b64(out["result_image"])


def generate(base_image_b64: str, params: dict) -> dict:
    """
    Generate one plastic-flow defect image.

    params keys (common):
      intensity       float 0-1 (default 0.6)
      seed            int
      sdxl_refine     bool (default False)
      ref_image_b64   str  (optional; NG reference for IP-Adapter)

    Returns:
      dict: {result_image, result_pre_refine, mask_b64, engine, metadata}
    """
    img_rgb = decode_b64(base_image_b64)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    try:
        cv_bgr, mask_gray = _cv_synthesize(img_bgr, params)
    except Exception as e:
        return {"error": f"PlasticFlow CV error: {e}"}

    cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
    pre_b64 = encode_b64(cv_rgb)
    mask_b64 = _encode_gray_png_b64(mask_gray)

    do_refine = bool(params.get("sdxl_refine", False))
    if do_refine:
        try:
            refined_rgb = _sdxl_refine(cv_rgb, mask_gray, params)
            result_b64 = encode_b64(refined_rgb)
            engine = "cv+sdxl"
        except Exception as e:
            print(f"[plastic_flow] SDXL refine failed: {e} — returning CV result")
            result_b64 = pre_b64
            engine = "cv"
    else:
        result_b64 = pre_b64
        engine = "cv"

    return {
        "result_image": result_b64,
        "result_pre_refine": pre_b64,
        "mask_b64": mask_b64,
        "engine": engine,
        "metadata": {
            "defect_type": "plastic_flow",
            "sdxl_refine": do_refine,
            "params": params,
        },
    }

