"""
CV Defect Synthesis Experiments
================================
Chạy: python V:/defect_samples/experiments.py

Output: V:/defect_samples/<product>/<defect>/output/exp_XXX.jpg
"""

import cv2
import numpy as np
from PIL import Image
import os, glob, random, sys
from datetime import datetime

# ── Import fast_physics engine ────────────────────────────────────────────────
_ENGINE_ROOT  = "V:/HondaPlus/defect_dataset_generator"
_SCRIPTS_ROOT = "V:/HondaPlus/defect_dataset_generator/scripts"
for _p in [_ENGINE_ROOT, _SCRIPTS_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engines.fast_physics import generate as _fp_generate
from engines.utils import decode_b64

ROOT       = "V:/defect_samples"
_OVR = os.environ.get("DEFECT_SAMPLES_RESULT_DIR", "").strip()
RESULT_DIR = _OVR if _OVR else os.path.join(ROOT, "results", datetime.now().strftime("%Y%m%d_%H%M%S"))
rng        = np.random.default_rng(42)

# ── scratch/scuff tuning (plastic) ───────────────────────────────────────────
PLASTIC_SCRATCH = {
    # matte adaptation (reduces specular so scuff visible)
    "matte_strength_range": (0.95, 1.45),
    # mix weights (delta-space). Keep near 50/50 for realism, allow small bias.
    "mix_ref_range": (0.20, 0.55),
    "mix_proc_range": (0.55, 1.05),
    # procedural scuff visibility
    "scuff_alpha_mult_range": (1.20, 1.95),
    "scuff_whiten_add_range": (95, 185),
    # geometry diversity
    "proc_strokes_range": (2, 5),
    "break_prob_range": (0.20, 0.65),
    "warp_range": (0.20, 0.60),
    # extra feature modes
    "extra_mode_prob": 0.60,      # probability to add a second scuff mode overlay
    "crosshatch_prob": 0.35,      # within extra overlay, chance to be crosshatch
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_ok(defect_path):
    ok_dir = os.path.join(defect_path, "ok")
    imgs   = glob.glob(os.path.join(ok_dir, "*"))
    return cv2.imread(imgs[0]) if imgs else None

def detect_product_bbox(img):
    """Otsu + largest contour → product bounding box (x0,y0,x1,y1)."""
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    h, w    = gray.shape
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, th   = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = np.ones((20, 20), np.uint8)
    th      = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    th      = cv2.morphologyEx(th, cv2.MORPH_OPEN,  kernel)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_bbox, best_area = None, 0
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, cw, ch = cv2.boundingRect(c)
        ar = cw / max(ch, 1)
        if area > h*w*0.05 and area < h*w*0.8 and 0.5 < ar < 3.0 and area > best_area:
            best_area = area
            best_bbox = (x, y, x+cw, y+ch)
    if best_bbox is None:
        mx, my = int(w*0.2), int(h*0.2)
        best_bbox = (mx, my, w-mx, h-my)
    return best_bbox

def extract_defect_shape(mask):
    """Crop the defect shape (tight bbox) from mask → returns small shape array."""
    ys, xs = np.where(mask > 127)
    if not len(ys):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return mask[y0:y1+1, x0:x1+1].copy()   # shape patch

def place_mask_random(shape_patch, product_bbox, ok_shape, rng2, rotate=True):
    """
    Place defect shape at random position inside product bbox.
    rotate=True: random rotation 0-360° so orientation differs from NG mask.
    Returns full-size mask (same size as OK image).
    """
    px0, py0, px1, py1 = product_bbox
    oh, ow = ok_shape[:2]

    patch = shape_patch.copy()

    if rotate:
        angle = float(rng2.uniform(0, 360))
        ph, pw = patch.shape
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
        # Expand canvas so rotated shape fits without clipping
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(ph * sin_a + pw * cos_a)
        new_h = int(ph * cos_a + pw * sin_a)
        M[0, 2] += new_w / 2 - pw / 2
        M[1, 2] += new_h / 2 - ph / 2
        patch = cv2.warpAffine(patch, M, (new_w, new_h),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    sh, sw = patch.shape

    # Valid placement range: shape must fit fully inside product bbox
    max_x = px1 - sw
    max_y = py1 - sh
    if max_x < px0 or max_y < py0:
        cx, cy = (px0+px1)//2, (py0+py1)//2
        ox, oy = cx - sw//2, cy - sh//2
    else:
        ox = int(rng2.integers(px0, max(px0+1, max_x)))
        oy = int(rng2.integers(py0, max(py0+1, max_y)))

    new_mask = np.zeros((oh, ow), dtype=np.uint8)
    ox = max(0, min(ox, ow-sw))
    oy = max(0, min(oy, oh-sh))
    new_mask[oy:oy+sh, ox:ox+sw] = patch
    return new_mask

def save_out(product, defect, name, img, ok=None, mask=None, ng_ref=None, ng_mask=None):
    """
    mask     : randomly placed mask (OK image coordinates) — for panels 2-6
    ng_mask  : original drawn mask on NG image — for panel 1 NG REF
    """
    out_dir = os.path.join(RESULT_DIR, product, defect)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, img)
    print(f"  → {path}")

    if ok is None or mask is None:
        return

    ys, xs = np.where(mask > 127)
    if not len(ys):
        return

    TH  = 320   # panel height
    PAD = 100   # context around defect
    oh, ow = ok.shape[:2]
    x0 = max(0, xs.min()-PAD); x1 = min(ow, xs.max()+PAD)
    y0 = max(0, ys.min()-PAD); y1 = min(oh, ys.max()+PAD)

    def rh(p, h=TH):
        ih, iw = p.shape[:2]
        return cv2.resize(p, (max(1, int(iw*h/ih)), h))

    def label(p, txt, color=(0,255,255)):
        p = p.copy()
        cv2.putText(p, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(p, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,   2, cv2.LINE_AA)
        return p

    # Panel 1 — NG ref: use ng_mask (original drawn mask) for overlay + crop
    p1 = None
    if ng_ref is not None:
        ng_vis = cv2.resize(ng_ref, (ow, oh))
        # Use original drawn mask for overlay (shows real defect location on NG)
        m_show = ng_mask if ng_mask is not None else mask
        ng_ys, ng_xs = np.where(m_show > 127)
        if len(ng_ys):
            ov = ng_vis.copy(); ov[m_show > 127] = (0, 0, 220)
            ng_vis = cv2.addWeighted(ov, 0.4, ng_vis, 0.6, 0)
            cv2.rectangle(ng_vis,
                          (int(ng_xs.min())-4, int(ng_ys.min())-4),
                          (int(ng_xs.max())+4, int(ng_ys.max())+4),
                          (0, 255, 255), 2)
            # Crop NG panel to NG defect location (not placed mask location)
            ng_pad = 80
            nx0 = max(0, ng_xs.min()-ng_pad); nx1 = min(ow, ng_xs.max()+ng_pad)
            ny0 = max(0, ng_ys.min()-ng_pad); ny1 = min(oh, ng_ys.max()+ng_pad)
            p1 = rh(label(ng_vis[ny0:ny1, nx0:nx1], "NG REF"))
        else:
            p1 = rh(label(ng_vis[y0:y1, x0:x1], "NG REF"))

    # Panel 2 — OK crop
    p2 = rh(label(ok[y0:y1,x0:x1].copy(), "OK"))

    # Panel 3 — mask overlay on OK
    ov_ok = ok[y0:y1,x0:x1].copy()
    mc    = mask[y0:y1,x0:x1]
    tmp   = ov_ok.copy(); tmp[mc>127] = (0,0,220)
    ov_ok = cv2.addWeighted(tmp, 0.45, ov_ok, 0.55, 0)
    cv2.rectangle(ov_ok, (xs.min()-x0-3, ys.min()-y0-3),
                          (xs.max()-x0+3, ys.max()-y0+3), (0,255,0), 1)
    p3 = rh(label(ov_ok, "MASK"))

    # Panel 4 — result crop
    p4 = rh(label(img[y0:y1,x0:x1].copy(), "RESULT"))

    # Panel 5 — zoom on defect bbox: OK (top) | RESULT (bottom)
    zpad = 24
    zx0 = max(0, xs.min()-zpad); zx1 = min(ow, xs.max()+zpad)
    zy0 = max(0, ys.min()-zpad); zy1 = min(oh, ys.max()+zpad)
    zoom_res = img[zy0:zy1, zx0:zx1].copy()
    zoom_ok  = ok[zy0:zy1, zx0:zx1].copy()
    zh = TH // 2
    z_ok_r  = cv2.resize(zoom_ok,  (TH, zh))
    z_res_r = cv2.resize(zoom_res, (TH, zh))
    cv2.putText(z_ok_r,  "OK",     (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200), 1, cv2.LINE_AA)
    cv2.putText(z_res_r, "RESULT", (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,200,100), 1, cv2.LINE_AA)
    p5 = np.vstack([z_ok_r, z_res_r])

    # Panel 6 — DIFF: |result - ok| × 4 amplified, only in mask region
    diff = np.abs(img.astype(np.int16) - ok.astype(np.int16)).astype(np.float32)
    diff_amp = np.clip(diff * 4, 0, 255).astype(np.uint8)
    diff_crop = diff_amp[y0:y1, x0:x1].copy()
    # Draw mask contour on diff
    mc_diff = mask[y0:y1, x0:x1]
    diff_crop[mc_diff > 127] = np.maximum(diff_crop[mc_diff > 127], np.array([0, 40, 40]))
    p6 = rh(label(diff_crop, "DIFF ×4", color=(100,255,100)))

    panels = [p for p in [p1, p2, p3, p4, p5, p6] if p is not None]
    debug  = np.hstack(panels)
    cv2.imwrite(os.path.join(out_dir, "debug_" + name), debug)

def mask_bbox(m):
    ys, xs = np.where(m > 127)
    return xs.min(), ys.min(), xs.max(), ys.max()


def gen_dent_patch(rng2, product_bbox):
    """
    Generate a random small dent shape patch (not yet placed).
    4 variants: circle / wide ellipse / tall ellipse / irregular blob.
    Size = 3–9% of product dimension.
    """
    px0, py0, px1, py1 = product_bbox
    prod_min = min(px1 - px0, py1 - py0)

    min_r = max(10, int(prod_min * 0.03))
    max_r = max(min_r + 5, int(prod_min * 0.09))

    kind = rng2.integers(0, 4)

    if kind == 0:                              # ── circle
        r = int(rng2.integers(min_r, max_r))
        size = r * 2 + 8
        canvas = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(canvas, (size//2, size//2), (r, r), 0, 0, 360, 255, -1)

    elif kind == 1:                            # ── wide ellipse
        rx = int(rng2.integers(min_r, max_r))
        ry = int(rng2.integers(max(6, min_r//2), max(7, rx * 2 // 3)))
        ang = float(rng2.uniform(0, 180))
        size = max(rx, ry) * 2 + 8
        canvas = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(canvas, (size//2, size//2), (rx, ry), ang, 0, 360, 255, -1)

    elif kind == 2:                            # ── tall ellipse
        ry = int(rng2.integers(min_r, max_r))
        rx = int(rng2.integers(max(6, min_r//2), max(7, ry * 2 // 3)))
        ang = float(rng2.uniform(0, 180))
        size = max(rx, ry) * 2 + 8
        canvas = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(canvas, (size//2, size//2), (rx, ry), ang, 0, 360, 255, -1)

    else:                                      # ── irregular blob
        r = int(rng2.integers(min_r, max_r))
        size = r * 2 + 16
        n_pts = int(rng2.integers(7, 14))
        angs  = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        noise = rng2.uniform(0.72, 1.28, n_pts)
        pts   = np.array([
            [int(size//2 + r * noise[i] * np.cos(a)),
             int(size//2 + r * noise[i] * np.sin(a))]
            for i, a in enumerate(angs)
        ], dtype=np.int32)
        canvas = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(canvas, [pts], 255)
        canvas = cv2.GaussianBlur(canvas, (9, 9), 3)
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)

    return canvas


def gen_nhựa_patch(rng2, product_bbox):
    """
    Generate compact organic plastic-flash shapes.
    3 variants: smooth-blob / soft-ellipse / gentle-wavy-blob.
    Size = 3–8% of product dimension.
    All variants stay near-circular (compact) — no crescents, kidneys, or multi-lobes.
    """
    px0, py0, px1, py1 = product_bbox
    prod_min = min(px1 - px0, py1 - py0)

    min_r = max(12, int(prod_min * 0.03))
    max_r = max(min_r + 8, int(prod_min * 0.08))
    r     = int(rng2.integers(min_r, max_r))

    kind = rng2.integers(0, 3)

    if kind == 0:                                   # ── smooth organic blob
        size  = r * 2 + 20
        n_pts = int(rng2.integers(14, 22))
        angs  = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        noise = rng2.uniform(0.82, 1.18, n_pts)    # ±18% → compact, not amoeba
        noise = np.convolve(np.tile(noise, 3), np.ones(5) / 5, mode='same')[n_pts:2*n_pts]
        pts   = np.array([
            [int(size // 2 + r * noise[i] * np.cos(a)),
             int(size // 2 + r * noise[i] * np.sin(a))]
            for i, a in enumerate(angs)
        ], dtype=np.int32)
        canvas = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(canvas, [pts], 255)
        canvas = cv2.GaussianBlur(canvas, (11, 11), 4)
        _, canvas = cv2.threshold(canvas, 100, 255, cv2.THRESH_BINARY)

    elif kind == 1:                                 # ── soft ellipse (near-round)
        size  = r * 2 + 20
        canvas = np.zeros((size, size), dtype=np.uint8)
        rx = max(4, int(r * float(rng2.uniform(0.82, 1.00))))
        ry = max(4, int(r * float(rng2.uniform(0.82, 1.00))))
        ang = float(rng2.uniform(0, 180))
        cv2.ellipse(canvas, (size // 2, size // 2), (rx, ry), ang, 0, 360, 255, -1)
        canvas = cv2.GaussianBlur(canvas, (9, 9), 3)
        _, canvas = cv2.threshold(canvas, 100, 255, cv2.THRESH_BINARY)

    else:                                           # ── gentle wavy-blob (low-amplitude waves)
        size  = r * 2 + 20
        n_pts = int(rng2.integers(20, 28))
        angs  = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        freq  = int(rng2.integers(2, 5))            # fewer, gentler waves
        wave  = 1.0 + 0.10 * np.sin(freq * angs + float(rng2.uniform(0, 2 * np.pi)))
        base_noise = rng2.uniform(0.90, 1.10, n_pts)
        radii = wave * base_noise
        pts   = np.array([
            [int(size // 2 + r * radii[i] * np.cos(a)),
             int(size // 2 + r * radii[i] * np.sin(a))]
            for i, a in enumerate(angs)
        ], dtype=np.int32)
        canvas = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(canvas, [pts], 255)
        canvas = cv2.GaussianBlur(canvas, (9, 9), 3)
        _, canvas = cv2.threshold(canvas, 100, 255, cv2.THRESH_BINARY)

    return canvas


# ── METHOD: Dị vật đen — nhiều chấm nhỏ rải trong product region ─────────────

def synth_dark_spots(ok, product_bbox, seed=0, n_spots_range=(1, 1), r_range=(1, 2),
                     bump=True):
    """
    Scatter dark spots (circle/ellipse/blob) inside product bbox.
    Returns (result_bgr, composite_mask).
    """
    rng2 = np.random.default_rng(seed)
    out  = ok.copy().astype(np.float32)
    H, W = ok.shape[:2]
    px0, py0, px1, py1 = product_bbox

    n0, n1 = int(n_spots_range[0]), int(n_spots_range[1])
    n_spots = int(rng2.integers(n0, n1 + 1)) if n1 >= n0 else n0
    composite = np.zeros((H, W), dtype=np.float32)

    # Build placement candidates: moderately bright pixels on cap surface.
    # Avoid over-bright halo (>210) and dark shadow grooves (<80).
    roi_gray = cv2.cvtColor(ok[py0:py1, px0:px1], cv2.COLOR_BGR2GRAY)
    bright_ys, bright_xs = np.where((roi_gray > 80) & (roi_gray < 210))
    _has_bright = len(bright_ys) > 0

    for _ in range(n_spots):
        if _has_bright:
            idx = int(rng2.integers(0, len(bright_ys)))
            cx = int(bright_xs[idx]) + px0
            cy = int(bright_ys[idx]) + py0
        else:
            cx = int(rng2.integers(px0 + 5, px1 - 5))
            cy = int(rng2.integers(py0 + 5, py1 - 5))
        r0, r1 = int(r_range[0]), int(r_range[1])
        r  = int(rng2.integers(r0, r1 + 1))
        # Compact round/oval shapes only — no streaks (ref image shows round blobs).
        # kind 1 = smooth ellipse, kind 2 = organic blob (compact)
        kind = int(rng2.choice([1, 2], p=[0.40, 0.60]))

        S = 7  # supersample factor
        pad = 6
        pr = max(2, int(r) + pad)
        ph = pr * 2 + 1
        pw = pr * 2 + 1
        phs, pws = ph * S, pw * S

        spot_s = np.zeros((phs, pws), dtype=np.float32)
        cc = (pr * S, pr * S)

        if kind == 1:  # smooth ellipse — compact, aspect ratio close to 1
            rx = int(max(1, r) * S)
            ry = int(max(1, r) * float(rng2.uniform(0.65, 1.0)) * S)
            angle = float(rng2.uniform(0, 180))
            cv2.ellipse(spot_s, cc, (rx, ry), angle, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
        else:  # organic blob — compact, mild jitter only
            n_pts = int(rng2.integers(8, 13))
            base_angs = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            ang_step = 2 * np.pi / n_pts
            ang_noise = rng2.uniform(-0.25, 0.25, n_pts) * ang_step
            angs = np.sort((base_angs + ang_noise) % (2 * np.pi))
            # Tight radial jitter — stays compact, never elongated
            noise = rng2.uniform(0.80, 1.20, n_pts)
            rr = float(max(1, r) * S)
            pts = np.array([
                [int(cc[0] + rr * noise[i] * np.cos(a)),
                 int(cc[1] + rr * noise[i] * np.sin(a))]
                for i, a in enumerate(angs)
            ], dtype=np.int32)
            cv2.fillPoly(spot_s, [pts], 1.0, lineType=cv2.LINE_AA)

        # Downsample then apply SOFT Gaussian falloff — edges fade gradually like ref image.
        spot_p = cv2.resize(spot_s, (pw, ph), interpolation=cv2.INTER_AREA)
        # Strong blur = soft diffuse edge (matches ref: spot fades into surface)
        sigma_soft = max(0.8, r * 0.35)
        spot_p = cv2.GaussianBlur(spot_p, (0, 0), sigma_soft)
        # No hard threshold — keep the gradient so edges are naturally soft
        spot_p = np.clip(spot_p / (spot_p.max() + 1e-6), 0.0, 1.0)

        spot = np.zeros((H, W), dtype=np.float32)
        y0 = cy - pr; y1 = cy + pr + 1
        x0 = cx - pr; x1 = cx + pr + 1
        py0 = 0; px0 = 0; py1 = ph; px1 = pw
        if y0 < 0:
            py0 = -y0; y0 = 0
        if x0 < 0:
            px0 = -x0; x0 = 0
        if y1 > H:
            py1 = ph - (y1 - H); y1 = H
        if x1 > W:
            px1 = pw - (x1 - W); x1 = W
        spot[y0:y1, x0:x1] = spot_p[py0:py1, px0:px1]

        if bump:
            # Opaque dark particle: replace metal texture with near-black flat color
            dark_val = float(rng2.uniform(20, 55))
            for c in range(3):
                out[:, :, c] = out[:, :, c] * (1.0 - spot) + dark_val * spot

            # Surface grain: fine noise on particle body (rough foreign body texture)
            grain = rng2.standard_normal((H, W)).astype(np.float32) * 5.0
            for c in range(3):
                out[:, :, c] += grain * spot

            # Light direction (consistent across highlight + shadow)
            lx = float(rng2.uniform(-1.0, 1.0))
            ly = float(rng2.uniform(-1.0, 1.0))
            nrm = (lx**2 + ly**2) ** 0.5 + 1e-6
            lx, ly = lx / nrm, ly / nrm
            gy, gx = np.gradient(spot)

            # Specular highlight: bright rim on light-facing edge
            highlight = np.clip(-(gx * lx + gy * ly), 0.0, 1.0)
            rim = np.clip((spot - 0.50) / 0.50, 0.0, 1.0)
            highlight = cv2.GaussianBlur(highlight * rim, (0, 0), 0.5)

            # Cast shadow: thin dark crescent on surface outside particle, shadow side.
            # A raised particle blocks light → shadow falls on surface just beside it.
            spot_u8 = np.clip(spot * 255, 0, 255).astype(np.uint8)
            dil_px = max(2, int(r * 0.5))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dil_px * 2 + 1, dil_px * 2 + 1))
            dilated = cv2.dilate(spot_u8, kernel).astype(np.float32) / 255.0
            shadow_ring = np.clip(dilated - spot, 0.0, 1.0)
            # Directional weight: heavier on the side opposite the light source
            yy, xx = np.mgrid[0:H, 0:W]
            dir_w = np.clip(
                ((xx - cx) * (-lx) + (yy - cy) * (-ly)) / max(r, 1),
                0.0, 1.0).astype(np.float32)
            shadow_cast = cv2.GaussianBlur(
                shadow_ring * (0.3 + 0.7 * dir_w), (0, 0), dil_px * 0.5 + 0.5)

            for c in range(3):
                out[:, :, c] = out[:, :, c] + highlight * 28.0
                out[:, :, c] *= (1.0 - shadow_cast * 0.55)
        else:
            # Pure dark speck
            darkness = float(rng2.uniform(0.01, 0.05))   # 1-5% brightness
            for c in range(3):
                out[:, :, c] *= (1.0 - spot * (1.0 - darkness))

        composite = np.maximum(composite, spot)

    mask_out = (composite * 255).clip(0, 255).astype(np.uint8)
    return np.clip(out, 0, 255).astype(np.uint8), mask_out


# ── METHOD: Xước — đường thẳng gấp khúc nhiều nét ────────────────────────────

def synth_scratch_lines(ok, mask, seed=0):
    """
    Multiple angular straight scratch lines with sharp corners (not curved).
    Dark groove + 1px bright highlight offset.
    """
    rng2  = np.random.default_rng(seed)
    out   = ok.copy().astype(np.float32)
    H, W  = ok.shape[:2]
    x0, y0, x1, y1 = mask_bbox(mask)
    span  = max(x1-x0, y1-y0, 30)

    layer = np.zeros((H, W), dtype=np.float32)
    n_sc  = int(rng2.integers(4, 9))   # 4-8 vết xước

    for _ in range(n_sc):
        # Điểm bắt đầu ngẫu nhiên trong mask bbox
        sx = int(rng2.integers(x0, max(x0+1, x1)))
        sy = int(rng2.integers(y0, max(y0+1, y1)))
        n_segs = int(rng2.integers(1, 5))   # 1-4 đoạn thẳng gấp khúc
        thick  = int(rng2.integers(1, 3))
        cx, cy = sx, sy

        for _ in range(n_segs):
            angle  = float(rng2.uniform(0, 2 * np.pi))
            length = int(rng2.integers(max(15, span//6), max(16, span//2)))
            ex = int(np.clip(cx + np.cos(angle) * length, 0, W-1))
            ey = int(np.clip(cy + np.sin(angle) * length, 0, H-1))

            p0, p1 = (cx, cy), (ex, ey)
            # Dark groove
            cv2.line(layer, p0, p1, -1.0, thick, cv2.LINE_AA)
            # Bright highlight: 1px perpendicular offset
            dx, dy = ex - cx, ey - cy
            n = np.sqrt(dx**2 + dy**2) + 1e-6
            nx, ny = int(round(-dy / n)), int(round(dx / n))
            cv2.line(layer, (cx+nx, cy+ny), (ex+nx, ey+ny), 0.5, 1, cv2.LINE_AA)

            cx, cy = ex, ey

    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (9, 9), 3)
    layer = np.clip(layer, -1, 1)
    for c in range(3):
        out[:, :, c] += layer * alpha * 75
    return np.clip(out, 0, 255).astype(np.uint8)


def synth_plastic_scuff(ok, mask, seed=0, alpha_mult=1.35, whiten_add=120, mode="auto"):
    """
    Plastic scuff: soft whitening streaks (less dark groove, softer edges).
    Produces 1–3 thin strokes, then blurs to look matte.
    """
    rng2 = np.random.default_rng(seed)
    out  = ok.copy().astype(np.float32)
    H, W = ok.shape[:2]
    x0, y0, x1, y1 = mask_bbox(mask)
    span = max(x1 - x0, y1 - y0, 40)

    layer = np.zeros((H, W), dtype=np.float32)
    n_sc  = int(rng2.integers(2, 6))

    # Pick a scuff style (feature diversity)
    if mode == "auto":
        mode = ["whiten_streak", "micro_haze", "gouge_whiten", "parallel_micro", "crosshatch"][int(rng2.integers(0, 5))]

    def _draw_parallel_bundle(layer_img, angle, count, spacing_px, length_rng, thick=1):
        """Draw a bundle of near-parallel micro scratches."""
        ca, sa = np.cos(angle), np.sin(angle)
        # perpendicular direction for offsets
        px, py = -sa, ca
        for i in range(count):
            # center point in mask bbox
            cx = int(rng2.integers(x0, max(x0 + 1, x1)))
            cy = int(rng2.integers(y0, max(y0 + 1, y1)))
            # offset line by i*spacing with jitter
            off = (i - (count - 1) / 2.0) * spacing_px + float(rng2.uniform(-0.6, 0.6) * spacing_px)
            cx2 = int(np.clip(cx + px * off, 0, W - 1))
            cy2 = int(np.clip(cy + py * off, 0, H - 1))
            length = int(rng2.integers(length_rng[0], length_rng[1]))
            # Make it less "perfect": slight curvature + broken segments + varying intensity
            n_pts = int(rng2.integers(6, 10))
            t = np.linspace(-1.0, 1.0, n_pts).astype(np.float32)
            # perpendicular wobble
            amp = spacing_px * float(rng2.uniform(0.20, 0.85))
            freq = float(rng2.uniform(0.8, 1.6))
            phase = float(rng2.uniform(0, 2 * np.pi))
            wob = (np.sin((t + 1) * np.pi * freq + phase) * amp).astype(np.float32)

            ax = (cx2 + ca * length * t).astype(np.float32)
            ay = (cy2 + sa * length * t).astype(np.float32)
            bx = np.clip(ax + px * wob, 0, W - 1).astype(np.int32)
            by = np.clip(ay + py * wob, 0, H - 1).astype(np.int32)

            drop_p = float(rng2.uniform(0.10, 0.35))
            base_i = float(rng2.uniform(0.55, 1.0))
            for j in range(n_pts - 1):
                if float(rng2.random()) < drop_p:
                    continue
                p0 = (int(bx[j]), int(by[j]))
                p1 = (int(bx[j + 1]), int(by[j + 1]))
                inten = float(np.clip(base_i * rng2.uniform(0.85, 1.10), 0.35, 1.0))
                cv2.line(layer_img, p0, p1, inten, thick, cv2.LINE_AA)

    for _ in range(n_sc):
        sx = int(rng2.integers(x0, max(x0 + 1, x1)))
        sy = int(rng2.integers(y0, max(y0 + 1, y1)))
        n_segs = int(rng2.integers(2, 5))
        thick  = 1 if mode != "micro_haze" else int(rng2.integers(2, 4))
        cx, cy = sx, sy
        for _ in range(n_segs):
            angle  = float(rng2.uniform(0, 2 * np.pi))
            length = int(rng2.integers(max(22, span // 4), max(23, span)))
            ex = int(np.clip(cx + np.cos(angle) * length, 0, W - 1))
            ey = int(np.clip(cy + np.sin(angle) * length, 0, H - 1))
            cv2.line(layer, (cx, cy), (ex, ey), 1.0, thick, cv2.LINE_AA)
            cx, cy = ex, ey

    # Additional feature modes
    if mode == "parallel_micro":
        # Many thin, near-parallel hairlines
        base_ang = float(rng2.uniform(0, np.pi))
        count = int(rng2.integers(6, 16))
        spacing = float(rng2.uniform(2.0, 5.0))
        length_rng = (max(25, span // 3), max(26, span))
        _draw_parallel_bundle(layer, base_ang, count=count, spacing_px=spacing, length_rng=length_rng, thick=1)
        # add a second faint bundle with slight angle change
        if rng2.random() < 0.55:
            _draw_parallel_bundle(layer, base_ang + float(rng2.uniform(-0.25, 0.25)),
                                  count=int(rng2.integers(4, 10)),
                                  spacing_px=float(rng2.uniform(2.5, 6.0)),
                                  length_rng=(max(18, span // 4), max(19, span // 2)),
                                  thick=1)

    elif mode == "crosshatch":
        # Two bundles crossing each other (common on plastic scuff)
        a0 = float(rng2.uniform(0, np.pi))
        a1 = a0 + float(rng2.uniform(0.6, 1.2))
        count0 = int(rng2.integers(6, 14))
        count1 = int(rng2.integers(5, 12))
        spacing0 = float(rng2.uniform(2.0, 5.0))
        spacing1 = float(rng2.uniform(2.0, 5.5))
        length_rng0 = (max(22, span // 3), max(23, span))
        length_rng1 = (max(18, span // 4), max(19, span // 2))
        _draw_parallel_bundle(layer, a0, count=count0, spacing_px=spacing0, length_rng=length_rng0, thick=1)
        _draw_parallel_bundle(layer, a1, count=count1, spacing_px=spacing1, length_rng=length_rng1, thick=1)

    # Soft edges + matte appearance
    if mode == "whiten_streak":
        layer = cv2.GaussianBlur(layer, (0, 0), 1.7)
    elif mode == "micro_haze":
        layer = cv2.GaussianBlur(layer, (0, 0), 3.3)
    elif mode in ("parallel_micro", "crosshatch"):
        # keep thin but not razor sharp
        layer = cv2.GaussianBlur(layer, (0, 0), 1.35)
    else:  # gouge_whiten
        layer = cv2.GaussianBlur(layer, (0, 0), 1.2)
    layer = np.clip(layer, 0, 1)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (13, 13), 3.8)
    a = np.clip(alpha * layer * float(np.clip(alpha_mult, 0.5, 3.0)), 0, 1)

    # Whitening: lift toward local bright, avoid harsh contrast
    gray = out.mean(axis=2)
    # Micro-haze is more subtle but wider; gouge has stronger local whitening
    if mode == "micro_haze":
        target = np.clip(gray + float(whiten_add) * 0.75, 0, 252)
    elif mode == "gouge_whiten":
        target = np.clip(gray + float(whiten_add) * 1.05, 0, 255)
    elif mode in ("parallel_micro", "crosshatch"):
        # micro scratches: less whitening but more edge density
        target = np.clip(gray + float(whiten_add) * 0.85, 0, 252)
    else:
        target = np.clip(gray + float(whiten_add), 0, 255)

    # Optional shallow groove for gouge (plastic can show slight dark edge)
    if mode == "gouge_whiten":
        edge = cv2.Laplacian(layer, cv2.CV_32F, ksize=3)
        edge = np.clip(edge, 0, 1)
        groove = cv2.GaussianBlur(edge, (0, 0), 0.9) * (a * 0.6)
        for c in range(3):
            out[:, :, c] = out[:, :, c] - groove * 35.0

    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1 - a) + target * a
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_plastic_matte(ok, mask, seed=0, strength=0.75):
    """
    Reduce specular/contrast in mask region to mimic plastic surface (matte),
    so whitening scratches remain visible even on shiny-looking OK.
    """
    rng2 = np.random.default_rng(seed)
    out = ok.copy().astype(np.float32)
    H, W = ok.shape[:2]

    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (23, 23), 6.5)
    a = np.clip(a * float(np.clip(strength, 0.0, 1.5)), 0.0, 1.0)

    # Local blur removes sharp specular reflections
    blur = cv2.GaussianBlur(out, (0, 0), 3.6)

    # Compress highlights in ROI
    gray = out.mean(axis=2)
    hi = np.percentile(gray[mask > 127], 92) if (mask > 127).any() else 235.0
    hi = float(np.clip(hi, 160.0, 245.0))
    comp = out.copy()
    comp = np.minimum(comp, hi)

    # Blend toward blurred + compressed for matte look
    matte = (blur * 0.72 + comp * 0.28)

    # Add tiny micro-texture noise (very subtle) to avoid "airbrushed" look
    n = rng2.normal(0.0, 1.0, (H, W)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 1.2)
    n = (n - n.min()) / (n.max() - n.min() + 1e-6) - 0.5
    matte = matte + n[:, :, None] * 2.0

    out = out * (1 - a[:, :, None]) + matte * a[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ── METHOD: Procedural thread (Dị vật chỉ) ───────────────────────────────────

def synth_thread(ok, mask, seed=0):
    """Procedural thread (Dị vật chỉ) — bright organic curve on ring surface."""
    rng2 = np.random.default_rng(seed)
    out  = ok.copy().astype(np.float32)
    x0, y0, x1, y1 = mask_bbox(mask)
    w, h = max(x1-x0, 10), max(y1-y0, 10)
    H, W = ok.shape[:2]

    bg_vals  = ok[mask > 127].astype(np.float32)
    bg_mean  = float(bg_vals.mean()) if len(bg_vals) else 160.0
    bg_float = ok.astype(np.float32).mean(axis=2)

    thread_mult = rng2.uniform(1.30, 1.50)
    thread_base = float(np.clip(bg_mean * thread_mult + 12, 160, 238))

    # ── Thread path ────────────────────────────────────────────────────────────
    cx_m, cy_m = (x0+x1)/2.0, (y0+y1)/2.0
    angle    = rng2.uniform(0, np.pi)
    half_len = np.sqrt(w**2 + h**2) / 2 * 2.5
    px_v, py_v = -np.sin(angle), np.cos(angle)
    N = 900
    t_vals = np.linspace(0, 1, N)
    main_x = cx_m - np.cos(angle)*half_len + t_vals * 2*np.cos(angle)*half_len
    main_y = cy_m - np.sin(angle)*half_len + t_vals * 2*np.sin(angle)*half_len

    # ── Curvature: sine-wave nhẹ + noise mượt → cong tự nhiên, không gấp ────
    # ── Curvature: tổng nhiều sin tần số khác nhau → spacing bất đều ────────
    amp = max(w, h) * rng2.uniform(0.08, 0.15)   # biên độ vừa, không gấp

    # Layer 1: sóng chính — 5-8 uốn
    f1 = rng2.uniform(5, 9)
    offset = amp * np.sin(t_vals * np.pi * f1 + rng2.uniform(0, 2*np.pi))

    # Layer 2: tần số lệch → phách tạo spacing không đều
    f2 = f1 * rng2.uniform(0.55, 0.80)
    offset += amp * rng2.uniform(0.5, 0.85) * np.sin(t_vals * np.pi * f2 + rng2.uniform(0, 2*np.pi))

    # Layer 3: tần số cao hơn → thêm uốn nhỏ xen giữa
    f3 = f1 * rng2.uniform(1.4, 2.0)
    offset += amp * rng2.uniform(0.2, 0.40) * np.sin(t_vals * np.pi * f3 + rng2.uniform(0, 2*np.pi))

    # Smooth vừa đủ: giữ uốn nhỏ, loại góc gấp
    final_sigma = max(8, N // 40)
    fk = int(final_sigma * 4) | 1
    fker = np.exp(-np.linspace(-2, 2, fk)**2 / 2); fker /= fker.sum()
    offset = np.convolve(offset, fker, mode='same')

    bx = np.clip((main_x + px_v * offset).astype(int), 0, W-1)
    by = np.clip((main_y + py_v * offset).astype(int), 0, H-1)
    abs_off_norm = np.abs(offset) / (amp + 1e-6)

    # ── Per-segment brightness ─────────────────────────────────────────────────
    micro_noise = np.convolve(rng2.uniform(0.93, 1.07, N), np.ones(11)/11, mode='same')
    seg_bright  = np.clip(thread_base * (0.72 + 0.28*abs_off_norm) * micro_noise, 0, 232)

    # ── Taper: computed on VISIBLE range inside clip region ───────────────────
    inflate = max(w, h) // 3
    cx_lo = max(0, x0 - inflate); cx_hi = min(W, x1 + inflate)
    cy_lo = max(0, y0 - inflate); cy_hi = min(H, y1 + inflate)

    in_clip = (bx >= cx_lo) & (bx < cx_hi) & (by >= cy_lo) & (by < cy_hi)
    if in_clip.any():
        first_vis = int(np.argmax(in_clip))
        last_vis  = int(N - 1 - np.argmax(in_clip[::-1]))
    else:
        first_vis, last_vis = 0, N - 1

    vis_len = max(last_vis - first_vis + 1, 2)
    tn      = max(2, int(vis_len * 0.18))   # 18% taper on each end

    taper = np.zeros(N, dtype=np.float32)
    taper[first_vis:last_vis + 1] = 1.0
    taper[first_vis: first_vis + tn] = np.linspace(0.0, 1.0, tn) ** 1.5   # ease-in → sharp tip
    taper[last_vis + 1 - tn: last_vis + 1] = np.linspace(1.0, 0.0, tn) ** 1.5
    # Light smooth to avoid 1-sample jumps (keep tips sharp — small kernel)
    taper = cv2.GaussianBlur(taper.reshape(1, -1), (1, 7), 0).flatten()

    # ── Draw: supersampling 6× + pre-blur → downscale INTER_AREA ─────────────
    S = 6
    presence_s = np.zeros((H*S, W*S), dtype=np.float32)
    body_s     = np.zeros((H*S, W*S), dtype=np.float32)
    center_s   = np.zeros((H*S, W*S), dtype=np.float32)

    for i in range(N - 1):
        t = float(taper[i])
        if t < 0.02:
            continue
        b  = float(seg_bright[i]) / 255.0
        p0 = (int(bx[i]*S),   int(by[i]*S))
        p1 = (int(bx[i+1]*S), int(by[i+1]*S))
        body_w  = max(S, round(2*t) * S)
        spec_bv = min(1.0, b * 1.50) * t
        cv2.line(presence_s, p0, p1, t,        body_w, cv2.LINE_AA)
        cv2.line(body_s,     p0, p1, b*0.30*t, body_w, cv2.LINE_AA)
        cv2.line(center_s,   p0, p1, spec_bv,  S,      cv2.LINE_AA)

    # Blur trên SS canvas (σ = 0.6 pixel output) → khử aliasing trước downscale
    sig = S * 0.6
    ks  = int(sig * 4) | 1
    presence_s = cv2.GaussianBlur(presence_s, (ks, ks), sig)
    body_s     = cv2.GaussianBlur(body_s,     (ks, ks), sig)
    center_s   = cv2.GaussianBlur(center_s,   (ks, ks), sig * 0.5)

    def _ds(layer):
        return cv2.resize(layer, (W, H), interpolation=cv2.INTER_AREA)

    presence_layer = _ds(presence_s)
    body_layer     = _ds(body_s)
    center_layer   = _ds(center_s)

    # ── Clip to inflated bbox ──────────────────────────────────────────────────
    clip_r = np.zeros((H, W), dtype=np.uint8)
    clip_r[cy_lo:cy_hi, cx_lo:cx_hi] = 255

    def _clip(layer):
        return cv2.bitwise_and(
            (layer*255).clip(0,255).astype(np.uint8), clip_r
        ).astype(np.float32) / 255.0

    presence_layer = _clip(presence_layer)
    body_layer     = _clip(body_layer)
    center_layer   = _clip(center_layer)

    # Blur nhẹ sau downscale để blend mượt; center rất tight giữ highlight sắc
    alpha      = cv2.GaussianBlur(presence_layer, (3, 3), 0.6)
    brt_body   = cv2.GaussianBlur(body_layer,     (3, 3), 0.6)
    brt_center = cv2.GaussianBlur(center_layer,   (3, 3), 0.35)

    # Composite: specular chiếm 80% → bóng cao
    brt_map = np.where(brt_center > 0.008,
                       brt_body * 0.20 + brt_center * 0.80,
                       brt_body)

    alpha_h    = np.clip(alpha * 1.7, 0.0, 1.0)
    brt_target = np.where(alpha > 0.02, brt_map / (alpha + 1e-6) * 250, 0.0)
    brt_floor  = np.clip(bg_float + 42, 0, 232)
    brt_target = np.clip(np.maximum(brt_target, brt_floor * alpha_h), 0, 250)

    for c in range(3):
        ch = out[:, :, c]
        ch = ch * (1 - alpha_h) + brt_target * alpha_h
        out[:, :, c] = ch

    return np.clip(out, 0, 255).astype(np.uint8)


# ── Ref-based injection (transfer real defect texture) ────────────────────────

def _rotate_crop(img, angle_deg, border_value=0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle_deg), 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )


def inject_defect_delta(ok, placed_mask, ng_img, ng_mask, seed=0,
                        strength=0.9, alpha_blur_ks=9, inpaint_radius=5,
                        dark_gain=140.0, bright_gain=85.0, thr=0.12,
                        n_strokes=2, warp=0.25, break_prob=0.25,
                        material="metal", placed_mask_loose=None):
    """
    Transfer a scratch-like signal from NG → OK by extracting a signed,
    high-frequency difference between NG and an inpainted background.
    This keeps grayscale scratch character (dark groove + bright highlight)
    without pasting large-scale reflection patterns.

    - placed_mask: mask on OK coordinates (uint8 HxW)
    - ng_img/ng_mask: NG ref image + its original mask (both resized to OK size)
    """
    if ng_img is None or ng_mask is None:
        return ok

    rng2 = np.random.default_rng(seed)
    H, W = ok.shape[:2]

    ng_r = cv2.resize(ng_img, (W, H), interpolation=cv2.INTER_LINEAR)
    ngm  = cv2.resize(ng_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # NG defect crop
    ng_ys, ng_xs = np.where(ngm > 127)
    if not len(ng_ys):
        return ok
    nx0, nx1 = int(ng_xs.min()), int(ng_xs.max())
    ny0, ny1 = int(ng_ys.min()), int(ng_ys.max())
    pad = int(rng2.integers(8, 18))
    nx0 = max(0, nx0 - pad); ny0 = max(0, ny0 - pad)
    nx1 = min(W - 1, nx1 + pad); ny1 = min(H - 1, ny1 + pad)

    ng_crop  = ng_r[ny0:ny1+1, nx0:nx1+1].astype(np.float32)
    ngm_crop = ngm[ny0:ny1+1, nx0:nx1+1]

    # Estimate background on NG by inpainting, then extract a thin scratch map.
    ip_mask = (ngm_crop > 127).astype(np.uint8) * 255
    ip_r = int(max(1, inpaint_radius))
    ng_u8 = np.clip(ng_crop, 0, 255).astype(np.uint8)
    ng_g  = cv2.cvtColor(ng_u8, cv2.COLOR_BGR2GRAY)
    bg_g  = cv2.inpaint(ng_g, ip_mask, ip_r, cv2.INPAINT_TELEA)

    # Scratch signal: signed high-pass of (NG - BG_inpaint)
    diff = (ng_g.astype(np.float32) - bg_g.astype(np.float32))
    diff_hp = diff - cv2.GaussianBlur(diff, (0, 0), 2.0)
    diff_hp[ip_mask == 0] = 0.0

    # Robust normalize inside mask to avoid zebra / over-amplification
    vals = diff_hp[ip_mask > 0]
    if vals.size < 20:
        return ok
    scale = float(np.percentile(np.abs(vals), 95)) + 1e-6
    diff_n = np.clip(diff_hp / scale, -1.0, 1.0)

    # Slight thinning: keep only stronger responses (scratch is sparse)
    thr = float(np.clip(thr, 0.02, 0.60))
    diff_n[np.abs(diff_n) < thr] = 0.0
    diff_n = cv2.GaussianBlur(diff_n, (0, 0), 0.65)

    # Soft alpha from NG mask for later gating
    a_ng = cv2.GaussianBlur((ip_mask.astype(np.float32) / 255.0), (9, 9), 2.0)

    # Target crop on OK by (optionally looser) placed mask bbox
    base_mask = placed_mask_loose if placed_mask_loose is not None else placed_mask
    ys, xs = np.where(base_mask > 127)
    if not len(ys):
        return ok
    tx0, tx1 = int(xs.min()), int(xs.max())
    ty0, ty1 = int(ys.min()), int(ys.max())
    tx0 = max(0, tx0); ty0 = max(0, ty0)
    tx1 = min(W - 1, tx1); ty1 = min(H - 1, ty1)
    th, tw = (ty1 - ty0 + 1), (tx1 - tx0 + 1)

    def _warp_small(img_f32, rngw, s=warp):
        """Small elastic-ish warp to diversify shape without thickening too much."""
        if s <= 0:
            return img_f32
        hh, ww = img_f32.shape[:2]
        # coarse grid displacement -> resize -> smooth
        g = 5
        dx = rngw.uniform(-1, 1, (g, g)).astype(np.float32)
        dy = rngw.uniform(-1, 1, (g, g)).astype(np.float32)
        dx = cv2.resize(dx, (ww, hh), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(dy, (ww, hh), interpolation=cv2.INTER_CUBIC)
        sigma = max(1.0, min(hh, ww) / 18)
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * (s * min(hh, ww) * 0.04)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * (s * min(hh, ww) * 0.04)
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        map_x = (xx + dx).astype(np.float32)
        map_y = (yy + dy).astype(np.float32)
        return cv2.remap(img_f32, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Prepare ROI
    out = ok.copy().astype(np.float32)
    roi = out[ty0:ty1+1, tx0:tx1+1]

    # Draw multiple thin strokes per placed mask to increase diversity/features
    n_strokes = int(np.clip(n_strokes, 1, 5))
    layer_sum = np.zeros((th, tw), dtype=np.float32)

    for si in range(n_strokes):
        rngs = np.random.default_rng(int(seed) * 131 + si * 17 + 3)

        # Random rotate + anisotropic scaling (thinness/length variety)
        angle = float(rngs.uniform(-55, 55))
        diff_r = _rotate_crop(diff_n.astype(np.float32), angle_deg=angle, border_value=0)
        a_r    = _rotate_crop((a_ng * 255).astype(np.uint8), angle_deg=angle, border_value=0).astype(np.float32) / 255.0

        # Anisotropic scale via resize before fitting bbox
        sy = float(rngs.uniform(0.55, 1.10))   # thinner/thicker vertically
        sx = float(rngs.uniform(0.85, 1.55))   # longer/shorter horizontally
        dh, dw = diff_r.shape[:2]
        diff_rs = cv2.resize(diff_r, (max(2, int(dw * sx)), max(2, int(dh * sy))), interpolation=cv2.INTER_LINEAR)
        a_rs    = cv2.resize(a_r,    (diff_rs.shape[1], diff_rs.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Optional warp to change curvature/shape
        diff_rw = _warp_small(diff_rs, rngs, s=float(np.clip(warp, 0.0, 1.0)))
        a_rw    = _warp_small(a_rs,   rngs, s=float(np.clip(warp, 0.0, 1.0)))

        # Resize to target bbox
        diff_fit = cv2.resize(diff_rw, (tw, th), interpolation=cv2.INTER_LINEAR)
        a_fit    = cv2.resize(a_rw,    (tw, th), interpolation=cv2.INTER_LINEAR)

        # Alpha gates final injection:
        # - loose mask allows wide bleed
        # - core mask boosts center strength (more visible in the middle)
        pm_loose = base_mask[ty0:ty1+1, tx0:tx1+1].astype(np.float32) / 255.0
        pm_core  = placed_mask[ty0:ty1+1, tx0:tx1+1].astype(np.float32) / 255.0
        k2 = int(max(5, alpha_blur_ks) | 1)
        pm_loose = cv2.GaussianBlur(pm_loose, (k2, k2), k2 / 4.0)
        pm_core  = cv2.GaussianBlur(pm_core,  (max(3, k2 // 2) | 1, max(3, k2 // 2) | 1), max(0.8, k2 / 8.0))
        gate = np.clip(pm_loose * 0.85 + pm_core * 0.75, 0.0, 1.0)
        alpha = np.clip(gate * a_fit, 0.0, 1.0)

        # Signed diff -> groove + highlight; keep thin by blurring lightly only
        d = cv2.GaussianBlur(diff_fit.astype(np.float32), (0, 0), 0.55)

        dark = np.clip(-d, 0.0, 1.0)
        bright_src = np.clip(d, 0.0, 1.0)
        # Highlight offset direction randomized a bit
        oy = int(rngs.integers(-1, 2))
        ox = int(rngs.integers(-1, 2))
        bright = np.roll(np.roll(bright_src, oy, axis=0), ox, axis=1)

        # Break into segments sometimes (more realistic variety)
        if float(rngs.random()) < float(np.clip(break_prob, 0.0, 1.0)):
            seg = np.ones((th, tw), dtype=np.float32)
            # random horizontal-ish gaps
            n_gaps = int(rngs.integers(2, 6))
            for _ in range(n_gaps):
                gx0 = int(rngs.integers(0, tw))
                gw  = int(rngs.integers(max(6, tw // 18), max(7, tw // 7)))
                seg[:, gx0: min(tw, gx0 + gw)] *= float(rngs.uniform(0.0, 0.35))
            seg = cv2.GaussianBlur(seg, (0, 0), 1.2)
            alpha *= seg

        s = float(np.clip(strength, 0.0, 2.0))
        mat = str(material).lower()
        if mat in ("plastic", "polymer"):
            # Plastic scratches tend to look like whitening / scuff (less sharp specular).
            # Reduce dark groove, increase soft bright scuff.
            scuff = cv2.GaussianBlur(np.maximum(dark, bright), (0, 0), 0.9)
            layer = (-0.35 * float(dark_gain) * dark + 1.15 * float(bright_gain) * scuff) * s
            # Keep it matte: suppress extreme peaks that look metallic
            layer = np.clip(layer, -90.0, 90.0)
        else:
            layer = (-float(dark_gain) * dark + float(bright_gain) * bright) * s
        layer_sum += layer * alpha

    # Apply accumulated layer
    # De-speckle / soften so scratches aren't "noisy" or "stiff"
    if str(material).lower() in ("plastic", "polymer"):
        # remove isolated dots then soften
        m = (np.abs(layer_sum) > 2.0).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        keep = (m.astype(np.float32) / 255.0)
        layer_sum = layer_sum * keep
        layer_sum = cv2.GaussianBlur(layer_sum, (0, 0), 1.0)
        # Global boost after smoothing (to keep visibility)
        layer_sum *= 1.35

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] + layer_sum
    out[ty0:ty1+1, tx0:tx1+1] = roi
    return np.clip(out, 0, 255).astype(np.uint8)


def _blend_deltas(ok, a, b, wa=0.5, wb=0.5):
    """Blend two synthesized results by mixing their deltas vs OK.
    Note: wa+wb can be >1 to intentionally amplify visibility.
    """
    okf = ok.astype(np.float32)
    af  = a.astype(np.float32)
    bf  = b.astype(np.float32)
    out = okf + (af - okf) * float(wa) + (bf - okf) * float(wb)
    return np.clip(out, 0, 255).astype(np.uint8)

def make_loose_mask(mask, seed=0, dilate_min=20, dilate_max=75, blur_sigma=9.5):
    """
    Feathered/dilated mask so scuffs can extend outside the original mask.
    """
    rng2 = np.random.default_rng(seed)
    k = int(rng2.integers(dilate_min, dilate_max + 1))
    k = max(3, k | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dm = cv2.dilate(mask, ker, iterations=1)
    a = cv2.GaussianBlur(dm.astype(np.float32) / 255.0, (0, 0), float(blur_sigma))
    a = np.clip(a, 0, 1)
    return (a * 255).astype(np.uint8)


def make_core_mask(mask, blur_sigma=2.2, gain=1.35):
    """
    Core emphasis mask: strong near original mask, fades quickly.
    Used to make scratches stronger in the center while still allowing wide bleed.
    """
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), float(blur_sigma))
    a = np.clip(a * float(gain), 0.0, 1.0)
    return (a * 255).astype(np.uint8)


# ── Crop ref image using its mask (extract defect region only) ────────────────

def crop_ref_with_mask(ref_path, mask_path, pad=10):
    """Crop ref image to mask bounding box — gives tight defect patch for signal injection."""
    ref  = cv2.imread(ref_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if ref is None or mask is None:
        return None
    # Resize mask to ref size
    rh, rw = ref.shape[:2]
    mask_r = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask_r > 127)
    if len(ys) == 0:
        return None
    x0 = max(0, xs.min() - pad)
    y0 = max(0, ys.min() - pad)
    x1 = min(rw, xs.max() + pad)
    y1 = min(rh, ys.max() + pad)
    return ref[y0:y1, x0:x1]


def encode_ref(ref_crop_bgr):
    """Encode cropped ref as base64 PNG for fast_physics."""
    import base64
    ref_rgb = cv2.cvtColor(ref_crop_bgr, cv2.COLOR_BGR2RGB)
    _, buf  = cv2.imencode(".png", ref_rgb)
    return base64.b64encode(buf).decode()


# ── METHOD: Nhựa chảy — raised plastic drip/flow blob ────────────────────────

def _nhựa_surface_texture(rng2, H, W, cx, cy, mask_r, alpha_gate, intensity):
    """
    Generate a surface texture that looks like solidified/raised plastic.
    Returns brightness offset array (H×W float32), gated to alpha_gate region.

    4 texture types (randomly selected):
      0 ripple   — concentric rings from centroid (frozen drip/splash)
      1 grain    — multi-scale Gaussian noise (rough plastic surface)
      2 streak   — directional flow lines (plastic flowed one way)
      3 cellular — random micro-bumps array (crystalline/bubble texture)
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    tex_type = int(rng2.integers(0, 4))

    if tex_type == 0:                               # ── ripple
        dist  = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        freq  = float(rng2.uniform(0.12, 0.30))    # rings per px-radius
        phase = float(rng2.uniform(0, 2 * np.pi))
        tex   = np.sin(dist * freq * 2 * np.pi + phase)
        tex   = cv2.GaussianBlur(tex, (0, 0), 0.8)

    elif tex_type == 1:                             # ── grain (2 scales)
        n1  = rng2.normal(0, 1, (H, W)).astype(np.float32)
        n2  = rng2.normal(0, 1, (H, W)).astype(np.float32)
        tex = (cv2.GaussianBlur(n1, (0, 0), 1.5) * 0.65 +
               cv2.GaussianBlur(n2, (0, 0), 3.5) * 0.35)
        mx  = np.abs(tex).max() + 1e-6
        tex = np.clip(tex / mx, -1, 1)

    elif tex_type == 2:                             # ── directional streak
        ang  = float(rng2.uniform(0, np.pi))
        proj = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
        # Low freq → fewer, wider bands (less geometric-looking)
        freq = float(rng2.uniform(0.025, 0.07))
        tex  = np.sin(proj * freq * 2 * np.pi) * 0.45
        # Break regularity with significant noise
        n1  = rng2.normal(0, 1, (H, W)).astype(np.float32)
        n2  = rng2.normal(0, 1, (H, W)).astype(np.float32)
        tex += cv2.GaussianBlur(n1, (0, 0), 2.5) * 0.40
        tex += cv2.GaussianBlur(n2, (0, 0), 5.0) * 0.20
        tex  = cv2.GaussianBlur(tex, (0, 0), 1.0)   # final smooth

    else:                                           # ── cellular micro-bumps
        n_cells = int(rng2.integers(10, 28))
        spread  = mask_r * float(rng2.uniform(0.7, 1.0))
        tex     = np.zeros((H, W), dtype=np.float32)
        for _ in range(n_cells):
            bx  = cx + float(rng2.uniform(-spread, spread))
            by  = cy + float(rng2.uniform(-spread, spread))
            r   = max(1.5, mask_r * float(rng2.uniform(0.06, 0.18)))
            d   = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
            b   = np.exp(-0.5 * (d / r) ** 2)       # Gaussian bump
            tex += b * float(rng2.choice([-1.0, 1.0]))
        mx  = np.abs(tex).max() + 1e-6
        tex = np.clip(tex / mx, -1, 1)
        tex = cv2.GaussianBlur(tex, (0, 0), 0.6)   # slight smoothing

    # Strength: ±10-22 DN, scale with intensity
    strength = 10.0 + intensity * 12.0
    tex_out  = tex * strength * alpha_gate
    return tex_out.astype(np.float32)


def _warp_texture_inside_mask(ok_bgr, mask, rng, mask_r):
    """
    Warp/displace pixel coordinates inside mask to break background texture.
    Simulates excess plastic distorting the underlying surface.
    Amplitude = 18-30% of blob radius.
    """
    H, W = ok_bgr.shape[:2]
    amplitude = mask_r * float(rng.uniform(0.18, 0.30))

    # Low-frequency flow-direction warp
    sigma = max(3.0, mask_r * 0.55)
    raw_x = rng.normal(0, 1, (H, W)).astype(np.float32)
    raw_y = rng.normal(0, 1, (H, W)).astype(np.float32)
    dx = cv2.GaussianBlur(raw_x, (0, 0), sigma) * amplitude
    dy = cv2.GaussianBlur(raw_y, (0, 0), sigma) * amplitude

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = np.clip(xx + dx, 0, W - 1)
    map_y = np.clip(yy + dy, 0, H - 1)

    warped = cv2.remap(ok_bgr, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    # Blend: warped inside mask, original outside (soft edge)
    sig_blend = float(np.clip(mask_r * 0.14, 1.0, 9.0))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sig_blend)
    alpha = np.clip(alpha, 0, 1)[:, :, np.newaxis]

    result = warped.astype(np.float32) * alpha + ok_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def _flatten_metallic_inside_mask(ok_bgr, mask, rng, mask_r, flatten_strength=0.45):
    """
    Partially flatten metallic reflections inside mask → matte plastic appearance.
    Blends toward darkened local mean (removes metallic sheen, adds base darkness).
    The base should be visibly darker than surrounding so specular peak stands out.
    """
    H, W = ok_bgr.shape[:2]
    # Very large blur = ambient "plastic body color" without metallic details
    sig_mean = max(8.0, mask_r * 1.0)
    mean_bgr = cv2.GaussianBlur(ok_bgr.astype(np.float32), (0, 0), sig_mean)

    # Darker base: plastic matte is significantly darker than polished metal
    dark = float(rng.uniform(0.22, 0.35))   # was 0.04-0.12 → much darker
    mean_bgr = mean_bgr * (1.0 - dark)

    sig_blend = float(np.clip(mask_r * 0.14, 1.0, 9.0))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sig_blend)
    # flatten_strength pushed higher — really suppress metallic highlights inside
    alpha = np.clip(alpha * flatten_strength, 0, 1)[:, :, np.newaxis]

    result = mean_bgr * alpha + ok_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def synth_nhựa_chảy(ok, mask, seed=0, intensity=0.5):
    """
    Nhựa chảy (plastic flow): raised plastic blob with surface texture.

    Pipeline:
      Step 1 — Base shading (brightness bump matching real ref appearance):
        blob    (50%): smooth raised cosine, bright rim, edge shadow
        cluster (33%): 3-7 crystalline micro-bumps
        haze    (17%): directional wave brightening

      Step 2 — Surface texture overlay (ALL modes get this):
        ripple / grain / streak / cellular  →  ±10-22 DN within mask
    """
    rng2 = np.random.default_rng(seed)
    out  = ok.copy().astype(np.float32)
    H, W = ok.shape[:2]

    ys, xs = np.where(mask > 127)
    if not len(ys):
        return ok

    cx     = float(xs.mean())
    cy     = float(ys.mean())
    mask_r = max(5.0, float(np.sqrt(np.count_nonzero(mask > 127) / np.pi)))

    mode = rng2.choice(["blob", "blob", "blob", "cluster", "cluster", "haze"])

    # ── Alpha masks ───────────────────────────────────────────────────────────
    # Sharp alpha: clear blob boundary (không quá mờ)
    sig_sharp  = float(np.clip(mask_r * 0.15, 0.8, 8.0))
    alpha_sharp = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sig_sharp)
    alpha_sharp = np.clip(alpha_sharp, 0, 1)

    # Medium alpha: for outer shadow (slightly wider than blob)
    sig_med   = float(np.clip(mask_r * 0.35, 1.5, 16.0))
    alpha_med = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sig_med)
    alpha_med = np.clip(alpha_med, 0, 1)

    # Strict alpha: for texture (tight inside shape)
    sig_strict   = float(np.clip(mask_r * 0.12, 0.6, 5.0))
    alpha_strict = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sig_strict)
    alpha_strict = np.clip(alpha_strict, 0, 1)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # ── Step 1: base shading ──────────────────────────────────────────────────

    if mode == "blob":
        # ── Crumpled thin plastic film model (from ref image analysis) ────────
        # Ref shows: thin plastic sheet that's crumpled/folded → NOT a smooth blob.
        # Visual signature:
        #   - Multiple fine fold ridges (bright ridge + dark valley, like folded cellophane)
        #   - Overall brightness ≈ surrounding (not much lift), texture is the key difference
        #   - Irregular organic shape (already from gen_dent_patch)
        #   - Thin hard shadow ring just outside

        mask_bin = (mask > 127).astype(np.uint8)
        mask_inv = 1 - mask_bin
        dist_out = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5).astype(np.float32)

        rng_p = np.random.default_rng(seed + 9191)

        # ── 1. Subtle base lift (slight — crumpled plastic ≈ same tone as surround) ──
        lift = alpha_sharp * (8.0 + intensity * 10.0)
        for c in range(3):
            out[:, :, c] += lift

        # ── 2. Fold ridges — core of crumpled plastic look ────────────────────
        # Each fold: a thin bright ridge with dark sides, at random angle & position.
        # Bright = fold crest facing light; dark = fold valley in shadow.
        n_folds  = int(rng_p.integers(10, 22))
        fold_map = np.zeros((H, W), dtype=np.float32)

        for _ in range(n_folds):
            fold_ang = float(rng_p.uniform(0, np.pi))          # fold line direction
            # Fold center: random inside mask (biased toward center)
            ang2 = float(rng_p.uniform(0, 2 * np.pi))
            rad2 = mask_r * float(rng_p.beta(1.2, 2.0))
            fcx  = cx + rad2 * np.cos(ang2)
            fcy  = cy + rad2 * np.sin(ang2)

            # Perpendicular distance from fold axis
            perp  = (xx - fcx) * np.sin(fold_ang) - (yy - fcy) * np.cos(fold_ang)
            # Along-axis fade (fold has finite length)
            along = (xx - fcx) * np.cos(fold_ang) + (yy - fcy) * np.sin(fold_ang)
            fold_len  = mask_r * float(rng_p.uniform(0.25, 0.85))
            along_fade = np.exp(-0.5 * (along / fold_len) ** 2)

            # Ridge profile: sharp bright crest with dark flanks
            fw    = max(1.2, mask_r * float(rng_p.uniform(0.03, 0.10)))
            crest = np.exp(-0.5 * (perp / fw) ** 2)           # bright crest
            flank = np.exp(-0.5 * (perp / (fw * 2.5)) ** 2)  # wider dark shadow
            ridge = crest - flank * 0.45                       # net: bright center, dark sides

            amp = float(rng_p.uniform(0.4, 1.0))
            fold_map += ridge * along_fade * amp

        # Normalize fold map to [-1, 1] then scale
        fm_max = np.abs(fold_map).max() + 1e-6
        fold_map = fold_map / fm_max
        # Scale: folds ±30-55 DN inside mask
        fold_add = fold_map * alpha_strict * (30.0 + intensity * 25.0)

        # ── 3. Fine grain (plastic surface micro-texture) ─────────────────────
        n1   = rng_p.normal(0, 1, (H, W)).astype(np.float32)
        grain = cv2.GaussianBlur(n1, (0, 0), max(1.0, mask_r * 0.05))
        grain = grain / (np.abs(grain).max() + 1e-6)
        grain_add = grain * alpha_strict * (5.0 + intensity * 5.0)

        for c in range(3):
            out[:, :, c] += fold_add + grain_add

        # ── 4. Thin hard shadow ring just outside boundary ────────────────────
        shad_w    = max(1.2, mask_r * 0.08)
        shad_prof = np.exp(-0.5 * (dist_out / shad_w) ** 2)
        shad_prof = shad_prof * (1.0 - alpha_sharp)
        contact_shadow = shad_prof * (110.0 + intensity * 60.0)
        for c in range(3):
            out[:, :, c] -= contact_shadow

    elif mode == "cluster":
        # Multiple shiny micro-bumps (each with its own Phong highlight)
        n_bumps        = int(rng2.integers(3, 8))
        cluster_spread = mask_r * float(rng2.uniform(0.40, 0.75))
        la = float(rng2.uniform(0, 2 * np.pi))          # shared light angle
        total = np.zeros((H, W), dtype=np.float32)

        for _ in range(n_bumps):
            bx = cx + float(rng2.uniform(-cluster_spread, cluster_spread))
            by = cy + float(rng2.uniform(-cluster_spread, cluster_spread))
            r  = max(2.0, mask_r * float(rng2.uniform(0.12, 0.28)))
            d  = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
            b  = np.clip(1.0 - d / r, 0.0, 1.0) ** 2   # quadratic

            gy_b, gx_b = np.gradient(b)
            nx_b = -gx_b * 5.0;  ny_b = -gy_b * 5.0;  nz_b = np.ones_like(nx_b)
            nn   = np.sqrt(nx_b**2 + ny_b**2 + nz_b**2) + 1e-6
            nx_b, ny_b, nz_b = nx_b/nn, ny_b/nn, nz_b/nn
            lx_d = np.cos(la) * 0.5;  ly_d = np.sin(la) * 0.5;  lz_d = 0.75
            ndl  = nx_b * lx_d + ny_b * ly_d + nz_b * lz_d
            total += (np.clip(ndl, 0, 1) * b * (45.0 + intensity * 35.0) +
                      np.clip(ndl, 0, 1) ** 10 * b * (70.0 + intensity * 60.0))

        total = np.clip(total, 0, 230.0)
        # Base shadow under cluster
        dist  = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        outer = np.clip((dist - mask_r * 0.90) / (mask_r * 0.35), 0, 1)
        outer_shad = outer * (1 - outer) * 4.0 * (20.0 + intensity * 20.0)
        sub_out = outer_shad * (alpha_med - alpha_sharp)
        for c in range(3):
            out[:, :, c] += total * alpha_sharp - sub_out

    else:  # haze — broad flow mark
        sig_haze   = float(np.clip(mask_r * 0.65, 2.0, 28.0))
        alpha_haze = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sig_haze)
        alpha_haze = np.clip(alpha_haze, 0, 1)

        wave_angle = float(rng2.uniform(0, np.pi))
        proj = (xx - cx) * np.cos(wave_angle) + (yy - cy) * np.sin(wave_angle)
        freq = float(rng2.uniform(0.04, 0.09))
        wave = 0.35 * np.sin(proj * freq * 2.0 * np.pi) + 0.65

        noise = rng2.normal(0, 0.14, (H, W)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), 2.5)

        haze = np.clip((wave + noise) * (25.0 + intensity * 30.0), 0, 85.0)
        for c in range(3):
            out[:, :, c] += haze * alpha_haze

    # ── Step 2: surface texture overlay ──────────────────────────────────────
    tex = _nhựa_surface_texture(rng2, H, W, cx, cy, mask_r, alpha_strict, intensity)
    for c in range(3):
        out[:, :, c] += tex * 2.0    # stronger texture — plastic flow marks

    return np.clip(out, 0, 255).astype(np.uint8)


# ── MKA defect → fast_physics mapping ────────────────────────────────────────
#  defect_type follows fast_physics._resolve_method:
#    scratch/crack/foreign → signal_injection (needs ref)
#    dent/bulge            → shaded_warp      (no ref)
#    chip/rust             → ref_paste        (needs ref)

MKA_DEFECTS = {
    # defect_folder  : (fp_defect_type, needs_ref)  — None = use custom method
    "Lõm"           : ("dent",    False),
    "Nhựa_chảy"     : ("bulge",   True),
    "Xước"          : ("scratch", True),
    "Dị_vật_chỉ"    : ("foreign", True),
    "Dị_vật_đen"    : (None,      True),   # use synth_blob
    # "Cấn_miệng"  skipped — ok/ref size mismatch, handle later
}

INTENSITIES  = [0.5, 0.8]   # 2 variants only
MAX_MASKS    = 2             # max 2 masks per defect


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output → {RESULT_DIR}\n")
    for defect_folder, (fp_type, needs_ref) in MKA_DEFECTS.items():
        defect_path = os.path.join(ROOT, "MKA", defect_folder)
        ok = load_ok(defect_path)
        if ok is None:
            print(f"[SKIP] No ok: {defect_path}"); continue

        oh, ow = ok.shape[:2]

        # Pair ref images with their masks (same stem)
        ref_dir  = os.path.join(defect_path, "ref")
        mask_dir = os.path.join(defect_path, "mask")
        ref_files = sorted(glob.glob(os.path.join(ref_dir, "*"))  )
        ref_files = [r for r in ref_files if not r.endswith("mask.png") and
                     os.path.splitext(os.path.basename(r))[0] != "mask"]

        pairs = []  # (mask_arr, ref_crop_b64 or None, ng_img or None)
        for ref_path in ref_files:
            stem      = os.path.splitext(os.path.basename(ref_path))[0]
            mask_path = os.path.join(mask_dir, stem + ".png")
            if not os.path.exists(mask_path):
                continue
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_NEAREST)
            if not (m > 127).any():
                continue
            ng_img  = cv2.imread(ref_path)
            ref_b64 = None
            if needs_ref:
                crop = crop_ref_with_mask(ref_path, mask_path)
                if crop is not None:
                    ref_b64 = encode_ref(crop)
            pairs.append((m, ref_b64, ng_img))

        if not pairs:
            print(f"[SKIP] No mask-ref pairs: {defect_folder}"); continue

        pairs = pairs[:MAX_MASKS]

        # Detect product region once for this OK image
        product_bbox = detect_product_bbox(ok)
        px0,py0,px1,py1 = product_bbox
        print(f"[MKA/{defect_folder}] fp_type={fp_type}  pairs={len(pairs)}  product=({px0},{py0})-({px1},{py1})")

        for mi, (ng_mask_orig, ref_b64, ng_img) in enumerate(pairs):

            # Lõm + Nhựa chảy: 4 procedural shapes per pair; others: 1 shape from NG ref
            n_shapes = 4 if fp_type in ("dent", "bulge") else 1

            for si in range(n_shapes):
                for vi, intensity in enumerate(INTENSITIES):
                    base_seed = mi * 1000 + si * 100 + vi
                    rng2 = np.random.default_rng(base_seed)

                    if fp_type == "bulge":
                        shape_patch = gen_nhựa_patch(rng2, product_bbox)
                    elif fp_type == "dent":
                        shape_patch = gen_dent_patch(rng2, product_bbox)
                    if fp_type in ("dent", "bulge"):
                        mask = place_mask_random(shape_patch, product_bbox, ok.shape,
                                                np.random.default_rng(base_seed + 1), rotate=True)
                    else:
                        shape_patch = extract_defect_shape(ng_mask_orig)
                        if shape_patch is None:
                            print(f"  [SKIP] mask{mi:02d} empty"); continue
                        mask = place_mask_random(shape_patch, product_bbox, ok.shape, rng2)

                    seed = base_seed
                    tag  = f"m{mi:02d}_s{si}_i{int(intensity*10)}"

                    def _save(nm, result_bgr, _mask=mask):
                        save_out("MKA", defect_folder, nm, result_bgr,
                                 ok=ok, mask=_mask, ng_ref=ng_img, ng_mask=ng_mask_orig)

                    # ── Dispatch by defect type ───────────────────────────────────

                    if fp_type is None:
                        # Dị vật đen: chấm nhỏ rải trong product, bỏ qua mask ngẫu nhiên
                        # Generate ONE foreign black spot per image
                        result_bgr, spot_mask = synth_dark_spots(
                            ok, product_bbox,
                            seed=seed,
                            n_spots_range=(1, 1),
                            r_range=(1, 2),
                            bump=True,
                        )
                        save_out("MKA", defect_folder, f"dark_{tag}.jpg", result_bgr,
                                 ok=ok, mask=spot_mask, ng_ref=ng_img, ng_mask=ng_mask_orig)

                    elif fp_type == "foreign":
                        # Dị vật chỉ
                        result_bgr = synth_thread(ok, mask, seed=seed)
                        _save(f"thread_{tag}.jpg", result_bgr)

                    else:
                        # Xước: inject real defect delta from NG (more natural than stampy signal_injection)
                        if fp_type == "scratch":
                            rngp = np.random.default_rng(seed + 2026)
                            # Matte adaptation first (plastic scratches are visible on matte polymer)
                            matte_s = float(rngp.uniform(*PLASTIC_SCRATCH["matte_strength_range"]))
                            ok_for_scratch = apply_plastic_matte(ok, mask, seed=seed + 123, strength=matte_s)
                            # Allow scuff/scratch to bleed outside the strict mask
                            loose_mask = make_loose_mask(mask, seed=seed + 77)
                            core_mask  = make_core_mask(mask, blur_sigma=2.0, gain=1.45)

                            # Use original NG mask + NG image as texture donor
                            # Increase geometric diversity (feature-rich) within plausible bounds
                            warp_v = float(rngp.uniform(*PLASTIC_SCRATCH["warp_range"]))
                            brk_v  = float(rngp.uniform(*PLASTIC_SCRATCH["break_prob_range"]))
                            nst    = int(rngp.integers(PLASTIC_SCRATCH["proc_strokes_range"][0],
                                                      PLASTIC_SCRATCH["proc_strokes_range"][1] + 1))

                            ref_bgr = inject_defect_delta(
                                ok_for_scratch, mask,
                                ng_img=ng_img,
                                ng_mask=ng_mask_orig,
                                seed=seed,
                                strength=1.55 if intensity >= 0.8 else 1.20,
                                dark_gain=210.0 if intensity >= 0.8 else 175.0,
                                bright_gain=130.0 if intensity >= 0.8 else 105.0,
                                thr=0.10 if intensity >= 0.8 else 0.12,
                                n_strokes=nst,
                                warp=warp_v,
                                break_prob=brk_v,
                                material="plastic" if defect_folder == "Xước" else "metal",
                                placed_mask_loose=loose_mask,
                            )
                            # Procedural component adds diverse geometry even if ref is weak
                            a_mult = float(rngp.uniform(*PLASTIC_SCRATCH["scuff_alpha_mult_range"]))
                            w_add  = int(rngp.integers(PLASTIC_SCRATCH["scuff_whiten_add_range"][0],
                                                      PLASTIC_SCRATCH["scuff_whiten_add_range"][1] + 1))
                            proc_bgr = synth_plastic_scuff(
                                ok_for_scratch, loose_mask,
                                seed=seed + 999,
                                alpha_mult=a_mult,
                                whiten_add=w_add,
                                mode="auto",
                            )
                            # Boost center visibility (core) without killing wide bleed
                            proc_core = synth_plastic_scuff(
                                ok_for_scratch, core_mask,
                                seed=seed + 1002,
                                alpha_mult=float(np.clip(a_mult * 1.05, 0.6, 2.2)),
                                whiten_add=int(np.clip(w_add + 15, 70, 210)),
                                mode="whiten_streak",
                            )
                            proc_bgr = _blend_deltas(ok_for_scratch, proc_bgr, proc_core, wa=0.72, wb=0.28)
                            # Optionally overlay an extra scuff mode for feature richness
                            if float(rngp.random()) < float(PLASTIC_SCRATCH["extra_mode_prob"]):
                                extra_mode = "crosshatch" if float(rngp.random()) < float(PLASTIC_SCRATCH["crosshatch_prob"]) else "parallel_micro"
                                proc2 = synth_plastic_scuff(
                                    ok_for_scratch, loose_mask,
                                    seed=seed + 1999,
                                    alpha_mult=float(np.clip(a_mult * rngp.uniform(0.7, 1.05), 0.6, 2.2)),
                                    whiten_add=int(np.clip(w_add + int(rngp.integers(-20, 25)), 70, 210)),
                                    mode=extra_mode,
                                )
                                # Blend procedural layers lightly to avoid overblown whiten
                                proc_bgr = _blend_deltas(ok_for_scratch, proc_bgr, proc2, wa=0.65, wb=0.35)

                            # Mix in delta space. Keep realism: weights near 1.0 total, small bias allowed.
                            wa = float(rngp.uniform(*PLASTIC_SCRATCH["mix_ref_range"]))
                            wb = float(rngp.uniform(*PLASTIC_SCRATCH["mix_proc_range"]))
                            # Prevent extreme overblown results (cap total amplification)
                            total = wa + wb
                            if total > 1.45:
                                wa *= 1.45 / total
                                wb *= 1.45 / total
                            result_bgr = _blend_deltas(ok_for_scratch, ref_bgr, proc_bgr, wa=wa, wb=wb)
                            _save(f"scratch_{tag}.jpg", result_bgr)
                            continue

                        elif fp_type == "bulge":
                            # Nhựa chảy — dùng synth_nhựa_chảy trực tiếp.
                            # Không blend với injection vì cộng delta 2 pass gây overexposed.
                            # Fold-ridge synthesis đã đủ → DIFF đúng = RESULT đúng.
                            result_bgr = synth_nhựa_chảy(ok, mask, seed=seed, intensity=intensity)
                            _save(f"nhựa_{tag}.jpg", result_bgr)
                            continue

                        # Lõm (dent) → fast_physics shaded_warp
                        fp_intensity = intensity
                        params = {
                            "intensity":         fp_intensity,
                            "naturalness":       0.6,
                            "position_jitter":   0.0,
                            "seed":              seed,
                            "sdxl_refine":       False,
                            "skip_struct_adapt": True,
                        }
                        if ref_b64:
                            params["ref_image_b64"] = ref_b64
                        ok_rgb     = cv2.cvtColor(ok, cv2.COLOR_BGR2RGB)
                        res        = _fp_generate(ok_rgb, mask, fp_type, "metal", params)
                        result_bgr = cv2.cvtColor(decode_b64(res["result_pre_refine"]),
                                                  cv2.COLOR_RGB2BGR)
                        _save(f"{fp_type}_{tag}.jpg", result_bgr)

    print(f"\nDone → {RESULT_DIR}")
