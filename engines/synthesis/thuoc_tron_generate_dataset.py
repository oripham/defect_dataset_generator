import argparse
import glob
import os
from datetime import datetime

import cv2
import numpy as np


def detect_pill_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Mask viên thuốc trên nền trắng, output uint8 0/255."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 5)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(g, dtype=np.uint8)
    if not cnts:
        return m
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(m, [c], -1, 255, -1)
    return m


def bbox_from_mask(mask_u8: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_u8 > 127)
    if ys.size < 10:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _soft_mask(mask_u8: np.ndarray, dilate: int, blur_k: int) -> np.ndarray:
    """Approx `create_soft_mask` from fast_physics (Gaussian blur of dilated mask)."""
    m = mask_u8
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate | 1, dilate | 1))
        m = cv2.dilate(m, k, iterations=1)
    blur_k = int(blur_k) | 1
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (blur_k, blur_k), 0)


def _save_debug(
    out_dir: str,
    name: str,
    *,
    ok: np.ndarray,
    result: np.ndarray,
    mask_u8: np.ndarray,
    circle: tuple[int, int, int] | None = None,
    arc: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> None:
    """Save debug panels similar to `experiments.py`."""
    ys, xs = np.where(mask_u8 > 127)
    if ys.size < 10:
        return

    TH = 320
    PAD = 90
    oh, ow = ok.shape[:2]
    x0 = max(0, int(xs.min()) - PAD)
    x1 = min(ow, int(xs.max()) + PAD)
    y0 = max(0, int(ys.min()) - PAD)
    y1 = min(oh, int(ys.max()) + PAD)

    def rh(p: np.ndarray, h: int = TH) -> np.ndarray:
        ih, iw = p.shape[:2]
        return cv2.resize(p, (max(1, int(iw * h / max(1, ih))), h))

    def label(p: np.ndarray, txt: str, color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        p = p.copy()
        cv2.putText(p, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(p, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return p

    ok_crop = ok[y0:y1, x0:x1].copy()
    res_crop = result[y0:y1, x0:x1].copy()
    m_crop = mask_u8[y0:y1, x0:x1]

    # OK
    p2 = rh(label(ok_crop, "OK"))

    # MASK overlay
    ov_ok = ok_crop.copy()
    tmp = ov_ok.copy()
    tmp[m_crop > 127] = (0, 0, 220)
    ov_ok = cv2.addWeighted(tmp, 0.45, ov_ok, 0.55, 0)
    # Optional: show Hough circle + arc for "Nứt"
    if circle is not None:
        cx, cy, r = circle
        cv2.circle(ov_ok, (cx - x0, cy - y0), int(r), (255, 180, 0), 2, cv2.LINE_AA)
        cv2.circle(ov_ok, (cx - x0, cy - y0), 2, (255, 180, 0), -1, cv2.LINE_AA)
    if arc is not None:
        (xA, yA), (xB, yB) = arc
        cv2.line(ov_ok, (xA - x0, yA - y0), (xB - x0, yB - y0), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(
        ov_ok,
        (int(xs.min()) - x0 - 3, int(ys.min()) - y0 - 3),
        (int(xs.max()) - x0 + 3, int(ys.max()) - y0 + 3),
        (0, 255, 0),
        1,
    )
    p3 = rh(label(ov_ok, "MASK"))

    # RESULT
    res_vis = res_crop.copy()
    if circle is not None:
        cx, cy, r = circle
        cv2.circle(res_vis, (cx - x0, cy - y0), int(r), (255, 180, 0), 2, cv2.LINE_AA)
        cv2.circle(res_vis, (cx - x0, cy - y0), 2, (255, 180, 0), -1, cv2.LINE_AA)
    if arc is not None:
        (xA, yA), (xB, yB) = arc
        cv2.line(res_vis, (xA - x0, yA - y0), (xB - x0, yB - y0), (0, 255, 255), 2, cv2.LINE_AA)
    p4 = rh(label(res_vis, "RESULT"))

    # Zoom OK/RESULT at tight bbox
    zpad = 24
    zx0 = max(0, int(xs.min()) - zpad)
    zx1 = min(ow, int(xs.max()) + zpad)
    zy0 = max(0, int(ys.min()) - zpad)
    zy1 = min(oh, int(ys.max()) + zpad)
    zoom_ok = ok[zy0:zy1, zx0:zx1].copy()
    zoom_res = result[zy0:zy1, zx0:zx1].copy()
    zh = TH // 2
    z_ok_r = cv2.resize(zoom_ok, (TH, zh))
    z_res_r = cv2.resize(zoom_res, (TH, zh))
    cv2.putText(z_ok_r, "OK", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(
        z_res_r, "RESULT", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1, cv2.LINE_AA
    )
    p5 = np.vstack([z_ok_r, z_res_r])

    # DIFF ×4 inside crop
    diff = np.abs(result.astype(np.int16) - ok.astype(np.int16)).astype(np.float32)
    diff_amp = np.clip(diff * 4, 0, 255).astype(np.uint8)
    diff_crop = diff_amp[y0:y1, x0:x1].copy()
    diff_crop[m_crop > 127] = np.maximum(diff_crop[m_crop > 127], np.array([0, 40, 40], dtype=np.uint8))
    p6 = rh(label(diff_crop, "DIFF x4", color=(100, 255, 100)))

    debug = np.hstack([p2, p3, p4, p5, p6])
    cv2.imwrite(os.path.join(out_dir, "debug_" + name), debug)


def _hough_circle_on_pill(ok_bgr: np.ndarray, pill_mask_u8: np.ndarray) -> tuple[int, int, int] | None:
    """Return (cx,cy,r) in image coords if detected."""
    H, W = ok_bgr.shape[:2]
    g = cv2.cvtColor(ok_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.6)

    # Focus on pill area: fill outside with median so edges inside pill dominate
    if (pill_mask_u8 > 127).any():
        med = float(np.median(g[pill_mask_u8 > 127]))
    else:
        med = float(np.median(g))
    g2 = g.copy()
    g2[pill_mask_u8 <= 127] = med

    # Estimate plausible radius range from bbox
    bb = bbox_from_mask(pill_mask_u8)
    if bb is None:
        return None
    x, y, w, h = bb
    r0 = max(10, int(0.38 * min(w, h)))
    r1 = max(r0 + 2, int(0.60 * min(w, h)))

    circles = cv2.HoughCircles(
        g2,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, int(min(w, h) * 0.35)),
        param1=120,
        param2=24,
        minRadius=r0,
        maxRadius=r1,
    )
    if circles is None:
        return None
    c = circles[0]
    # pick strongest: first
    cx, cy, r = c[0]
    cx = int(np.clip(int(round(cx)), 0, W - 1))
    cy = int(np.clip(int(round(cy)), 0, H - 1))
    r = int(np.clip(int(round(r)), 5, max(H, W)))
    return cx, cy, r


def synth_dent(ok_bgr: np.ndarray, seed: int, intensity: float,
               dent_strength: float = 1.0, dent_size: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """
    Lõm thuốc bột nén — physical spherical-cap model.

    Nguyên tắc vật lý (ánh sáng khuếch tán từ trên):
      - Vành dent (rim): bề mặt nghiêng → pháp tuyến lệch khỏi nguồn sáng → tối hơn
      - Tâm dent: bề mặt phẳng → pháp tuyến hướng lên → sáng như bình thường hoặc hơi sáng hơn
      - Kết quả: vòng tối xung quanh dent, tâm sáng → trông có chiều sâu 3D thật

    Trước đây: tô đen đều → trông như vết bẩn, không có texture lõm.
    """
    rng = np.random.default_rng(seed)
    mask = detect_pill_mask(ok_bgr)
    bb = bbox_from_mask(mask)
    if bb is None:
        return ok_bgr.copy(), np.zeros(ok_bgr.shape[:2], dtype=np.uint8)
    x, y, w, h = bb
    H, W = ok_bgr.shape[:2]

    # ── 1. Vị trí dent (trong viên thuốc, tránh viền) ──────────────────────────
    circ = _hough_circle_on_pill(ok_bgr, mask)
    if circ is None:
        ccx, ccy, rr = x + w * 0.5, y + h * 0.5, 0.48 * float(min(w, h))
    else:
        ccx, ccy, rr = float(circ[0]), float(circ[1]), float(circ[2])

    ang_pos = float(rng.uniform(0, 2 * np.pi))
    rad_frac = float(np.clip(rng.beta(2.0, 2.2), 0.20, 0.65))
    cx = float(np.clip(ccx + rr * rad_frac * np.cos(ang_pos), x + 3, x + w - 4))
    cy = float(np.clip(ccy + rr * rad_frac * np.sin(ang_pos), y + 3, y + h - 4))

    # ── 2. Kích thước dent — có thể điều chỉnh qua dent_size ───────────────────
    # dent_size: tỉ lệ so với đường kính viên (0.04–0.20), default 0.08
    size_frac = float(np.clip(dent_size, 0.04, 0.20))
    r_dent = max(5.0, float(min(w, h)) * size_frac * float(rng.uniform(0.85, 1.15)))
    ax = r_dent * float(rng.uniform(0.85, 1.20))
    ay = r_dent * float(rng.uniform(0.85, 1.20))
    ang_tilt = float(rng.uniform(0.0, 180.0))

    # ── 3. Hệ tọa độ chuẩn hóa trong ellipse của dent ─────────────────────────
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cos_a, sin_a = float(np.cos(np.radians(ang_tilt))), float(np.sin(np.radians(ang_tilt)))
    dx = xx - cx;  dy = yy - cy
    dx_r =  cos_a * dx + sin_a * dy
    dy_r = -sin_a * dx + cos_a * dy
    r_norm = np.sqrt((dx_r / (ax + 1e-6))**2 + (dy_r / (ay + 1e-6))**2)  # 0=tâm, 1=viền

    inside = r_norm < 1.0

    # ── 4. Spherical-cap height field: h(r)=sqrt(1-r²), 1 ở tâm, 0 ở viền ────
    h_cap = np.where(inside, np.sqrt(np.clip(1.0 - r_norm**2, 0.0, 1.0)), 0.0).astype(np.float32)

    inten = float(np.clip(intensity, 0.0, 1.0))

    # ── 5. Alpha mask — Gaussian mềm, không hard edge, không viền tối ──────────
    # Dent chỉ lõm xuống → viền mờ dần tự nhiên như ảnh thật
    sigma_r = float(rng.uniform(0.38, 0.52))   # sigma theo đơn vị r_norm
    alpha_dent = np.exp(-(r_norm**2) / (2.0 * sigma_r**2))
    blur_k = max(3, int(r_dent * 0.6)) | 1
    alpha_dent = cv2.GaussianBlur(alpha_dent.astype(np.float32), (blur_k, blur_k), 0)
    alpha_dent *= mask.astype(np.float32) / 255.0
    alpha_dent = np.clip(alpha_dent, 0.0, 1.0)

    # ── 6. Tối toàn vùng — scale bởi dent_strength ──────────────────────────────
    ds = float(np.clip(dent_strength, 0.2, 4.0))
    base_dn = float(rng.uniform(25, 45)) * (0.5 + 0.8 * inten) * ds

    # Shading hướng nhẹ (1 phía tối hơn một chút) → cảm giác có chiều sâu
    light_ang = float(rng.uniform(0, 2 * np.pi))
    side = (dx_r / (ax + 1e-6)) * np.cos(light_ang) + (dy_r / (ay + 1e-6)) * np.sin(light_ang)
    directional = np.clip(side, -1.0, 1.0) * base_dn * 0.30

    # ── 7. Texture bột bên trong — đa tầng, mờ dần ra ngoài ────────────────────
    # Tầng coarse (σ~4-7px)
    n_c = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
    n_c = cv2.GaussianBlur(n_c, (0, 0), float(rng.uniform(4.0, 7.0)))
    n_c = (n_c - n_c.min()) / (n_c.max() - n_c.min() + 1e-6) - 0.5

    # Tầng medium (σ~2-3.5px)
    n_m = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
    n_m = cv2.GaussianBlur(n_m, (0, 0), float(rng.uniform(2.0, 3.5)))
    n_m = (n_m - n_m.min()) / (n_m.max() - n_m.min() + 1e-6) - 0.5

    # Tầng fine (σ~1px)
    n_f = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
    n_f = cv2.GaussianBlur(n_f, (0, 0), 1.0)
    n_f = (n_f - n_f.min()) / (n_f.max() - n_f.min() + 1e-6) - 0.5

    grain = n_c * 0.50 + n_m * 0.35 + n_f * 0.15
    grain_amp = float(rng.uniform(35, 55)) * (0.7 + 0.7 * inten) * ds
    texture = grain * grain_amp * alpha_dent

    # ── 8. Tổng hợp ────────────────────────────────────────────────────────────
    total_shift = (-(base_dn + directional) * alpha_dent + texture)

    result = ok_bgr.astype(np.float32).copy()
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] + total_shift, 0.0, 255.0)

    out = result.astype(np.uint8)
    dmask_dbg = ((alpha_dent > 0.25) & (mask > 127)).astype(np.uint8) * 255
    return out, dmask_dbg


def synth_chip(ok_bgr: np.ndarray, seed: int, intensity: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Nứt: dùng Hough circle để xác định viên tròn, tạo 1 phần trên vành (arc crack/chip).
    Debug will include detected circle + arc endpoints.
    """
    rng = np.random.default_rng(seed)
    out = ok_bgr.astype(np.float32)
    mask = detect_pill_mask(ok_bgr)
    bb = bbox_from_mask(mask)
    if bb is None:
        return ok_bgr.copy(), np.zeros(ok_bgr.shape[:2], dtype=np.uint8), {"circle": None, "arc": None}
    x, y, w, h = bb
    H, W = out.shape[:2]
    m = (mask > 127).astype(np.uint8)

    circ = _hough_circle_on_pill(ok_bgr, mask)
    if circ is None:
        # fallback: bbox center/radius
        cx = x + w * 0.5
        cy = y + h * 0.5
        r = 0.48 * float(min(w, h))
        circ = (int(cx), int(cy), int(r))
    cx, cy, r = circ

    # pick an arc segment on circumference
    ang = float(rng.uniform(0, 2 * np.pi))
    arc_len = float(rng.uniform(0.35, 0.75)) * (0.7 + 0.45 * intensity)  # radians
    a0 = ang - arc_len * 0.5
    a1 = ang + arc_len * 0.5
    bx = int(np.clip(cx + r * np.cos(ang), 0, W - 1))
    by = int(np.clip(cy + r * np.sin(ang), 0, H - 1))

    chip = np.zeros((H, W), np.uint8)
    # draw a thick arc "crack" region on rim (not full chip wedge)
    thick = int(rng.integers(max(2, int(r * 0.020)), max(3, int(r * 0.040)) + 1))
    rim_r = int(r * float(rng.uniform(0.93, 1.01)))
    # approximate arc by polyline points
    ts = np.linspace(a0, a1, int(rng.integers(20, 38))).astype(np.float32)
    pts = np.stack([cx + rim_r * np.cos(ts), cy + rim_r * np.sin(ts)], axis=1).astype(np.int32)
    cv2.polylines(chip, [pts], isClosed=False, color=255, thickness=thick, lineType=cv2.LINE_AA)
    chip = cv2.GaussianBlur(chip, (0, 0), 0.9)
    _, chip = cv2.threshold(chip, 20, 255, cv2.THRESH_BINARY)
    chip = cv2.bitwise_and(chip, (m * 255))

    # darken along crack + tiny highlight offset
    edge_f = cv2.GaussianBlur(chip.astype(np.float32) / 255.0, (0, 0), 1.2)
    dv = float(rng.uniform(-65.0, -35.0)) * (0.85 + 0.35 * intensity)
    out += dv * edge_f[:, :, None]

    # highlight slightly inward (specular on crack lip)
    inward = int(max(1, thick // 2))
    chip_in = np.zeros_like(chip)
    pts_in = np.stack([cx + (rim_r - inward) * np.cos(ts), cy + (rim_r - inward) * np.sin(ts)], axis=1).astype(np.int32)
    cv2.polylines(chip_in, [pts_in], isClosed=False, color=255, thickness=max(1, thick - 1), lineType=cv2.LINE_AA)
    chip_in = cv2.GaussianBlur(chip_in.astype(np.float32), (0, 0), 1.0) / 255.0
    out += float(rng.uniform(6.0, 18.0)) * chip_in[:, :, None] * (0.6 + 0.5 * intensity)

    # dark edge around chip (fracture boundary)
    blur = cv2.GaussianBlur(out, (0, 0), 0.7)
    a = cv2.GaussianBlur((chip > 0).astype(np.float32), (0, 0), 1.3)[:, :, None]
    out = out * (1.0 - 0.40 * a) + blur * (0.40 * a)

    # endpoints for debug
    pA = (int(np.clip(cx + rim_r * np.cos(a0), 0, W - 1)), int(np.clip(cy + rim_r * np.sin(a0), 0, H - 1)))
    pB = (int(np.clip(cx + rim_r * np.cos(a1), 0, W - 1)), int(np.clip(cy + rim_r * np.sin(a1), 0, H - 1)))
    dbg = {"circle": (int(cx), int(cy), int(r)), "arc": (pA, pB)}
    return np.clip(out, 0, 255).astype(np.uint8), chip, dbg


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Thuốc_tròn defect dataset (no YOLO labels yet)")
    ap.add_argument("--root", default=r"V:\defect_samples\Thuốc_tròn", help="Thuốc_tròn root")
    ap.add_argument("--out", default=r"V:\defect_samples\results", help="Output root")
    ap.add_argument("--num", type=int, default=1, help="Aug per OK image per intensity")
    ap.add_argument("--max-ok", type=int, default=2, help="Limit number of OK images per class (for quick preview)")
    ap.add_argument(
        "--ok-seed",
        type=int,
        default=None,
        help="Seed for random OK selection (default: random each run)",
    )
    ap.add_argument(
        "--intensities",
        default="0.8",
        help="Comma-separated list, e.g. '0.8' or '0.5,0.8' (default: 0.8)",
    )
    ap.add_argument("--debug", action="store_true", help="Write debug panels like experiments.py")
    args = ap.parse_args()

    _ovr = os.environ.get("DEFECT_SAMPLES_RESULT_DIR", "").strip()
    if _ovr:
        out_root = os.path.join(_ovr, "Thuốc_tròn")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = os.path.join(args.out, ts, "Thuốc_tròn")
    os.makedirs(out_root, exist_ok=True)

    configs = [
        ("Lõm", os.path.join(args.root, "Lõm", "ok"), synth_dent),
        ("Nứt", os.path.join(args.root, "Nứt", "ok"), synth_chip),
    ]
    intensities = []
    for s in str(args.intensities).split(","):
        s = s.strip()
        if not s:
            continue
        intensities.append(float(s))
    if not intensities:
        intensities = [0.8]

    total = 0
    for defect, ok_dir, fn in configs:
        out_dir = os.path.join(out_root, defect)
        os.makedirs(out_dir, exist_ok=True)
        ok_paths = sorted(glob.glob(os.path.join(ok_dir, "*.png")))
        # Randomly sample OK images (so we don't always use the same first files)
        max_ok = max(0, int(args.max_ok))
        if max_ok > 0 and len(ok_paths) > max_ok:
            rng_ok = np.random.default_rng(args.ok_seed)
            idx = rng_ok.choice(len(ok_paths), size=max_ok, replace=False)
            ok_paths = [ok_paths[int(i)] for i in sorted(idx)]
        for ok_i, p in enumerate(ok_paths):
            ok = cv2.imread(p)
            if ok is None:
                continue
            pill_mask = detect_pill_mask(ok)
            circ_ok = _hough_circle_on_pill(ok, pill_mask)
            ok_stem = os.path.splitext(os.path.basename(p))[0]
            for inten in intensities:
                for k in range(args.num):
                    seed = (ok_i + 1) * 10000 + int(inten * 10) * 100 + k * 7 + (0 if defect == "Lõm" else 999)
                    dbg = None
                    if defect == "Nứt":
                        aug, dmask, dbg = synth_chip(ok, seed=seed, intensity=float(inten))
                    else:
                        aug, dmask = synth_dent(ok, seed=seed, intensity=float(inten))
                    name = f"{defect}_{ok_stem}_i{int(inten*10)}_{k}.jpg"
                    cv2.imwrite(os.path.join(out_dir, name), aug)
                    if args.debug:
                        if dbg is not None:
                            _save_debug(out_dir, name, ok=ok, result=aug, mask_u8=dmask, circle=dbg["circle"], arc=dbg["arc"])
                        else:
                            _save_debug(out_dir, name, ok=ok, result=aug, mask_u8=dmask, circle=circ_ok, arc=None)
                    total += 1

    print(out_root)
    print(f"images={total}")


if __name__ == "__main__":
    main()

