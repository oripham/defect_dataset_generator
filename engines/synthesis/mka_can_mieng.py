"""
engines/synthesis/mka_can_mieng.py — Cấn miệng (Rim crush) synthesis
=====================================================================
Defect: local surface nick/crush on the dark inner oval rim of MKA plastic bottle.
Visually: a small bright streak (6-10 px) on the dark ring with adjacent shadow.
          The crushed rim surface changes local reflection angle → bright highlight
          on one side, shadow crescent on the other.

Real NG characteristics (from 2048×2048 close-up scans):
  - Defect size: ~20-30 px in 2048×2048 → ~6-10 px in 890×510 full bottle image
  - Appearance: small elongated bright spot along ring tangent + one-sided shadow
  - Location: on the dark oval rim ring (r≈50-70 px from bottle mouth center)
  - NOT a large protrusion — subtle surface brightness change only

Algorithm:
  1. Detect inner oval rim via threshold + ellipse fit
  2. Pick random point on ellipse perimeter
  3. Apply micro elastic warp (inward crush, 1.5-4 px) — shape signal only
  4. Inject asymmetric brightness: bright streak + one-sided shadow
  5. Tight Gaussian blend (radius 10-16 px)
"""

from __future__ import annotations
import cv2
import numpy as np


# ── Internal: detect inner oval rim ─────────────────────────────────────────

def _detect_inner_rim(gray: np.ndarray) -> tuple | None:
    """
    Find the bottle mouth rim (vành miệng chai) — the dark oval ring
    surrounding the inner opening, which is subject to 'cấn miệng' defects.

    Targets contour with:
      - area: 5000–15000 px   (inner mouth ring, nearly circular)
      - axis ratio > 0.80     (nearly circular, not a thin arc)
      - nearest to image center

    Returns (cx, cy, semi_x, semi_y, angle_deg) in full image coords.
    """
    h, w = gray.shape
    icx, icy = w // 2, h // 2

    _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark    = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best, best_dist = None, 1e9
    for c in cnts:
        area = cv2.contourArea(c)
        # Inner mouth ring: area ~5000-15000, nearly circular (ratio > 0.80)
        if area < 5000 or area > 15000 or len(c) < 5:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        mcx = M['m10'] / M['m00']
        mcy = M['m01'] / M['m00']

        ell = cv2.fitEllipse(c)
        (ex, ey), (ea, eb), eang = ell
        ratio = min(ea, eb) / (max(ea, eb) + 1e-6)
        if ratio < 0.80:
            continue  # must be nearly circular — inner mouth ring

        dist = np.hypot(mcx - icx, mcy - icy)
        if dist < best_dist:
            best_dist = dist
            best      = (ex, ey, ea / 2, eb / 2, eang)

    return best


# ── Internal: local elastic warp ─────────────────────────────────────────────

def _local_warp(img: np.ndarray, px: int, py: int,
                dx: float, dy: float, radius: int) -> np.ndarray:
    """
    Push pixels near (px, py) by (dx, dy) with Gaussian falloff.
    """
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    dist2 = (xs - px) ** 2 + (ys - py) ** 2
    sigma2 = (radius * 0.5) ** 2
    weight = np.exp(-dist2 / (2 * sigma2))

    map_x = (xs - dx * weight).clip(0, w - 1).astype(np.float32)
    map_y = (ys - dy * weight).clip(0, h - 1).astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ── Public: synth_can_mieng ───────────────────────────────────────────────────

def synth_can_mieng(
    ok_bgr: np.ndarray,
    seed: int = 42,
    intensity: float = 0.7,
    warp_strength: float = 1.0,
    streak_length: float = 1.0,
    thickness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesise a Rim Crush defect on the MKA bottle image.

    Parameters
    ----------
    ok_bgr        : OK image (BGR)
    seed          : RNG seed for reproducibility
    intensity     : 0–1, brightness of the highlight
    warp_strength : 0.5–2.0, pixel displacement magnitude of the elastic warp.
                    Controls how far the ring boundary is pushed outward.
                    0.5 = very subtle shape change  |  2.0 = clearly visible bump
    streak_length : 0.5–4.0, tangential extent (along the ring).
                    1.0 ≈ short nick (5-10 px)  |  4.0 = long streak (20-40 px)
    thickness     : 0.3–3.0, radial width of the defect (across the ring).
                    0.3 = razor-thin stripe  |  1.0 = normal  |  3.0 = wide blob

    Returns
    -------
    (result_bgr, defect_mask)  — both same shape as ok_bgr
    """
    rng = np.random.default_rng(seed)
    img = ok_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # 1. Detect inner oval rim (the ~101×101 inner dark ring)
    rim = _detect_inner_rim(gray)
    if rim is None:
        # Fallback: assume rim at image center with guessed size (~10% of min dim)
        rim = (w / 2, h / 2, min(h, w) * 0.12, min(h, w) * 0.10, 0.0)

    ecx, ecy, semi_a, semi_b, eang = rim
    ang_rad = np.deg2rad(eang)

    # 2. Pick random angle on ellipse perimeter
    theta = rng.uniform(0, 2 * np.pi)

    # Point on ellipse (axis-aligned), then rotate by eang
    ex_local = semi_a * np.cos(theta)
    ey_local = semi_b * np.sin(theta)
    px = int(ecx + ex_local * np.cos(ang_rad) - ey_local * np.sin(ang_rad))
    py = int(ecy + ex_local * np.sin(ang_rad) + ey_local * np.cos(ang_rad))
    px = int(np.clip(px, 5, w - 6))
    py = int(np.clip(py, 5, h - 6))

    # Local coordinate system at defect point
    rad_x = px - ecx;  rad_y = py - ecy
    rad_len = np.hypot(rad_x, rad_y) + 1e-6
    rad_nx, rad_ny = rad_x / rad_len, rad_y / rad_len   # unit radial (outward)
    tan_nx, tan_ny = -rad_ny, rad_nx                      # unit tangential

    ys_g, xs_g = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xs_g - px;  dy = ys_g - py
    d_tan = dx * tan_nx + dy * tan_ny   # signed distance along ring
    d_rad = dx * rad_nx + dy * rad_ny   # signed distance across ring (outward +)
    dist2 = dx**2 + dy**2

    # 3. Visible warp — physical rim deformation (5–12 px)
    #    NG shows clear bump/ridge: needs enough warp to shift ring boundary visibly
    sl = float(streak_length)
    th = float(thickness)
    sig_tan = rng.uniform(5.0, 10.0) * sl   # tangential extent (along rim)
    sig_rad = rng.uniform(3.0,  5.0) * th   # radial width (across rim)

    crush_r   = max(int(sig_tan * 1.6), 14)
    crush_mag = rng.uniform(5.0, 12.0) * float(warp_strength) * intensity
    # Push outward along radial — plastic rim crushed inward, material bulges outward
    warped = _local_warp(img, px, py,
                         -rad_nx * crush_mag,
                         -rad_ny * crush_mag,
                         crush_r)

    # 4. Organic ridge appearance: bright core + dark edge bands + smooth noise
    #    NG shows: uneven bright highlight with texture, dark shadow on both sides
    #    Model as: bright Gaussian − two dark flanks (inner + outer radial edges)

    core_mask = np.exp(-(d_tan**2 / (2 * sig_tan**2) +
                          d_rad**2 / (2 * sig_rad**2)))

    # Dark band on inner radial edge (shadow cast by raised rim toward center)
    d_rad_inner = d_rad + sig_rad * rng.uniform(0.8, 1.4)
    inner_shadow = np.exp(-(d_tan**2 / (2 * sig_tan**2) +
                             d_rad_inner**2 / (2 * (sig_rad * 0.6)**2)))

    # Dark band on outer edge (shadow on far side of bump)
    shadow_sign   = rng.choice([-1.0, 1.0])
    shadow_offset = sig_tan * rng.uniform(1.0, 1.8)
    outer_shadow  = np.exp(-((d_tan - shadow_sign * shadow_offset)**2 / (2 * sig_tan**2) +
                               d_rad**2 / (2 * sig_rad**2)))

    bright_amp = rng.uniform(70, 130) * intensity
    dark_amp   = bright_amp * rng.uniform(0.40, 0.65)

    # Smooth noise texture — gives organic, non-perfect surface look
    ksize = max(int(sig_tan * 1.2) | 1, 5)   # must be odd
    raw_noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    noise_sm  = cv2.GaussianBlur(raw_noise, (ksize, ksize), sig_tan * 0.4)
    texture   = core_mask * noise_sm * (bright_amp * 0.30)   # ±30% variation inside core

    delta  = (core_mask * bright_amp
              - inner_shadow * dark_amp
              - outer_shadow * dark_amp * 0.6
              + texture)[..., np.newaxis]
    result = np.clip(warped.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    # 5. Blend — covers full deformation zone
    sigma_blend  = float(crush_r) * rng.uniform(0.85, 1.15)
    blend_mask3d = np.exp(-dist2 / (2 * sigma_blend**2))[..., np.newaxis]
    final = (result.astype(np.float32) * blend_mask3d +
             img.astype(np.float32)    * (1 - blend_mask3d)).astype(np.uint8)

    # 6. Defect mask (tight around bright core)
    dmask = (core_mask * 255).clip(0, 255).astype(np.uint8)
    _, dmask = cv2.threshold(dmask, 20, 255, cv2.THRESH_BINARY)

    return final, dmask
