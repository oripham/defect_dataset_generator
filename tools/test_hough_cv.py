"""
tools/test_hough_cv.py
======================
Test CV engine voi Hough structure-aware placement.

Mode 1 -- Manual (default):
    Ve mask tay tren anh OK, gen NUM_IMAGES anh.
    python tools/test_hough_cv.py --case scratch

Mode 2 -- Auto placement (--auto N):
    Hough tu sinh mask ngau nhien tren rim, gen N anh, khong can ve.
    python tools/test_hough_cv.py --case scratch --auto 6
    python tools/test_hough_cv.py --case dent    --auto 9 --zone outer
    python tools/test_hough_cv.py --case scratch --auto 6 --zone inner

Zones (--zone):
    outer  -- outer rim (r = r_outer)          default cho chip/scratch
    inner  -- inner gray area (r = r_outer*0.6)
    random -- random radius giua inner va outer

Controls annotator (manual mode):
    Trai=ve | Phai=xoa | Scroll=brush | S/Enter=Luu | Q/Esc=Bo qua
"""

import sys, os, io, base64, argparse, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
from collections import defaultdict

from mask_annotator import Annotator
from engines.fast_physics import generate
from engines.structure_adapt import structure_adapt, _BG_BRIGHTNESS_THRESHOLD
from engines.utils import decode_b64

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCRIPTS))
from geometry.ring_detector import detect_ring

SAMPLE_DIR = "V:/demo_webapp/sample_1"
OK_IMAGE   = f"{SAMPLE_DIR}/good_images/ok_001.jpg"
TAG        = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = f"{SAMPLE_DIR}/output/hough_{TAG}"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_IMAGES = 3

DEFAULT_CASES = [
    ("scratch", f"{SAMPLE_DIR}/ref_scratch/ng_002.png"),
    ("chip",    f"{SAMPLE_DIR}/ref_chip/\uff11\uff3fNG_crop.png"),
    ("dent",    f"{SAMPLE_DIR}/ref_dent/\uff11\uff3fNG_crop.png"),
]

# Kich thuoc mask auto (w, h) theo defect type
_AUTO_MASK_SIZE = {
    "scratch": (65,  50),   # vung nho bi xuoc (area scratch), khong phai 1 duong
    "crack":   (100, 12),   # crack = duong nut, giu mong
    "chip":    (55,  45),
    "dent":    (70,  55),
    "bulge":   (70,  55),
    "foreign": (40,  40),
    "rust":    (60,  50),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_ref(ref_rgb):
    buf = io.BytesIO()
    Image.fromarray(ref_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def rotate_ref(ref_rgb: np.ndarray, angle_deg: float) -> np.ndarray:
    """Xoay ref image quanh tam, giu kich thuoc, fill black."""
    if angle_deg == 0.0:
        return ref_rgb
    h, w = ref_rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(ref_rgb, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def draw_label(arr, text, color=(255, 255, 0)):
    img = Image.fromarray(arr)
    d   = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    d.text((4, 4), text, fill=color)
    return np.array(img)


def mask_info(mask):
    ys, xs = np.where(mask > 127)
    if not len(ys):
        return "empty"
    cy, cx = int(ys.mean()), int(xs.mean())
    return f"{len(ys)}px  ctr({cx},{cy})"


def overlay_mask(base, mask, color_bgr=(0, 200, 0)):
    out = base.copy()
    out[mask > 127] = (
        out[mask > 127] * 0.35 + np.array(color_bgr) * 0.65
    ).astype(np.uint8)
    ys, xs = np.where(mask > 127)
    if len(ys):
        cv2.drawMarker(out, (int(xs.mean()), int(ys.mean())),
                       (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
    return out


# ── Auto mask generation ──────────────────────────────────────────────────────

# Zone mac dinh theo defect type — override bang --zone neu can
_DEFAULT_ZONE = {
    "scratch": "inner",   # be mat phang xam ben trong rim
    "dent":    "inner",
    "bulge":   "inner",
    "rust":    "inner",
    "foreign": "inner",
    "chip":    "outer",   # canh ngoai rim
    "crack":   "outer",
}

def elastic_deform_mask(mask: np.ndarray, strength: float, seed: int) -> np.ndarray:
    """
    Bien dang nhe mask bang random displacement field.
    strength=0 → khong bien dang
    strength=1 → bien dang manh (natural irregular scratch shape)
    """
    if strength <= 0:
        return mask
    rng = np.random.default_rng(seed)
    h, w = mask.shape

    # Random displacement field, scale theo mask size
    ys, xs = np.where(mask > 127)
    if not len(ys):
        return mask
    mask_size = max(ys.max()-ys.min(), xs.max()-xs.min(), 1)
    amplitude = strength * mask_size * 0.25   # max shift = 25% of mask size

    # Coarse random field → blur → smooth displacement
    coarse_h, coarse_w = max(4, h//8), max(4, w//8)
    dx = rng.uniform(-amplitude, amplitude, (coarse_h, coarse_w)).astype(np.float32)
    dy = rng.uniform(-amplitude, amplitude, (coarse_h, coarse_w)).astype(np.float32)
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC)

    # Warp grid
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x  = np.clip(xx + dx, 0, w - 1)
    map_y  = np.clip(yy + dy, 0, h - 1)
    warped = cv2.remap(mask.astype(np.float32), map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (warped > 64).astype(np.uint8) * 255


def auto_mask(base_rgb, defect_type, zone, seed, ref_rgb=None, shape_jitter=0.0):
    """
    Sinh mask tu dong tren rim dua vao Hough detection.
    zone: 'outer' | 'inner' | 'random' | 'auto' (dung default theo defect type)
    """
    rng  = random.Random(seed)
    gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    H, W = base_rgb.shape[:2]

    cx, cy, r_outer = detect_ring(gray, tag=f"_auto_{defect_type}")

    # Resolve zone: 'auto' -> dung default theo defect type
    effective_zone = zone
    if zone == "auto" or zone is None:
        effective_zone = _DEFAULT_ZONE.get(defect_type, "inner")

    # Chon ban kinh theo zone
    # inner: be mat phang xam ~65% r_outer
    # outer: canh rim sang ~100% r_outer
    r_inner = int(r_outer * 0.65)
    if effective_zone == "outer":
        radius = r_outer
    elif effective_zone == "inner":
        radius = rng.randint(int(r_outer * 0.55), int(r_outer * 0.75))  # mid inner zone — tránh vùng tối sát tâm và tránh sát rim
    else:  # random
        radius = rng.randint(r_inner, r_outer)

    # Random goc, tranh 4 goc (~background)
    angle = rng.uniform(0, 2 * math.pi)

    # Vi tri tam mask
    mx = int(cx + radius * math.cos(angle))
    my = int(cy + radius * math.sin(angle))
    mx = int(np.clip(mx, 0, W - 1))
    my = int(np.clip(my, 0, H - 1))

    # Kich thuoc mask: neu co ref_rgb thi xap xi theo ref size, fallback ve _AUTO_MASK_SIZE
    # Cap theo base image de tranh mask qua lon khi ref la full NG image
    if ref_rgb is not None:
        rh, rw = ref_rgb.shape[:2]
        raw_mw = max(30, int(rw * 0.5))
        raw_mh = max(20, int(rh * 0.5))
        # Cap: toi da 10% chieu ngang va 12% chieu doc cua base image
        cap_w  = int(W * 0.10)
        cap_h  = int(H * 0.12)
        mw     = min(raw_mw, cap_w)
        mh     = min(raw_mh, cap_h)
    else:
        mw, mh = _AUTO_MASK_SIZE.get(defect_type, (60, 40))

    # Khong xoay o day — structure_adapt._orient_tangential() se xoay dung
    # (neu xoay 2 lan se bi double-rotation, mask quay ra ngoai san pham)
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (mx, my), (mw // 2, mh // 2), 0, 0, 360, 255, -1)

    # Elastic deformation: bien dang nhe hinh dang → moi sample co shape khac nhau
    if shape_jitter > 0:
        mask = elastic_deform_mask(mask, shape_jitter, seed + 2000)

    angle_deg = math.degrees(angle) % 360
    clock_h   = int(round((angle_deg / 360.0) * 12 + 3) % 12) or 12
    print(f"  [AUTO] angle={angle_deg:.0f}deg ({clock_h} o'clock)"
          f"  radius={radius}  ctr=({mx},{my})  size={mw}x{mh}")
    return mask


# ── Panel helpers ─────────────────────────────────────────────────────────────

def make_panel(base, ref, mask_orig, mask_snap, result_pre, result_post,
               case_name, clock_str, mode, sdxl_ran=False):
    h, w   = base.shape[:2]
    pw, ph = w // 3, h // 2

    def thumb(arr, lbl, col=(255, 255, 0)):
        a = arr if arr.ndim == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return draw_label(cv2.resize(a, (pw, ph)), lbl, col)

    rh, rw  = ref.shape[:2]
    ref_pad = np.zeros((ph, pw, 3), np.uint8)
    ref_pad[:min(rh,ph), :min(rw,pw)] = ref[:min(rh,ph), :min(rw,pw)]

    ov_orig = overlay_mask(base, mask_orig, (0, 60, 220))
    ov_snap = overlay_mask(base, mask_snap, (0, 200, 0))

    # Zoom crop tai vung defect
    ys, xs = np.where(mask_snap > 127)
    def zoom_crop(img):
        if len(ys):
            pad = 60
            y1 = max(0, ys.min()-pad); y2 = min(h, ys.max()+pad)
            x1 = max(0, xs.min()-pad); x2 = min(w, xs.max()+pad)
            return cv2.resize(img[y1:y2, x1:x2], (pw, ph), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(img, (pw, ph))

    sdxl_lbl = "(+SDXL)" if sdxl_ran else "(CV only)"
    orig_lbl = "Mask AUTO" if mode == "auto" else "Mask user ve"

    row1 = np.hstack([thumb(base,      "Base (OK)"),
                      draw_label(ref_pad, f"Ref {rw}x{rh}"),
                      thumb(ov_orig,   orig_lbl, (100, 160, 255))])
    row2 = np.hstack([thumb(ov_snap,   f"Mask final ({clock_str})",   (0, 255, 128)),
                      thumb(result_pre, "CV result (pre-refiner)",     (255, 200, 0)),
                      draw_label(zoom_crop(result_pre), "Zoom CV",     (255, 200, 0))])
    row3 = np.hstack([thumb(result_post, f"Final {sdxl_lbl}",         (0, 255, 128)),
                      draw_label(zoom_crop(result_post), "Zoom Final", (0, 255, 200)),
                      np.zeros((ph, pw, 3), np.uint8)])   # placeholder

    panel = np.vstack([row1, row2, row3])
    bar   = np.zeros((26, panel.shape[1], 3), np.uint8)
    ImageDraw.Draw(Image.fromarray(bar)).text(
        (4, 5), f"[{mode.upper()}] {case_name}  pos={clock_str}  {sdxl_lbl}",
        fill=(200, 200, 200)
    )
    return np.vstack([bar, panel])


def make_strip(results_rgb, masks_snap, base_rgb, case_name, n):
    h, w    = base_rgb.shape[:2]
    thumb_w = max(w // n, 1)
    thumb_h = h // 2

    cols = []
    for i, (res, msnap) in enumerate(zip(results_rgb, masks_snap)):
        ys, xs = np.where(msnap > 127)
        if len(ys):
            pad = 50
            zy1=max(0,ys.min()-pad); zy2=min(h,ys.max()+pad)
            zx1=max(0,xs.min()-pad); zx2=min(w,xs.max()+pad)
            zoom_crop = res[zy1:zy2, zx1:zx2]
        else:
            zoom_crop = res
        top = draw_label(cv2.resize(res, (thumb_w, thumb_h)),
                         f"#{i+1} seed={42+i}", (0,255,128))
        bot = draw_label(cv2.resize(zoom_crop, (thumb_w, thumb_h),
                                    interpolation=cv2.INTER_LINEAR),
                         "Zoom", (0,200,255))
        cols.append(np.vstack([top, bot]))

    strip = np.hstack(cols)
    bar   = np.zeros((24, strip.shape[1], 3), np.uint8)
    ImageDraw.Draw(Image.fromarray(bar)).text(
        (4, 4), f"{case_name}  {n} samples", fill=(220,220,220)
    )
    return np.vstack([bar, strip])


# ── Ref prep helper (paint + rotate) ─────────────────────────────────────────

def prep_ref_interactive(ref_rgb: np.ndarray) -> np.ndarray:
    """
    Chuan bi ref image truoc khi gen:
      - To vung can giu (paint white mask)  → loai bo vien sang, background
      - Xoay ref de align huong scratch voi rim

    Controls:
        Chuot trai (giu)  — to vung can giu (brush xanh)
        Chuot phai (giu)  — xoa vung da to
        Scroll / [ ]      — doi brush size
        R / L             — xoay ref +5 / -5 do
        Z                 — undo 1 buoc (mask)
        S / Enter         — xac nhan, apply mask + rotation
        Q / Esc           — bo qua, dung ref nguyen goc
    """
    DISP_SIZE  = 500   # max display size
    BRUSH_DEF  = 12
    BRUSH_MIN  = 2
    BRUSH_MAX  = 80
    OVERLAY_A  = 0.45
    ROTATE_STEP = 5.0

    orig_h, orig_w = ref_rgb.shape[:2]
    scale = min(DISP_SIZE / orig_w, DISP_SIZE / orig_h, 1.0)
    dw    = max(int(orig_w * scale), 1)
    dh    = max(int(orig_h * scale), 1)

    state = {
        "angle":    0.0,
        "mask":     np.zeros((orig_h, orig_w), np.uint8),
        "mask_prev":np.zeros((orig_h, orig_w), np.uint8),
        "brush":    BRUSH_DEF,
        "drawing":  False,
        "erasing":  False,
    }

    def rotated_ref():
        if state["angle"] == 0.0:
            return ref_rgb
        cx, cy = orig_w / 2, orig_h / 2
        M = cv2.getRotationMatrix2D((cx, cy), state["angle"], 1.0)
        return cv2.warpAffine(ref_rgb, M, (orig_w, orig_h),
                              borderMode=cv2.BORDER_REFLECT)

    def render():
        rot  = cv2.cvtColor(rotated_ref(), cv2.COLOR_RGB2BGR)
        disp = cv2.resize(rot, (dw, dh), interpolation=cv2.INTER_LINEAR)
        # overlay mask (xanh la = vung duoc giu)
        dmask = cv2.resize(state["mask"], (dw, dh), interpolation=cv2.INTER_NEAREST)
        region = dmask > 127
        disp[region] = (disp[region] * (1 - OVERLAY_A) +
                        np.array([0, 200, 80]) * OVERLAY_A).astype(np.uint8)
        n_px  = int(np.sum(state["mask"] > 127))
        pct   = 100.0 * n_px / (orig_w * orig_h)
        hud   = (f"brush:{state['brush']}  rot:{state['angle']:.0f}deg  "
                 f"painted:{pct:.1f}%  |  R/L=xoay  S=OK  Q=skip  Z=undo  scroll=brush")
        cv2.putText(disp, hud, (4, dh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(disp, hud, (4, dh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0),     1, cv2.LINE_AA)
        return disp

    def paint(x, y, erase):
        ox = int(np.clip(x / scale, 0, orig_w - 1))
        oy = int(np.clip(y / scale, 0, orig_h - 1))
        b  = max(1, int(state["brush"] / scale))
        cv2.circle(state["mask"], (ox, oy), b, 0 if erase else 255, -1)

    def mouse_cb(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["mask_prev"] = state["mask"].copy()
            state["drawing"]   = True;  paint(x, y, False)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["mask_prev"] = state["mask"].copy()
            state["erasing"]   = True;  paint(x, y, True)
        elif event == cv2.EVENT_MOUSEMOVE:
            if state["drawing"]: paint(x, y, False)
            if state["erasing"]: paint(x, y, True)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            state["drawing"] = state["erasing"] = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 3 if flags > 0 else -3
            state["brush"] = int(np.clip(state["brush"] + delta, BRUSH_MIN, BRUSH_MAX))

    win = "Prep Ref  |  to vung can giu  |  R/L=xoay  S=OK  Q=skip"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    cv2.setMouseCallback(win, mouse_cb)

    result = ref_rgb   # default: giu nguyen
    confirmed = False

    while True:
        cv2.imshow(win, render())
        key = cv2.waitKey(15) & 0xFF

        if key in (ord('s'), 13):          # S / Enter → confirm
            confirmed = True;  break
        elif key in (ord('q'), 27):        # Q / Esc → skip
            break
        elif key == ord('z'):              # Undo
            state["mask"] = state["mask_prev"].copy()
        elif key == ord('r'):              # Rotate CW
            state["angle"] = (state["angle"] - ROTATE_STEP) % 360
        elif key == ord('l'):              # Rotate CCW
            state["angle"] = (state["angle"] + ROTATE_STEP) % 360
        elif key == ord('['):
            state["brush"] = max(state["brush"] - 3, BRUSH_MIN)
        elif key == ord(']'):
            state["brush"] = min(state["brush"] + 3, BRUSH_MAX)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(win)

    if not confirmed:
        print("  [PREP REF] Bo qua — dung ref nguyen")
        return ref_rgb

    # Apply rotation
    rot_rgb = rotated_ref()

    # Neu co vung duoc to: crop tight theo bbox cua mask (KHONG dung soft mask)
    # Soft/hard mask tren ref tao gradient nhan tao → HF extraction ra vien den/trang
    # Crop bbox thay the: lay dung phan ref co defect, loai context/rim, khong co artifact
    if np.any(state["mask"] > 127):
        ys, xs = np.where(state["mask"] > 127)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        # Padding nho de tranh cut sat canh
        pad = 3
        y0 = max(0, y0 - pad);  y1 = min(orig_h - 1, y1 + pad)
        x0 = max(0, x0 - pad);  x1 = min(orig_w - 1, x1 + pad)
        result = rot_rgb[y0:y1+1, x0:x1+1]
        nh, nw = result.shape[:2]
        print(f"  [PREP REF] angle={state['angle']:.0f}deg  bbox crop ({x0},{y0})-({x1},{y1}) → {nw}x{nh}")
    else:
        result = rot_rgb
        print(f"  [PREP REF] angle={state['angle']:.0f}deg  (no mask, giu ref nguyen)")

    return result


# ── Run ───────────────────────────────────────────────────────────────────────

def _open_image_safe(path: str) -> Image.Image:
    """Open image, fallback to numpy fromfile for non-ASCII paths."""
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, OSError):
        raw = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot open: {path}")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def run_ref(defect_type, ref_path, base_rgb, mode, n_images, zone,
            crop_ref=False, ref_rotate=0, shape_jitter=0.4):
    ref_rgb   = np.array(_open_image_safe(ref_path))
    rh, rw    = ref_rgb.shape[:2]

    if crop_ref:
        print(f"  [PREP REF] Hien thi ref — to vung can giu + xoay neu can, S=OK")
        ref_rgb = prep_ref_interactive(ref_rgb)
        rh, rw  = ref_rgb.shape[:2]

    raw_name  = os.path.basename(ref_path).replace('.png','')
    safe_name = raw_name.encode('ascii', errors='replace').decode('ascii').replace('?','x')
    case_name = f"{defect_type}_{safe_name}"
    ref_b64   = encode_ref(ref_rgb)

    print(f"\n{'='*60}")
    print(f"[{mode.upper()}] {case_name}  ref={rw}x{rh}  n={n_images}")

    results_rgb     = []
    results_pre_rgb = []   # CV only, truoc SDXL
    masks_snap      = []

    if mode == "manual":
        # Ve mask 1 lan, reuse cho tat ca seeds
        mask_path = os.path.join(OUT_DIR, f"mask_{defect_type}.png")
        print(f"  Ve mask tren anh OK (S=Luu, Q=Bo qua)")
        ann   = Annotator(OK_IMAGE, mask_path)
        saved = ann.run()
        if not saved or not os.path.exists(mask_path):
            print("  [SKIP]"); return
        mask_orig = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_orig is None or np.sum(mask_orig > 127) == 0:
            print("  [SKIP] mask trong"); return

        mask_snap, light_dir, clock_str = structure_adapt(base_rgb, mask_orig.copy(), defect_type)
        print(f"  mask ORIG: {mask_info(mask_orig)}")
        print(f"  mask SNAP: {mask_info(mask_snap)}  clock={clock_str}")

        for i in range(n_images):
            seed_i = 42 + i
            if ref_rotate > 0:
                rng_r   = random.Random(seed_i + 1000)
                angle_i = rng_r.uniform(-ref_rotate, ref_rotate)
                cur_b64 = encode_ref(rotate_ref(ref_rgb, angle_i))
                print(f"  gen {i+1}/{n_images} rot={angle_i:.0f}deg...", end=" ", flush=True)
            else:
                cur_b64 = ref_b64
                print(f"  gen {i+1}/{n_images}...", end=" ", flush=True)
            out = generate(base_rgb, mask_orig.copy(), defect_type, "metal", {
                "intensity": 0.9, "naturalness": 0.6,
                "sdxl_refine": True, "ref_image_b64": cur_b64, "seed": seed_i,
            })
            r     = decode_b64(out["result_image"])
            r_pre = decode_b64(out["result_pre_refine"])
            results_rgb.append(r)
            results_pre_rgb.append(r_pre)
            masks_snap.append(mask_snap)
            Image.fromarray(r).save(os.path.join(OUT_DIR, f"{case_name}_s{42+i}.jpg"))
            Image.fromarray(r_pre).save(os.path.join(OUT_DIR, f"{case_name}_s{42+i}_cv.jpg"))
            print(f"OK ({out['metadata'].get('method','?')}  sdxl={out['metadata'].get('sdxl_refined')})")

        sdxl_ran = out["metadata"].get("sdxl_refined", False)
        info = make_panel(base_rgb, ref_rgb, mask_orig, mask_snap,
                          results_pre_rgb[0], results_rgb[0],
                          case_name, clock_str or "?", mode, sdxl_ran)

    else:  # auto
        # Moi seed dung 1 mask ngau nhien khac nhau
        masks_orig = []
        clock_strs = []
        for i in range(n_images):
            seed_i    = 42 + i
            mask_orig = auto_mask(base_rgb, defect_type, zone, seed_i,
                                  ref_rgb=ref_rgb, shape_jitter=shape_jitter)
            masks_orig.append(mask_orig)

            mask_snap, light_dir, clock_str = structure_adapt(
                base_rgb, mask_orig.copy(), defect_type)
            masks_snap.append(mask_snap)
            clock_strs.append(clock_str or "?")

            # Xoay ref ngau nhien theo seed → moi sample co huong scratch khac nhau
            if ref_rotate > 0:
                rng_r   = random.Random(seed_i + 1000)
                angle_i = rng_r.uniform(-ref_rotate, ref_rotate)
                ref_i   = rotate_ref(ref_rgb, angle_i)
                cur_b64 = encode_ref(ref_i)
                print(f"  gen {i+1}/{n_images} ({clock_str}) rot={angle_i:.0f}deg...",
                      end=" ", flush=True)
            else:
                cur_b64 = ref_b64
                print(f"  gen {i+1}/{n_images} ({clock_str})...", end=" ", flush=True)

            out = generate(base_rgb, mask_orig.copy(), defect_type, "metal", {
                "intensity": 0.9, "naturalness": 0.6,
                "sdxl_refine": True, "ref_image_b64": cur_b64, "seed": seed_i,
            })
            r     = decode_b64(out["result_image"])
            r_pre = decode_b64(out["result_pre_refine"])
            results_rgb.append(r)
            results_pre_rgb.append(r_pre)
            Image.fromarray(r).save(os.path.join(OUT_DIR, f"{case_name}_s{seed_i}.jpg"))
            Image.fromarray(r_pre).save(os.path.join(OUT_DIR, f"{case_name}_s{seed_i}_cv.jpg"))
            print(f"OK ({out['metadata'].get('method','?')}  sdxl={out['metadata'].get('sdxl_refined')})")

        # Info panel dung sample dau tien
        sdxl_ran = out["metadata"].get("sdxl_refined", False)
        info = make_panel(base_rgb, ref_rgb, masks_orig[0], masks_snap[0],
                          results_pre_rgb[0], results_rgb[0],
                          case_name, clock_strs[0], mode, sdxl_ran)

    strip = make_strip(results_rgb, masks_snap, base_rgb, case_name, n_images)

    w_i, w_s = info.shape[1], strip.shape[1]
    if w_i != w_s:
        strip = cv2.resize(strip, (w_i, strip.shape[0]))
    full = np.vstack([info, strip])

    panel_path = os.path.join(OUT_DIR, f"{case_name}_panel.jpg")
    Image.fromarray(full).save(panel_path)
    print(f"  Saved: {panel_path}")

    max_w, max_h = 1600, 900
    sh = min(max_h, full.shape[0])
    sw = min(max_w, int(full.shape[1] * sh / full.shape[0]))
    cv2.imshow(f"Result - {case_name}",
               cv2.cvtColor(cv2.resize(full, (sw, sh)), cv2.COLOR_RGB2BGR))
    print("  Nhan phim bat ky de tiep tuc...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",   default=None,     help="scratch/chip/dent/...")
    parser.add_argument("--ref",    default=None,     help="Custom ref image path")
    parser.add_argument("--ok",     default=None,     help="Custom OK base image path")
    parser.add_argument("--output", default=None,     help="Custom output directory")
    parser.add_argument("--auto",   type=int, default=0,
                        help="Auto-placement mode: so anh can gen (0=manual)")
    parser.add_argument("--zone",     default="auto",
                        help="outer|inner|random|auto (default: auto=theo defect type)")
    parser.add_argument("--no-crop-ref", action="store_true",
                        help="Bo qua buoc crop ref (mac dinh luon hien UI crop)")
    parser.add_argument("--ref-rotate",   type=float, default=0,
                        help="Max goc xoay ref ngau nhien moi sample (deg), e.g. 45")
    parser.add_argument("--shape-jitter", type=float, default=0.4,
                        help="Muc do bien dang hinh dang mask 0=khong, 1=manh (default 0.4)")
    args = parser.parse_args()

    ok_path  = args.ok if args.ok else OK_IMAGE
    base_rgb = np.array(Image.open(ok_path).convert("RGB"))
    mode     = "auto" if args.auto > 0 else "manual"
    n_images = args.auto if args.auto > 0 else NUM_IMAGES

    global OUT_DIR
    if args.output:
        OUT_DIR = args.output
        os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Mode  : {mode.upper()}  n={n_images}")
    print(f"Base  : {base_rgb.shape[1]}x{base_rgb.shape[0]}")
    print(f"Output: {OUT_DIR}")

    cases = DEFAULT_CASES
    if args.case:
        cases = [(d, r) for d, r in cases if d == args.case]
    if args.ref:
        if not cases:
            print("[ERROR] --ref requires --case"); return
        cases = [(cases[0][0], args.ref)]

    for defect_type, ref_path in cases:
        run_ref(defect_type, ref_path, base_rgb, mode, n_images, args.zone,
                crop_ref=not args.no_crop_ref, ref_rotate=args.ref_rotate,
                shape_jitter=args.shape_jitter)

    print(f"\nDone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
