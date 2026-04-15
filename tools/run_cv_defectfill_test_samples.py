"""
tools/run_cv_defectfill_test_samples.py
======================================

Batch runner for CV DefectFill (engines.fast_physics) on V:\\dataHondatPlus\\test_samples.

Two modes:
  - Interactive (default): draw placement mask on OK, draw defect mask on NG (crop ref patch),
    then generate outputs.
  - File-based (--use-mask-files): reuse masks under test_samples/masks/*.png (legacy).

Outputs:
  - result image + 4-panel comparison (base | ref-crop | mask | result)
    into test_samples/output/
"""

import base64
import os
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engines.fast_physics import generate  # noqa: E402
from engines.utils import decode_b64, encode_b64  # noqa: E402


BASE_DIR = r"V:\dataHondatPlus\test_samples"
OK_DIR = os.path.join(BASE_DIR, "ok")
NG_DIR = os.path.join(BASE_DIR, "ng_mka")
MASK_DIR = os.path.join(BASE_DIR, "masks")
OUT_DIR = os.path.join(BASE_DIR, "output")

WIN_MAX_W = 1400
WIN_MAX_H = 860
BRUSH_DEFAULT = 18
BRUSH_MIN = 2
BRUSH_MAX = 120
OVERLAY_ALPHA = 0.45


def imread_w(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype="uint8")
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imread_gray_w(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype="uint8")
    g = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    return g


def _crop_patch_and_mask(ref_rgb: np.ndarray, ref_mask: np.ndarray, pad: int = 0):
    ys, xs = np.where(ref_mask > 127)
    if len(ys) == 0:
        raise ValueError("empty ref_mask")
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    H, W = ref_rgb.shape[:2]
    y1 = max(0, y1 - pad)
    y2 = min(H, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(W, x2 + pad)
    patch = ref_rgb[y1:y2, x1:x2].copy()
    pmask = ref_mask[y1:y2, x1:x2].copy()
    return patch, pmask


def _make_mask_overlay(base_rgb: np.ndarray, mask: np.ndarray, alpha=0.4) -> np.ndarray:
    out = base_rgb.copy().astype(np.float32)
    m = mask > 127
    if m.any():
        out[m] = out[m] * (1 - alpha) + np.array([0, 180, 0], dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_panel(base_rgb: np.ndarray, ref_patch_rgb: np.ndarray, placement_mask: np.ndarray,
                result_rgb: np.ndarray, out_path: str):
    H, W = base_rgb.shape[:2]
    pw = min(W, 560)
    ph = int(H * pw / W)

    def resz(rgb):
        return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (pw, ph), interpolation=cv2.INTER_AREA)

    base_bgr = resz(base_rgb)
    mask_bgr = resz(_make_mask_overlay(base_rgb, placement_mask))
    res_bgr = resz(result_rgb)

    # ref patch panel (centered on gray background)
    ref_bgr_small = cv2.cvtColor(ref_patch_rgb, cv2.COLOR_RGB2BGR)
    rh, rw = ref_patch_rgb.shape[:2]
    scale = min(pw / max(rw, 1), ph / max(rh, 1), 1.0)
    nw, nh = max(1, int(rw * scale)), max(1, int(rh * scale))
    ref_bgr_small = cv2.resize(ref_bgr_small, (nw, nh), interpolation=cv2.INTER_AREA)
    ref_panel = np.full((ph, pw, 3), 42, dtype=np.uint8)
    y0 = (ph - nh) // 2
    x0 = (pw - nw) // 2
    ref_panel[y0:y0 + nh, x0:x0 + nw] = ref_bgr_small
    cv2.putText(ref_panel, f"ref {rw}x{rh}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    sep = np.full((ph, 4, 3), 60, dtype=np.uint8)
    row = np.hstack([base_bgr, sep, ref_panel, sep, mask_bgr, sep, res_bgr])
    cv2.imencode(".jpg", row)[1].tofile(out_path)


def encode_rgb_b64(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode("utf-8")

class MaskDrawer:
    def __init__(self, image_rgb: np.ndarray, title: str, color_bgr=(0, 180, 0)):
        self.img = image_rgb
        self.title = title
        self.color_bgr = tuple(int(x) for x in color_bgr)
        self.brush = BRUSH_DEFAULT
        H, W = image_rgb.shape[:2]
        self.mask = np.zeros((H, W), dtype=np.uint8)
        self.scale = min(WIN_MAX_W / W, WIN_MAX_H / H, 1.0)
        self.dw = int(W * self.scale)
        self.dh = int(H * self.scale)
        self.disp = cv2.resize(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            (self.dw, self.dh),
            interpolation=cv2.INTER_AREA,
        )
        self.drawing = False
        self.erasing = False

    def _orig(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def _render(self):
        out = self.disp.copy()
        m = cv2.resize(self.mask, (self.dw, self.dh), interpolation=cv2.INTER_NEAREST) > 127
        overlay = out.copy()
        overlay[m] = (overlay[m] * (1.0 - OVERLAY_ALPHA) + np.array(self.color_bgr) * OVERLAY_ALPHA).astype(np.uint8)
        out = overlay
        info = f"brush={self.brush}px  LMB=draw  RMB=erase  [ / ]  Z=clear  S/Enter=OK  Q=quit"
        cv2.putText(out, info, (8, self.dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(out, info, (8, self.dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return out

    def _mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = False
            self.erasing = False
        elif event == cv2.EVENT_MOUSEMOVE and (self.drawing or self.erasing):
            ox, oy = self._orig(x, y)
            v = 255 if self.drawing else 0
            cv2.circle(self.mask, (ox, oy), int(self.brush), v, -1)

    def run(self) -> np.ndarray:
        win = self.title
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.dw, self.dh)
        cv2.setMouseCallback(win, self._mouse)
        while True:
            cv2.imshow(win, self._render())
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("s"), 13):
                break
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                raise SystemExit(0)
            if key == ord("z"):
                self.mask.fill(0)
            if key == ord("["):
                self.brush = max(BRUSH_MIN, self.brush - 2)
            if key == ord("]"):
                self.brush = min(BRUSH_MAX, self.brush + 2)
        cv2.destroyWindow(win)
        return self.mask


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--use-mask-files", action="store_true",
                    help="Dùng mask trong test_samples/masks (legacy). Mặc định = vẽ mask & crop ref trong lúc chạy.")
    ap.add_argument("--intensity", type=float, default=0.7)
    ap.add_argument("--naturalness", type=float, default=0.6)
    ap.add_argument("--ref-pad", type=int, default=0, help="Pad bbox khi crop ref patch từ NG (interactive)")
    ap.add_argument("--out", default=None,
                    help="Output folder. Default: test_samples/output/cv_<timestamp>/")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    ok_files = [f for f in os.listdir(OK_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if not ok_files:
        raise SystemExit(f"[!] No OK images in {OK_DIR}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out or os.path.join(OUT_DIR, f"cv_{ts}")
    os.makedirs(out_root, exist_ok=True)
    # Cases: (label, defect_type, ref_filename, material)
    cases = [
        ("foreign", "foreign", "ng_foreign.bmp", "metal"),
        ("thread",  "thread",  "ng_thread.bmp",  "metal"),
        ("scratch", "scratch", "ng_scratch.bmp", "metal"),
        ("crater",  "foreign", "ng_crater.bmp",  "metal"),
    ]

    if args.use_mask_files:
        # Legacy: reuse mask files in MASK_DIR and ref crop derived from those masks
        CASES = {
            "ng_foreign_mask.png": ("foreign", "ng_foreign.bmp", "metal"),
            "ng_thread_mask.png": ("thread", "ng_thread.bmp", "metal"),
            "ng_scratch_mask.png": ("scratch", "ng_scratch.bmp", "metal"),
            "ng_crater_mask.png": ("foreign", "ng_crater.bmp", "metal"),
        }
        mask_files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        selected = [f for f in mask_files if f in CASES]
        if not selected:
            raise SystemExit(f"[!] No recognized masks in {MASK_DIR}. Known: {list(CASES.keys())}")
        print(f"[RUN] mode=mask-files  OK images={len(ok_files)}  masks={len(selected)}  out={out_root}")

        for mf in selected:
            defect_type, ref_fn, material = CASES[mf]
            mask_path = os.path.join(MASK_DIR, mf)
            ref_path = os.path.join(NG_DIR, ref_fn)
            if not os.path.isfile(ref_path):
                print(f"[SKIP] missing ref: {ref_path}")
                continue

            placement_mask_raw = imread_gray_w(mask_path)
            ref_rgb = imread_w(ref_path)

            if placement_mask_raw.shape[:2] != ref_rgb.shape[:2]:
                placement_mask_ng = cv2.resize(
                    placement_mask_raw,
                    (ref_rgb.shape[1], ref_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                placement_mask_ng = placement_mask_raw

            ref_patch, ref_patch_mask = _crop_patch_and_mask(ref_rgb, placement_mask_ng, pad=0)
            ref_b64 = encode_rgb_b64(ref_patch)
            ref_pm_b64 = encode_b64(ref_patch_mask.astype(np.uint8))

            for okf in ok_files:
                ok_path = os.path.join(OK_DIR, okf)
                base_rgb = imread_w(ok_path)
                if placement_mask_raw.shape[:2] != base_rgb.shape[:2]:
                    placement_mask_ok = cv2.resize(
                        placement_mask_raw,
                        (base_rgb.shape[1], base_rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    placement_mask_ok = placement_mask_raw

                params = {
                    "intensity": float(args.intensity),
                    "naturalness": float(args.naturalness),
                    "position_jitter": 0.0,
                    "diversity": 0.0,
                    "lighting_match": True,
                    "sdxl_refine": False,
                    "ref_image_b64": ref_b64,
                    "ref_patch_mask_b64": ref_pm_b64,
                    "ref_is_patch": True,
                }

                result = generate(
                    base_image=base_rgb,
                    mask=placement_mask_ok,
                    defect_type=defect_type,
                    material=material,
                    params=params,
                )

                out_rgb = decode_b64(result["result_image"])
                ok_stem = os.path.splitext(okf)[0]
                case_stem = os.path.splitext(mf)[0].replace("ng_", "").replace("_mask", "")
                stem = f"cv_{case_stem}_{defect_type}_{ok_stem}_{ts}"
                out_img = os.path.join(out_root, f"{stem}.jpg")
                out_pan = os.path.join(out_root, f"{stem}_panel.jpg")

                bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                cv2.imencode(".jpg", bgr)[1].tofile(out_img)
                _save_panel(base_rgb, ref_patch, placement_mask_ok, out_rgb, out_pan)
                print(f"  saved → {os.path.basename(out_img)}")
    else:
        # Interactive: draw both masks during run
        print(f"[RUN] mode=interactive  OK images={len(ok_files)}  cases={len(cases)}  out={out_root}")
        for label, defect_type, ref_fn, material in cases:
            ref_path = os.path.join(NG_DIR, ref_fn)
            if not os.path.isfile(ref_path):
                print(f"[SKIP] missing ref: {ref_path}")
                continue
            ref_rgb = imread_w(ref_path)
            print(f"\n[NG] {label}: draw defect region on {ref_fn}")
            ref_mask = MaskDrawer(ref_rgb, f"NG crop [{label}] — ve vung loi — S=OK Q=quit", color_bgr=(0, 0, 255)).run()
            if int(np.sum(ref_mask > 127)) < 8:
                print("  [SKIP] ref mask too small")
                continue
            ref_patch, ref_patch_mask = _crop_patch_and_mask(ref_rgb, ref_mask, pad=int(args.ref_pad))
            ref_b64 = encode_rgb_b64(ref_patch)
            ref_pm_b64 = encode_b64(ref_patch_mask.astype(np.uint8))

            for okf in ok_files:
                ok_path = os.path.join(OK_DIR, okf)
                base_rgb = imread_w(ok_path)
                print(f"[OK] {okf}: draw placement mask for {label}")
                placement_mask_ok = MaskDrawer(base_rgb, f"OK place [{label}] — ve vi tri dat loi — S=OK Q=quit").run()
                if int(np.sum(placement_mask_ok > 127)) < 16:
                    print("  [SKIP] placement mask too small")
                    continue

                params = {
                    "intensity": float(args.intensity),
                    "naturalness": float(args.naturalness),
                    "position_jitter": 0.0,
                    "diversity": 0.0,
                    "lighting_match": True,
                    "sdxl_refine": False,
                    "ref_image_b64": ref_b64,
                    "ref_patch_mask_b64": ref_pm_b64,
                    "ref_is_patch": True,
                }

                result = generate(
                    base_image=base_rgb,
                    mask=placement_mask_ok,
                    defect_type=defect_type,
                    material=material,
                    params=params,
                )

                out_rgb = decode_b64(result["result_image"])
                ok_stem = os.path.splitext(okf)[0]
                stem = f"cv_{label}_{defect_type}_{ok_stem}_{ts}"
                out_img = os.path.join(out_root, f"{stem}.jpg")
                out_pan = os.path.join(out_root, f"{stem}_panel.jpg")
                bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                cv2.imencode(".jpg", bgr)[1].tofile(out_img)
                _save_panel(base_rgb, ref_patch, placement_mask_ok, out_rgb, out_pan)
                print(f"  saved → {os.path.basename(out_img)}")

    print(f"[DONE] outputs in {out_root}")


if __name__ == "__main__":
    main()

