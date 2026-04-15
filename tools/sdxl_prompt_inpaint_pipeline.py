"""
tools/sdxl_prompt_inpaint_pipeline.py — SDXL prompt-inpaint quick runner
=======================================================================

Generate natural-looking defects on an OK image with prompt-first SDXL inpainting.

Inputs:
  - OK image (--base)
  - placement mask: either provide a mask image (--mask) or draw interactively (--draw)
  - optional NG reference image (--ref) used only when --ip-scale > 0 (IP-Adapter)

This tool calls engines.fast_physics.generate(..., params={"sdxl_inpaint": True, ...}).
"""

import argparse
import base64
import os
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from engines.fast_physics import generate
from engines.utils import decode_b64


WIN_MAX_W = 1400
WIN_MAX_H = 860
BRUSH_DEFAULT = 18
BRUSH_MIN = 2
BRUSH_MAX = 120
OVERLAY_ALPHA = 0.45
COLOR_PLACE = (0, 180, 0)  # BGR


def _require_image_path(flag: str, p: str) -> str:
    if not p:
        raise SystemExit(f"[!] Thiếu {flag}.")
    if not os.path.isfile(p):
        raise SystemExit(f"[!] Không tìm thấy {flag}: {p}")
    return p


def imread_w(path: str) -> np.ndarray:
    arr = np.fromfile(path, dtype="uint8")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path: str, rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imencode(os.path.splitext(path)[1] or ".jpg", bgr)[1].tofile(path)


def encode_rgb_b64(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode("utf-8")

def _crop_from_mask(rgb: np.ndarray, mask_u8: np.ndarray, pad: int = 10):
    """
    Crop a bbox around the painted defect mask, and also return the cropped mask.
    (Crop is rectangular, but mask keeps the original painted shape.)
    """
    ys, xs = np.where(mask_u8 > 127)
    if len(ys) == 0:
        return rgb, mask_u8
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    H, W = rgb.shape[:2]
    y1 = max(0, y1 - pad)
    y2 = min(H, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(W, x2 + pad)
    return rgb[y1:y2, x1:x2].copy(), mask_u8[y1:y2, x1:x2].copy()


def _apply_mask_to_ref_crop(crop_rgb: np.ndarray, crop_mask_u8: np.ndarray,
                            feather_px: int = 7) -> np.ndarray:
    """
    Keep pixels inside mask, neutralize outside mask so IP-Adapter focuses on defect.
    """
    m = (crop_mask_u8 > 127).astype(np.float32)
    if feather_px > 0 and m.any():
        k = feather_px * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        m = np.clip(m, 0.0, 1.0)

    if not m.any():
        return crop_rgb

    # Neutral background: per-channel median of non-masked area (fallback 127)
    bg = np.full(3, 127.0, dtype=np.float32)
    inv = (crop_mask_u8 <= 127)
    if inv.any():
        for c in range(3):
            bg[c] = float(np.median(crop_rgb[:, :, c][inv]))

    crop_f = crop_rgb.astype(np.float32)
    out = crop_f * m[:, :, None] + bg[None, None, :] * (1.0 - m[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)

def _mask_overlay(rgb: np.ndarray, mask_u8: np.ndarray, color_rgb=(0, 180, 0), alpha=0.45) -> np.ndarray:
    out = rgb.copy()
    m = mask_u8 > 127
    if m.any():
        c = np.array(color_rgb, dtype=np.float32)
        out_f = out.astype(np.float32)
        out_f[m] = out_f[m] * (1.0 - alpha) + c * alpha
        out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out

def _make_4panel(base_rgb: np.ndarray,
                 ref_rgb: np.ndarray | None,
                 mask_u8: np.ndarray,
                 result_rgb: np.ndarray,
                 ref_title: str = "ref(IP)") -> np.ndarray:
    """Create 4-panel image: base | ref | mask overlay | result (BGR for saving)."""
    H, W = base_rgb.shape[:2]
    pw = min(W, 560)
    ph = int(H * pw / W)

    def resz(rgb):
        return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (pw, ph), interpolation=cv2.INTER_AREA)

    base_bgr = resz(base_rgb)
    mask_bgr = resz(_mask_overlay(base_rgb, mask_u8, color_rgb=(0, 180, 0), alpha=0.45))
    res_bgr  = resz(result_rgb)

    if ref_rgb is None:
        ref_bgr = np.full((ph, pw, 3), 42, dtype=np.uint8)
        cv2.putText(ref_bgr, "no ref (prompt-only)", (10, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    else:
        ref_bgr = resz(ref_rgb)

    sep = np.full((ph, 4, 3), 60, dtype=np.uint8)
    row = np.hstack([base_bgr, sep, ref_bgr, sep, mask_bgr, sep, res_bgr])

    labels = [("base(OK)", 0), (ref_title, pw + 4), ("mask", (pw + 4) * 2), ("result", (pw + 4) * 3)]
    for txt, x in labels:
        cv2.putText(row, txt, (x + 10, ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(row, txt, (x + 10, ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return row


class MaskDrawer:
    def __init__(self, base_rgb: np.ndarray, title: str):
        self.base_rgb = base_rgb
        self.title = title
        self.brush = BRUSH_DEFAULT
        H, W = base_rgb.shape[:2]
        self.mask = np.zeros((H, W), dtype=np.uint8)

        self.scale = min(WIN_MAX_W / W, WIN_MAX_H / H, 1.0)
        self.dw = int(W * self.scale)
        self.dh = int(H * self.scale)
        self.base_bgr_disp = cv2.resize(
            cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR),
            (self.dw, self.dh),
            interpolation=cv2.INTER_AREA,
        )
        self.drawing = False
        self.erasing = False

    def _orig(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def _render(self):
        out = self.base_bgr_disp.copy()
        m = cv2.resize(self.mask, (self.dw, self.dh), interpolation=cv2.INTER_NEAREST) > 127
        overlay = out.copy()
        overlay[m] = (overlay[m] * (1.0 - OVERLAY_ALPHA) + np.array(COLOR_PLACE) * OVERLAY_ALPHA).astype(np.uint8)
        out = overlay
        info = "brush=%dpx  LMB=draw  RMB=erase  [ / ]  Z=clear  S/Enter=OK  Q=quit" % self.brush
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Ảnh OK")
    ap.add_argument("--defect", default="foreign", help="scratch|crack|dent|bulge|chip|rust|foreign|burn")
    ap.add_argument("--out", default=r"V:\dataHondatPlus\test_samples\output", help="Output folder")

    ap.add_argument("--mask", default=None, help="Mask PNG/JPG (white=inpaint). Nếu không có, dùng --draw.")
    ap.add_argument("--draw", action="store_true", help="Vẽ mask placement trên OK (interactive)")

    ap.add_argument("--ref", default=None, help="Ảnh NG optional (dùng khi --ip-scale > 0)")
    ap.add_argument(
        "--no-ref-crop-draw",
        action="store_true",
        help="Tắt bước vẽ crop trên NG; dùng full --ref làm IP reference.",
    )
    ap.add_argument(
        "--ref-crop-pad",
        type=int,
        default=12,
        help="Padding (px) quanh bbox crop từ mask lỗi NG (khi dùng --ref-crop-draw).",
    )
    ap.add_argument(
        "--ref-crop-feather",
        type=int,
        default=7,
        help="Feather (px) cho mask khi tạo ref-crop (giúp IP-Adapter không học nền).",
    )
    ap.add_argument("--ip-scale", type=float, default=0.0, help="IP-Adapter scale (0=prompt-only)")

    ap.add_argument("--strength", type=float, default=0.78)
    ap.add_argument("--guidance", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--mask-dilate", type=int, default=18)
    ap.add_argument("--prompt", default=None, help="Override positive prompt (tùy chọn)")
    ap.add_argument("--negative", default=None, help="Override negative prompt (tùy chọn)")
    ap.add_argument("--no-lighting-match", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base_path = _require_image_path("--base", args.base)
    ref_path = _require_image_path("--ref", args.ref) if args.ref else None
    if args.ip_scale > 1e-6 and not ref_path:
        raise SystemExit("[!] --ip-scale > 0 cần kèm --ref.")
    if not args.draw and not args.mask:
        raise SystemExit("[!] Cần --mask hoặc --draw.")

    base_rgb = imread_w(base_path)
    ref_rgb = imread_w(ref_path) if ref_path else None
    ref_ip_rgb = ref_rgb
    if ref_rgb is not None and not args.no_ref_crop_draw:
        ref_mask = MaskDrawer(ref_rgb, "REF crop: draw DEFECT region on NG — S=OK  Q=quit").run()
        if int(np.sum(ref_mask > 127)) >= 8:
            ref_crop, ref_crop_mask = _crop_from_mask(ref_rgb, ref_mask, pad=int(args.ref_crop_pad))
            ref_ip_rgb = _apply_mask_to_ref_crop(ref_crop, ref_crop_mask, feather_px=int(args.ref_crop_feather))
            print(f"[REF] Using masked IP ref-crop: {ref_ip_rgb.shape[1]}x{ref_ip_rgb.shape[0]}")
        else:
            print("[REF] Crop mask too small → using full ref for IP.")

    if args.mask:
        m_rgb = imread_w(_require_image_path("--mask", args.mask))
        mask = cv2.cvtColor(m_rgb, cv2.COLOR_RGB2GRAY)
    else:
        drawer = MaskDrawer(base_rgb, "SDXL inpaint: draw placement mask — S=OK  Q=quit")
        mask = drawer.run()

    if int(np.sum(mask > 127)) < 16:
        raise SystemExit("[!] Mask quá nhỏ hoặc trống.")

    params = {
        "sdxl_inpaint": True,
        "sdxl_inpaint_ip_scale": float(args.ip_scale),
        "sdxl_inpaint_strength": float(args.strength),
        "sdxl_inpaint_guidance_scale": float(args.guidance),
        "sdxl_inpaint_steps": int(args.steps),
        "sdxl_inpaint_mask_dilate": int(args.mask_dilate),
        "lighting_match": not bool(args.no_lighting_match),
        "sdxl_refine": False,
        "position_jitter": 0.0,
        "diversity": 0.0,
    }
    if args.prompt:
        params["sdxl_inpaint_prompt"] = args.prompt
    if args.negative:
        params["sdxl_inpaint_negative_prompt"] = args.negative
    if ref_rgb is not None:
        params["ref_image_b64"] = encode_rgb_b64(ref_ip_rgb)

    result = generate(
        base_image=base_rgb,
        mask=mask.astype(np.uint8),
        defect_type=args.defect,
        material="metal",
        params=params,
    )
    out_rgb = decode_b64(result["result_image"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_stem = os.path.splitext(os.path.basename(base_path))[0]
    stem = f"sdxl_inpaint_{args.defect}_{base_stem}_{ts}"
    out_img = os.path.join(args.out, f"{stem}.jpg")
    out_pan = os.path.join(args.out, f"{stem}_panel.jpg")
    save_rgb(out_img, out_rgb)
    panel = _make_4panel(
        base_rgb=base_rgb,
        ref_rgb=ref_ip_rgb,
        mask_u8=mask.astype(np.uint8),
        result_rgb=out_rgb,
        ref_title=("ref(IP crop)" if ref_ip_rgb is not None else "ref"),
    )
    cv2.imencode(".jpg", panel)[1].tofile(out_pan)
    print(f"[OK] Saved:\n  {out_img}\n  {out_pan}")
    print(f"     pipeline={result.get('metadata', {}).get('pipeline')} engine={result.get('engine')}")


if __name__ == "__main__":
    main()

