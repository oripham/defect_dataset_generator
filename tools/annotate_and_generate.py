"""
tools/annotate_and_generate.py — Annotate + Generate trong 1 tool
==================================================================

Flow 3 bước (mặc định — Poisson chỉ blend đúng vùng lỗi, tránh viền bbox):
  [1] Vẽ vị trí đặt lỗi trên ảnh OK (mask xanh) — quyết định lỗi nằm ở đâu trên OK
  [2] Trên ảnh NG: chỉ tô (brush) đúng vùng lỗi — engine gửi kèm mask chặt cho seamlessClone
  [3] Chạy DefectFill + xem kết quả

  --fixed-ng-frame: bước 2 dùng khung W×H cố định trên NG (hành vi cũ, dễ sót viền chữ nhật).
  --legacy-order: [1] NG trước, [2] OK sau.
  --genx: chỉ vẽ mask trên OK → SDXL inpaint theo prompt (kiểu GenX, lỗi tự nhiên hơn, ít dán ref).
          --ref tùy chọn; nếu --genx-ip-scale > 0 thì cần --ref (IP-Adapter).

Dùng (thay NG_PATH và OK_PATH bằng đường dẫn file thật — không dùng chữ <ng>):
    cd V:\\HondaPlus\\defect_dataset_generator
    python tools/annotate_and_generate.py --ref NG_PATH --base OK_PATH [--defect scratch]

Ví dụ:
    python tools/annotate_and_generate.py ^
        --ref  "V:\\dataHondatPlus\\test_samples\\ng_mka\\ng_scratch.bmp" ^
        --base "V:\\dataHondatPlus\\test_samples\\ok\\ok_mka.jpg" ^
        --defect scratch
    Thêm --sdxl-refine để bật SDXLRefiner sau CV (cần GPU).

Annotate mask (OK / NG legacy):
    Chuột trái/phải — Vẽ / Xóa  |  Scroll / [ ] — brush  |  Z R  |  S Enter  |  Q Esc
Bước khung cố định trên NG (chỉ khi --fixed-ng-frame):
    Chuột trái kéo — dời khung  |  I K J L — dịch 1px  |  S Enter  |  Q Esc
"""

import argparse
import base64
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from engines.fast_physics import generate
from engines.utils import decode_b64, encode_b64

# Alias từ UI cũ / tên ngắn → giá trị engine chuẩn
_DEFECT_ALIASES = {"crater": "foreign"}

# ── Constants ─────────────────────────────────────────────────────────────────
WIN_MAX_W    = 1400
WIN_MAX_H    = 860
BRUSH_DEFAULT = 18
BRUSH_MIN     = 2
BRUSH_MAX     = 120
OVERLAY_ALPHA = 0.45
COLOR_DEFECT  = (0, 0, 255)    # đỏ — vùng lỗi trên ref
COLOR_PLACE   = (0, 180, 0)    # xanh lá — vị trí đặt trên base


# ── Image I/O ─────────────────────────────────────────────────────────────────

def imread_w(path):
    arr = np.fromfile(path, dtype="uint8")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _defect_crop_and_mask(ref_rgb: np.ndarray, ref_mask: np.ndarray):
    """BBox từ mask lỗi trên NG → crop RGB + mask cùng kích thước (gửi engine)."""
    ys, xs = np.where(ref_mask > 127)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return (
        ref_rgb[y1:y2, x1:x2].copy(),
        ref_mask[y1:y2, x1:x2].copy(),
    )


def encode_rgb_b64(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode()

def save_rgb(path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def _require_image_path(label: str, path: str) -> str:
    """Resolve path; exit with hint if missing or placeholder like <ng>."""
    if any(c in path for c in "<>"):
        print(
            f"Lỗi: {label} chứa '<' hoặc '>' — đó chỉ là ví dụ trong tài liệu, không phải tên file.\n"
            "  Dùng đường dẫn thật, ví dụ:\n"
            r'  --ref "V:\dataHondatPlus\test_samples\ng_mka\ng_scratch.bmp" '
            r'--base "V:\dataHondatPlus\test_samples\ok\ok_mka.jpg"',
            file=sys.stderr,
        )
        sys.exit(2)
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        print(f"Lỗi: không tìm thấy file {label}:\n  {path}\n  (đã kiểm tra: {resolved})", file=sys.stderr)
        sys.exit(2)
    return resolved


# ── Annotator (dùng lại cho cả 2 bước) ────────────────────────────────────────

class Annotator:
    def __init__(self, image_rgb, title, overlay_color=COLOR_DEFECT):
        self.img     = image_rgb
        self.title   = title
        self.color   = overlay_color
        self.mask    = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        self.mask_bk = self.mask.copy()
        self.brush   = BRUSH_DEFAULT
        self.drawing = False
        self.erasing = False

        oh, ow = image_rgb.shape[:2]
        self.scale  = min(WIN_MAX_W / ow, WIN_MAX_H / oh, 1.0)
        self.dw     = int(ow * self.scale)
        self.dh     = int(oh * self.scale)
        self.disp   = cv2.resize(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            (self.dw, self.dh), interpolation=cv2.INTER_AREA,
        )

    def _orig(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def _paint(self, x, y, erase=False):
        ox, oy = self._orig(x, y)
        r = max(1, int(self.brush / self.scale))
        cv2.circle(self.mask, (ox, oy), r, 0 if erase else 255, -1)

    def _render(self):
        dmask = cv2.resize(self.mask, (self.dw, self.dh), interpolation=cv2.INTER_NEAREST)
        out   = self.disp.copy()
        region = dmask > 127
        for c, cv in enumerate(self.color):
            out[:, :, c][region] = (
                out[:, :, c][region] * (1 - OVERLAY_ALPHA) + cv * OVERLAY_ALPHA
            ).astype(np.uint8)

        pct  = 100.0 * np.sum(self.mask > 127) / self.mask.size
        info = (f"brush:{self.brush}px  {pct:.2f}%  "
                f"[S/Enter]=OK  [Z]=Undo  [R]=Reset  [Q]=Quit")
        cv2.putText(out, info, (8, self.dh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 2)
        cv2.putText(out, info, (8, self.dh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
        return out

    def _mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mask_bk = self.mask.copy(); self.drawing = True; self._paint(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mask_bk = self.mask.copy(); self.erasing = True; self._paint(x, y, True)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing: self._paint(x, y)
            elif self.erasing: self._paint(x, y, True)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = self.erasing = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.brush = min(BRUSH_MAX, self.brush + 3) if flags > 0 \
                    else max(BRUSH_MIN, self.brush - 3)

    def run(self):
        win = self.title
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.dw, self.dh)
        cv2.setMouseCallback(win, self._mouse)

        while True:
            cv2.imshow(win, self._render())
            key = cv2.waitKey(15) & 0xFF
            if key in (ord('s'), 13):
                if np.sum(self.mask > 127) < 10:
                    print("  [!] Mask trống — hãy vẽ vùng lỗi trước.")
                    continue
                break
            elif key in (ord('q'), 27):
                cv2.destroyAllWindows(); sys.exit(0)
            elif key == ord('z'):
                self.mask = self.mask_bk.copy()
            elif key == ord('r'):
                self.mask_bk = self.mask.copy()
                self.mask[:] = 0
            elif key == ord('['):
                self.brush = max(BRUSH_MIN, self.brush - 3)
            elif key == ord(']'):
                self.brush = min(BRUSH_MAX, self.brush + 3)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows(); sys.exit(0)

        cv2.destroyWindow(win)
        return self.mask


def _placement_bbox(mask: np.ndarray):
    """Trả về (x1,y1,x2,y2,bw,bh) từ mask >127; None nếu rỗng."""
    ys, xs = np.where(mask > 127)
    if len(ys) < 1:
        return None
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return x1, y1, x2, y2, x2 - x1, y2 - y1


class FixedCropPicker:
    """
    Khung chữ nhật cố định (cw×ch) trên ảnh NG; kéo chuột trái để dời.
    Trả về crop + ref_mask (full size, rect=255).
    """
    def __init__(self, image_rgb, title, crop_w: int, crop_h: int):
        self.img   = image_rgb
        self.title = title
        oh, ow     = image_rgb.shape[:2]
        self.cw    = max(1, min(int(crop_w), ow))
        self.ch    = max(1, min(int(crop_h), oh))
        self.rx    = max(0, (ow - self.cw) // 2)
        self.ry    = max(0, (oh - self.ch) // 2)

        self.scale = min(WIN_MAX_W / ow, WIN_MAX_H / oh, 1.0)
        self.dw    = int(ow * self.scale)
        self.dh    = int(oh * self.scale)
        self.disp  = cv2.resize(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            (self.dw, self.dh), interpolation=cv2.INTER_AREA,
        )
        self.drag = False
        self._drag_start_disp = (0, 0)
        self._drag_start_rect = (0, 0)

    def _orig(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def _clamp_rect(self):
        oh, ow = self.img.shape[:2]
        self.rx = max(0, min(self.rx, ow - self.cw))
        self.ry = max(0, min(self.ry, oh - self.ch))

    def _render(self):
        out = self.disp.copy()
        x0, y0 = int(self.rx * self.scale), int(self.ry * self.scale)
        x1, y1 = int((self.rx + self.cw) * self.scale), int((self.ry + self.ch) * self.scale)
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 255), -1)
        out = cv2.addWeighted(out, 0.82, overlay, 0.18, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)

        info = (f"khung {self.cw}x{self.ch}px  keo chuot trai  "
                f"[S/Enter]=OK  [I/K/J/L]=dich 1px  [Q]=Thoat")
        cv2.putText(out, info, (8, self.dh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
        cv2.putText(out, info, (8, self.dh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        return out

    def _mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = True
            self._drag_start_disp = (x, y)
            self._drag_start_rect = (self.rx, self.ry)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            ox, oy = self._orig(x, y)
            ox0, oy0 = self._orig(*self._drag_start_disp)
            self.rx = self._drag_start_rect[0] + (ox - ox0)
            self.ry = self._drag_start_rect[1] + (oy - oy0)
            self._clamp_rect()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = False

    def run(self):
        win = self.title
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.dw, self.dh)
        cv2.setMouseCallback(win, self._mouse)

        while True:
            cv2.imshow(win, self._render())
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("s"), 13):
                break
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                sys.exit(0)
            if key == ord("i"):
                self.ry -= 1
            elif key == ord("k"):
                self.ry += 1
            elif key == ord("j"):
                self.rx -= 1
            elif key == ord("l"):
                self.rx += 1
            if key in (ord("i"), ord("k"), ord("j"), ord("l")):
                self._clamp_rect()
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                sys.exit(0)

        cv2.destroyWindow(win)

        oh, ow = self.img.shape[:2]
        ref_mask = np.zeros((oh, ow), dtype=np.uint8)
        ref_mask[self.ry : self.ry + self.ch, self.rx : self.rx + self.cw] = 255
        ref_crop = self.img[self.ry : self.ry + self.ch, self.rx : self.rx + self.cw].copy()
        return ref_crop, ref_mask


# ── Result viewer ─────────────────────────────────────────────────────────────

def show_result(base_rgb, ref_rgb, ref_mask, place_mask, result_rgb, ref_column_title="ref+defect"):
    """Hiển thị 4 panel: base | ref+crop (hoặc GenX placeholder) | place mask | result."""
    H, W = base_rgb.shape[:2]
    pw = min(W, 540)
    ph = int(H * pw / W)

    def resz(img_rgb):
        return cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                          (pw, ph), interpolation=cv2.INTER_AREA)

    def overlay_resize_bgr(img_rgb, mask, color_bgr):
        """Resize ảnh + mask cùng (pw,ph) rồi blend — mask/ảnh có thể full-res khác nhau."""
        out = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                         (pw, ph), interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST) > 127
        for c, v in enumerate(color_bgr):
            out[:, :, c][m] = (out[:, :, c][m] * 0.5 + v * 0.5).astype(np.uint8)
        return out

    b = resz(base_rgb)
    if ref_rgb is None:
        ref_panel = np.full((ph, pw, 3), 42, dtype=np.uint8)
        t1 = "GenX: SDXL inpaint"
        t2 = "(khong ref / prompt only)"
        cv2.putText(ref_panel, t1, (10, ph // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(ref_panel, t2, (10, ph // 2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    else:
        ref_panel = overlay_resize_bgr(ref_rgb, ref_mask, (255, 0, 0))
    m = overlay_resize_bgr(base_rgb, place_mask, (0, 180, 0))
    r = resz(result_rgb)

    sep = np.full((ph, 3, 3), 50, dtype=np.uint8)
    row = np.hstack([b, sep, ref_panel, sep, m, sep, r])

    for txt, x in [("base(OK)", 6), (ref_column_title, pw+9),
                   ("placement", (pw+3)*2+6), ("result", (pw+3)*3+6)]:
        cv2.putText(row, txt, (x, ph - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,0), 2)
        cv2.putText(row, txt, (x, ph - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1)

    win = "Result — S=save  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(row.shape[1], 1800), ph)
    cv2.imshow(win, row)

    saved = False
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord('s'), 13):
            saved = True; break

    cv2.destroyWindow(win)
    return row, saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref",
        default=None,
        help="Ảnh NG (DefectFill bắt buộc; với --sdxl-inpaint là tùy chọn, cần nếu --sdxl-inpaint-ip-scale > 0)",
    )
    ap.add_argument("--base",   required=True,  help="Đường dẫn file ảnh OK (nền sạch), không dùng <ok>")
    ap.add_argument("--defect", default="scratch",
                    help="Loại lỗi: scratch|crack|dent|bulge|chip|rust|foreign|burn (crater→foreign)")
    ap.add_argument("--intensity",   type=float, default=0.7)
    ap.add_argument("--naturalness", type=float, default=0.6)
    ap.add_argument("--out",    default=r"V:\dataHondatPlus\test_samples\output")
    ap.add_argument(
        "--sdxl-refine",
        action="store_true",
        help="Bật SDXLRefiner sau DefectFill (cần GPU + model trong scripts/sdxl_refiner)",
    )
    ap.add_argument(
        "--legacy-order",
        action="store_true",
        help="Flow cũ: vẽ NG trước, OK sau (crop ref tự do, dễ upscale mạnh)",
    )
    ap.add_argument(
        "--fixed-ng-frame",
        action="store_true",
        help="Bước 2: khung cố định W×H trên NG (không brush chỉ lỗi). Mặc định = brush lỗi + mask chặt.",
    )
    ap.add_argument(
        "--sdxl-inpaint",
        action="store_true",
        help="SDXL prompt-inpaint trên mask OK (prompt), không bước vẽ NG trừ khi dùng IP (--sdxl-inpaint-ip-scale)",
    )
    ap.add_argument(
        "--sdxl-inpaint-ip-scale",
        type=float,
        default=0.0,
        help="SDXL inpaint: IP-Adapter scale từ --ref (0=chỉ prompt). Cần --ref khi > 0.",
    )
    # Backward compatible (deprecated) flags
    ap.add_argument("--genx", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--genx-ip-scale", type=float, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.genx or (args.genx_ip_scale is not None):
        print("  [!] --genx/--genx-ip-scale đã deprecated. Dùng --sdxl-inpaint/--sdxl-inpaint-ip-scale.")
        if args.genx:
            args.sdxl_inpaint = True
        if args.genx_ip_scale is not None:
            args.sdxl_inpaint_ip_scale = float(args.genx_ip_scale)

    if not args.sdxl_inpaint and args.ref is None:
        print("[!] Thiếu --ref (bắt buộc trừ khi dùng --sdxl-inpaint).")
        sys.exit(2)
    if args.sdxl_inpaint and args.sdxl_inpaint_ip_scale > 1e-6 and args.ref is None:
        print("[!] --sdxl-inpaint-ip-scale > 0 cần kèm --ref.")
        sys.exit(2)
    if args.sdxl_inpaint and args.sdxl_refine:
        print("  [!] --sdxl-refine bị bỏ qua khi dùng --sdxl-inpaint.")
    if args.sdxl_inpaint and (args.legacy_order or args.fixed_ng_frame):
        print("  [!] --legacy-order / --fixed-ng-frame bị bỏ qua khi dùng --sdxl-inpaint.")

    defect_type = _DEFECT_ALIASES.get(args.defect, args.defect)

    os.makedirs(args.out, exist_ok=True)

    base_path = _require_image_path("--base (ảnh OK)", args.base)
    ref_path = _require_image_path("--ref (ảnh NG)", args.ref) if args.ref else None

    print(f"\n{'='*60}")
    print(f"  Defect: {defect_type}" + (f"  (alias từ {args.defect})" if defect_type != args.defect else ""))
    print(f"  Ref:    {ref_path or '(không — chế độ SDXL prompt-inpaint)'}")
    print(f"  Base:   {base_path}")
    if args.sdxl_inpaint:
        flow_desc = f"SDXL prompt-inpaint  (ip_scale={args.sdxl_inpaint_ip_scale})"
    else:
        if args.legacy_order and args.fixed_ng_frame:
            print("  [!] --fixed-ng-frame bị bỏ qua khi dùng --legacy-order")
        if args.legacy_order:
            flow_desc = "legacy (NG trước)"
        else:
            flow_desc = (
                "OK trước → khung NG cố định"
                if args.fixed_ng_frame
                else "OK trước → brush chỉ vùng lỗi trên NG"
            )
    print(f"  Flow:   {flow_desc}")
    print(f"{'='*60}\n")
    base_rgb = imread_w(base_path)
    ref_rgb = imread_w(ref_path) if ref_path else None

    ref_mask = None

    # ── SDXL prompt-inpaint: chỉ mask trên OK ────────────────────────────────
    if args.sdxl_inpaint:
        print("[Bước 1] Vẽ vùng cần TẠO LỖI trên ảnh OK (màu xanh)")
        print("         → SDXL sẽ vẽ lại vùng này theo prompt + loại --defect\n")
        ann_base = Annotator(
            base_rgb, "SDXL inpaint: Ve vung inpaint tren OK — S=OK  Q=Thoat",
            overlay_color=COLOR_PLACE,
        )
        place_mask = ann_base.run()
        if int(np.sum(place_mask > 127)) < 16:
            print("  [!] Mask quá nhỏ.")
            sys.exit(1)
        print(f"  Inpaint mask: {int(np.sum(place_mask > 127))} px")

        inpaint_params = {
            "sdxl_inpaint":   True,
            "sdxl_refine":    False,
            "lighting_match": True,
            "position_jitter": 0.0,
            "diversity":       0.0,
            "sdxl_inpaint_ip_scale":   args.sdxl_inpaint_ip_scale,
        }
        if ref_rgb is not None:
            inpaint_params["ref_image_b64"] = encode_rgb_b64(ref_rgb)

        print("\n[Bước 2] Chạy SDXL prompt-inpaint…\n")
        result = generate(
            base_image  = base_rgb,
            mask        = place_mask,
            defect_type = defect_type,
            material    = "metal",
            params      = inpaint_params,
        )
        result_arr = decode_b64(result["result_image"])
        print(f"  Done — engine={result['engine']}  pipeline={result['metadata']['pipeline']}")

        ref_mask = np.zeros((base_rgb.shape[0], base_rgb.shape[1]), dtype=np.uint8)
        ref_col_title = "ref (IP)" if ref_rgb is not None else "sdxl-inpaint"
        panel, save_it = show_result(
            base_rgb, ref_rgb, ref_mask, place_mask, result_arr,
            ref_column_title=ref_col_title,
        )

        if save_it:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_stem = os.path.splitext(os.path.basename(base_path))[0]
            stem = f"sdxl_inpaint_{defect_type}_{base_stem}_{ts}"
            out_img = os.path.join(args.out, f"{stem}_result.jpg")
            out_pan = os.path.join(args.out, f"{stem}_panel.jpg")
            save_rgb(out_img, result_arr)
            cv2.imwrite(out_pan, panel)
            print(f"\n  Saved:\n    {out_img}\n    {out_pan}")
        else:
            print("\n  Không lưu.")
        return

    if args.legacy_order:
        # ── Flow cũ: NG → OK ───────────────────────────────────────────────
        print("[Bước 1] Vẽ vùng LỖI trên ảnh NG ref (màu đỏ)")
        print("         → Chỉ bôi đúng chỗ lỗi (không cả khung lớn), nhấn S khi xong\n")
        ann_ref = Annotator(
            ref_rgb, "Buoc 1 (legacy): Ve vung loi tren NG — S=OK  Q=Thoat",
            overlay_color=COLOR_DEFECT,
        )
        ref_mask = ann_ref.run()
        if int(np.sum(ref_mask > 127)) < 8:
            print("  [!] Mask lỗi trên NG quá nhỏ hoặc trống.")
            sys.exit(1)
        ref_crop, ref_pm = _defect_crop_and_mask(ref_rgb, ref_mask)
        print(f"  Defect crop: {ref_crop.shape[1]}x{ref_crop.shape[0]} px")

        print("\n[Bước 2] Vẽ VỊ TRÍ đặt lỗi trên ảnh OK base (màu xanh)")
        print("         → Bôi chỗ muốn tạo lỗi, nhấn S khi xong\n")
        ann_base = Annotator(
            base_rgb, "Buoc 2 (legacy): Ve vi tri dat loi tren OK — S=OK  Q=Thoat",
            overlay_color=COLOR_PLACE,
        )
        place_mask = ann_base.run()
        print(f"  Placement mask: {int(np.sum(place_mask > 127))} px")
    else:
        # ── Mặc định: OK trước ──────────────────────────────────────────────
        print("[Bước 1] Vẽ VỊ TRÍ đặt lỗi trên ảnh OK base (màu xanh)")
        print("         → Bôi vùng trên OK nơi lỗi sẽ xuất hiện, nhấn S khi xong\n")
        ann_base = Annotator(
            base_rgb, "Buoc 1: Ve vi tri dat loi tren OK — S=OK  Q=Thoat",
            overlay_color=COLOR_PLACE,
        )
        place_mask = ann_base.run()
        npx = int(np.sum(place_mask > 127))
        print(f"  Placement mask: {npx} px")

        bb = _placement_bbox(place_mask)
        if bb is None or bb[4] < 2 or bb[5] < 2:
            print("  [!] Mask OK trống hoặc bbox quá nhỏ.")
            sys.exit(1)
        x1, y1, x2, y2, bw, bh = bb
        print(f"  Bbox placement: ({x1},{y1})→({x2},{y2})  size={bw}x{bh}")

        if args.fixed_ng_frame:
            rh, rw = ref_rgb.shape[:2]
            cw, ch = min(bw, rw), min(bh, rh)
            if cw < bw or ch < bh:
                print(
                    f"  [!] Ảnh NG ({rw}x{rh}) nhỏ hơn bbox ({bw}x{bh}) "
                    f"→ khung thu về {cw}x{ch} (engine vẫn resize lên bbox — có thể mềm nét)."
                )
            print("\n[Bước 2] Chọn vùng texture trên ảnh NG (khung cố định)")
            print(f"         → Khung {cw}x{ch}px — kéo chuột trái, I/K/J/L dịch 1px, S khi xong\n")
            picker = FixedCropPicker(
                ref_rgb,
                f"Buoc 2: Khung {cw}x{ch} tren NG — S=OK  Q=Thoat",
                crop_w=cw,
                crop_h=ch,
            )
            ref_crop, ref_mask = picker.run()
            ys, xs = np.where(ref_mask > 127)
            ry1, ry2 = int(ys.min()), int(ys.max()) + 1
            rx1, rx2 = int(xs.min()), int(xs.max()) + 1
            ref_pm = ref_mask[ry1:ry2, rx1:rx2].copy()
            print(
                f"  Ref crop: {ref_crop.shape[1]}x{ref_crop.shape[0]}  "
                f"(target bbox OK: {bw}x{bh})"
            )
        else:
            print("\n[Bước 2] Vẽ vùng LỖI trên ảnh NG (màu đỏ)")
            print("         → Chỉ bôi đúng phần lỗi (không khoanh cả vùng lớn), S khi xong\n")
            ann_ref = Annotator(
                ref_rgb, "Buoc 2: Chi bo vung loi tren NG — S=OK  Q=Thoat",
                overlay_color=COLOR_DEFECT,
            )
            ref_mask = ann_ref.run()
            if int(np.sum(ref_mask > 127)) < 8:
                print("  [!] Mask lỗi trên NG quá nhỏ hoặc trống.")
                sys.exit(1)
            ref_crop, ref_pm = _defect_crop_and_mask(ref_rgb, ref_mask)
            print(f"  Defect crop: {ref_crop.shape[1]}x{ref_crop.shape[0]} px")

    # ── Bước 3: Chạy pipeline ──────────────────────────────────────────────────
    print("\n[Bước 3] Chạy DefectFill pipeline...")
    if args.sdxl_refine:
        print("         SDXLRefiner: BẬT (vùng trong mask vẫn giữ pixel CV sau refine)\n")

    # Encode ref_crop (đã crop đúng vùng lỗi)
    ref_b64 = encode_rgb_b64(ref_crop)
    ref_pm_b64 = encode_b64(ref_pm)

    result = generate(
        base_image  = base_rgb,
        mask        = place_mask,
        defect_type = defect_type,
        material    = "metal",
        params      = {
            "intensity":       args.intensity,
            "naturalness":     args.naturalness,
            "position_jitter": 0.0,
            "diversity":       0.0,
            "lighting_match":  True,
            "sdxl_refine":     args.sdxl_refine,
            "ref_image_b64":   ref_b64,
            "ref_patch_mask_b64": ref_pm_b64,
            # Ref là crop bbox(mask NG) — không dùng bbox mask(base) làm index vào ref
            "ref_is_patch":    True,
        },
    )

    result_arr = decode_b64(result["result_image"])
    print(f"  Done — engine={result['engine']}  pipeline={result['metadata']['pipeline']}")

    # ── Hiển thị kết quả ──────────────────────────────────────────────────────
    panel, save_it = show_result(base_rgb, ref_rgb, ref_mask, place_mask, result_arr)

    if save_it:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        ref_stem = os.path.splitext(os.path.basename(args.ref))[0]
        stem = f"{defect_type}_{ref_stem}_{ts}"
        out_img = os.path.join(args.out, f"{stem}_result.jpg")
        out_pan = os.path.join(args.out, f"{stem}_panel.jpg")
        save_rgb(out_img, result_arr)
        cv2.imwrite(out_pan, panel)
        print(f"\n  Saved:\n    {out_img}\n    {out_pan}")
    else:
        print("\n  Không lưu.")


if __name__ == "__main__":
    main()
