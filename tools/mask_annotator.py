"""
tools/mask_annotator.py — Vẽ mask cho ảnh lỗi (defect annotation)
===================================================================

Cách dùng:
    python tools/mask_annotator.py <image_path> [--out <mask_path>]

Ví dụ:
    python tools/mask_annotator.py "V:/dataHondatPlus/test_samples/ng_ref/ng_scratch.bmp"
    python tools/mask_annotator.py "V:/dataHondatPlus/test_samples/ng_ref/ng_scratch.bmp" --out my_mask.png

Nếu không có --out thì mask tự lưu cạnh ảnh gốc với suffix _mask.png

Controls:
    Chuột trái giữ    — Vẽ (brush trắng = vùng lỗi)
    Chuột phải giữ   — Xóa (erase)
    Scroll up/down   — Tăng/giảm kích thước brush
    Z                — Undo (1 bước)
    R                — Reset mask (xóa hết)
    F                — Fill toàn bộ mask (paint all)
    [ ]              — Giảm / tăng brush size (thay thế scroll)
    S / Enter        — Lưu mask và thoát
    Q / Esc          — Thoát không lưu
"""

import argparse
import os
import sys
import platform
import numpy as np
import cv2

# Fix Windows DPI scaling — phải gọi trước khi tạo bất kỳ window nào
if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
    except Exception:
        pass


# ── Config ────────────────────────────────────────────────────────────────────
OVERLAY_ALPHA = 0.45          # độ trong của mask overlay
OVERLAY_COLOR = (0, 0, 255)   # màu overlay (BGR) — đỏ
BRUSH_DEFAULT = 20
BRUSH_MIN     = 2
BRUSH_MAX     = 120
WIN_MAX_H     = 900           # max window height (auto-scale nếu ảnh to)
WIN_MAX_W     = 1600


# ── State ─────────────────────────────────────────────────────────────────────
class Annotator:
    def __init__(self, image_path: str, out_path: str):
        self.image_path = image_path
        self.out_path   = out_path

        raw = np.fromfile(image_path, dtype='uint8')
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[ERROR] Không đọc được ảnh: {image_path}")
            sys.exit(1)

        self.orig_h, self.orig_w = img_bgr.shape[:2]

        # Auto-scale display nếu ảnh quá lớn
        scale = min(WIN_MAX_W / self.orig_w, WIN_MAX_H / self.orig_h, 1.0)
        self.scale     = scale
        self.disp_w    = int(self.orig_w * scale)
        self.disp_h    = int(self.orig_h * scale)

        if scale < 1.0:
            self.disp_img = cv2.resize(img_bgr, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        else:
            self.disp_img = img_bgr.copy()

        self.mask      = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
        self.mask_prev = self.mask.copy()   # for undo
        self.brush     = BRUSH_DEFAULT
        self.drawing   = False
        self.erasing   = False
        self.saved     = False

        # Detect ring guide (Hough) — hien thi vong tron len anh de de ve dung vi tri
        self.ring_guide = None   # (cx, cy, r) in display coords
        self._detect_ring_guide(img_bgr)

    # ── ring guide detection ─────────────────────────────────────────────────

    def _detect_ring_guide(self, img_bgr):
        """Detect outer ring bang Hough, luu ket qua theo display coords."""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            h, w = gray.shape
            r_min = max(int(min(h, w) * 0.15), 10)
            r_max = int(max(h, w) * 0.48)
            circles = cv2.HoughCircles(
                blur, cv2.HOUGH_GRADIENT_ALT, dp=1.0,
                minDist=50, param1=100, param2=0.7,
                minRadius=r_min, maxRadius=r_max,
            )
            if circles is not None:
                c = sorted(circles[0], key=lambda x: -x[2])[0]
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                # Chuyen sang display coords
                self.ring_guide = (
                    int(cx * self.scale),
                    int(cy * self.scale),
                    int(r  * self.scale),
                )
                print(f"[Guide] Ring detected: cx={cx} cy={cy} r={r}")
        except Exception as e:
            print(f"[Guide] Ring detection failed: {e}")

    # ── draw / erase ──────────────────────────────────────────────────────────

    def _to_orig(self, x, y):
        """Display coords → original image coords."""
        return int(x / self.scale), int(y / self.scale)

    def _paint(self, x, y, erase=False):
        ox, oy = self._to_orig(x, y)
        # brush in original coords
        brush_orig = max(1, int(self.brush / self.scale))
        val = 0 if erase else 255
        cv2.circle(self.mask, (ox, oy), brush_orig, val, -1)

    # ── render ────────────────────────────────────────────────────────────────

    def _render(self):
        # Scale mask to display size
        disp_mask = cv2.resize(self.mask, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)

        overlay = self.disp_img.copy()
        region  = disp_mask > 127
        for c, cv in enumerate(OVERLAY_COLOR):
            overlay[:, :, c][region] = (
                overlay[:, :, c][region] * (1 - OVERLAY_ALPHA) + cv * OVERLAY_ALPHA
            ).astype(np.uint8)

        # Ring guide overlay
        if self.ring_guide is not None:
            gx, gy, gr = self.ring_guide
            cv2.circle(overlay, (gx, gy), gr, (0, 255, 255), 2)   # vong ngoai — vang
            cv2.circle(overlay, (gx, gy), 4,  (0, 255, 255), -1)  # tam

        # HUD
        n_px    = int(np.sum(self.mask > 127))
        pct     = 100.0 * n_px / (self.orig_w * self.orig_h)
        info    = (f"brush:{self.brush}px | mask:{pct:.2f}% | "
                   f"[S/Enter]=Save  [Z]=Undo  [R]=Reset  [Q/Esc]=Quit  scroll=brush size")
        cv2.putText(overlay, info, (8, self.disp_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, info, (8, self.disp_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),   1, cv2.LINE_AA)
        return overlay

    # ── mouse callback ────────────────────────────────────────────────────────

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mask_prev = self.mask.copy()
            self.drawing   = True
            self._paint(x, y, erase=False)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mask_prev = self.mask.copy()
            self.erasing   = True
            self._paint(x, y, erase=True)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self._paint(x, y, erase=False)
            elif self.erasing:
                self._paint(x, y, erase=True)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = self.erasing = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.brush = min(self.brush + 3, BRUSH_MAX)
            else:
                self.brush = max(self.brush - 3, BRUSH_MIN)

    # ── save ──────────────────────────────────────────────────────────────────

    def save(self):
        _, buf = cv2.imencode(".png", self.mask)
        buf.tofile(self.out_path)
        n_px = int(np.sum(self.mask > 127))
        pct  = 100.0 * n_px / (self.orig_w * self.orig_h)
        print(f"[OK] Mask saved → {self.out_path}")
        print(f"     {self.orig_w}×{self.orig_h} | {n_px} px painted ({pct:.2f}%)")
        self.saved = True

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        win = "Mask Annotator — " + os.path.basename(self.image_path)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.disp_w, self.disp_h)
        cv2.setMouseCallback(win, self.mouse_cb)

        print(f"[INFO] Ảnh: {self.image_path}  ({self.orig_w}×{self.orig_h})")
        print(f"[INFO] Scale hiển thị: {self.scale:.2f}x → window {self.disp_w}×{self.disp_h}")
        print(f"[INFO] Mask sẽ lưu ra: {self.out_path}")
        print()
        print("  Chuột trái = vẽ vùng lỗi  |  Chuột phải = xóa")
        print("  Scroll = đổi brush size    |  [ ] = giảm / tăng brush")
        print("  Z = Undo  |  R = Reset  |  S/Enter = Lưu  |  Q/Esc = Thoát")
        print()

        while True:
            frame = self._render()
            cv2.imshow(win, frame)
            key = cv2.waitKey(15) & 0xFF

            if key in (ord('s'), 13):        # S or Enter
                self.save()
                break
            elif key in (ord('q'), 27):      # Q or Esc
                print("[INFO] Thoát không lưu.")
                break
            elif key == ord('z'):            # Undo
                self.mask = self.mask_prev.copy()
            elif key == ord('r'):            # Reset
                self.mask_prev = self.mask.copy()
                self.mask = np.zeros_like(self.mask)
                print("[INFO] Reset mask.")
            elif key == ord('f'):            # Fill all
                self.mask_prev = self.mask.copy()
                self.mask[:] = 255
            elif key == ord('['):
                self.brush = max(self.brush - 3, BRUSH_MIN)
            elif key == ord(']'):
                self.brush = min(self.brush + 3, BRUSH_MAX)

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        return self.saved


# ── CLI ───────────────────────────────────────────────────────────────────────

def _default_out(image_path: str) -> str:
    base, _ = os.path.splitext(image_path)
    return base + "_mask.png"


def main():
    parser = argparse.ArgumentParser(description="Vẽ mask defect cho ảnh lỗi")
    parser.add_argument("image",        help="Đường dẫn ảnh lỗi (NG)")
    parser.add_argument("--out", "-o",  help="Đường dẫn lưu mask PNG (default: <image>_mask.png)")
    args = parser.parse_args()

    image_path = args.image
    out_path   = args.out if args.out else _default_out(image_path)

    if not os.path.isfile(image_path):
        print(f"[ERROR] Không tìm thấy file: {image_path}")
        sys.exit(1)

    ann = Annotator(image_path, out_path)
    ann.run()


if __name__ == "__main__":
    main()
