"""
Đọc góc + cắt ROI từ ảnh kiểu thumb (nền tối, viên sáng), ví dụ:
  V:/dataHondatPlus/thumb_long.jpg

Chạy:
  python capsule_thumb_roi.py --input V:/dataHondatPlus/thumb_long.jpg --out V:/dataHondatPlus/rois
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def order_box_points(pts: np.ndarray) -> np.ndarray:
    """4 điển boxPoints → [tl, tr, br, bl] float32."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def perspective_crop_from_contour(
    img: np.ndarray, cnt: np.ndarray, pad: int = 4
) -> tuple[np.ndarray, float, tuple]:
    """
    Cắt ROI theo minAreaRect (phối cảnh → chữ nhật), trả về (warped_bgr, angle_deg, rect).
    rect = ((cx,cy), (w,h), angle) như OpenCV minAreaRect.
    """
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    o = order_box_points(box)

    tl, tr, br, bl = o
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))
    max_w = int(round(max(w_top, w_bot))) + 2 * pad
    max_h = int(round(max(h_left, h_right))) + 2 * pad

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(o, dst)
    warped = cv2.warpPerspective(
        img, M, (max_w, max_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return warped, float(rect[2]), rect


def normalize_angle_for_horizontal(long_side_w: int, long_side_h: int, angle_ocv: float) -> float:
    """
    Góc minAreaRect OpenCV (độ): diễn giải gần với 'lệch so với trục ngang'.
    Nếu cạnh dài là chiều cao của rect OpenCV, bổ sung 90°.
    """
    # rect[1] order có thể w < h; ta quan tâm viên nằm ngang → cạnh dài ~ chiều rộng crop
    a = angle_ocv
    if long_side_h > long_side_w:
        a = a + 90.0
    # map về [-45, 45] cho dễ đọc
    while a > 45:
        a -= 90
    while a < -45:
        a += 90
    return float(a)


def find_capsule_contours(
    gray: np.ndarray,
    thresh_val: int | None,
    min_area_ratio: float,
    min_aspect: float,
    close_kernel: tuple[int, int] | None,
):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if thresh_val is None:
        t, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = float(thresh_val)
        _, th = cv2.threshold(blur, int(thresh_val), 255, cv2.THRESH_BINARY)
    if close_kernel is not None and close_kernel[0] > 0 and close_kernel[1] > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, ker)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    img_area = float(h * w)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < img_area * min_area_ratio:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        ar = cw / max(ch, 1)
        if ar < min_aspect:
            continue
        out.append(c)
    out.sort(key=lambda c: cv2.boundingRect(c)[0])
    return out, th, t


def main():
    ap = argparse.ArgumentParser(description="Góc + ROI từ ảnh thumb viên nang (nền tối)")
    ap.add_argument("--input", default="V:/dataHondatPlus/thumb_long.jpg", help="Ảnh đầu vào")
    ap.add_argument("--out", default="", help="Thư mục lưu ROI (để trống = chỉ in góc)")
    ap.add_argument(
        "--thresh",
        type=int,
        default=None,
        help="Ngưỡng nhị phân (0-255). Mặc định: Otsu",
    )
    ap.add_argument(
        "--min-area",
        type=float,
        default=0.005,
        help="Tối thiểu diện tích contour / diện tích ảnh (thumb 400²: ~0.005 để giữ cả viên nhỏ)",
    )
    ap.add_argument(
        "--min-aspect",
        type=float,
        default=1.2,
        help="Tối thiểu w/h của bbox (viên nằm ngang)",
    )
    ap.add_argument(
        "--close",
        type=int,
        nargs=2,
        default=(17, 5),
        metavar=("W", "H"),
        help="Kernel morphology close (0 0 = tắt). Nối vùng sáng hai đầu khi giữa tối (thiếu hàm lượng)",
    )
    ap.add_argument("--pad", type=int, default=6, help="Padding perspective crop")
    ap.add_argument("--viz", default="", help="Lưu ảnh debug box+angle (path .jpg/.png)")
    args = ap.parse_args()

    path = args.input
    if not os.path.isfile(path):
        raise SystemExit(f"Không thấy file: {path}")

    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"Không đọc được ảnh: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    ck = tuple(args.close) if args.close != (0, 0) else None
    cnts, th_bin, t_used = find_capsule_contours(
        gray, args.thresh, args.min_area, args.min_aspect, ck
    )
    print(f"threshold_used={t_used}  n_contours={len(cnts)}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    vis = bgr.copy() if args.viz else None

    for i, c in enumerate(cnts):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box_i = np.intp(box)
        angle_raw = float(rect[2])
        (rw, rh) = rect[1]
        rw, rh = float(rw), float(rh)
        long_w = int(max(rw, rh))
        long_h = int(min(rw, rh))
        angle_h = normalize_angle_for_horizontal(long_w, long_h, angle_raw)

        if vis is not None:
            cv2.drawContours(vis, [box_i], 0, (0, 255, 0), 2)
            cx, cy = map(int, rect[0])
            cv2.putText(
                vis,
                f"#{i} ang={angle_h:.1f}",
                (cx - 40, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        warped, _, _ = perspective_crop_from_contour(bgr, c, pad=args.pad)
        print(f"  capsule[{i}]: minAreaRect_angle_ocv={angle_raw:.2f}  angle_norm_deg~{angle_h:.2f}  crop_size={warped.shape[1]}x{warped.shape[0]}")

        if args.out:
            out_path = os.path.join(args.out, f"roi_{i:02d}.jpg")
            cv2.imwrite(out_path, warped)
            print(f"    saved {out_path}")

    if vis is not None and args.viz:
        cv2.imwrite(args.viz, vis)
        print(f"saved viz {args.viz}")

    if args.out:
        cv2.imwrite(os.path.join(args.out, "_binary.jpg"), th_bin)


if __name__ == "__main__":
    main()
