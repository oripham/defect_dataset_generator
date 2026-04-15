import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Pill:
    mask: np.ndarray  # uint8 0/255 in full image coords
    bbox: tuple[int, int, int, int]  # x,y,w,h
    area: float


def _find_components(gray: np.ndarray) -> list[Pill]:
    # pills are darker than background
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 5)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    out: list[Pill] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 1800:  # drop dust
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 28 or h < 28:
            continue
        m = np.zeros((H, W), np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        out.append(Pill(mask=m, bbox=(x, y, w, h), area=area))
    out.sort(key=lambda p: (p.bbox[1], p.bbox[0]))
    return out


def _crop_with_pad(img: np.ndarray, bbox: tuple[int, int, int, int], pad: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = bbox
    H, W = img.shape[:2]
    padx = int(max(2, round(w * pad)))
    pady = int(max(2, round(h * pad)))
    x0 = max(0, x - padx)
    y0 = max(0, y - pady)
    x1 = min(W, x + w + padx)
    y1 = min(H, y + h + pady)
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Split Thuốc_tròn images into per-pill crops")
    ap.add_argument("--ok", required=True, help="OK image path (bmp)")
    ap.add_argument("--ng", required=True, help="NG/defect image path (bmp)")
    ap.add_argument("--out-root", default="V:/defect_samples/Thuốc_tròn", help="Output root folder")
    ap.add_argument("--pad", type=float, default=0.18, help="Padding ratio around detected bbox")
    ap.add_argument("--min-area-ok", type=float, default=2500, help="Min contour area for OK pills")
    ap.add_argument("--min-area-ng", type=float, default=1200, help="Min contour area for NG components (broken pieces smaller)")
    args = ap.parse_args()

    ok = cv2.imread(args.ok)
    ng = cv2.imread(args.ng)
    if ok is None or ng is None:
        raise RuntimeError("Cannot read OK/NG image")

    # OK pills
    ok_gray = cv2.cvtColor(ok, cv2.COLOR_BGR2GRAY)
    ok_pills = _find_components(ok_gray)
    ok_pills = [p for p in ok_pills if p.area >= float(args.min_area_ok)]

    # NG components (may include broken pieces)
    ng_gray = cv2.cvtColor(ng, cv2.COLOR_BGR2GRAY)
    ng_pills = _find_components(ng_gray)
    ng_pills = [p for p in ng_pills if p.area >= float(args.min_area_ng)]

    out_ok = os.path.join(args.out_root, "crops_ok")
    out_ng = os.path.join(args.out_root, "crops_ng")
    out_dbg = os.path.join(args.out_root, "crops_debug")
    _ensure_dir(out_ok)
    _ensure_dir(out_ng)
    _ensure_dir(out_dbg)

    # Save OK crops + debug overlay
    dbg_ok = ok.copy()
    for i, p in enumerate(ok_pills):
        crop, win = _crop_with_pad(ok, p.bbox, args.pad)
        cv2.imwrite(os.path.join(out_ok, f"ok_{i:03d}.png"), crop)
        x0, y0, x1, y1 = win
        cv2.rectangle(dbg_ok, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(dbg_ok, f"ok_{i:03d}", (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(out_dbg, "debug_ok_windows.png"), dbg_ok)

    # Save NG crops + debug overlay
    dbg_ng = ng.copy()
    for i, p in enumerate(ng_pills):
        crop, win = _crop_with_pad(ng, p.bbox, args.pad)
        cv2.imwrite(os.path.join(out_ng, f"ng_{i:03d}.png"), crop)
        x0, y0, x1, y1 = win
        cv2.rectangle(dbg_ng, (x0, y0), (x1, y1), (255, 0, 255), 2)
        cv2.putText(dbg_ng, f"ng_{i:03d}", (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.imwrite(os.path.join(out_dbg, "debug_ng_windows.png"), dbg_ng)

    print(out_ok)
    print(out_ng)
    print(out_dbg)
    print(f"ok_pills={len(ok_pills)} ng_components={len(ng_pills)}")


if __name__ == "__main__":
    main()

