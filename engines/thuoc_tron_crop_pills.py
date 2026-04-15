import argparse
import os
from datetime import datetime

import cv2
import numpy as np


def find_pills(gray: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """
    Detect multiple pill blobs on white background.
    Returns list of (mask_u8, bbox) in full-image coordinates.
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 5)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    pills: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 30 or h < 30:
            continue
        m = np.zeros((H, W), np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        pills.append((m, (x, y, w, h)))
    pills.sort(key=lambda t: (t[1][1], t[1][0]))
    return pills


def crop_with_pad(img: np.ndarray, bbox: tuple[int, int, int, int], pad_ratio: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = bbox
    H, W = img.shape[:2]
    padx = int(max(2, round(w * pad_ratio)))
    pady = int(max(2, round(h * pad_ratio)))
    x0 = max(0, x - padx)
    y0 = max(0, y - pady)
    x1 = min(W, x + w + padx)
    y1 = min(H, y + h + pady)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Crop each Thuốc_tròn pill into separate images")
    ap.add_argument("--ok", required=True, help="OK image path (bmp)")
    ap.add_argument("--out-dir", default=r"V:\defect_samples\Thuốc_tròn\cropped", help="Output directory")
    ap.add_argument("--pad", type=float, default=0.18, help="Padding ratio around pill bbox")
    args = ap.parse_args()

    ok = cv2.imread(args.ok)
    if ok is None:
        raise RuntimeError("Cannot read OK image")
    gray = cv2.cvtColor(ok, cv2.COLOR_BGR2GRAY)
    pills = find_pills(gray)
    if not pills:
        raise RuntimeError("No pills detected")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.out_dir, ts)
    out_ok = os.path.join(root, "ok")
    out_mask = os.path.join(root, "mask")
    os.makedirs(out_ok, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    # Save overview debug
    dbg = ok.copy()
    for i, (_, bb) in enumerate(pills):
        x, y, w, h = bb
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(dbg, f"{i}", (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(root, "debug_overview.jpg"), dbg)

    # Crop each pill + its mask
    for i, (m, bb) in enumerate(pills):
        crop_img, (x0, y0, x1, y1) = crop_with_pad(ok, bb, args.pad)
        crop_m = m[y0:y1, x0:x1]
        cv2.imwrite(os.path.join(out_ok, f"pill_{i:03d}.png"), crop_img)
        cv2.imwrite(os.path.join(out_mask, f"pill_{i:03d}_mask.png"), crop_m)
        # Save bbox metadata for later reinsertion if needed
        with open(os.path.join(root, f"pill_{i:03d}.txt"), "w", encoding="utf-8") as f:
            f.write(f"x0={x0}\n")
            f.write(f"y0={y0}\n")
            f.write(f"x1={x1}\n")
            f.write(f"y1={y1}\n")

    print(root)
    print(f"pills={len(pills)}")


if __name__ == "__main__":
    main()

