"""Visualize YOLO segmentation labels for B6/MKA dataset."""
import cv2
import numpy as np
from pathlib import Path

SRC = Path(r"V:\defect_samples\results\cap\b6_full_100")
IMG_DIR = SRC / "images"
LBL_DIR = SRC / "labels"
OUT_DIR = SRC / "label_visual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "dark_spots": (0, 0, 255),
    "dent": (0, 255, 0),
    "thread": (255, 255, 0),
    "plastic_flow": (255, 0, 255),
    "scratch": (0, 165, 255),
}


def draw_yolo_labels(img, label_path, color):
    h, w = img.shape[:2]
    overlay = img.copy()
    text = label_path.read_text().strip()
    if not text:
        return img, 0
    lines = text.split("\n")
    n_polys = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = list(map(float, parts[1:]))
        pts = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        n_polys += 1
    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    return result, n_polys


for defect_name, color in COLORS.items():
    imgs = sorted(IMG_DIR.glob(f"{defect_name}_*.png"))[:2]
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        result, n_polys = draw_yolo_labels(img, lbl_path, color)
        cv2.putText(result, f"{defect_name}: {n_polys} polys",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out_path = OUT_DIR / f"{defect_name}_{img_path.stem}.png"
        cv2.imwrite(str(out_path), result)
        print(f"  {out_path.name}  ({n_polys} polys)")

print(f"\nSaved to {OUT_DIR}")
