"""Visualize YOLO segmentation labels overlaid on images."""
import cv2
import numpy as np
from pathlib import Path

DEFECTS = {
    "scratch": {
        "img_dir": Path(r"V:\HondaPlus\napchai_yolo\scratch\images"),
        "lbl_dir": Path(r"V:\HondaPlus\napchai_yolo\scratch\labels"),
        "color": (0, 0, 255),
    },
    "mc_deform": {
        "img_dir": Path(r"V:\HondaPlus\napchai_yolo\mc_deform\images"),
        "lbl_dir": Path(r"V:\HondaPlus\napchai_yolo\mc_deform\labels"),
        "color": (0, 255, 0),
    },
    "ring_fracture": {
        "img_dir": Path(r"V:\HondaPlus\napchai_yolo\ring_fracture\images"),
        "lbl_dir": Path(r"V:\HondaPlus\napchai_yolo\ring_fracture\labels"),
        "color": (255, 0, 0),
    },
}

OUT_DIR = Path(r"V:\HondaPlus\napchai_yolo\label_visual")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_yolo_labels(img, label_path, color):
    h, w = img.shape[:2]
    overlay = img.copy()
    lines = label_path.read_text().strip().split("\n")
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


for defect_name, cfg in DEFECTS.items():
    imgs = sorted(cfg["img_dir"].glob("*.png"))[:3]
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        stem = img_path.stem
        lbl_path = cfg["lbl_dir"] / (stem + ".txt")
        if not lbl_path.exists():
            continue
        result, n_polys = draw_yolo_labels(img, lbl_path, cfg["color"])
        cv2.putText(result, f"{defect_name}: {n_polys} polygons",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out_path = OUT_DIR / f"{defect_name}_{stem}.png"
        cv2.imwrite(str(out_path), result)
        print(f"  {out_path.name}  ({n_polys} polygons)")

print(f"\nSaved to {OUT_DIR}")
