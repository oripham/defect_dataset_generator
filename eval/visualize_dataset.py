"""
Visualize ALL dataset images with bbox + segmentation overlays.
Saves annotated images grouped by class for visual QC.
"""
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(r"V:\HondaPlus\defect_dataset_generator\batch_output\yolo")
OUT = Path(r"V:\HondaPlus\defect_dataset_generator\eval\output\visual_check")

COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
]


def parse_yolo_seg(label_path, img_w, img_h):
    """Parse YOLO-seg label → list of (class_id, polygon_pts, bbox)."""
    results = []
    if not label_path.exists():
        return results
    text = label_path.read_text().strip()
    if not text:
        return results
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = []
        for i in range(0, len(coords) - 1, 2):
            x = coords[i] * img_w
            y = coords[i + 1] * img_h
            pts.append((int(x), int(y)))
        if len(pts) < 3:
            continue
        pts_arr = np.array(pts, dtype=np.int32)
        x1, y1 = pts_arr.min(axis=0)
        x2, y2 = pts_arr.max(axis=0)
        bw = x2 - x1
        bh = y2 - y1
        results.append({
            "cls": cls_id,
            "polygon": pts_arr,
            "bbox": (x1, y1, x2, y2),
            "size_px": (bw, bh),
            "area_px": cv2.contourArea(pts_arr),
        })
    return results


def visualize_dataset(dataset_name):
    ds_root = ROOT / dataset_name
    yaml_path = ds_root / "dataset.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg["names"]

    out_root = OUT / dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Per-class output dirs
    for cid, cname in class_names.items():
        (out_root / cname).mkdir(exist_ok=True)
    (out_root / "_overview").mkdir(exist_ok=True)

    stats = {cname: {"count": 0, "sizes": [], "areas": []} for cname in class_names.values()}

    for split in ["train", "val"]:
        img_dir = ds_root / split / "images"
        lbl_dir = ds_root / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            label_path = lbl_dir / (img_path.stem + ".txt")
            anns = parse_yolo_seg(label_path, w, h)

            vis = img.copy()

            for ann in anns:
                cls_id = ann["cls"]
                cls_name = class_names.get(cls_id, f"cls{cls_id}")
                color = COLORS[cls_id % len(COLORS)]
                polygon = ann["polygon"]
                x1, y1, x2, y2 = ann["bbox"]
                bw, bh = ann["size_px"]
                area = ann["area_px"]

                # Draw filled polygon (semi-transparent)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [polygon], color)
                cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

                # Draw polygon outline
                cv2.polylines(vis, [polygon], True, color, 2)

                # Draw bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # Label with class name + size
                label = f"{cls_name} {bw}x{bh}px area={area:.0f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(vis, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                stats[cls_name]["count"] += 1
                stats[cls_name]["sizes"].append((bw, bh))
                stats[cls_name]["areas"].append(area)

                # Save per-class crop (zoomed in on defect)
                pad = max(bw, bh) * 2
                cx1 = max(0, x1 - int(pad))
                cy1 = max(0, y1 - int(pad))
                cx2 = min(w, x2 + int(pad))
                cy2 = min(h, y2 + int(pad))
                crop = vis[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    crop_name = f"{split}_{img_path.stem}_{cls_name}.jpg"
                    cv2.imwrite(str(out_root / cls_name / crop_name), crop)

            # Save full annotated image
            fname = f"{split}_{img_path.stem}.jpg"
            cv2.imwrite(str(out_root / "_overview" / fname), vis)

    # Print stats
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    img_area = None
    for split in ["train", "val"]:
        img_dir = ds_root / split / "images"
        if img_dir.exists():
            sample = next(img_dir.glob("*.*"), None)
            if sample:
                s = cv2.imread(str(sample))
                if s is not None:
                    img_area = s.shape[0] * s.shape[1]
                    print(f"Image size: {s.shape[1]}x{s.shape[0]} ({img_area} px)")
                    break

    for cls_name, s in stats.items():
        if s["count"] == 0:
            print(f"\n  {cls_name}: NO ANNOTATIONS!")
            continue
        sizes = np.array(s["sizes"])
        areas = np.array(s["areas"])
        avg_w = sizes[:, 0].mean()
        avg_h = sizes[:, 1].mean()
        min_w = sizes[:, 0].min()
        min_h = sizes[:, 1].min()
        max_w = sizes[:, 0].max()
        max_h = sizes[:, 1].max()
        avg_area = areas.mean()
        pct = (avg_area / img_area * 100) if img_area else 0

        print(f"\n  {cls_name}: {s['count']} annotations")
        print(f"    Bbox size: avg {avg_w:.0f}x{avg_h:.0f}, "
              f"min {min_w}x{min_h}, max {max_w}x{max_h}")
        print(f"    Seg area: avg {avg_area:.0f} px ({pct:.2f}% of image)")

    print(f"\nVisualized → {out_root}")


if __name__ == "__main__":
    visualize_dataset("napchai")
    visualize_dataset("mka")
    print("\nDONE — check output/visual_check/ for annotated images")
