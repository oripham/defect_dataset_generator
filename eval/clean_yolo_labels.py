"""Clean YOLO labels: remove tiny noise polygons, keep only significant ones."""
from pathlib import Path
import math

DATASET_DIR = Path(r"V:\HondaPlus\napchai_yolo\scratch_dataset")
MIN_POLYGON_AREA = 0.0005  # minimum normalized area (relative to image)


def polygon_area(coords):
    """Shoelace formula for polygon area from flat coord list."""
    n = len(coords) // 2
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        x_i, y_i = coords[2*i], coords[2*i+1]
        x_j, y_j = coords[2*j], coords[2*j+1]
        area += x_i * y_j - x_j * y_i
    return abs(area) / 2


def clean_label_file(label_path):
    text = label_path.read_text().strip()
    if not text:
        return 0, 0
    lines = text.split("\n")
    kept = []
    removed = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            removed += 1
            continue
        coords = list(map(float, parts[1:]))
        area = polygon_area(coords)
        if area >= MIN_POLYGON_AREA:
            kept.append(line.strip())
        else:
            removed += 1
    label_path.write_text("\n".join(kept) + "\n" if kept else "", encoding="utf-8")
    return len(kept), removed


total_kept = 0
total_removed = 0
for split in ["train", "val"]:
    lbl_dir = DATASET_DIR / split / "labels"
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        kept, removed = clean_label_file(lbl_path)
        if removed > 0:
            print(f"  {lbl_path.name}: kept {kept}, removed {removed}")
        total_kept += kept
        total_removed += removed

print(f"\nTotal: kept {total_kept}, removed {total_removed} noise polygons")
