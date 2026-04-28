"""Clean YOLO labels for all 3 napchai classes: remove tiny noise polygons."""
from pathlib import Path

NAPCHAI_DIR = Path(r"V:\HondaPlus\napchai_yolo")
CLASSES = ["scratch", "mc_deform", "ring_fracture"]
MIN_POLYGON_AREA = 0.0005


def polygon_area(coords):
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


for cls in CLASSES:
    lbl_dir = NAPCHAI_DIR / cls / "labels"
    total_kept = 0
    total_removed = 0
    files_changed = 0
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        kept, removed = clean_label_file(lbl_path)
        if removed > 0:
            files_changed += 1
        total_kept += kept
        total_removed += removed
    print(f"{cls}: kept {total_kept}, removed {total_removed} noise polys, {files_changed} files changed")

    # verify
    poly_counts = []
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        text = lbl_path.read_text().strip()
        n = len(text.split("\n")) if text else 0
        poly_counts.append(n)
    print(f"  After cleaning: min={min(poly_counts)}, max={max(poly_counts)}, "
          f"mean={sum(poly_counts)/len(poly_counts):.1f}")
    zero_count = sum(1 for x in poly_counts if x == 0)
    if zero_count > 0:
        print(f"  WARNING: {zero_count} files with 0 polygons!")
