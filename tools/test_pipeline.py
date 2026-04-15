"""
tools/test_pipeline.py — CV pipeline test trên test_samples
============================================================

Chạy fast_physics engine (không cần GPU / SDXL) trên từng cặp
  OK base + NG ref + synthetic mask
và lưu ảnh so sánh:  base | mask overlay | result

Dùng:
    cd V:\\HondaPlus\\defect_dataset_generator
    python tools/test_pipeline.py
    python tools/test_pipeline.py --sdxl   # bật SDXLRefiner (cần GPU + model)

Output:
    V:\\dataHondatPlus\\test_samples\\output\\<defect>_<method>.jpg
"""

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import cv2
from PIL import Image

from engines.fast_physics import generate
from engines.utils import decode_b64
import base64

# ── Paths ─────────────────────────────���──────────────────���────────────────────
BASE_DIR  = r"V:\dataHondatPlus\test_samples"
OK_IMG    = os.path.join(BASE_DIR, r"ok\ok_mka.jpg")
NG_DIR    = os.path.join(BASE_DIR, "ng_mka")
OUT_DIR   = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def imread_w(p):
    return cv2.imdecode(np.fromfile(p, dtype="uint8"), cv2.IMREAD_COLOR)

def to_rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
def to_bgr(rgb): return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def encode_image_b64(arr_rgb):
    _, buf = cv2.imencode(".png", to_bgr(arr_rgb))
    return base64.b64encode(buf).decode()

def make_mask_overlay(base_rgb, mask, alpha=0.4):
    overlay = base_rgb.copy()
    overlay[mask > 127] = (
        overlay[mask > 127] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    ).astype(np.uint8)
    return overlay

def save_comparison(base_rgb, mask, result_rgb, label, out_path, ref_rgb=None):
    """Save 4-panel: base | ref (or mask) | mask overlay | result."""
    H, W = base_rgb.shape[:2]
    panel_w = min(W, 600)
    ph = int(H * panel_w / W)

    def resz(img):
        return cv2.resize(img, (panel_w, ph), interpolation=cv2.INTER_AREA)

    sep = np.full((ph, 4, 3), 60, dtype=np.uint8)

    b = resz(to_bgr(base_rgb))
    m = resz(to_bgr(make_mask_overlay(base_rgb, mask)))
    r = resz(to_bgr(result_rgb))

    if ref_rgb is not None:
        # Hiển thị CROP thực tế đang inject (mask bbox trong ref), không phải full ref
        from engines.fast_physics import _align_ref_to_mask
        ref_crop = _align_ref_to_mask(ref_rgb, mask)  # crop đúng vùng defect
        crop_h, crop_w = ref_crop.shape[:2]
        scale = min(panel_w / max(crop_w, 1), ph / max(crop_h, 1), 1.0)
        nw, nh = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
        ref_small = cv2.resize(to_bgr(ref_crop), (nw, nh), interpolation=cv2.INTER_AREA)
        panel_ref = np.full((ph, panel_w, 3), 30, dtype=np.uint8)
        y0 = (ph - nh) // 2
        x0 = (panel_w - nw) // 2
        panel_ref[y0:y0+nh, x0:x0+nw] = ref_small
        cv2.putText(panel_ref, f"ref crop {crop_w}x{crop_h}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        row = np.hstack([b, sep, panel_ref, sep, m, sep, r])
        labels = [("base", 0), ("ref-crop", panel_w+4),
                  ("mask", (panel_w+4)*2), (label, (panel_w+4)*3)]
    else:
        row = np.hstack([b, sep, m, sep, r])
        labels = [("base", 0), ("mask", panel_w+4), (label, (panel_w+4)*2)]

    for txt, x in labels:
        cv2.putText(row, txt, (x+8, ph-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
        cv2.putText(row, txt, (x+8, ph-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    cv2.imwrite(out_path, row)
    print(f"  saved → {os.path.basename(out_path)}")


# ── Synthetic mask generators ─────────────────────────────────────────────────

def mask_scratch(H, W):
    """Diagonal scratch across center."""
    m = np.zeros((H, W), dtype=np.uint8)
    cx, cy = W // 2, H // 2
    cv2.line(m, (cx - W//5, cy - H//8), (cx + W//5, cy + H//8), 255, 6)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(m, kern)

def mask_dent(H, W):
    """Ellipse dent slightly off-center."""
    m = np.zeros((H, W), dtype=np.uint8)
    cx, cy = int(W * 0.55), int(H * 0.45)
    cv2.ellipse(m, (cx, cy), (W//9, H//7), 15, 0, 360, 255, -1)
    return m

def mask_chip(H, W):
    """Irregular polygon chip at edge."""
    m = np.zeros((H, W), dtype=np.uint8)
    cx, cy = W // 2, H // 4
    pts = np.array([
        [cx - 30, cy], [cx + 10, cy - 25], [cx + 45, cy - 5],
        [cx + 40, cy + 30], [cx + 5, cy + 35], [cx - 25, cy + 20],
    ], dtype=np.int32)
    cv2.fillPoly(m, [pts], 255)
    return m

def mask_foreign(H, W):
    """Small blob (foreign object)."""
    m = np.zeros((H, W), dtype=np.uint8)
    cx, cy = int(W * 0.45), int(H * 0.52)
    cv2.ellipse(m, (cx, cy), (W//18, H//14), 0, 0, 360, 255, -1)
    return m


# ── Test cases ────────────────────────────────────────────────────────────────

CASES = [
    # (label,       defect_type, material, ng_file,          mask_fn,      needs_ref)
    ("scratch",     "scratch",   "metal",  "ng_scratch.bmp", mask_scratch,  True),
    ("dent",        "dent",      "metal",  None,             mask_dent,     False),
    ("chip",        "chip",      "metal",  "ng_foreign.bmp", mask_chip,     True),
    ("foreign",     "foreign",   "metal",  "ng_foreign.bmp", mask_foreign,  True),
    ("mouth_dent",  "dent",      "metal",  None,             mask_dent,     False),
    ("crater",      "foreign",   "metal",  "ng_crater.bmp",  mask_foreign,  True),
]

PARAMS_BASE = {
    "intensity":       0.7,
    "naturalness":     0.6,
    "position_jitter": 0.0,
    "diversity":       0.0,
    "poisson_blend":   True,
    "lighting_match":  True,
    "sdxl_refine":     False,   # no GPU locally
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Test fast_physics / DefectFill on test_samples")
    ap.add_argument(
        "--sdxl",
        action="store_true",
        help="Bật SDXLRefiner sau bước CV (cần CUDA + diffusers model như Docker)",
    )
    cli = ap.parse_args()

    base_bgr = imread_w(OK_IMG)
    if base_bgr is None:
        print(f"[ERROR] Cannot read base image: {OK_IMG}"); return
    base_rgb = to_rgb(base_bgr)
    H, W = base_rgb.shape[:2]
    print(f"Base: {W}x{H}  {os.path.basename(OK_IMG)}\n")

    results = []

    for label, defect_type, material, ng_file, mask_fn, needs_ref in CASES:
        print(f"[{label}]")

        # Load ref
        ref_b64 = None
        if needs_ref and ng_file:
            ng_path = os.path.join(NG_DIR, ng_file)
            ng_bgr  = imread_w(ng_path)
            if ng_bgr is None:
                print(f"  [SKIP] Cannot read ref: {ng_path}"); continue
            ref_rgb = to_rgb(ng_bgr)
            ref_b64 = encode_image_b64(ref_rgb)
            print(f"  ref: {ng_bgr.shape[1]}x{ng_bgr.shape[0]}  {ng_file}")

        # Build mask
        mask = mask_fn(H, W)
        n_px = int(np.sum(mask > 127))
        print(f"  mask: {n_px} px  ({100*n_px//(H*W)}%)")

        # Encode base as b64 for reference (engine takes np.ndarray directly)
        params = {**PARAMS_BASE, "sdxl_refine": cli.sdxl}
        if ref_b64:
            params["ref_image_b64"] = ref_b64

        try:
            result = generate(
                base_image  = base_rgb,
                mask        = mask,
                defect_type = defect_type,
                material    = material,
                params      = params,
            )
        except Exception as e:
            print(f"  [ERROR] {e}"); import traceback; traceback.print_exc(); continue

        # Decode result
        result_arr = decode_b64(result["result_image"])
        meta       = result.get("metadata", {})
        has_ref    = meta.get("has_ref", False)
        print(f"  engine=cv  pipeline=defectfill  has_ref={has_ref}")

        # Load ref for display if used
        ref_rgb_display = None
        if needs_ref and ng_file:
            ng_path = os.path.join(NG_DIR, ng_file)
            ng_bgr  = imread_w(ng_path)
            if ng_bgr is not None:
                ref_rgb_display = to_rgb(ng_bgr)

        out_path = os.path.join(OUT_DIR, f"{label}.jpg")
        save_comparison(base_rgb, mask, result_arr,
                        f"result [{label}]", out_path,
                        ref_rgb=ref_rgb_display)
        results.append(out_path)

    print(f"\n{'='*50}")
    print(f"Done. {len(results)} results in {OUT_DIR}")


if __name__ == "__main__":
    main()
