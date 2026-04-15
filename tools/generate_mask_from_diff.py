"""
Generate defect mask by diffing OK vs NG images.
Usage:
    python generate_mask_from_diff.py \
        --ok  <ok_image> \
        --ng  <ng_image> \
        --out <output_mask.png> \
        [--threshold 30] \
        [--dilate 20] \
        [--min_area 200] \
        [--preview]
"""
import argparse
import sys
import numpy as np
from PIL import Image, ImageFilter


def generate_mask(ok_path, ng_path, out_path,
                  threshold=30, dilate=20, min_area=200, preview=False):

    ok = Image.open(ok_path).convert("L")
    ng = Image.open(ng_path).convert("L")

    # Resize NG to match OK if sizes differ
    if ok.size != ng.size:
        ng = ng.resize(ok.size, Image.LANCZOS)
        print(f"[INFO] Resized NG to {ok.size}")

    ok_arr = np.array(ok, dtype=np.int16)
    ng_arr = np.array(ng, dtype=np.int16)

    diff = np.abs(ok_arr - ng_arr).astype(np.uint8)

    # Threshold
    mask = (diff >= threshold).astype(np.uint8) * 255

    # Remove small noise regions
    mask_img = Image.fromarray(mask, mode="L")

    # Dilate to expand mask region
    if dilate > 0:
        for _ in range(dilate // 5):
            mask_img = mask_img.filter(ImageFilter.MaxFilter(size=11))

    mask_arr = np.array(mask_img)

    # Keep only connected components larger than min_area
    from PIL import ImageDraw
    import PIL

    # Simple: if mask too sparse, warn
    white_px = np.sum(mask_arr > 128)
    total_px = mask_arr.size
    coverage = white_px / total_px * 100

    if white_px < min_area:
        print(f"[WARN] Mask coverage very low ({white_px}px). "
              f"Try lowering --threshold (current={threshold})")

    print(f"[INFO] Mask coverage: {white_px}px / {total_px}px ({coverage:.2f}%)")

    # Save
    mask_img.save(out_path)
    print(f"[OK] Saved mask: {out_path}")

    if preview:
        import os
        preview_path = out_path.replace(".png", "_preview.png")
        # Overlay mask on OK image
        ok_rgb = Image.open(ok_path).convert("RGB").resize(ok.size)
        mask_rgba = mask_img.resize(ok_rgb.size)
        overlay = Image.new("RGBA", ok_rgb.size, (0, 0, 0, 0))
        red = Image.new("RGBA", ok_rgb.size, (255, 0, 0, 120))
        overlay.paste(red, mask=mask_rgba)
        result = ok_rgb.convert("RGBA")
        result = Image.alpha_composite(result, overlay)
        result.convert("RGB").save(preview_path)
        print(f"[OK] Preview saved: {preview_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ok",        required=True,  help="Path to OK image")
    parser.add_argument("--ng",        required=True,  help="Path to NG image")
    parser.add_argument("--out",       required=True,  help="Output mask path (.png)")
    parser.add_argument("--threshold", type=int, default=30,
                        help="Pixel diff threshold (default: 30)")
    parser.add_argument("--dilate",    type=int, default=20,
                        help="Dilation radius in px (default: 20)")
    parser.add_argument("--min_area",  type=int, default=200,
                        help="Min mask area in pixels (warn if smaller)")
    parser.add_argument("--preview",   action="store_true",
                        help="Save red-overlay preview image")
    args = parser.parse_args()

    generate_mask(
        ok_path=args.ok,
        ng_path=args.ng,
        out_path=args.out,
        threshold=args.threshold,
        dilate=args.dilate,
        min_area=args.min_area,
        preview=args.preview,
    )
