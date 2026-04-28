"""
Mask drawing tool for Napchai defects (MC Deform, Ring Fracture, Scratch).
Usage: python draw_mask_napchai.py [mc|ring|scratch]

Controls:
  Left click + drag : Draw mask (white)
  Right click + drag: Erase mask
  Mouse wheel       : Change brush size
  R                 : Toggle reference NG image overlay
  C                 : Clear mask
  S                 : Save mask and exit
  Q / ESC           : Quit without saving
"""
import sys
import cv2
import numpy as np
from pathlib import Path

NAPCHAI = Path(r"V:\defect_samples\Napchai")

DEFECTS = {
    "mc": {
        "name": "MC Deform",
        "ok": NAPCHAI / "Biến_dạng_MC" / "ok" / "ok_001.jpg",
        "mask_out": NAPCHAI / "Biến_dạng_MC" / "mask" / "drawn_mask.png",
        "ref_dir": NAPCHAI / "Biến_dạng_MC" / "ref",
    },
    "ring": {
        "name": "Ring Fracture",
        "ok": NAPCHAI / "Vỡ_vòng" / "ok" / "ok_001.jpg",
        "mask_out": NAPCHAI / "Vỡ_vòng" / "mask" / "drawn_mask.png",
        "ref_dir": NAPCHAI / "Vỡ_vòng" / "ref",
    },
    "scratch": {
        "name": "Scratch",
        "ok": NAPCHAI / "Xước" / "ok" / "ok_001.jpg",
        "mask_out": NAPCHAI / "Xước" / "mask" / "drawn_mask.png",
        "ref_dir": NAPCHAI / "Xước" / "ref",
    },
}

drawing = False
erasing = False
brush_size = 15
ix, iy = -1, -1
show_ref = False


def mouse_cb(event, x, y, flags, param):
    global drawing, erasing, ix, iy, mask, brush_size

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(mask, (ix, iy), (x, y), 255, brush_size)
            ix, iy = x, y
            update_display()
        elif erasing:
            cv2.line(mask, (ix, iy), (x, y), 0, brush_size)
            ix, iy = x, y
            update_display()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONUP:
        erasing = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            brush_size = min(100, brush_size + 2)
        else:
            brush_size = max(3, brush_size - 2)
        update_display()


def update_display():
    global display
    if show_ref and ref_img is not None:
        overlay = cv2.addWeighted(base_img, 0.5, ref_resized, 0.5, 0)
    else:
        overlay = base_img.copy()

    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    mask_3ch = cv2.merge([mask, mask, mask])
    overlay = np.where(mask_3ch > 0,
                       cv2.addWeighted(overlay, 1 - 0.35, red, 0.35, 0),
                       overlay)

    # Brush size indicator
    cv2.putText(overlay, f"Brush: {brush_size}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    nz = np.count_nonzero(mask)
    cv2.putText(overlay, f"Pixels: {nz}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if show_ref:
        cv2.putText(overlay, "REF ON", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    display = overlay


def main():
    global mask, base_img, display, brush_size, show_ref, ref_img, ref_resized

    defect_key = sys.argv[1] if len(sys.argv) > 1 else "mc"
    if defect_key not in DEFECTS:
        print(f"Usage: python draw_mask_napchai.py [{' | '.join(DEFECTS.keys())}]")
        return

    cfg = DEFECTS[defect_key]
    print(f"\n=== Draw Mask: {cfg['name']} ===\n")

    img = cv2.imread(str(cfg["ok"]))
    if img is None:
        print(f"ERROR: Cannot read {cfg['ok']}")
        return

    h, w = img.shape[:2]
    scale = min(1200 / w, 800 / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    base_img = img.copy()
    h, w = base_img.shape[:2]

    # Load reference NG image
    ref_img = None
    ref_resized = None
    ref_files = sorted(cfg["ref_dir"].glob("*.*")) if cfg["ref_dir"].exists() else []
    for rf in ref_files:
        ref_img = cv2.imread(str(rf))
        if ref_img is not None:
            ref_resized = cv2.resize(ref_img, (w, h))
            print(f"Ref NG: {rf.name}")
            break

    # Load existing mask
    if cfg["mask_out"].exists():
        existing = cv2.imread(str(cfg["mask_out"]), cv2.IMREAD_GRAYSCALE)
        if existing is not None:
            mask = cv2.resize(existing, (w, h))
            print(f"Loaded existing mask ({np.count_nonzero(mask)} px)")
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    update_display()

    win = f"Draw Mask: {cfg['name']} | L:draw R:erase Wheel:size S:save R:ref Q:quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    cv2.setMouseCallback(win, mouse_cb)

    print(f"Image: {w}x{h}")
    print(f"Brush size: {brush_size}")
    print("Left=draw | Right=erase | Wheel=size | R=toggle ref | S=save | Q=quit\n")

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            orig = cv2.imread(str(cfg["ok"]))
            oh, ow = orig.shape[:2]
            if (ow, oh) != (w, h):
                mask_full = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
            else:
                mask_full = mask
            cfg["mask_out"].parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cfg["mask_out"]), mask_full)
            print(f"\nMask saved: {cfg['mask_out']}")
            print(f"Size: {mask_full.shape}, non-zero: {np.count_nonzero(mask_full)}")
            break
        elif key == ord('r'):
            if ref_img is not None:
                show_ref = not show_ref
                update_display()
                print(f"Ref overlay: {'ON' if show_ref else 'OFF'}")
            else:
                print("No ref image available")
        elif key == ord('c'):
            mask[:] = 0
            update_display()
            print("Mask cleared")
        elif key == ord('q') or key == 27:
            print("Quit without saving")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
