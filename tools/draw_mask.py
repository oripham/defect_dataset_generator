"""
Simple mask drawing tool for Napchai scratch generation.
- Left click + drag: Draw mask (white on black)
- Right click + drag: Erase mask
- Mouse wheel: Change brush size
- 'c': Clear mask
- 's': Save mask and exit
- 'q': Quit without saving
"""
import cv2
import numpy as np
from pathlib import Path

OK_IMG = Path(r"V:\defect_samples\Napchai\Xước\ok\ok_001.jpg")
MASK_OUT = Path(r"V:\defect_samples\Napchai\Xước\mask\drawn_mask.png")

drawing = False
erasing = False
brush_size = 15
ix, iy = -1, -1


def mouse_cb(event, x, y, flags, param):
    global drawing, erasing, ix, iy, mask, display, brush_size

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
        print(f"Brush size: {brush_size}")


def update_display():
    global display
    overlay = base_img.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    mask_3ch = cv2.merge([mask, mask, mask])
    alpha = 0.35
    overlay = np.where(mask_3ch > 0,
                       cv2.addWeighted(overlay, 1 - alpha, red, alpha, 0),
                       overlay)
    display = overlay


def main():
    global mask, base_img, display, brush_size

    img = cv2.imread(str(OK_IMG))
    if img is None:
        print(f"ERROR: Cannot read {OK_IMG}")
        return

    h, w = img.shape[:2]
    scale = min(1200 / w, 800 / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    base_img = img.copy()
    h, w = base_img.shape[:2]

    # Load existing mask if present
    if MASK_OUT.exists():
        existing = cv2.imread(str(MASK_OUT), cv2.IMREAD_GRAYSCALE)
        if existing is not None:
            mask = cv2.resize(existing, (w, h))
            print("Loaded existing mask")
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    update_display()

    win = "Draw Mask - Left:draw | Right:erase | Wheel:size | S:save | Q:quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    cv2.setMouseCallback(win, mouse_cb)

    print(f"\nImage: {w}x{h}")
    print(f"Brush size: {brush_size}")
    print("Left click+drag = draw mask")
    print("Right click+drag = erase")
    print("Mouse wheel = change brush size")
    print("'s' = save, 'c' = clear, 'q' = quit\n")

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            # Save at original resolution
            orig = cv2.imread(str(OK_IMG))
            oh, ow = orig.shape[:2]
            if (ow, oh) != (w, h):
                mask_full = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
            else:
                mask_full = mask
            MASK_OUT.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(MASK_OUT), mask_full)
            print(f"\nMask saved: {MASK_OUT}")
            print(f"Size: {mask_full.shape}, non-zero pixels: {np.count_nonzero(mask_full)}")
            break
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
