"""
draw_mask_scratch.py — Draw mask interactively, then generate scratch napchai
==============================================================================
Usage:
    python tools/draw_mask_scratch.py

Controls:
    Left mouse  — draw (paint white on mask)
    Right mouse — erase
    +/-         — brush size
    R           — reset mask
    G           — generate scratch with current mask
    S           — save mask to defect_samples
    Q / ESC     — quit
"""
import sys, os, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

OK_IMG = r"V:\defect_samples\Napchai\Xước\ok\ok_001.jpg"
MASK_SAVE_DIR = r"V:\defect_samples\Napchai\Xước\mask"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_all_output")

drawing = False
erasing = False
brush_r = 8
mask = None
img_disp = None
scale = 1.0


def mouse_cb(event, x, y, flags, _):
    global drawing, erasing, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x, y), brush_r, 255, -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
        cv2.circle(mask, (x, y), brush_r, 0, -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_r, 255, -1)
        elif erasing:
            cv2.circle(mask, (x, y), brush_r, 0, -1)
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        drawing = False
        erasing = False


def make_overlay(img, msk, alpha=0.4):
    out = img.copy()
    green = np.zeros_like(img); green[:] = (0, 255, 0)
    mf = msk.astype(np.float32) / 255.0
    m3 = mf[:, :, np.newaxis]
    return (out * (1 - m3 * alpha) + green * m3 * alpha).astype(np.uint8)


def generate_scratch(img_bgr, mask_gray, seed=42):
    from engines.metal_cap.scratch_napchai_engine import generate
    from engines.utils import encode_b64

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ok_b64 = encode_b64(img_rgb)

    _, mask_buf = cv2.imencode(".png", mask_gray)
    mask_b64 = base64.b64encode(mask_buf).decode()

    result = generate(ok_b64, {"seed": seed, "intensity": 0.7}, mask_b64=mask_b64)
    return result


def main():
    global mask, img_disp, brush_r, scale

    img_bgr = cv2.imread(OK_IMG)
    if img_bgr is None:
        print(f"Cannot read {OK_IMG}")
        return

    h, w = img_bgr.shape[:2]
    scale = min(900 / w, 700 / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)
    img_disp = cv2.resize(img_bgr, (disp_w, disp_h))

    mask = np.zeros((disp_h, disp_w), dtype=np.uint8)

    win = "Draw Scratch Mask  |  L=draw R=erase +/-=brush G=gen S=save Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, mouse_cb)

    print("Draw mask on the area where scratches should appear.")
    print("Press G to generate, S to save mask, Q to quit.")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MASK_SAVE_DIR, exist_ok=True)
    seed = 42

    while True:
        overlay = make_overlay(img_disp, mask)
        cv2.putText(overlay, f"brush={brush_r} seed={seed}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, overlay)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('+') or key == ord('='):
            brush_r = min(brush_r + 3, 80)
        elif key == ord('-'):
            brush_r = max(brush_r - 3, 2)
        elif key == ord('r'):
            mask[:] = 0
            print("Mask reset.")
        elif key == ord('n'):
            seed += 1
            print(f"Seed = {seed}")
        elif key == ord('s'):
            mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            path = os.path.join(MASK_SAVE_DIR, "drawn_mask.png")
            cv2.imwrite(path, mask_full)
            print(f"Mask saved: {path}")
        elif key == ord('g'):
            if mask.sum() == 0:
                print("Draw something first!")
                continue
            print(f"Generating scratch (seed={seed})...", flush=True)
            mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            result = generate_scratch(img_bgr, mask_full, seed=seed)
            if "error" in result:
                print(f"ERROR: {result['error']}")
                continue

            res_data = base64.b64decode(result.get("result_image") or result.get("result_pre_refine", ""))
            res_img = cv2.imdecode(np.frombuffer(res_data, np.uint8), cv2.IMREAD_COLOR)
            mask_data = base64.b64decode(result.get("mask_b64", ""))
            mask_out = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)

            cv2.imwrite(os.path.join(OUT_DIR, "scratch_result.png"), res_img)
            cv2.imwrite(os.path.join(OUT_DIR, "scratch_drawn_mask.png"), mask_full)
            if mask_out is not None:
                cv2.imwrite(os.path.join(OUT_DIR, "scratch_output_mask.png"), mask_out)
                ovl = make_overlay(res_img, mask_out)
            else:
                ovl = res_img

            res_disp = cv2.resize(res_img, (disp_w, disp_h))
            ovl_disp = cv2.resize(ovl, (disp_w, disp_h))
            cv2.imshow("Scratch Result", res_disp)
            cv2.imshow("Scratch Overlay", ovl_disp)
            pct = (mask_out > 0).sum() / mask_out.size * 100 if mask_out is not None else 0
            print(f"Done! mask={pct:.2f}%  Saved to {OUT_DIR}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
