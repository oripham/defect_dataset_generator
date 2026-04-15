# scripts/ref_loader.py  v3 — pass-through (user provides tight crop)
import os
import random
import numpy as np
from PIL import Image

REF_SIZE = 512


def load_reference_image(ref_root, class_name):
    """
    Load NG reference image for IP-Adapter / classical methods.

    Design: user is responsible for providing a tight crop around the defect
    area (not the whole product). ref_loader just pads to square and resizes.
    No auto-cropping logic — it caused fill=0.99 on complex textures.
    """
    class_dir = os.path.join(ref_root, class_name)
    files = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'))
    ])
    if not files:
        raise RuntimeError(f"No reference images in {class_dir}")

    img_path = os.path.join(class_dir, random.choice(files))
    img = Image.open(img_path).convert("RGB")
    img_arr = np.array(img)
    print(f"[REF] Loaded {os.path.basename(img_path)} ({img.width}x{img.height})")

    # Pad to square with edge-fill to avoid aspect ratio distortion
    ch, cw = img_arr.shape[:2]
    if ch != cw:
        side  = max(ch, cw)
        pad_t = (side - ch) // 2
        pad_b = side - ch - pad_t
        pad_l = (side - cw) // 2
        pad_r = side - cw - pad_l
        img_arr = np.pad(img_arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='edge')

    return Image.fromarray(img_arr).resize((REF_SIZE, REF_SIZE), Image.LANCZOS)
