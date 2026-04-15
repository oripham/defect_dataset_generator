# scripts/metrics.py — Quality metrics for generated defect images
# Usage:
#   python metrics.py --run_dir /workspace/jobs/画像1/output/run_XXXXXXXX \
#                     --good_dir /workspace/jobs/画像1/good_images
#
# Outputs per-image SSIM + mean/std summary for each defect class.
import argparse
import os
import yaml
import cv2
import numpy as np


def ssim_gray(img_a, img_b):
    """Compute SSIM between two uint8 grayscale images (same size)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2 = mu_a ** 2
    mu_b2 = mu_b ** 2
    mu_ab = mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sig_ab  = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    ssim_map = num / (den + 1e-10)
    return float(ssim_map.mean())


def load_images(folder, exts=(".jpg", ".jpeg", ".png")):
    paths = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            imgs.append((p, img))
    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",  required=True, help="Path to run_XXXXXXXX output folder")
    parser.add_argument("--good_dir", required=True, help="Path to good_images folder")
    args = parser.parse_args()

    run_dir  = args.run_dir
    good_dir = args.good_dir

    img_dir  = os.path.join(run_dir, "images")
    lbl_dir  = os.path.join(run_dir, "labels")
    cfg_path = os.path.join(run_dir, "config.yaml")

    if not os.path.isdir(img_dir):
        raise RuntimeError(f"images/ folder not found in {run_dir}")

    # Load class names from saved config
    class_names = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        for cls in cfg.get("classes", []):
            class_names[cls["class_id"]] = cls["name"]

    # Load good images (reference), resize to match generated
    good_imgs_raw = load_images(good_dir)
    if not good_imgs_raw:
        raise RuntimeError(f"No good images found in {good_dir}")

    # Load generated images
    gen_imgs = load_images(img_dir)
    if not gen_imgs:
        raise RuntimeError(f"No images found in {img_dir}")

    # Get target size from first generated image
    h_gen, w_gen = gen_imgs[0][1].shape[:2]

    # Prepare good image pool (resize to match generated)
    good_pool = [
        cv2.resize(img, (w_gen, h_gen), interpolation=cv2.INTER_LANCZOS4)
        for _, img in good_imgs_raw
    ]

    # Parse labels to get class_id per image
    def get_class(img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                line = f.readline().strip()
                if line:
                    return int(line.split()[0])
        return -1

    # Compute SSIM: each generated image vs closest good image (mean over pool)
    results = {}  # class_id → list of ssim
    print(f"\n{'Image':<20} {'Class':<15} {'SSIM':>6}")
    print("-" * 45)

    for img_path, gen_img in gen_imgs:
        cls_id = get_class(img_path)
        cls_label = class_names.get(cls_id, f"cls_{cls_id}")
        gen_gray  = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)

        # SSIM against each good image, take mean
        ssim_vals = []
        for good_img in good_pool:
            good_gray = cv2.cvtColor(good_img, cv2.COLOR_BGR2GRAY)
            ssim_vals.append(ssim_gray(gen_gray, good_gray))

        mean_ssim = float(np.mean(ssim_vals))
        fname = os.path.basename(img_path)
        print(f"{fname:<20} {cls_label:<15} {mean_ssim:>6.4f}")

        results.setdefault(cls_id, []).append(mean_ssim)

    # Summary per class
    print("\n" + "=" * 45)
    print(f"{'Class':<20} {'Mean SSIM':>10} {'Std':>8} {'N':>4}")
    print("-" * 45)
    for cls_id in sorted(results.keys()):
        vals = results[cls_id]
        cls_label = class_names.get(cls_id, f"cls_{cls_id}")
        print(f"{cls_label:<20} {np.mean(vals):>10.4f} {np.std(vals):>8.4f} {len(vals):>4}")
    print("=" * 45)

    overall = [v for vals in results.values() for v in vals]
    print(f"\nOverall SSIM: mean={np.mean(overall):.4f}  std={np.std(overall):.4f}  n={len(overall)}")
    print("\nNote: Higher SSIM = generated image is more similar to clean good image")
    print("      For defect augmentation, SSIM should be high (structure preserved)")
    print("      but NOT 1.0 (defect is visible).")


if __name__ == "__main__":
    main()
