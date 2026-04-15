"""
photometry/texture.py
Match local noise/texture statistics of defect region to background.

Ensures defect region does not look "too smooth" or "too sharp"
compared to the surrounding metal surface texture.
"""
import cv2
import numpy as np


def measure_texture(
    img: np.ndarray,
    mask_f: np.ndarray,
    region: str = "background",
) -> dict:
    """
    Measure mean, std, and high-frequency power of a region.

    Parameters
    ----------
    img    : (H, W, 3) uint8 or (H, W) float32
    mask_f : (H, W) float32 [0, 1]
    region : "background" → pixels where mask_f < 0.1
             "defect"     → pixels where mask_f > 0.5

    Returns dict with keys: mean, std, hf_std (high-frequency std)
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    if region == "background":
        sel = mask_f < 0.1
    else:
        sel = mask_f > 0.5

    if sel.sum() < 5:
        return {"mean": 128.0, "std": 5.0, "hf_std": 2.0}

    pixels = gray[sel]

    # High-frequency component: original − smoothed
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    hf     = gray - smooth
    hf_pixels = hf[sel]

    return {
        "mean":   float(pixels.mean()),
        "std":    float(pixels.std()),
        "hf_std": float(hf_pixels.std()),
    }


def match_texture_stats(
    rendered: np.ndarray,
    orig: np.ndarray,
    mask_f: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """
    Make the noise distribution inside the defect region match the background.

    Steps:
      1. Measure background HF std (σ_bg)
      2. Measure defect HF std (σ_def)
      3. If σ_def < σ_bg (defect too smooth): add Gaussian noise with
         std = sqrt(σ_bg² − σ_def²) to equalize power
      4. If σ_def > σ_bg (defect too sharp): apply mild Gaussian blur
         inside defect only to soften

    The adjustment is applied only inside the mask region (mask_f > 0.5).
    """
    rng = np.random.RandomState(seed)
    out = rendered.astype(np.float32)

    stats_bg  = measure_texture(orig,     mask_f, region="background")
    stats_def = measure_texture(rendered, mask_f, region="defect")

    sigma_bg  = stats_bg["hf_std"]
    sigma_def = stats_def["hf_std"]

    defect_region = (mask_f > 0.5)[:, :, np.newaxis]

    if sigma_def < sigma_bg - 0.5:
        # Defect too smooth → add matching noise
        add_sigma = float(np.sqrt(max(sigma_bg ** 2 - sigma_def ** 2, 0)))
        noise = rng.normal(0, add_sigma, out.shape).astype(np.float32)
        out = out + noise * defect_region

    elif sigma_def > sigma_bg + 1.0:
        # Defect too sharp → blur inside mask
        blurred = cv2.GaussianBlur(out.astype(np.uint8), (3, 3), 0).astype(np.float32)
        out = blurred * defect_region + out * (1.0 - defect_region)

    return np.clip(out, 0, 255).astype(np.uint8)
