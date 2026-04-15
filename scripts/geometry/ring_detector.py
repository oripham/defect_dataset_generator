"""
geometry/ring_detector.py
Detect the dominant concentric ring in a grayscale crop.
Returns (cx, cy, r) — the ring center and radius in pixel coordinates.
"""
import cv2
import numpy as np


def detect_ring(gray: np.ndarray, tag: str = "") -> tuple[int, int, int]:
    """
    Detect the bright concentric ring using Hough Circle Transform.

    Validation rules (reject bad detections):
      - Center must be at least 5 px from any image edge
      - Radius must be > 0.15 × min(h, w)   (too small → probably noise)
      - Radius must be < 0.95 × max(h, w)/2 (too large → wraps outside)

    Falls back to bright-pixel centroid + median-radius estimate if HoughCircles
    finds nothing valid.

    Returns
    -------
    (cx, cy, r) : int, int, int
    """
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    r_min = max(int(min(h, w) * 0.15), 10)
    r_max = int(max(h, w) * 0.48)
    margin = 5

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT_ALT, dp=1.0,
        minDist=50, param1=100, param2=0.7,
        minRadius=r_min, maxRadius=r_max,
    )

    if circles is not None:
        # Sort by descending radius; take first that passes validation
        for c in sorted(circles[0], key=lambda x: -x[2]):
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            if (cx > margin and cy > margin
                    and cx < w - margin and cy < h - margin):
                print(f"[RING{tag}] HoughCircles cx={cx} cy={cy} r={r}")
                return cx, cy, r

    # Fallback: centroid of top-20% bright pixels
    bright = (blurred > np.percentile(blurred, 80)).astype(np.uint8)
    m = cv2.moments(bright)
    if m["m00"] > 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        ys, xs = np.where(bright > 0)
        r = int(np.median(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)))
        r = max(r, r_min)
        print(f"[RING{tag}] fallback-centroid cx={cx} cy={cy} r={r}")
        return cx, cy, r

    # Last resort: image center
    cx, cy, r = w // 2, h // 2, min(h, w) // 3
    print(f"[RING{tag}] fallback-center cx={cx} cy={cy} r={r}")
    return cx, cy, r
