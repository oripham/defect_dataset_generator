"""
engines/other_engine.py — "Other" (Unclassified) Defect Engine
================================================================

Pipeline for defects that don't fit the hardcoded cap/pharma/metal_cap
categories.  Uses a **reference-image-based** approach:

  1.  User provides OK image + NG reference image + mask
  2.  IP-Adapter learns the *style* of the defect from the NG image
  3.  SDXL inpaints the defect region on the OK image

If GenAI is not available (no GPU), falls back to a simple alpha-blend
paste of the NG crop onto the masked region of the OK image.

generate(
    base_image_b64,   # OK product image (base64 PNG)
    mask_b64,         # user-drawn mask on OK image (base64 PNG grayscale)
    ref_image_b64,    # NG reference image (base64 PNG) — style source
    params,           # {seed, intensity, ...}
) -> dict
"""

from __future__ import annotations

import base64
import traceback

import cv2
import numpy as np

from ..utils import encode_b64, decode_b64, decode_b64_gray


# ── Helpers ──────────────────────────────────────────────────────────────────

def _b64_to_bgr(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _bgr_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


def _make_debug_panel(ok_bgr, mask_gray, result_bgr, panel_h=240) -> np.ndarray:
    """Build 4-panel debug image: OK | Mask | Result | Diff×4."""
    diff = cv2.absdiff(ok_bgr, result_bgr)
    diff_bright = cv2.convertScaleAbs(diff, alpha=4.0)
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    panels = []
    for img in [ok_bgr, mask_bgr, result_bgr, diff_bright]:
        h, w = img.shape[:2]
        pw = int(w * panel_h / h)
        panels.append(cv2.resize(img, (pw, panel_h)))
    return np.hstack(panels)


# ── Notebook Helpers ─────────────────────────────────────────────────────────

def get_circle_hough(img_bgr):
    """Fallback circle detection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20,
        param1=50, param2=30, minRadius=10, maxRadius=0
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, 0] # [x, y, r]
    return None

def get_product_contour(img_bgr):
    """Find largest outer contour."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    return max(contours, key=cv2.contourArea)

def get_bbox(mask):
    coords = cv2.findNonZero(mask)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

def extract_defect_pixels(ref_bgr, mask_ng):
    """
    Sử dụng kĩ thuật xử lý ảnh để lọc ra các pixel thực sự là lỗi trong vùng mask.
    Giúp tránh việc dán cả màu nền của ảnh NG vào ảnh OK.
    """
    gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    # Lấy vùng mask
    region = cv2.bitwise_and(gray, mask_ng)
    
    # Sử dụng Otsu hoặc Adaptive Threshold để tìm các pixel nổi bật (lỗi thường có tương phản cao)
    # Ở đây dùng kĩ thuật đơn giản: tính giá trị trung bình vùng mask và lọc các pixel lệch xa trung bình
    mean_val = cv2.mean(gray, mask=mask_ng)[0]
    diff = cv2.absdiff(gray, int(mean_val))
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    
    # Kết hợp với mask gốc của người dùng
    refined_mask = cv2.bitwise_and(thresh, mask_ng)
    return refined_mask

def align_and_paste_mask_to_mask(ok_bgr, ref_bgr, mask_ok, mask_ng, intensity=0.6):
    """
    Dán trực tiếp vùng lỗi từ NG sang OK thông qua mapping BBox của 2 mask.
    Không xoay, không tìm tâm sản phẩm.
    """
    bbox_g = get_bbox(mask_ok)
    bbox_n = get_bbox(mask_ng)
    
    if bbox_g is None or bbox_n is None:
        return ok_bgr.copy()

    gx, gy, gw, gh = bbox_g
    nx, ny, nw, nh = bbox_n

    # 1. Cắt vùng lỗi từ ảnh NG
    crop_ref = ref_bgr[ny:ny+nh, nx:nx+nw]
    # Lọc mask tinh (chỉ lấy pixel lỗi thực sự)
    refined_mask_ng = extract_defect_pixels(ref_bgr, mask_ng)
    crop_mask = refined_mask_ng[ny:ny+nh, nx:nx+nw]

    # 2. Resize vùng lỗi cho khớp với BBox trên ảnh OK
    crop_ref_res = cv2.resize(crop_ref, (gw, gh))
    crop_mask_res = cv2.resize(crop_mask, (gw, gh))

    # Làm mờ biên mask để blend mượt
    crop_mask_res = cv2.GaussianBlur(crop_mask_res, (7, 7), 0)
    alpha = (crop_mask_res.astype(np.float32) / 255.0) * intensity
    alpha_3ch = np.stack([alpha] * 3, axis=-1)

    # 3. Thực hiện dán đè lên ảnh OK
    result = ok_bgr.copy()
    target_roi = result[gy:gy+gh, gx:gx+gw].astype(np.float32)
    source_roi = crop_ref_res.astype(np.float32)

    blended_roi = target_roi * (1.0 - alpha_3ch) + source_roi * alpha_3ch
    result[gy:gy+gh, gx:gx+gw] = blended_roi.clip(0, 255).astype(np.uint8)

    return result

# ── Main Entry Point ─────────────────────────────────────────────────────────

def generate(
    base_image_b64: str,
    mask_b64: str | None,
    ref_image_b64: str | None = None,
    ref_mask_b64: str | None = None,
    params: dict | None = None,
) -> dict:
    if params is None: params = {}

    ok_bgr = _b64_to_bgr(base_image_b64)
    h, w = ok_bgr.shape[:2]
    
    # Decode masks
    mask_ok = decode_b64_gray(mask_b64) if mask_b64 else np.zeros((h, w), dtype=np.uint8)
    if mask_ok.shape[:2] != (h, w):
        mask_ok = cv2.resize(mask_ok, (w, h), interpolation=cv2.INTER_NEAREST)
        
    mask_ng = decode_b64_gray(ref_mask_b64) if ref_mask_b64 else None
    ref_bgr = _b64_to_bgr(ref_image_b64) if ref_image_b64 else None

    # 1. Bước CV: Dán trực tiếp Mask-to-Mask
    intensity = float(params.get("intensity", 0.6))
    if ref_bgr is not None and mask_ng is not None:
        cv_result_bgr = align_and_paste_mask_to_mask(ok_bgr, ref_bgr, mask_ok, mask_ng, intensity)
    else:
        # Nếu không có reference, chỉ làm tối vùng mask
        cv_result_bgr = _cv_paste_defect(ok_bgr, mask_ok, None, params)

    final_result_bgr = cv_result_bgr
    engine_used = "cv_paste"

    # 2. Bước AI Refine: Dùng SDXL để hòa trộn lại ảnh sau khi đã dán bằng CV
    use_ai = params.get("use_sdxl") or params.get("use_ai") or params.get("use_genai")
    if use_ai:
        # Convert ảnh đã dán bằng CV sang RGB để AI xử lý
        cv_result_rgb = cv2.cvtColor(cv_result_bgr, cv2.COLOR_BGR2RGB)
        
        # Gọi AI với base_image là ảnh đã dán sẵn (Refinement mode)
        genai_result = _try_genai(cv_result_rgb, mask_ok, ref_image_b64, params)
        
        if genai_result and "result_image" in genai_result:
            final_result_bgr = _b64_to_bgr(genai_result["result_image"])
            engine_used = "genai_refine"

    # Build debug panel
    debug_panel = _make_debug_panel(ok_bgr, mask_ok, final_result_bgr)

    return {
        "result_image": _bgr_to_b64(final_result_bgr),
        "mask_b64":     _bgr_to_b64(cv2.cvtColor(mask_ok, cv2.COLOR_GRAY2BGR)),
        "engine":       engine_used,
        "debug_panel":  _bgr_to_b64(debug_panel),
        "metadata": {
            "engine":    engine_used,
            "intensity": intensity,
            "has_ref":   ref_image_b64 is not None,
        },
    }
