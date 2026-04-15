"""
tools/run_daily_plastic_flow_demo.py
===================================

Run a single Plastic Flow generation demo and save artifacts for daily report.

Default inputs (Windows drive):
  V:\\defect_samples\\MKA\\Nhựa_chảy\\ok\\*   (OK)
  V:\\defect_samples\\MKA\\Nhựa_chảy\\ref\\*  (NG ref crop or full NG)

Outputs:
  defect_dataset_generator/test_output/daily_<YYYYMMDD>_plastic_flow/
    ok.png
    ng_ref.png
    mask.png
    result.png
    pre_refine.png
    debug_panel.png   (OK | Mask | Result | Diff×4)
    summary.md
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


def _b64_to_bgr(b64: str) -> np.ndarray | None:
    if not b64:
        return None
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _make_debug_panel(ok_bgr: np.ndarray, mask_gray: np.ndarray | None,
                      result_bgr: np.ndarray, panel_h: int = 320) -> np.ndarray:
    diff = cv2.absdiff(ok_bgr, result_bgr)
    diff = cv2.convertScaleAbs(diff, alpha=4.0)

    if mask_gray is None:
        mask_gray = np.zeros(ok_bgr.shape[:2], np.uint8)
    if mask_gray.shape[:2] != ok_bgr.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (ok_bgr.shape[1], ok_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)

    panels = []
    for img in [ok_bgr, mask_bgr, result_bgr, diff]:
        h, w = img.shape[:2]
        pw = max(1, int(w * panel_h / h))
        panels.append(cv2.resize(img, (pw, panel_h)))
    return np.hstack(panels)


def _first_image(folder: Path) -> Path | None:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files += sorted(folder.glob(e))
    return files[0] if files else None


def main():
    root = Path(__file__).resolve().parent.parent  # defect_dataset_generator/
    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = root / "test_output" / f"daily_{date_tag}_plastic_flow"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(r"V:\defect_samples\MKA\Nhựa_chảy")
    ok_path = _first_image(data_root / "ok")
    ref_path = _first_image(data_root / "ref")
    if not ok_path or not ref_path:
        raise SystemExit(f"Missing inputs under {data_root} (need ok/ and ref/ images)")

    ok_bgr = cv2.imread(str(ok_path))
    ref_bgr = cv2.imread(str(ref_path))
    if ok_bgr is None:
        raise SystemExit(f"Cannot read OK image: {ok_path}")
    if ref_bgr is None:
        raise SystemExit(f"Cannot read NG ref image: {ref_path}")

    # Import engine
    import sys
    sys.path.insert(0, str(root))
    from engines.plastic_flow_engine import generate as gen_plastic_flow

    params = {
        "seed": 42,
        "intensity": 0.7,
        "ref_image_b64": _img_to_b64(ref_bgr),
        "pixel_warp_strength": 2.5,
        "use_synth": True,
        "synth_core_shrink_px": 12,
        "synth_intensity": 0.35,
        # extraction tuning (safe defaults)
        "ref_thresh_pct": 92.0,
        "core_r_px": 8.0,
        "fade_r_px": 10.0,
        "ref_pad_px": 14,
        "warp_disp_strength": 1.2,
        "warp_rot_deg": 0.0,
        "warp_scale": 1.0,
        # keep mask small / avoid halo
        "mask_shrink_px": 2,
        "max_area_frac": 0.012,
    }

    res = gen_plastic_flow(base_image_b64=_img_to_b64(ok_bgr), params=params)
    if "error" in res:
        raise SystemExit(res["error"])

    # Decode outputs
    result_bgr = _b64_to_bgr(res.get("result_image", ""))
    pre_bgr = _b64_to_bgr(res.get("result_pre_refine", ""))
    mask_bgr = _b64_to_bgr(res.get("mask_b64", ""))
    mask_gray = mask_bgr[:, :, 0] if mask_bgr is not None else None

    if result_bgr is None:
        raise SystemExit("Engine returned no result_image")

    # Save artifacts
    cv2.imwrite(str(out_dir / "ok.png"), ok_bgr)
    cv2.imwrite(str(out_dir / "ng_ref.png"), ref_bgr)
    if mask_gray is not None:
        cv2.imwrite(str(out_dir / "mask.png"), mask_gray)
    cv2.imwrite(str(out_dir / "result.png"), result_bgr)
    if pre_bgr is not None:
        cv2.imwrite(str(out_dir / "pre_refine.png"), pre_bgr)

    debug = _make_debug_panel(ok_bgr, mask_gray, result_bgr)
    cv2.imwrite(str(out_dir / "debug_panel.png"), debug)

    # Write summary
    # Filter params for report (avoid embedding huge base64 blobs)
    safe_params = {k: v for k, v in params.items() if not k.endswith("_b64")}

    summary = out_dir / "summary.md"
    summary.write_text(
        "\n".join([
            "## Daily demo — Plastic Flow generation",
            "",
            f"- **OK**: `{ok_path}`",
            f"- **NG ref**: `{ref_path}`",
            f"- **Output dir**: `{out_dir}`",
            f"- **Engine**: `{res.get('engine', 'cv')}`",
            "",
            "### Params",
            "```json",
            __import__("json").dumps(safe_params, ensure_ascii=False, indent=2),
            "```",
        ]) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Saved daily demo to: {out_dir}")


if __name__ == "__main__":
    main()

