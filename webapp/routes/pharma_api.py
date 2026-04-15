"""
webapp/routes/pharma_api.py — Pharma Defect API
================================================

Endpoints:
  POST /api/pharma/auto-mask          → detect tablet mask (Otsu)
  POST /api/pharma/preview            → generate 1 image, return immediately
  POST /api/pharma/batch              → start batch job (background thread)
  GET  /api/pharma/batch/<job_id>     → poll batch progress
  GET  /api/pharma/results            → list result files (for gallery strip)

No mask drawing required — auto-mask via Otsu for all pharma products.
"""

from __future__ import annotations

import os
import sys
import uuid
import json
import base64
import threading
import glob
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

# ── Locate capsule_engine ──────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent   # defect_dataset_generator/
sys.path.insert(0, str(_ROOT))

from utils import engine_post as _engine_post

try:
    from engines.pharma.capsule_engine import generate as _ce_generate_local, auto_mask as _ce_auto_mask_local
    _HAS_LOCAL_PHARMA = True
except Exception:
    _HAS_LOCAL_PHARMA = False

pharma_bp = Blueprint("pharma_api", __name__)

# ── Data root ─────────────────────────────────────────────────────────────────
_DEFECT_SAMPLES = Path(os.environ.get("PHARMA_DATA_ROOT",
    str(_ROOT.parent.parent / "defect_samples")))

PRODUCTS = {
    "elongated_capsule": {
        "display": "Elongated Capsule",
        "data_dir": _DEFECT_SAMPLES / "Thuốc_dài",
        "defects": {
            "hollow":    {"display": "Hollow",    "dir": "Rỗng"},
            "underfill": {"display": "Underfill", "dir": "Thiếu_hàm_lượng"},
        },
    },
    "round_tablet": {
        "display": "Round Tablet",
        "data_dir": _DEFECT_SAMPLES / "Thuốc_tròn",
        "defects": {
            "crack": {"display": "Crack", "dir": "Nứt"},
            "dent":  {"display": "Dent",  "dir": "Lõm"},
        },
    },
    "napchai": {
        "display": "Napchai",
        "locked":  True,
        "data_dir": _DEFECT_SAMPLES / "Napchai",
        "defects": {},
    },
}

RESULTS_ROOT = _DEFECT_SAMPLES / "results"

# ── Batch job store ────────────────────────────────────────────────────────────
_pharma_batch_jobs: dict = {}
_pharma_batch_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok_images_for(product_key: str, defect_key: str) -> list[Path]:
    p = PRODUCTS.get(product_key, {})
    d = p.get("defects", {}).get(defect_key, {})
    ok_dir = p.get("data_dir", Path()) / d.get("dir", "") / "ok"
    exts = ["*.png", "*.jpg", "*.bmp"]
    files = []
    for ext in exts:
        files += sorted(ok_dir.glob(ext))
    return files


def _img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


def _b64_to_bgr(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr  = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


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


# ── API: products list ─────────────────────────────────────────────────────────

@pharma_bp.get("/api/pharma/products")
def api_products():
    out = {}
    for pk, pv in PRODUCTS.items():
        defects = {}
        for dk, dv in pv.get("defects", {}).items():
            ok_dir = pv.get("data_dir", Path()) / dv.get("dir", "") / "ok"
            n_ok = len(list(ok_dir.glob("*.png")) + list(ok_dir.glob("*.jpg"))) if ok_dir.exists() else 0
            defects[dk] = {**dv, "n_ok": n_ok}
        out[pk] = {
            "display": pv["display"],
            "locked":  pv.get("locked", False),
            "defects": defects,
        }
    return jsonify(out)


# ── API: auto-mask ─────────────────────────────────────────────────────────────

@pharma_bp.post("/api/pharma/auto-mask")
def api_auto_mask():
    """
    Detect mask automatically from uploaded OK image.
    Body: { image_b64: str }
    """
    body = request.get_json(force=True, silent=True) or {}
    img_b64 = body.get("image_b64") or body.get("base_image")
    if not img_b64:
        return jsonify(error="image_b64 required"), 400

    result = _engine_post("/api/pharma/auto-mask", {"image_b64": img_b64})
    if result.get("_fallback") and _HAS_LOCAL_PHARMA:
        result = _ce_auto_mask_local(img_b64)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


# ── API: preview (1 image) ─────────────────────────────────────────────────────

@pharma_bp.post("/api/pharma/preview")
def api_pharma_preview():
    """
    Generate 1 pharma defect image.

    Body:
    {
      "image_b64":    str,           # base64 PNG OK image
      "mask_b64":     str | null,    # pre-computed mask (null = auto-detect)
      "product":      str,           # "elongated_capsule" | "round_tablet"
      "defect_type":  str,           # "hollow" | "crack" | ...
      "params":       {              # defect params
        "intensity":   float,
        "seed":        int,
        ...defect-specific...
      }
    }

    Returns:
    {
      "result_image": str,     # base64 PNG result
      "debug_panel":  str,     # base64 PNG 4-panel debug
      "mask_b64":     str,
      "engine":       "cv",
      "qc":           {verdict, ...},
      "metadata":     {}
    }
    """
    body = request.get_json(force=True, silent=True) or {}
    img_b64     = body.get("image_b64") or body.get("base_image")
    mask_b64    = body.get("mask_b64")
    defect_type = body.get("defect_type", "crack")
    params      = body.get("params", {})

    if not img_b64:
        return jsonify(error="image_b64 required"), 400

    result = _engine_post("/api/pharma/preview", {
        "image_b64":   img_b64,
        "mask_b64":    mask_b64,
        "product":     body.get("product", "round_tablet"),
        "defect_type": defect_type,
        "params":      params,
    })
    if result.get("_fallback") and _HAS_LOCAL_PHARMA:
        result = _ce_generate_local(
            base_image_b64=img_b64,
            mask_b64=mask_b64,
            defect_type=defect_type,
            params=params,
        )

    if "error" in result:
        return jsonify(result), 500

    # Build debug panel
    try:
        ok_bgr   = _b64_to_bgr(img_b64)
        res_bgr  = _b64_to_bgr(result["result_image"])
        mask_b64_used = result.get("mask_b64", "")
        mask_gray = _b64_to_bgr(mask_b64_used)[:, :, 0] if mask_b64_used else np.zeros(ok_bgr.shape[:2], np.uint8)
        panel    = _make_debug_panel(ok_bgr, mask_gray, res_bgr)
        result["debug_panel"] = _img_to_b64(panel)
    except Exception:
        result["debug_panel"] = None

    return jsonify(result)


# ── API: batch ─────────────────────────────────────────────────────────────────

def _batch_worker(job_id: str, payload: dict):
    """Background thread: generate N images and save to results/."""
    with _pharma_batch_lock:
        _pharma_batch_jobs[job_id]["status"] = "running"

    product     = payload["product"]
    defect_type = payload["defect_type"]
    params_base = payload.get("params", {})
    n_images    = int(payload.get("n_images", 10))

    # break_types / angles / depths for crack variety
    break_types = payload.get("break_types",
        ["straight", "corner", "curved", "concave", "zigzag2", "zigzag3"])
    depths      = payload.get("depths", [0.15, 0.30, 0.45, 0.60])
    angles      = payload.get("angles", [0, 45, 90, 135, 180, 225, 270, 315])

    ok_files = _ok_images_for(product, defect_type)
    if not ok_files:
        with _pharma_batch_lock:
            _pharma_batch_jobs[job_id].update(
                status="error", error="No OK images found")
        return

    import itertools, random as _random
    from datetime import datetime

    out_dir = RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S") / product / defect_type
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    errors    = 0

    # Build param combinations
    if defect_type == "crack":
        combos = list(itertools.product(break_types, depths, angles))
        _random.shuffle(combos)
    else:
        combos = [(None, None, None)] * n_images

    ok_cycle = itertools.cycle(ok_files)

    for i, (bt, depth, angle) in enumerate(combos):
        if generated >= n_images:
            break

        ok_path = next(ok_cycle)
        ok_bgr  = cv2.imread(str(ok_path))
        if ok_bgr is None:
            errors += 1
            continue

        _, buf = cv2.imencode(".png", ok_bgr)
        img_b64 = base64.b64encode(buf).decode("utf-8")

        params = dict(params_base)
        params.setdefault("seed", i * 7 + 42)
        if bt:    params["break_type"] = bt
        if depth: params["depth"]      = depth
        if angle is not None: params["angle"] = float(angle)

        result = _engine_post("/api/pharma/preview", {
            "image_b64": img_b64, "mask_b64": None,
            "product": product, "defect_type": defect_type, "params": params,
        })
        if result.get("_fallback") and _HAS_LOCAL_PHARMA:
            result = _ce_generate_local(
                base_image_b64=img_b64, mask_b64=None,
                defect_type=defect_type, params=params,
            )

        if "error" in result:
            errors += 1
        else:
            ok_stem = ok_path.stem
            bt_tag  = f"_{bt}" if bt else ""
            d_tag   = f"_d{int((depth or 0)*100)}" if depth else ""
            a_tag   = f"_a{int(angle or 0)}" if angle is not None else ""
            s_tag   = f"_s{params['seed']}"
            fname   = f"{defect_type}{bt_tag}{d_tag}{a_tag}_{ok_stem}{s_tag}.png"

            res_bgr = _b64_to_bgr(result["result_image"])
            cv2.imwrite(str(out_dir / fname), res_bgr)

            # Save debug panel
            try:
                mask_b64_used = result.get("mask_b64", "")
                mask_gray = _b64_to_bgr(mask_b64_used)[:, :, 0] if mask_b64_used else np.zeros(ok_bgr.shape[:2], np.uint8)
                panel = _make_debug_panel(ok_bgr, mask_gray, res_bgr)
                cv2.imwrite(str(out_dir / f"debug_{fname}"), panel)
            except Exception:
                pass

            generated += 1

        with _pharma_batch_lock:
            _pharma_batch_jobs[job_id].update(
                generated=generated,
                errors=errors,
                total=n_images,
                progress=int(generated / n_images * 100),
            )

    with _pharma_batch_lock:
        _pharma_batch_jobs[job_id].update(
            status="done",
            out_dir=str(out_dir),
        )


@pharma_bp.post("/api/pharma/batch")
def api_pharma_batch():
    """
    Start batch generation.
    Body: same as preview but with n_images, break_types, depths, angles.
    """
    body = request.get_json(force=True, silent=True) or {}
    if not body.get("product") or not body.get("defect_type"):
        return jsonify(error="product and defect_type required"), 400

    job_id = str(uuid.uuid4())[:8]
    with _pharma_batch_lock:
        _pharma_batch_jobs[job_id] = {
            "status":    "queued",
            "generated": 0,
            "errors":    0,
            "total":     int(body.get("n_images", 10)),
            "progress":  0,
        }

    t = threading.Thread(target=_batch_worker, args=(job_id, body), daemon=True)
    t.start()

    return jsonify(job_id=job_id, status="queued")


@pharma_bp.get("/api/pharma/batch/<job_id>")
def api_pharma_batch_status(job_id):
    with _pharma_batch_lock:
        job = _pharma_batch_jobs.get(job_id)
    if job is None:
        return jsonify(error="job not found"), 404
    return jsonify(job)


# ── API: results list (for gallery strip) ─────────────────────────────────────

@pharma_bp.get("/api/pharma/ok-images")
def api_ok_images():
    """
    List OK images for a product/defect combo.
    Query params: product, defect
    """
    product     = request.args.get("product", "")
    defect_type = request.args.get("defect", "")

    files = _ok_images_for(product, defect_type)
    out = []
    for f in files[:20]:  # limit to 20 thumbnails
        ok_bgr = cv2.imread(str(f))
        if ok_bgr is None:
            continue
        # Thumbnail
        h, w = ok_bgr.shape[:2]
        th = 80
        tw = int(w * th / h)
        thumb = cv2.resize(ok_bgr, (tw, th))
        _, buf = cv2.imencode(".png", thumb)
        thumb_b64 = base64.b64encode(buf).decode("utf-8")
        out.append({"filename": f.name, "path": str(f), "thumb_b64": thumb_b64})

    return jsonify(images=out, total=len(files))


@pharma_bp.post("/api/pharma/save")
def api_pharma_save():
    """
    Save result image to disk.
    Body: { result_b64: str, product: str, defect_type: str, filename: str (optional) }
    """
    body        = request.get_json(force=True, silent=True) or {}
    result_b64  = body.get("result_b64") or body.get("result_image")
    product     = body.get("product", "unknown")
    defect_type = body.get("defect_type", "unknown")
    filename    = body.get("filename", "")

    if not result_b64:
        return jsonify(error="result_b64 required"), 400

    from datetime import datetime
    out_dir = RESULTS_ROOT / "manual" / product / defect_type
    out_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{defect_type}_{ts}.png"

    res_bgr = _b64_to_bgr(result_b64)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), res_bgr)

    return jsonify(saved=True, path=str(out_path), filename=filename)


@pharma_bp.get("/api/pharma/results")
def api_pharma_results():
    """
    List recent result files.
    Query params: product, defect_type, limit (default 40)
    """
    product     = request.args.get("product", "")
    defect_type = request.args.get("defect_type", "")
    limit       = int(request.args.get("limit", 40))

    pattern = str(RESULTS_ROOT / "**" / "*.png")
    files   = sorted(glob.glob(pattern, recursive=True), reverse=True)

    out = []
    for fpath in files:
        p = Path(fpath)
        if p.name.startswith("debug_"):
            continue
        parts = p.parts
        # path: results/<timestamp>/<product>/<defect>/file.png
        if len(parts) < 4:
            continue
        prod_dir   = parts[-3] if len(parts) >= 4 else ""
        defect_dir = parts[-2] if len(parts) >= 3 else ""

        if product and product.lower() not in prod_dir.lower():
            continue
        if defect_type and defect_type.lower() not in defect_dir.lower() \
                        and defect_type.lower() not in p.name.lower():
            continue

        out.append({
            "path":       fpath,
            "filename":   p.name,
            "product":    prod_dir,
            "defect":     defect_dir,
            "debug_path": str(p.parent / f"debug_{p.name}"),
        })
        if len(out) >= limit:
            break

    return jsonify(results=out, total=len(out))
