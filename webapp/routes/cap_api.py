"""
webapp/routes/cap_api.py — MKA Cap Defect API
==============================================

Endpoints:
  GET  /api/cap/products           → list products + defects + ok image counts
  GET  /api/cap/ok-images          → list OK image thumbnails for a defect
  POST /api/cap/detect-circle      → Hough circle overlay for uploaded image
  POST /api/cap/preview            → generate 1 image, return immediately
  POST /api/cap/batch              → start batch job (background thread)
  GET  /api/cap/batch/<job_id>     → poll batch progress
  POST /api/cap/save               → save result to disk
  GET  /api/cap/results            → list result files (for gallery)
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

_ROOT = Path(__file__).parent.parent.parent   # defect_dataset_generator/
sys.path.insert(0, str(_ROOT))

from utils import engine_post as _engine_post

# Local engine — used only when no server_url (offline mode)
try:
    from engines.metal_cap.cap_engine import generate as _cap_generate_local, detect_circle_info as _cap_detect_circle_local
    _HAS_LOCAL_CAP = True
except Exception:
    _HAS_LOCAL_CAP = False

cap_bp = Blueprint("cap_api", __name__)

# ── Data root ─────────────────────────────────────────────────────────────────
_DEFECT_SAMPLES = Path(os.environ.get("CAP_DATA_ROOT",
    str(_ROOT.parent.parent / "defect_samples")))

MKA_ROOT = _DEFECT_SAMPLES / "MKA"

PRODUCTS = {
    "mka": {
        "display": "MKA Cap",
        "data_dir": MKA_ROOT,
        "defects": {
            "scratch":      {"display": "Scratch (Xước)",          "dir": "Xước",       "engine": "mask"},
            "dent":         {"display": "Dent (Lõm)",              "dir": "Lõm",        "engine": "mask"},
            "plastic_flow": {"display": "Plastic Flow (Nhựa chảy)","dir": "Nhựa_chảy", "engine": "mask"},
            "thread":       {"display": "Thread (Dị vật chỉ)",     "dir": "Dị_vật_chỉ","engine": "mask"},
            "dark_spots":   {"display": "Dark Spots (Dị vật đen)", "dir": "Dị_vật_đen","engine": "mask"},
        },
    },
}

RESULTS_ROOT = _DEFECT_SAMPLES / "results" / "cap"

# ── Batch job store ───────────────────────────────────────────────────────────
_cap_batch_jobs: dict = {}
_cap_batch_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok_files_for(product_key: str, defect_key: str) -> list[Path]:
    p = PRODUCTS.get(product_key, {})
    d = p.get("defects", {}).get(defect_key, {})
    ok_dir = p.get("data_dir", Path()) / d.get("dir", "") / "ok"
    files = []
    for ext in ["*.png", "*.jpg", "*.bmp"]:
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
    diff = cv2.absdiff(ok_bgr, result_bgr)
    diff_bright = cv2.convertScaleAbs(diff, alpha=4.0)
    if mask_gray is None or mask_gray.size == 0:
        mask_gray = np.zeros(ok_bgr.shape[:2], np.uint8)
    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    panels = []
    for img in [ok_bgr, mask_bgr, result_bgr, diff_bright]:
        h, w = img.shape[:2]
        pw = int(w * panel_h / h)
        panels.append(cv2.resize(img, (pw, panel_h)))
    return np.hstack(panels)


# ── API: products list ────────────────────────────────────────────────────────

@cap_bp.get("/api/cap/products")
def api_cap_products():
    out = {}
    for pk, pv in PRODUCTS.items():
        defects = {}
        for dk, dv in pv.get("defects", {}).items():
            ok_dir = pv.get("data_dir", Path()) / dv.get("dir", "") / "ok"
            n_ok = len(list(ok_dir.glob("*.png")) + list(ok_dir.glob("*.jpg")) +
                       list(ok_dir.glob("*.bmp"))) if ok_dir.exists() else 0
            defects[dk] = {**dv, "n_ok": n_ok}
        out[pk] = {
            "display": pv["display"],
            "locked":  pv.get("locked", False),
            "defects": defects,
        }
    return jsonify(out)


# ── API: ok-images ────────────────────────────────────────────────────────────

@cap_bp.get("/api/cap/load-image")
def api_cap_load_image():
    """Serve a full OK image by absolute path."""
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        return jsonify(error="File not found"), 404
    # Safety: only serve files inside _DEFECT_SAMPLES
    try:
        Path(path).resolve().relative_to(_DEFECT_SAMPLES.resolve())
    except ValueError:
        return jsonify(error="Path not allowed"), 403
    from flask import send_file
    return send_file(path)


@cap_bp.get("/api/cap/ok-images")
def api_cap_ok_images():
    product     = request.args.get("product", "mka")
    defect_type = request.args.get("defect", "scratch")

    files = _ok_files_for(product, defect_type)
    out = []
    for f in files[:20]:
        ok_bgr = cv2.imread(str(f))
        if ok_bgr is None:
            continue
        h, w = ok_bgr.shape[:2]
        th = 80
        tw = int(w * th / h)
        thumb = cv2.resize(ok_bgr, (tw, th))
        _, buf = cv2.imencode(".png", thumb)
        out.append({
            "filename": f.name,
            "path":     str(f),
            "thumb_b64": base64.b64encode(buf).decode("utf-8"),
        })
    return jsonify(images=out, total=len(files))


# ── API: detect-circle ────────────────────────────────────────────────────────

@cap_bp.post("/api/cap/detect-circle")
def api_cap_detect_circle():
    """Detect Hough circle from uploaded OK image for UI overlay."""
    body    = request.get_json(force=True, silent=True) or {}
    img_b64 = body.get("image_b64") or body.get("base_image")
    if not img_b64:
        return jsonify(error="image_b64 required"), 400
    result = _engine_post("/api/cap/detect-circle", {"image_b64": img_b64})
    if result.get("_fallback") and _HAS_LOCAL_CAP:
        result = _cap_detect_circle_local(img_b64)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


# ── API: preview ──────────────────────────────────────────────────────────────

@cap_bp.post("/api/cap/preview")
def api_cap_preview():
    """
    Generate 1 MKA cap defect image.
    Body:
    {
      "image_b64":   str,
      "product":     str,   (default "mka")
      "defect_type": str,
      "params":      {}
    }
    """
    body        = request.get_json(force=True, silent=True) or {}
    img_b64     = body.get("image_b64") or body.get("base_image")
    product     = body.get("product", "mka")
    defect_type = body.get("defect_type", "scratch")
    params      = body.get("params", {})

    if not img_b64:
        return jsonify(error="image_b64 required"), 400

    # Pass data_root for mask lookup
    p = PRODUCTS.get(product, {})
    data_root = str(p.get("data_dir", "")) if p else ""

    result = _engine_post("/api/cap/preview", {
        "image_b64":   img_b64,
        "product":     product,
        "defect_type": defect_type,
        "params":      params,
    })
    if result.get("_fallback") and _HAS_LOCAL_CAP:
        result = _cap_generate_local(
            base_image_b64=img_b64,
            defect_type=defect_type,
            params=params,
            data_root=data_root,
        )

    if "error" in result:
        return jsonify(result), 500

    # Build debug panel
    try:
        ok_bgr   = _b64_to_bgr(img_b64)
        res_bgr  = _b64_to_bgr(result["result_image"])
        mask_b64_used = result.get("mask_b64", "")
        if mask_b64_used:
            mask_data = base64.b64decode(mask_b64_used)
            mask_arr  = np.frombuffer(mask_data, np.uint8)
            mask_gray = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        else:
            mask_gray = np.zeros(ok_bgr.shape[:2], np.uint8)
        panel = _make_debug_panel(ok_bgr, mask_gray, res_bgr)
        result["debug_panel"] = _img_to_b64(panel)
    except Exception:
        result["debug_panel"] = None

    return jsonify(result)


# ── API: save ─────────────────────────────────────────────────────────────────

@cap_bp.post("/api/cap/save")
def api_cap_save():
    body        = request.get_json(force=True, silent=True) or {}
    result_b64  = body.get("result_b64") or body.get("result_image")
    product     = body.get("product", "mka")
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


# ── API: batch ────────────────────────────────────────────────────────────────

def _batch_worker(job_id: str, payload: dict):
    with _cap_batch_lock:
        _cap_batch_jobs[job_id]["status"] = "running"

    product     = payload.get("product", "mka")
    defect_type = payload.get("defect_type", "scratch")
    params_base = payload.get("params", {})
    n_images    = int(payload.get("n_images", 10))

    p = PRODUCTS.get(product, {})
    data_root = str(p.get("data_dir", "")) if p else ""
    ok_files  = _ok_files_for(product, defect_type)

    if not ok_files:
        with _cap_batch_lock:
            _cap_batch_jobs[job_id].update(status="error", error="No OK images found")
        return

    import itertools
    from datetime import datetime

    out_dir = RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S") / product / defect_type
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    errors    = 0
    ok_cycle  = itertools.cycle(ok_files)

    for i in range(n_images):
        ok_path = next(ok_cycle)
        ok_bgr  = cv2.imread(str(ok_path))
        if ok_bgr is None:
            errors += 1
            continue

        _, buf  = cv2.imencode(".png", ok_bgr)
        img_b64 = base64.b64encode(buf).decode("utf-8")

        params = dict(params_base)
        params.setdefault("seed", i * 7 + 42)

        result = _engine_post("/api/cap/preview", {
            "image_b64": img_b64, "product": product,
            "defect_type": defect_type, "params": params,
        })
        if result.get("_fallback") and _HAS_LOCAL_CAP:
            result = _cap_generate_local(
                base_image_b64=img_b64,
                defect_type=defect_type,
                params=params,
                data_root=data_root,
            )

        if "error" in result:
            errors += 1
        else:
            fname   = f"{defect_type}_s{params['seed']}_{ok_path.stem}.png"
            res_bgr = _b64_to_bgr(result["result_image"])
            cv2.imwrite(str(out_dir / fname), res_bgr)

            # Debug panel
            try:
                mask_b64_used = result.get("mask_b64", "")
                if mask_b64_used:
                    mask_data = base64.b64decode(mask_b64_used)
                    mask_gray = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                else:
                    mask_gray = np.zeros(ok_bgr.shape[:2], np.uint8)
                panel = _make_debug_panel(ok_bgr, mask_gray, res_bgr)
                cv2.imwrite(str(out_dir / f"debug_{fname}"), panel)
            except Exception:
                pass

            generated += 1

        with _cap_batch_lock:
            _cap_batch_jobs[job_id].update(
                generated=generated, errors=errors, total=n_images,
                progress=int(generated / n_images * 100),
            )

    with _cap_batch_lock:
        _cap_batch_jobs[job_id].update(status="done", out_dir=str(out_dir))


@cap_bp.post("/api/cap/batch")
def api_cap_batch():
    body = request.get_json(force=True, silent=True) or {}
    if not body.get("defect_type"):
        return jsonify(error="defect_type required"), 400

    job_id = str(uuid.uuid4())[:8]
    with _cap_batch_lock:
        _cap_batch_jobs[job_id] = {
            "status": "queued", "generated": 0, "errors": 0,
            "total": int(body.get("n_images", 10)), "progress": 0,
        }

    t = threading.Thread(target=_batch_worker, args=(job_id, body), daemon=True)
    t.start()
    return jsonify(job_id=job_id, status="queued")


@cap_bp.get("/api/cap/batch/<job_id>")
def api_cap_batch_status(job_id):
    with _cap_batch_lock:
        job = _cap_batch_jobs.get(job_id)
    if job is None:
        return jsonify(error="job not found"), 404
    return jsonify(job)


# ── API: results ──────────────────────────────────────────────────────────────

@cap_bp.get("/api/cap/results")
def api_cap_results():
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
