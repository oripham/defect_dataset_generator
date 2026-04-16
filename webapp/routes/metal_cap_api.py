"""
webapp/routes/metal_cap_api.py — Metal Cap (Napchai) Defect API (Unified)
============================================================================

This replaces the older napchai_api.py route naming.

Endpoints:
  POST /api/metal_cap/preview            → generate 1 image, return immediately
  POST /api/metal_cap/batch              → start batch job (background thread)
  GET  /api/metal_cap/batch/<job_id>     → poll batch progress
  GET  /api/metal_cap/ok-images          → list OK images for a defect class
  GET  /api/metal_cap/load-image         → serve full OK image by absolute path (safe)
  POST /api/metal_cap/save               → save result to disk
  GET  /api/metal_cap/results            → list saved result files

Forward target (engine server):
  POST /api/metal_cap/preview  (FastAPI in defect_dataset_generator/engines/api.py)

Fallback:
  engines.napchai_engine.generate (CV-only) if engine server unavailable.
"""

from __future__ import annotations

import os
import sys
import uuid
import base64
import threading
import glob
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

_ROOT = Path(__file__).parent.parent.parent  # defect_dataset_generator/
sys.path.insert(0, str(_ROOT))

from utils import engine_post as _engine_post

try:
    from engines.metal_cap.napchai_engine import generate as _nc_generate_local
    _HAS_LOCAL_NAPCHAI = True
except Exception:
    _HAS_LOCAL_NAPCHAI = False

metal_cap_bp = Blueprint("metal_cap_api", __name__)

# ── Data directories ───────────────────────────────────────────────────────────
_DEFECT_SAMPLES = Path(os.environ.get(
    "PHARMA_DATA_ROOT",
    str(_ROOT.parent.parent / "defect_samples"),
))

NAPCHAI_ROOT = _DEFECT_SAMPLES / "Napchai"
RESULTS_ROOT = _DEFECT_SAMPLES / "results" / "napchai"

DEFECT_CLASSES = {
    "mc_deform":     {"display": "MC Deform / MC変形",     "dir": "Biến_dạng_MC"},
    "ring_fracture": {"display": "Ring Fracture / 割れ輪", "dir": "Vỡ_vòng"},
    "scratch":       {"display": "Scratch / 傷",           "dir": "Xước"},
}

# ── Batch job store ────────────────────────────────────────────────────────────
_batch_jobs: dict = {}
_batch_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ok_images_for(defect_key: str) -> list[Path]:
    d = DEFECT_CLASSES.get(defect_key, {})
    ok_dir = NAPCHAI_ROOT / d.get("dir", "") / "ok"
    files: list[Path] = []
    for ext in ["*.png", "*.jpg", "*.bmp"]:
        files += sorted(ok_dir.glob(ext))
    return files


def _img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


def _b64_to_bgr(b64: str) -> np.ndarray | None:
    if not b64:
        return None
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
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


# ── API: catalogue ────────────────────────────────────────────────────────────
@metal_cap_bp.get("/api/metal_cap/products")
def api_products():
    out = {}
    for dk, dv in DEFECT_CLASSES.items():
        ok_dir = NAPCHAI_ROOT / dv["dir"] / "ok"
        n_ok = len(list(ok_dir.glob("*.png")) +
                   list(ok_dir.glob("*.jpg")) +
                   list(ok_dir.glob("*.bmp"))) if ok_dir.exists() else 0
        out[dk] = {"display": dv["display"], "n_ok": n_ok}
    return jsonify(out)


# ── API: load full image ──────────────────────────────────────────────────────
@metal_cap_bp.get("/api/metal_cap/load-image")
def api_load_image():
    """Serve a full OK image by absolute path."""
    from flask import send_file
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        return jsonify(error="File not found"), 404
    try:
        Path(path).resolve().relative_to(_DEFECT_SAMPLES.resolve())
    except ValueError:
        return jsonify(error="Path not allowed"), 403
    return send_file(path)


# ── API: OK images ─────────────────────────────────────────────────────────────
@metal_cap_bp.get("/api/metal_cap/ok-images")
def api_ok_images():
    defect_key = request.args.get("defect", "")
    files = _ok_images_for(defect_key)
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


# ── API: preview ──────────────────────────────────────────────────────────────
@metal_cap_bp.post("/api/metal_cap/preview")
def api_preview():
    """
    Body:
    {
      "image_b64":    str,           # base64 PNG OK image
      "defect_type":  str,           # "mc_deform" | "ring_fracture" | "scratch"
      "params":       {
        "seed":           int,
        "intensity":      float,
        "sdxl_refine":    bool,      # default True
        "ref_image_b64":  str,       # optional NG crop for IP-Adapter
        ...defect-specific...
      }
    }
    """
    body = request.get_json(force=True, silent=True) or {}
    img_b64 = body.get("image_b64") or body.get("base_image")
    defect_type = body.get("defect_type", "mc_deform")
    params = dict(body.get("params", {}))   # mutable copy

    # Pass user-drawn Cartesian mask (if any) into params so napchai_engine can use it
    mask_b64 = body.get("mask_b64")
    if mask_b64:
        params["mask_b64"] = mask_b64

    if not img_b64:
        return jsonify(error="image_b64 required"), 400

    print(f"\n[metal_cap_api] --- Preview Request ---")
    print(f"[metal_cap_api] Defect: {defect_type}")
    
    # Forward to engine server
    result = _engine_post("/api/metal_cap/preview", {
        "image_b64":   img_b64,
        "mask_b64":    mask_b64,
        "defect_type": defect_type,
        "params":      params,
    })
    
    if "error" in result:
        print(f"[metal_cap_api] Engine returned error: {result['error']}")
    else:
        print(f"[metal_cap_api] Engine success! Result image size: {len(result.get('result_image', ''))} chars")

    # Fallback: old CV-only napchai_engine
    if result.get("_fallback") and _HAS_LOCAL_NAPCHAI:
        result = _nc_generate_local(
            base_image_b64=img_b64,
            defect_type=defect_type,
            params=params,
        )

    if "error" in result:
        return jsonify(result), 500

    # Debug panel
    try:
        ok_bgr = _b64_to_bgr(img_b64)
        res_bgr = _b64_to_bgr(result.get("result_image", ""))
        mask_bgr = _b64_to_bgr(result.get("mask_b64", ""))
        mask_gray = mask_bgr[:, :, 0] if mask_bgr is not None else np.zeros(ok_bgr.shape[:2], np.uint8)
        panel = _make_debug_panel(ok_bgr, mask_gray, res_bgr)
        result["debug_panel"] = _img_to_b64(panel)
    except Exception:
        result["debug_panel"] = None

    return jsonify(result)


# ── API: save ─────────────────────────────────────────────────────────────────
@metal_cap_bp.post("/api/metal_cap/save")
def api_save():
    body = request.get_json(force=True, silent=True) or {}
    result_b64 = body.get("result_b64") or body.get("result_image")
    defect_type = body.get("defect_type", "unknown")
    filename = body.get("filename", "")

    if not result_b64:
        return jsonify(error="result_b64 required"), 400

    from datetime import datetime
    out_dir = RESULTS_ROOT / "manual" / defect_type
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
    with _batch_lock:
        _batch_jobs[job_id]["status"] = "running"

    defect_type = payload["defect_type"]
    params_base = payload.get("params", {})
    n_images = int(payload.get("n_images", 10))

    ok_files = _ok_images_for(defect_type)
    if not ok_files:
        with _batch_lock:
            _batch_jobs[job_id].update(status="error", error="No OK images found")
        return

    import itertools
    from datetime import datetime

    out_dir = RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S") / defect_type
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_cycle = itertools.cycle(ok_files)
    generated = 0
    errors = 0

    for i in range(n_images):
        ok_path = next(ok_cycle)
        ok_bgr = cv2.imread(str(ok_path))
        if ok_bgr is None:
            errors += 1
            continue

        _, buf = cv2.imencode(".png", ok_bgr)
        img_b64 = base64.b64encode(buf).decode("utf-8")

        params = dict(params_base)
        params.setdefault("seed", i * 7 + 42)

        result = _engine_post("/api/metal_cap/preview", {
            "image_b64": img_b64, "defect_type": defect_type, "params": params,
        })
        if result.get("_fallback") and _HAS_LOCAL_NAPCHAI:
            result = _nc_generate_local(
                base_image_b64=img_b64, defect_type=defect_type, params=params,
            )

        if "error" in result:
            errors += 1
        else:
            s_tag = f"_s{params['seed']}"
            fname = f"{defect_type}_{ok_path.stem}{s_tag}.png"
            res_bgr = _b64_to_bgr(result.get("result_image", ""))
            cv2.imwrite(str(out_dir / fname), res_bgr)
            generated += 1

        with _batch_lock:
            _batch_jobs[job_id].update(
                generated=generated,
                errors=errors,
                total=n_images,
                progress=int((i + 1) / n_images * 100),
            )

    with _batch_lock:
        _batch_jobs[job_id].update(status="done", out_dir=str(out_dir))


@metal_cap_bp.post("/api/metal_cap/batch")
def api_batch():
    body = request.get_json(force=True, silent=True) or {}
    if not body.get("defect_type"):
        return jsonify(error="defect_type required"), 400

    job_id = str(uuid.uuid4())[:8]
    with _batch_lock:
        _batch_jobs[job_id] = {
            "status": "queued",
            "generated": 0,
            "errors": 0,
            "total": int(body.get("n_images", 10)),
            "progress": 0,
        }

    t = threading.Thread(target=_batch_worker, args=(job_id, body), daemon=True)
    t.start()
    return jsonify(job_id=job_id, status="queued")


@metal_cap_bp.get("/api/metal_cap/batch/<job_id>")
def api_batch_status(job_id):
    with _batch_lock:
        job = _batch_jobs.get(job_id)
    if job is None:
        return jsonify(error="job not found"), 404
    return jsonify(job)


# ── API: results list ─────────────────────────────────────────────────────────
@metal_cap_bp.get("/api/metal_cap/results")
def api_results():
    defect_type = request.args.get("defect_type", "")
    limit = int(request.args.get("limit", 40))

    pattern = str(RESULTS_ROOT / "**" / "*.png")
    files = sorted(glob.glob(pattern, recursive=True), reverse=True)

    out = []
    for fpath in files:
        p = Path(fpath)
        if p.name.startswith("debug_"):
            continue
        if defect_type and defect_type.lower() not in p.name.lower():
            continue
        out.append({"path": fpath, "filename": p.name})
        if len(out) >= limit:
            break

    return jsonify(results=out, total=len(out))

