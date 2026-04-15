"""
webapp/routes/other_api.py — "Other" (Unclassified) Defect API
================================================================

Endpoints:
  POST /api/other/preview          → generate 1 image (reference-based)
  POST /api/other/batch            → start batch job (background thread)
  GET  /api/other/batch/<job_id>   → poll batch progress
  GET  /api/other/ok-images        → list OK images for preview strip
  POST /api/other/save             → save result to disk
  GET  /api/other/results          → list saved result files

Unlike cap/pharma/metal_cap, this pipeline does NOT hardcode defect types.
It uses a reference NG image + user-drawn mask to generate similar defects.
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

# ── Locate engines ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent  # defect_dataset_generator/
sys.path.insert(0, str(_ROOT))

from utils import engine_post as _engine_post

try:
    from engines.other.other_engine import generate as _other_generate_local
    _HAS_LOCAL_OTHER = True
except Exception:
    _HAS_LOCAL_OTHER = False

other_bp = Blueprint("other_api", __name__)

# ── Data directories ──────────────────────────────────────────────────────────
_DEFECT_SAMPLES = Path(os.environ.get(
    "PHARMA_DATA_ROOT",
    str(_ROOT.parent.parent / "defect_samples"),
))

OTHER_ROOT   = _DEFECT_SAMPLES / "Other"
RESULTS_ROOT = _DEFECT_SAMPLES / "results" / "other"

# ── Batch job store ───────────────────────────────────────────────────────────
_batch_jobs: dict = {}
_batch_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok_images_for(category: str) -> list[Path]:
    ok_dir = OTHER_ROOT / category / "ok"
    files: list[Path] = []
    for ext in ["*.png", "*.jpg", "*.bmp"]:
        files += sorted(ok_dir.glob(ext))
    return files


def _ng_images_for(category: str) -> list[Path]:
    ng_dir = OTHER_ROOT / category / "ng"
    files: list[Path] = []
    for ext in ["*.png", "*.jpg", "*.bmp"]:
        files += sorted(ng_dir.glob(ext))
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


# ── API: categories list ──────────────────────────────────────────────────────

@other_bp.get("/api/other/categories")
def api_categories():
    """List available categories under defect_samples/Other/."""
    out = {}
    if OTHER_ROOT.exists():
        for d in sorted(OTHER_ROOT.iterdir()):
            if d.is_dir():
                ok_dir = d / "ok"
                ng_dir = d / "ng"
                n_ok = len(list(ok_dir.glob("*.png")) + list(ok_dir.glob("*.jpg"))) if ok_dir.exists() else 0
                n_ng = len(list(ng_dir.glob("*.png")) + list(ng_dir.glob("*.jpg"))) if ng_dir.exists() else 0
                out[d.name] = {"display": d.name, "n_ok": n_ok, "n_ng": n_ng}
    return jsonify(out)


# ── API: OK images ────────────────────────────────────────────────────────────

@other_bp.get("/api/other/ok-images")
def api_ok_images():
    category = request.args.get("category", "")
    files = _ok_images_for(category)
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


# ── API: NG images ────────────────────────────────────────────────────────────

@other_bp.get("/api/other/ng-images")
def api_ng_images():
    category = request.args.get("category", "")
    files = _ng_images_for(category)
    out = []
    for f in files[:20]:
        ng_bgr = cv2.imread(str(f))
        if ng_bgr is None:
            continue
        h, w = ng_bgr.shape[:2]
        th = 80
        tw = int(w * th / h)
        thumb = cv2.resize(ng_bgr, (tw, th))
        _, buf = cv2.imencode(".png", thumb)
        out.append({
            "filename": f.name,
            "path":     str(f),
            "thumb_b64": base64.b64encode(buf).decode("utf-8"),
        })
    return jsonify(images=out, total=len(files))


# ── API: load full image ─────────────────────────────────────────────────────

@other_bp.get("/api/other/load-image")
def api_load_image():
    from flask import send_file
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        return jsonify(error="File not found"), 404
    try:
        Path(path).resolve().relative_to(_DEFECT_SAMPLES.resolve())
    except ValueError:
        return jsonify(error="Path not allowed"), 403
    return send_file(path)


# ── API: preview ──────────────────────────────────────────────────────────────

@other_bp.post("/api/other/preview")
def api_preview():
    """
    Generate 1 defect image using reference-based approach.

    Body:
    {
      "image_b64":     str,       # base64 PNG OK image
      "mask_b64":      str|null,  # user-drawn mask (null = auto center)
      "ref_image_b64": str|null,  # NG reference image (null = darkening)
      "params": {
        "seed":       int,
        "intensity":  float (0-1),
        "use_genai":  bool (default true)
      }
    }
    """
    body = request.get_json(force=True, silent=True) or {}
    img_b64     = body.get("image_b64") or body.get("base_image")
    mask_b64    = body.get("mask_b64")
    ref_b64     = body.get("ref_image_b64")
    params      = body.get("params", {})

    if not img_b64:
        return jsonify(error="image_b64 required"), 400

    # Try remote engine first
    result = _engine_post("/api/other/preview", {
        "image_b64":     img_b64,
        "mask_b64":      mask_b64,
        "ref_image_b64": ref_b64,
        "params":        params,
    })

    # Fallback to local engine
    if result.get("_fallback") and _HAS_LOCAL_OTHER:
        result = _other_generate_local(
            base_image_b64=img_b64,
            mask_b64=mask_b64,
            ref_image_b64=ref_b64,
            params=params,
        )

    if "error" in result:
        return jsonify(result), 500

    # Build debug panel if not already present
    if not result.get("debug_panel"):
        try:
            ok_bgr  = _b64_to_bgr(img_b64)
            res_bgr = _b64_to_bgr(result.get("result_image", ""))
            m_b64   = result.get("mask_b64", "")
            m_bgr   = _b64_to_bgr(m_b64)
            m_gray  = m_bgr[:, :, 0] if m_bgr is not None else np.zeros(ok_bgr.shape[:2], np.uint8)
            panel   = _make_debug_panel(ok_bgr, m_gray, res_bgr)
            result["debug_panel"] = _img_to_b64(panel)
        except Exception:
            result["debug_panel"] = None

    return jsonify(result)


# ── API: save ─────────────────────────────────────────────────────────────────

@other_bp.post("/api/other/save")
def api_save():
    body       = request.get_json(force=True, silent=True) or {}
    result_b64 = body.get("result_b64") or body.get("result_image")
    category   = body.get("category", "unknown")
    filename   = body.get("filename", "")

    if not result_b64:
        return jsonify(error="result_b64 required"), 400

    from datetime import datetime
    out_dir = RESULTS_ROOT / "manual" / category
    out_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"other_{category}_{ts}.png"

    res_bgr = _b64_to_bgr(result_b64)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), res_bgr)
    return jsonify(saved=True, path=str(out_path), filename=filename)


# ── API: batch ────────────────────────────────────────────────────────────────

def _batch_worker(job_id: str, payload: dict):
    with _batch_lock:
        _batch_jobs[job_id]["status"] = "running"

    category    = payload.get("category", "default")
    params_base = payload.get("params", {})
    n_images    = int(payload.get("n_images", 10))
    ref_b64     = payload.get("ref_image_b64")
    mask_b64    = payload.get("mask_b64")

    ok_files = _ok_images_for(category)
    if not ok_files:
        with _batch_lock:
            _batch_jobs[job_id].update(status="error", error="No OK images found")
        return

    import itertools
    from datetime import datetime

    out_dir = RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S") / category
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

        if _HAS_LOCAL_OTHER:
            result = _other_generate_local(
                base_image_b64=img_b64,
                mask_b64=mask_b64,
                ref_image_b64=ref_b64,
                params=params,
            )
        else:
            result = {"error": "No local engine available"}

        if "error" in result:
            errors += 1
        else:
            s_tag = f"_s{params['seed']}"
            fname = f"other_{ok_path.stem}{s_tag}.png"
            res_bgr = _b64_to_bgr(result.get("result_image", ""))
            if res_bgr is not None:
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


@other_bp.post("/api/other/batch")
def api_batch():
    body = request.get_json(force=True, silent=True) or {}

    job_id = str(uuid.uuid4())[:8]
    with _batch_lock:
        _batch_jobs[job_id] = {
            "status":    "queued",
            "generated": 0,
            "errors":    0,
            "total":     int(body.get("n_images", 10)),
            "progress":  0,
        }

    t = threading.Thread(target=_batch_worker, args=(job_id, body), daemon=True)
    t.start()
    return jsonify(job_id=job_id, status="queued")


@other_bp.get("/api/other/batch/<job_id>")
def api_batch_status(job_id):
    with _batch_lock:
        job = _batch_jobs.get(job_id)
    if job is None:
        return jsonify(error="job not found"), 404
    return jsonify(job)


# ── API: results list ─────────────────────────────────────────────────────────

@other_bp.get("/api/other/results")
def api_results():
    limit = int(request.args.get("limit", 40))
    pattern = str(RESULTS_ROOT / "**" / "*.png")
    files = sorted(glob.glob(pattern, recursive=True), reverse=True)

    out = []
    for fpath in files:
        p = Path(fpath)
        if p.name.startswith("debug_"):
            continue
        out.append({"path": fpath, "filename": p.name})
        if len(out) >= limit:
            break

    return jsonify(results=out, total=len(out))
