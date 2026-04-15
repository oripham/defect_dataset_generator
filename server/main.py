"""
Defect Dataset Generator - FastAPI Server
Runs inside Docker container with GPU access.
Exposes HTTP API consumed by the GUI.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Dict

import asyncio

import numpy as np
import yaml
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Defect Dataset Generator Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
WORKSPACE.mkdir(parents=True, exist_ok=True)

# In-memory job store: job_id -> {status, logs, output_dir}
jobs: Dict[str, dict] = {}

# ── Engine imports (available after Docker COPY engines/ ./engines/) ─────────

try:
    from engines.router_engine import route as _route, get_default_engine
    from engines.utils import decode_b64, decode_b64_gray
    _ENGINES_AVAILABLE = True
except ImportError as _e:
    _ENGINES_AVAILABLE = False
    print(f"[WARN] engines not available: {_e}")

# In-memory store for inline generate jobs (preview/batch)
gen_jobs: Dict[str, dict] = {}


# ── Pydantic request models ───────────────────────────────────────────────────

class PreviewRequest(BaseModel):
    base_image:       str            # base64 PNG (RGB)
    mask:             str            # base64 PNG (grayscale)
    defect_type:      str
    material:         str
    intensity:        float = 0.6
    naturalness:      float = 0.6
    position_jitter:  float = 0.0
    seed:             Optional[int]   = None
    ref_image_b64:    Optional[str]   = None
    engine_override:  Optional[str]   = None
    prompts:          Optional[list]  = None
    negative_prompt:  Optional[str]   = None
    # Advanced Parameters (override formula defaults in deep_generative)
    strength:         Optional[float] = None
    guidance_scale:   Optional[float] = None
    steps:            Optional[int]   = None
    ip_scale:         Optional[float] = None
    controlnet_scale: Optional[float] = None
    inject_alpha:     Optional[float] = None
    epsilon_factor:   Optional[float] = None


class BatchRequest(BaseModel):
    base_image:       str
    mask:             str
    defect_type:      str
    material:         str
    count:            int   = 20
    intensity:        float = 0.6
    naturalness:      float = 0.6
    position_jitter:  float = 0.0
    seed:             Optional[int]   = None
    ref_image_b64:    Optional[str]   = None
    engine_override:  Optional[str]   = None
    prompts:          Optional[list]  = None
    negative_prompt:  Optional[str]   = None
    # Advanced Parameters
    strength:         Optional[float] = None
    guidance_scale:   Optional[float] = None
    steps:            Optional[int]   = None
    ip_scale:         Optional[float] = None
    controlnet_scale: Optional[float] = None
    inject_alpha:     Optional[float] = None
    epsilon_factor:   Optional[float] = None


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "none"
        gpu_mem = (
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            if gpu_ok else "N/A"
        )
    except Exception as e:
        gpu_ok, gpu_name, gpu_mem = False, f"error: {e}", "N/A"

    return {
        "status": "ok",
        "gpu_available": gpu_ok,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_mem,
    }


@app.get("/api/generate/progress")
def get_gen_progress():
    """Return current GenAI inference progress (status, step, queued count)."""
    if _ENGINES_AVAILABLE:
        try:
            from engines.deep_generative import _gen_progress
            return dict(_gen_progress)
        except Exception:
            pass
    return {"status": "idle", "queued": 0, "step": 0, "total_steps": 0, "defect_type": ""}


# --------------------------------------------------------------------------- #
# Job management
# --------------------------------------------------------------------------- #

def _job_dir(job_id: str) -> Path:
    return WORKSPACE / "jobs" / job_id


def _log(job_id: str, line: str):
    entry = jobs[job_id]["logs"]
    entry.append(line)
    # Keep last 2000 lines
    if len(entry) > 2000:
        jobs[job_id]["logs"] = entry[-2000:]


@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    dataset_zip: UploadFile = File(
        ..., description="Zip containing good_images/, mask_root/, defect_refs/"
    ),
    config_json: str = Form(
        ..., description="JSON with model/device/classes settings"
    ),
):
    """
    Create a new generation job.
    Returns {job_id}.
    """
    job_id = str(uuid.uuid4())[:8]
    jdir = _job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    jobs[job_id] = {"status": "uploading", "logs": [], "output_dir": str(jdir / "output")}

    # --- Save & extract zip ---
    zip_path = jdir / "upload.zip"
    zip_path.write_bytes(await dataset_zip.read())

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(jdir)
        zip_path.unlink()
    except Exception as e:
        jobs[job_id]["status"] = "error"
        _log(job_id, f"[ERROR] Failed to extract zip: {e}")
        raise HTTPException(status_code=400, detail=f"Bad zip file: {e}")

    # --- Parse config ---
    try:
        cfg_data = json.loads(config_json)
    except json.JSONDecodeError as e:
        jobs[job_id]["status"] = "error"
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")

    # --- Build config.yaml with server-side paths ---
    # Inject default signal_injection params into each class if not set
    classes = cfg_data.get("classes", [])
    for cls in classes:
        gen = cls.setdefault("generation", {})
        gen.setdefault("method", "signal_injection")
        # Common alpha params
        gen.setdefault("alpha_blur",   31)
        gen.setdefault("alpha_dilate",  5)
        # Method-specific defaults
        if gen["method"] == "shaded_warp":
            gen.setdefault("warp_strength", 25.0)
            gen.setdefault("warp_mode",     "dent")
            gen.setdefault("shading_gain",  110.0)
            gen.setdefault("amplitude",     0.8)
            gen.setdefault("normal_scale",  10.0)
            gen.setdefault("alpha_blur",    31)
            gen.setdefault("alpha_dilate",  10)
        elif gen["method"] == "elastic_warp":
            gen.setdefault("warp_strength", 8.0)
            gen.setdefault("warp_mode",     "dent")
            gen.setdefault("warp_blur",     31)
        elif gen["method"] == "ref_paste":
            gen.setdefault("blend_strength", 0.85)
        elif gen["method"] == "bright_inject":
            gen.setdefault("brightness_boost", 90.0)
        elif gen["method"] == "polar_paste":
            gen.setdefault("align_mode",       "tangent")
            gen.setdefault("band_width",        20)
            gen.setdefault("blend_mode",       "seamless")
            gen.setdefault("blend_levels",       4)
            gen.setdefault("brightness_boost",  80.0)
        else:  # signal_injection
            gen.setdefault("blur_kernel",   51)
            gen.setdefault("intensity_min", 0.80)
            gen.setdefault("intensity_max", 1.8)
            gen.setdefault("alpha_dilate",  10)

    config = {
        "paths": {
            "good_images": str(jdir / "good_images"),
            "mask_root":   str(jdir / "mask_root"),
            "defect_refs": str(jdir / "defect_refs"),
            "output_root": str(jdir / "output"),
        },
        "model": {
            "type":              cfg_data.get("generator_type", "classical"),
            "device":            cfg_data.get("device", "cuda"),
            "base_model":        cfg_data.get("model_name", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"),
            "controlnet_model":  "diffusers/controlnet-canny-sdxl-1.0",
            "ip_adapter_model":  "/models/models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729",
            "ip_adapter_weight": "ip-adapter_sdxl.bin",
            "ip_adapter_subfolder": "sdxl_models",
        },
        "product": {
            "image_size": [
                cfg_data.get("image_width", 1024),
                cfg_data.get("image_height", 1024),
            ],
        },
        "classes": classes,
        "sdxl_refine": {
            # src pipeline runs SDXL internally — skip refine to avoid double-processing
            "enabled":         cfg_data.get("generator_type", "classical") != "src",
            "strength":        0.14,
            "guidance_scale":  5.0,
            "steps":           20,
            "prompt":          cfg_data.get("sdxl_refine_prompt", None),
            "negative_prompt": cfg_data.get("sdxl_refine_negative", None),
            "model": "/models/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1"
                     "/snapshots/115134f363124c53c7d878647567d04daf26e41e",
        },
    }

    # Pre-create output_root so generate_dataset.py path check passes
    (jdir / "output").mkdir(parents=True, exist_ok=True)

    config_path = jdir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    jobs[job_id]["status"] = "pending"
    cls_names = [c["name"] for c in cfg_data.get("classes", [])]
    _log(job_id, f"[INFO] Job {job_id} created")
    _log(job_id, f"[INFO] Classes: {cls_names}")
    _log(job_id, f"[INFO] Device: {config['model']['device']}")
    _log(job_id, f"[INFO] Image size: {config['product']['image_size']}")

    background_tasks.add_task(_run_generation, job_id, config_path)

    return {"job_id": job_id}


def _run_generation(job_id: str, config_path: Path):
    jobs[job_id]["status"] = "running"
    _log(job_id, "[INFO] Starting generation pipeline...")

    script = Path("/app/scripts/generate_dataset.py")
    cmd = [sys.executable, str(script), "--config", str(config_path)]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd="/app",
        )
        for line in proc.stdout:
            _log(job_id, line.rstrip())
        proc.wait()

        if proc.returncode == 0:
            jobs[job_id]["status"] = "done"
            _log(job_id, "✅ Generation complete!")
        else:
            jobs[job_id]["status"] = "error"
            _log(job_id, f"❌ Process exited with code {proc.returncode}")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        _log(job_id, f"❌ Exception: {e}")


@app.get("/jobs/{job_id}/status")
def get_status(job_id: str, since: int = 0):
    """
    Poll job status.
    `since`: index of first new log line to return (pass log_count from previous call).
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    all_logs = job["logs"]
    return {
        "status": job["status"],
        "log_count": len(all_logs),
        "new_logs": all_logs[since:],
    }


@app.get("/jobs/{job_id}/results")
def get_results(job_id: str):
    """Download results as a zip file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = Path(jobs[job_id]["output_dir"])
    run_dirs = sorted(output_dir.glob("run_*")) if output_dir.exists() else []
    if not run_dirs:
        raise HTTPException(status_code=404, detail="No results yet")

    latest = run_dirs[-1]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(latest.rglob("*")):
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(latest))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=results_{job_id}.zip"
        },
    )


@app.get("/jobs")
def list_jobs():
    return {
        jid: {"status": j["status"], "log_lines": len(j["logs"])}
        for jid, j in jobs.items()
    }


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404)
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    del jobs[job_id]
    return {"deleted": job_id}


# --------------------------------------------------------------------------- #
# Inline Generate — Preview & Batch
# (uses engines/router_engine directly, no subprocess)
# --------------------------------------------------------------------------- #

def _check_engines():
    if not _ENGINES_AVAILABLE:
        raise HTTPException(status_code=503, detail="engines not available — rebuild Docker image")


@app.post("/api/generate/preview")
async def generate_preview(request: PreviewRequest, background_tasks: BackgroundTasks):
    """
    Async preview — returns job_id immediately to avoid proxy timeout (504).
    Client polls /api/generate/preview/status/{job_id} for result.
    """
    _check_engines()

    try:
        base_img = decode_b64(request.base_image)
        mask_arr = decode_b64_gray(request.mask)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode error: {e}")

    params = {
        "intensity":       request.intensity,
        "naturalness":     request.naturalness,
        "position_jitter": request.position_jitter,
        "seed":            request.seed,
        "ref_image_b64":   request.ref_image_b64,
        "prompts":         request.prompts,
        "negative_prompt": request.negative_prompt,
    }
    # Advanced Parameters — only add if explicitly provided (None = use formula default)
    for key in ("strength", "guidance_scale", "steps", "ip_scale",
                "controlnet_scale", "inject_alpha", "epsilon_factor"):
        val = getattr(request, key, None)
        if val is not None:
            params[key] = val

    job_id = str(uuid.uuid4())[:8]
    gen_jobs[job_id] = {"status": "queued", "result_image": None, "engine_used": None,
                        "metadata": {}, "error": None}

    print(f"[PREVIEW] job={job_id} defect={request.defect_type} engine={request.engine_override}",
          flush=True)
    background_tasks.add_task(
        _run_preview_job, job_id, base_img, mask_arr,
        request.defect_type, request.material, params, request.engine_override
    )
    return {"job_id": job_id}


def _run_preview_job(job_id, base_img, mask_arr, defect_type, material, params, engine_override):
    gen_jobs[job_id]["status"] = "running"
    try:
        result = _route(
            base_image=base_img, mask=mask_arr,
            defect_type=defect_type, material=material,
            params=params, engine_override=engine_override,
        )
        gen_jobs[job_id].update({
            "status":       "done",
            "result_image": result["result_image"],
            "engine_used":  result["engine"],
            "metadata":     result.get("metadata", {}),
        })
        print(f"[PREVIEW] job={job_id} done engine={result['engine']}", flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] preview job={job_id} failed:\n{tb}", flush=True)
        gen_jobs[job_id].update({"status": "error", "error": f"{type(e).__name__}: {e}"})


@app.get("/api/generate/preview/status/{job_id}")
def get_preview_status(job_id: str):
    """Poll preview job status. Returns result_image when done."""
    job = gen_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Preview job not found")
    return job


@app.post("/api/generate/batch")
async def generate_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Start a background batch generation job.
    Returns {job_id} immediately; poll /api/generate/status/{job_id}.
    """
    _check_engines()

    job_id = str(uuid.uuid4())[:8]
    gen_jobs[job_id] = {
        "status":   "queued",
        "progress": 0,
        "total":    request.count,
        "results":  [],
        "engine":   None,
        "error":    None,
    }
    background_tasks.add_task(_run_batch, job_id, request)
    return {"job_id": job_id}


@app.get("/api/generate/status/{job_id}")
def get_gen_status(job_id: str):
    """
    Poll batch job status.
    Oanh polls this every 2s to update the progress bar.
    """
    job = gen_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


async def _run_batch(job_id: str, request: BatchRequest):
    job = gen_jobs[job_id]
    job["status"] = "running"

    try:
        base_img = decode_b64(request.base_image)
        mask_arr = decode_b64_gray(request.mask)
    except Exception as e:
        job["status"] = "error"
        job["error"]  = f"Image decode error: {e}"
        return

    params = {
        "intensity":       request.intensity,
        "naturalness":     request.naturalness,
        "position_jitter": request.position_jitter,
        "ref_image_b64":   request.ref_image_b64,
    }
    for key in ("strength", "guidance_scale", "steps", "ip_scale",
                "controlnet_scale", "inject_alpha", "epsilon_factor"):
        val = getattr(request, key, None)
        if val is not None:
            params[key] = val

    results = []
    for i in range(request.count):
        # Per-image seed: base seed + index, or None
        if request.seed is not None:
            params["seed"] = request.seed + i
        else:
            params["seed"] = None

        try:
            r = _route(
                base_image      = base_img,
                mask            = mask_arr,
                defect_type     = request.defect_type,
                material        = request.material,
                params          = params,
                engine_override = request.engine_override,
            )
            results.append(r["result_image"])
            job["engine"]   = r["engine"]
            job["progress"] = int((i + 1) / request.count * 100)
            job["results"]  = results
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] batch img {i+1}/{request.count} failed:\n{tb}", flush=True)
            job["status"] = "error"
            job["error"]  = f"{type(e).__name__}: {e}"
            return

        await asyncio.sleep(0)   # yield so FastAPI can serve other requests

    job["status"]   = "done"
    job["progress"] = 100


# --------------------------------------------------------------------------- #
# Path validation (for Oanh's file browser replacement)
# --------------------------------------------------------------------------- #

@app.get("/api/validate-path")
def validate_path(path: str):
    """
    Check whether a directory path exists and is accessible.
    Oanh uses this to validate user-typed folder paths in the UI.

    Query param: ?path=/workspace/jobs/my_job/good_images
    Response: { "valid": bool, "is_dir": bool, "message": str }
    """
    try:
        p = Path(path)
        if p.exists():
            return {"valid": True, "is_dir": p.is_dir(), "message": "Path exists"}
        else:
            return {"valid": False, "is_dir": False, "message": "Path does not exist"}
    except Exception as e:
        return {"valid": False, "is_dir": False, "message": str(e)}
