"""
engines/api.py — FastAPI Standalone Generation Server
======================================================

Exposes the router engine as HTTP endpoints.
Runs independently from webapp/app.py — no conflict with Oanh's Flask work.

Usage:
    cd defect_dataset_generator
    uvicorn engines.api:app --port 8001 --reload

Endpoints:
    GET  /health
    GET  /api/default-engine?defect_type=dent&material=metal
    POST /api/generate/preview
    POST /api/generate/batch
    GET  /api/generate/status/{job_id}
"""

from __future__ import annotations

import asyncio
import sys
import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── sys.path: allow scripts/ imports when running from defect_dataset_generator/
_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
if os.path.abspath(_SCRIPTS) not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCRIPTS))

from .core.router_engine import route, get_default_engine
from .utils import decode_b64, decode_b64_gray

# ── Lazy engine imports (only load when first used) ───────────────────────────
_cap_generate        = None
_cap_detect_circle   = None
_pharma_generate     = None
_pharma_auto_mask    = None
# Metal Cap: 3 separate engines (one per defect type)
_mc_deform_generate      = None
_ring_fracture_generate  = None
_scratch_napchai_generate = None

def _load_cap():
    global _cap_generate, _cap_detect_circle
    from .metal_cap.mka_cap_engine import generate as _g
    from .metal_cap.cap_engine import detect_circle_info as _d
    _cap_generate = _g; _cap_detect_circle = _d

def _load_pharma():
    global _pharma_generate, _pharma_auto_mask
    from .pharma.capsule_engine import generate as _g, auto_mask as _a
    _pharma_generate = _g; _pharma_auto_mask = _a

def _load_mc_deform():
    global _mc_deform_generate
    from .metal_cap.mc_deform_engine import generate as _g
    _mc_deform_generate = _g

def _load_ring_fracture():
    global _ring_fracture_generate
    from .metal_cap.ring_fracture_engine import generate as _g
    _ring_fracture_generate = _g

def _load_scratch_napchai():
    global _scratch_napchai_generate
    from .metal_cap.scratch_napchai_engine import generate as _g
    _scratch_napchai_generate = _g


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="HondaPlus Defect Generation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev — restrict to server IP in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Defect type groups (for ref validation) ───────────────────────────────────

# Shape defects never need a ref image (ControlNet depth only).
_SHAPE_TYPES = {"dent", "bulge"}

# Appearance defects that require ref_image_b64 on BOTH CV and GenAI paths.
# Without ref, CV falls back to shaded_warp which produces a dent, not the correct defect.
_NEEDS_REF_CV = {"scratch", "crack", "rust", "foreign", "chip"}


# ── In-memory job store ───────────────────────────────────────────────────────

# job_store[job_id] = {
#   "status":   "queued" | "running" | "done" | "error",
#   "progress": int 0-100,
#   "total":    int,
#   "results":  list[str],   # base64 PNGs, filled when done
#   "engine":   str,
#   "error":    str | None,
# }
_job_store: dict[str, dict] = {}


# ── Pydantic request models ───────────────────────────────────────────────────

class PreviewRequest(BaseModel):
    base_image:      str            # base64 PNG, RGB
    mask:            str            # base64 PNG, grayscale
    defect_type:     str            # scratch | crack | dent | bulge | chip | rust | burn | ...
    material:        str            # metal | plastic | pharma
    intensity:       float = 0.6   # 0.0 – 1.0
    naturalness:     float = 0.7   # 0.0 – 1.0
    position_jitter: float = 0.0   # 0.0 – 1.0
    engine_override: Optional[str] = None   # "cv" | "genai" | null
    ref_image_b64:   Optional[str] = None   # required for appearance defects on genai path
    seed:            Optional[int] = None


class BatchRequest(BaseModel):
    base_image:      str
    mask:            str
    defect_type:     str
    material:        str
    count:           int   = 10
    intensity:       float = 0.6
    naturalness:     float = 0.7
    position_jitter: float = 0.0
    engine_override: Optional[str] = None
    ref_image_b64:   Optional[str] = None
    seed:            Optional[int] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_ref(defect_type: str, engine_override: Optional[str],
                  material: str, ref_b64: Optional[str]) -> None:
    """
    Raise 422 if an appearance defect is missing ref_image_b64.
    Both CV and GenAI paths require ref for appearance defects:
      - GenAI: IP-Adapter needs NG reference image
      - CV:    signal_injection/ref_paste need ref; without it falls back to
               shaded_warp which generates a dent, not the correct defect type
    """
    if defect_type in _SHAPE_TYPES:
        return  # dent/bulge: ControlNet depth only, no ref needed

    if ref_b64:
        return  # ref provided — all good

    # Determine which engine will actually run
    if engine_override in ("cv", "genai"):
        engine = engine_override
    else:
        engine = get_default_engine(defect_type, material)

    if engine == "genai":
        raise HTTPException(
            status_code=422,
            detail=(
                f"defect_type='{defect_type}' on material='{material}' routes to GenAI "
                f"(poisson_ipadapter pipeline). ref_image_b64 is required — "
                f"IP-Adapter needs a NG reference image. "
                f"Upload a cropped NG image or switch engine_override to 'cv'."
            ),
        )

    if engine == "cv" and defect_type in _NEEDS_REF_CV:
        raise HTTPException(
            status_code=422,
            detail=(
                f"defect_type='{defect_type}' on CV engine requires ref_image_b64. "
                f"Without a reference image the engine falls back to shaded_warp "
                f"(dent effect) which is incorrect for '{defect_type}'. "
                f"Upload a cropped NG image as reference."
            ),
        )


def _build_params(req, seed_offset: int = 0) -> dict:
    params = {
        "intensity":       req.intensity,
        "naturalness":     req.naturalness,
        "position_jitter": req.position_jitter,
    }
    if req.seed is not None:
        params["seed"] = req.seed + seed_offset
    if req.ref_image_b64:
        params["ref_image_b64"] = req.ref_image_b64
    return params


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name   = torch.cuda.get_device_name(0)
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory // (1024**2)} MB"
        else:
            gpu_name   = None
            gpu_memory = None
    except Exception:
        gpu_available = False
        gpu_name      = None
        gpu_memory    = None
    return {
        "status":        "ok",
        "gpu_available": gpu_available,
        "gpu_name":      gpu_name,
        "gpu_memory":    gpu_memory,
    }


@app.get("/api/default-engine")
def default_engine(defect_type: str, material: str):
    """
    Returns the default engine for a (defect_type, material) combination.
    Oanh calls this when user changes the defect/material dropdown to auto-set
    the CV/GenAI toggle without user having to manually choose.
    """
    engine = get_default_engine(defect_type, material)
    return {"engine": engine, "defect_type": defect_type, "material": material}


@app.post("/api/generate/preview")
def preview(req: PreviewRequest):
    """
    Generate a single defect image synchronously.
    Returns immediately with the result image as base64 PNG.
    CV path: < 1 second. GenAI path: ~3-10 seconds.
    """
    _validate_ref(req.defect_type, req.engine_override, req.material, req.ref_image_b64)

    base_img = decode_b64(req.base_image)
    mask     = decode_b64_gray(req.mask)       # grayscale — must use decode_b64_gray
    params   = _build_params(req)

    result = route(
        base_image      = base_img,
        mask            = mask,
        defect_type     = req.defect_type,
        material        = req.material,
        params          = params,
        engine_override = req.engine_override,
    )

    return {
        "result_image":    result["result_image"],
        "result_pre_refine": result.get("result_pre_refine"),  # pre-SDXL image (CV only)
        "engine_used":     result["engine"],
        "metadata":        result.get("metadata", {}),
    }


@app.post("/api/generate/batch")
async def start_batch(req: BatchRequest):
    """
    Start a batch generation job in the background.
    Returns job_id immediately. Poll /api/generate/status/{job_id} for progress.
    """
    _validate_ref(req.defect_type, req.engine_override, req.material, req.ref_image_b64)

    job_id = str(uuid.uuid4())
    _job_store[job_id] = {
        "status":   "queued",
        "progress": 0,
        "total":    req.count,
        "results":  [],
        "engine":   None,
        "error":    None,
    }

    # Start background task (non-blocking)
    asyncio.create_task(_run_batch(job_id, req))

    return {"job_id": job_id}


@app.get("/api/generate/progress")
def get_progress():
    """
    Returns live GenAI inference progress (step counter updated by deep_generative).
    Used by webapp to show step N/total during SDXL generation.
    """
    try:
        from .core.deep_generative import _gen_progress
        return _gen_progress
    except Exception:
        return {"status": "idle", "queued": 0, "defect_type": "",
                "step": 0, "total_steps": 0}


@app.get("/api/generate/status/{job_id}")
def get_status(job_id: str):
    """
    Poll batch job status.
    - status: "queued" | "running" | "done" | "error"
    - results: list of base64 PNGs (empty while running, filled when done)
    """
    job = _job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


# ═══════════════════════════════════════════════════════════════════════════════
# Cap / Pharma / Metal Cap endpoints — forward from Flask webapp to this server
# ═══════════════════════════════════════════════════════════════════════════════

from pydantic import BaseModel as _BM

class _CapPreviewReq(_BM):
    image_b64:   str
    mask_b64:    Optional[str] = None
    defect_type: str
    product:     str = "mka"
    params:      dict = {}

class _PharmaPreviewReq(_BM):
    image_b64:   str
    mask_b64:    Optional[str] = None
    ref_image_b64: Optional[str] = None
    product:     str = "round_tablet"
    defect_type: str = "crack"
    params:      dict = {}

@app.post("/api/cap/preview")
def cap_preview(req: _CapPreviewReq):
    _load_cap()
    import traceback as _tb
    try:
        result = _cap_generate(
            base_image_b64=req.image_b64,
            defect_type=req.defect_type,
            params=req.params,
            mask_b64=req.mask_b64,
        )
    except Exception as _e:
        _tb.print_exc()
        raise HTTPException(status_code=500, detail=f"Unhandled exception: {_e}")
    if "error" in result:
        print(f"[cap_preview] error: {result['error']}")
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/api/cap/detect-circle")
def cap_detect_circle(body: dict):
    _load_cap()
    img_b64 = body.get("image_b64") or body.get("base_image")
    if not img_b64:
        raise HTTPException(status_code=400, detail="image_b64 required")
    result = _cap_detect_circle(img_b64)
    return result


@app.post("/api/pharma/auto-mask")
def pharma_auto_mask(body: dict):
    _load_pharma()
    img_b64 = body.get("image_b64") or body.get("base_image")
    if not img_b64:
        raise HTTPException(status_code=400, detail="image_b64 required")
    result = _pharma_auto_mask(img_b64)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/api/pharma/preview")
def pharma_preview(req: _PharmaPreviewReq):
    _load_pharma()
    import traceback as _tb
    try:
        result = _pharma_generate(
            base_image_b64=req.image_b64,
            mask_b64=req.mask_b64,
            defect_type=req.defect_type,
            params=req.params,
            ref_image_b64=req.ref_image_b64,
        )
    except Exception as _e:
        _tb.print_exc()
        raise HTTPException(status_code=500, detail=f"Unhandled exception: {_e}")
    if "error" in result:
        print(f"[pharma_preview] error: {result['error']}")
        raise HTTPException(status_code=500, detail=result["error"])
    return result


class _MetalCapPreviewReq(_BM):
    image_b64:   str
    mask_b64:    Optional[str] = None
    defect_type: str = "mc_deform"   # "mc_deform" | "ring_fracture" | "scratch"
    params:      dict = {}


_METAL_CAP_LOADERS = {
    "mc_deform":     (_load_mc_deform,       lambda: _mc_deform_generate),
    "ring_fracture": (_load_ring_fracture,   lambda: _ring_fracture_generate),
    "scratch":       (_load_scratch_napchai, lambda: _scratch_napchai_generate),
}

# Other engine (unclassified defects)
_other_generate = None

def _load_other():
    global _other_generate
    if _other_generate is None:
        from .other.other_engine import generate as _g
        _other_generate = _g


class _OtherPreviewReq(_BM):
    image_b64:     str
    mask_b64:      Optional[str] = None
    ref_image_b64: Optional[str] = None
    ref_mask_b64:  Optional[str] = None
    params:        dict = {}


@app.post("/api/other/preview")
def other_preview(req: _OtherPreviewReq):
    import traceback as _tb
    _load_other()
    try:
        result = _other_generate(
            base_image_b64=req.image_b64,
            mask_b64=req.mask_b64,
            ref_image_b64=req.ref_image_b64,
            ref_mask_b64=req.ref_mask_b64,
            params=req.params or {},
        )
    except Exception as _e:
        _tb.print_exc()
        raise HTTPException(status_code=500, detail=f"Unhandled exception: {_e}")
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=str(result["error"]))
    return result


@app.post("/api/metal_cap/preview")
def metal_cap_preview(req: _MetalCapPreviewReq):
    import traceback as _tb
    print(f"[metal_cap_preview] Request --- Defect: {req.defect_type}, Has Mask: {bool(req.mask_b64)}")
    
    entry = _METAL_CAP_LOADERS.get(req.defect_type)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown defect_type '{req.defect_type}'. "
                   f"Valid: {list(_METAL_CAP_LOADERS)}",
        )

    loader, getter = entry
    loader()   # lazy-load the specific engine
    generate_fn = getter()

    try:
        result = generate_fn(
            base_image_b64=req.image_b64,
            params=req.params,
            mask_b64=req.mask_b64,
        )
    except Exception as _e:
        _tb.print_exc()
        raise HTTPException(status_code=500, detail=f"Unhandled exception: {_e}")
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


# ── Background batch runner ───────────────────────────────────────────────────

async def _run_batch(job_id: str, req: BatchRequest):
    job = _job_store[job_id]
    job["status"] = "running"

    try:
        base_img = decode_b64(req.base_image)
        mask     = decode_b64_gray(req.mask)
        results  = []

        for i in range(req.count):
            params = _build_params(req, seed_offset=i)

            r = route(
                base_image      = base_img,
                mask            = mask,
                defect_type     = req.defect_type,
                material        = req.material,
                params          = params,
                engine_override = req.engine_override,
            )

            results.append(r["result_image"])
            job["engine"]   = r["engine"]
            job["progress"] = int((i + 1) / req.count * 100)
            job["results"]  = results

            await asyncio.sleep(0)   # yield so server stays responsive during batch

        job["status"] = "done"
        job["progress"] = 100

    except Exception as e:
        job["status"] = "error"
        job["error"]  = str(e)
