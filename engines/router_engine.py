"""
engines/router_engine.py — Defect Generation Router
=====================================================

Receives a generation request and dispatches to the correct engine:

  CV engine   (fast_physics)     — simple defects, all materials, fast (<1s)
  GenAI engine (deep_generative) — complex metal defects, slower (~3s)

Routing priority:
  1. engine_override ("cv" | "genai") — set by user in UI toggle
  2. ROUTE_TABLE lookup (defect_type, material)
  3. Default fallback: "cv"
"""

from __future__ import annotations

import numpy as np

from .fast_physics    import generate as _cv_generate
from .deep_generative import generate as _genai_generate


# ── Routing Table ─────────────────────────────────────────────────────────────
# Key: (defect_type, material)  →  "cv" | "genai"
# Missing entries fall back to "cv" (safe default).

ROUTE_TABLE: dict[tuple[str, str], str] = {
    # ── Appearance defects — CV handles all materials ─────────────────────────
    ("scratch",     "metal"):   "cv",
    ("scratch",     "plastic"): "cv",
    ("scratch",     "pharma"):  "cv",

    ("foreign",     "metal"):   "genai",
    ("foreign",     "plastic"): "genai",
    ("foreign",     "pharma"):  "genai",

    ("chip",        "plastic"): "genai",
    ("chip",        "pharma"):  "genai",

    # ── Complex defects on shiny metal → GenAI ────────────────────────────────
    ("dent",        "metal"):   "cv",
    ("bulge",       "metal"):   "genai",
    ("chip",        "metal"):   "genai",
    # ── Shape defects on plastic/pharma → CV (GenAI not calibrated) ──────────
    ("dent",        "plastic"): "cv",
    ("dent",        "pharma"):  "cv",
    ("bulge",       "plastic"): "cv",
    ("bulge",       "pharma"):  "cv",
}


def get_default_engine(defect_type: str, material: str) -> str:
    """Return default engine for (defect_type, material) without running anything."""
    return ROUTE_TABLE.get((defect_type, material), "cv")


def route(
    base_image:      np.ndarray,
    mask:            np.ndarray,
    defect_type:     str,
    material:        str,
    params:          dict,
    engine_override: str | None = None,
) -> dict:
    """
    Route a generation request to the correct engine and return result.

    Parameters
    ----------
    base_image      : uint8 RGB np.ndarray (H, W, 3)
    mask            : uint8 grayscale np.ndarray (H, W), white = defect region
    defect_type     : "scratch" | "dent" | "bulge" | "chip" | "foreign"
    material        : "metal" | "plastic" | "pharma"
    params          : dict with keys:
                        intensity        float 0-1
                        naturalness      float 0-1
                        position_jitter  float 0-1
                        seed             int (optional)
                        ref_image_b64    str base64 PNG (optional)
    engine_override : "cv" | "genai" | None  — None means auto from table

    Returns
    -------
    dict: {
        "result_image": str,   # base64 PNG
        "engine":       str,   # "cv" or "genai"
        "metadata":     dict,
    }
    """
    # Priority: explicit user override > forced defaults > route table
    if engine_override in ("cv", "genai"):
        engine = engine_override
    elif defect_type == "scratch":
        engine = "cv"   # auto: scratch always CV
    else:
        engine = ROUTE_TABLE.get((defect_type, material), "cv")

    if engine == "genai":
        return _genai_generate(base_image, mask, defect_type, material, params)
    else:
        return _cv_generate(base_image, mask, defect_type, material, params)
