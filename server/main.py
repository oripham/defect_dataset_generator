"""
server/main.py — Entry point alias for local dev
=================================================
Delegates to engines/api.py which has all endpoints including /api/metal_cap/preview.

Usage (from repo root):
    cd /d v:\\HondaPlus\\defect_dataset_generator
    uvicorn server.main:app --host 0.0.0.0 --port 8001 --reload

Or directly:
    uvicorn engines.api:app --host 0.0.0.0 --port 8001 --reload
"""
import sys
import os

# Ensure repo root is in path so "engines.*" imports work
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Re-export the FastAPI app from engines/api.py
from engines.api import app  # noqa: F401  — this is the ASGI app uvicorn loads
