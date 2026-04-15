import os
import re

app_file = r"d:\HondaPlus\defect_dataset_generator\webapp\app.py"
with open(app_file, "r", encoding="utf-8") as f:
    text = f.read()

# We want to split app.py into:
# - core.py (globals, state proxy, helpers)
# - routes_pages.py (pages)
# - routes_api.py (all api endpoints)
# - app.py (just the setup, session stuff, and blueprint registration)

# By inspection, lines until `@app.route('/')` are setup and helpers.
# Let's split using regex for page routes.

core_content = """import os, json, time, threading, zipfile, io, base64
from pathlib import Path
import requests as req_lib
from flask import g, request, jsonify, send_file
from werkzeug.local import LocalProxy
from gui.app_state import AppState, ClassConfig
from gui.i18n import TRANSLATIONS

state = LocalProxy(lambda: getattr(g, 'state', None))

_preview_jobs = {}
_preview_lock = threading.Lock()
_current_job = {}
_poll_lock = threading.Lock()
_poll_thread = None

def _auth_headers() -> dict:
    if state.api_key:
        return {'Authorization': f'Bearer {state.api_key}'}
    return {}

def _server_url(path: str) -> str:
    base = state.server_url.rstrip('/')
    url = f'{base}{path}'
    if state.api_key and 'runpod.net' in base:
        sep = '&' if '?' in url else '?'
        url = f'{url}{sep}runpodApiKey={state.api_key}'
    return url

def T(key: str, **kwargs) -> str:
    lang = state.language
    table = TRANSLATIONS.get(lang, TRANSLATIONS['ja'])
    text = table.get(key) or TRANSLATIONS['ja'].get(key) or key
    if kwargs:
        try: return text.format(**kwargs)
        except: return text
    return text

def _auto_mask_dir(cls) -> str:
    if cls.ref_dir: return str(Path(cls.ref_dir).parent / 'mask')
    return ''

def _ensure_mask_dir(cls):
    if not cls.mask_dir and cls.ref_dir:
        cls.mask_dir = _auto_mask_dir(cls)

def _has_masks(cls) -> bool:
    _ensure_mask_dir(cls)
    if not cls.mask_dir: return False
    p = Path(cls.mask_dir)
    return p.exists() and any(p.rglob('mask_*.png'))

def _has_cropped_refs(cls) -> bool:
    if not cls.ref_dir: return False
    p = Path(cls.ref_dir).parent / 'cropped'
    return p.exists() and any(p.iterdir())
"""

# Actually, doing this via script is risky if I don't catch all helpers.
print("Script initialized.")
