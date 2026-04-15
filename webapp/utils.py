import json
import os
import shutil
import time
from pathlib import Path
from flask import g, request
from werkzeug.local import LocalProxy

from gui.i18n import TRANSLATIONS
from gui.app_state import AppState

state = LocalProxy(lambda: getattr(g, 'state', None))

SESSION_DIR = Path('/tmp/hondaplus_uploads/sessions')

def cleanup_old_sessions(max_age_hours=24):
    """Auto-delete sessions and upload folders older than 24 hours."""
    try:
        if not SESSION_DIR.exists(): return
        now = time.time()
        for sf in SESSION_DIR.glob('*.json'):
            if now - sf.stat().st_mtime > max_age_hours * 3600:
                sid = sf.stem
                sf.unlink(missing_ok=True)
                sess_dir = Path(f'/tmp/hondaplus_uploads/{sid}')
                if sess_dir.exists():
                    shutil.rmtree(sess_dir, ignore_errors=True)
    except Exception as e:
        print(f"[Garbage Collection] Failed to cleanup sessions: {e}")

# Run cleanup gracefully once on server import
import threading
threading.Thread(target=cleanup_old_sessions, daemon=True).start()

def _get_session_id() -> str:
    """Read session ID from request header X-Session-ID or query parameter."""
    sid = request.headers.get('X-Session-ID') or request.args.get('sid')
    if not sid or len(sid) < 4:
        sid = request.cookies.get('sid', 'default')
    return ''.join(c for c in str(sid) if c in '0123456789abcdef')[:16]

def _auth_headers() -> dict:
    """Bearer token header for RunPod proxy auth."""
    if state.api_key:
        return {'Authorization': f'Bearer {state.api_key}'}
    return {}

def _server_url(path: str) -> str:
    """Build full URL. For RunPod proxy, append API key as query param."""
    base = state.server_url.rstrip('/')
    url = f'{base}{path}'
    if state.api_key and 'runpod.net' in base:
        sep = '&' if '?' in url else '?'
        url = f'{url}{sep}runpodApiKey={state.api_key}'
    return url

def engine_post(path: str, body: dict, timeout_s: int = 900) -> dict:
    """
    POST to RunPod engine server if server_url is configured.
    Falls back to returning {"_fallback": True} when no server_url set.

    Usage in route handlers:
        result = engine_post("/api/cap/preview", body)
        if result.get("_fallback"):
            result = _local_generate(...)   # local fallback
    """
    try:
        srv = state.server_url
    except Exception:
        srv = None
    if not srv:
        return {"_fallback": True}

    import requests as _req
    url = _server_url(path)
    try:
        # Default was 120s, but first-run model loading on weak servers can exceed it.
        resp = _req.post(
            url,
            json=body,
            headers=_auth_headers(),
            timeout=(15, int(timeout_s)),
            verify=False,
        )
        # 404 = endpoint not deployed on RunPod yet → fall back to local engine
        if resp.status_code == 404:
            return {"_fallback": True}
        # Surface error details to UI (FastAPI returns JSON {"detail": "..."}).
        if resp.status_code >= 400:
            try:
                j = resp.json()
                detail = j.get("detail") if isinstance(j, dict) else None
            except Exception:
                detail = None
            msg = detail or resp.text or f"HTTP {resp.status_code}"
            return {"error": f"Engine HTTP {resp.status_code}: {msg}"}
        return resp.json()
    except Exception as e:
        return {"error": f"Engine proxy error: {e}"}


def T(key: str, **kwargs) -> str:
    lang = state.language
    table = TRANSLATIONS.get(lang, TRANSLATIONS['ja'])
    text = table.get(key) or TRANSLATIONS['ja'].get(key) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text

def _auto_mask_dir(cls) -> str:
    """mask_dir is sibling to bad/ (ref_dir)."""
    if cls.ref_dir:
        return str(Path(cls.ref_dir).parent / 'mask')
    return ''

def _ensure_mask_dir(cls):
    """Set cls.mask_dir from ref_dir if not set."""
    if not cls.mask_dir and getattr(cls, 'ref_dir', None):
        cls.mask_dir = _auto_mask_dir(cls)

def _has_masks(cls) -> bool:
    _ensure_mask_dir(cls)
    if not getattr(cls, 'mask_dir', None):
        return False
    p = Path(cls.mask_dir)
    return p.exists() and any(p.rglob('mask_*.png'))

def _has_cropped_refs(cls) -> bool:
    if not getattr(cls, 'ref_dir', None):
        return False
    p = Path(cls.ref_dir).parent / 'cropped'
    return p.exists() and any(p.iterdir())

def _classes_list():
    return [
        {
            'name': c.name,
            'class_id': c.class_id,
            'mask_dir': getattr(c, 'mask_dir', '') or _auto_mask_dir(c),
            'ref_dir': getattr(c, 'ref_dir', ''),
            'num_images': getattr(c, 'num_images', 10),
            'prompts': getattr(c, 'prompts', []),
            'negative_prompt': getattr(c, 'negative_prompt', ''),
            'guidance_scale': getattr(c, 'guidance_scale', 7.0),
            'steps': getattr(c, 'steps', 20),
            'strength': getattr(c, 'strength', 0.5),
            'ip_scale': getattr(c, 'ip_scale', 0.5),
            'controlnet_scale': getattr(c, 'controlnet_scale', 0.5),
            'inject_alpha': getattr(c, 'inject_alpha', 0.5),
            'mask_rotation_min': getattr(c, 'mask_rotation_min', 0),
            'mask_rotation_max': getattr(c, 'mask_rotation_max', 0),
            'has_masks': _has_masks(c),
            'has_cropped_refs': _has_cropped_refs(c),
        }
        for c in state.classes
    ]
