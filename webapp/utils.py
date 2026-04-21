import json
import os
import shutil
import time
from pathlib import Path
from flask import g, request
from werkzeug.local import LocalProxy
from gui.i18n import TRANSLATIONS

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
    POST to Engine server using urllib (more robust than requests in some venvs).
    Includes detailed diagnostic logging to the Flask terminal.
    """
    try:
        srv = state.server_url
    except Exception:
        srv = None
    
    if not srv:
        return {"_fallback": True}

    url = _server_url(path)
    # Ensure we use 127.0.0.1 instead of localhost to avoid IPv6 issues on Windows
    if "localhost" in url:
        url = url.replace("localhost", "127.0.0.1")

    print(f"\n[engine_post] >>> Calling Engine: {url}")
    print(f"[engine_post] >>> Payload keys: {list(body.keys())}")
    
    import urllib.request as _urlreq
    import urllib.error as _urlerr
    
    headers = _auth_headers()
    headers['Content-Type'] = 'application/json'
    
    req = _urlreq.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
    
    try:
        # We use a shorter connection timeout (10s) and a long read timeout (timeout_s)
        # Note: urllib.request.urlopen timeout is for the whole operation
        with _urlreq.urlopen(req, timeout=float(timeout_s)) as resp:
            status = resp.getcode()
            print(f"[engine_post] <<< Success: HTTP {status}")
            return json.loads(resp.read().decode('utf-8'))
            
    except _urlerr.HTTPError as e:
        status = e.code
        print(f"[engine_post] !!! HTTP Error {status}: {e.reason}")
        if status == 404:
            return {"_fallback": True}
        try:
            err_body = json.loads(e.read().decode('utf-8'))
            detail = err_body.get("detail") or err_body
        except Exception:
            detail = e.reason
        return {"error": f"Engine HTTP {status}: {detail}"}
        
    except _urlerr.URLError as e:
        print(f"[engine_post] !!! Connection Error: {e.reason}")
        return {"error": f"Engine Connection Error: {e.reason}"}
        
    except Exception as e:
        print(f"[engine_post] !!! Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
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
