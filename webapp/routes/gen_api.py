import os
import time
import uuid
import base64
import json
import random
import shutil
import threading
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file
import requests as req_lib

from utils import state, _get_session_id, _ensure_mask_dir, _has_masks, _auth_headers, _server_url


def _mask_to_yolo_polygons(mask):
    """Convert binary mask (H,W uint8) to list of YOLO normalized polygon coords."""
    try:
        import cv2
        import numpy as np
        h, w = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                continue
            poly = cnt.squeeze()
            if len(poly.shape) != 2:
                continue
            poly_norm = []
            for x, y in poly:
                poly_norm.extend([x / w, y / h])
            polygons.append(poly_norm)
        return polygons
    except Exception:
        return []

gen_bp = Blueprint('gen_api', __name__)

# ── GenAI default parameters (from scripts/generator.py) ──────────────────────
GENAI_DEFAULTS = {
    "strength":         0.28,
    "guidance_scale":   6.5,
    "steps":            30,
    "ip_scale":         0.6,
    "controlnet_scale": 0.35,
    "inject_alpha":     0.80,
}
# ──────────────────────────────────────────────────────────────────────────────

def _safe_log(resp_data: dict) -> str:
    """Safely serialize response dict for logging — skips non-serializable values."""
    try:
        filtered = {k: v for k, v in resp_data.items() if k not in ('result_image', 'image_b64')}
        return json.dumps(filtered, ensure_ascii=False, default=str)
    except Exception:
        return '(non-serializable)'

@gen_bp.get('/api/default-engine')
def api_default_engine():
    """Return recommended engine ('genai' or 'cv') for a given defect class.
    Uses 'genai' when ref images are configured (IP adapter can run), else 'cv'.
    """
    defect_type = request.args.get('defect_type', '')
    cls = next((c for c in state.classes if c.name == defect_type), None)
    if cls is None:
        return jsonify(engine='cv', reason='class not found')
    has_ref = bool(getattr(cls, 'ref_dir', '')) and os.path.isdir(getattr(cls, 'ref_dir', ''))
    has_crop = False
    if has_ref:
        crop_dir = Path(cls.ref_dir).parent / 'cropped'
        has_crop = crop_dir.is_dir() and any(crop_dir.iterdir())
    if has_ref or has_crop:
        return jsonify(engine='genai', reason='ref images available')
    return jsonify(engine='cv', reason='no ref images configured')

_preview_jobs: dict = {}
_preview_lock = threading.Lock()

_batch_jobs = {}
_batch_lock = threading.Lock()

@gen_bp.get('/api/download-batch/<job_id>')
def api_download_batch(job_id):
    with _batch_lock:
        job = _batch_jobs.get(job_id)
    if not job:
        return "Job not found", 404

    out_dir = job.get('output_dir')
    if not out_dir or not os.path.exists(out_dir):
        return "No images found for this batch", 404

    # Load class metadata saved at batch start
    meta_path = os.path.join(out_dir, '_class_meta.json')
    class_id_map = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as _f:
                class_id_map = json.load(_f)
        except Exception:
            pass

    # Build YOLO data.yaml
    id_to_name = {v: k for k, v in class_id_map.items()}
    sorted_ids = sorted(id_to_name.keys())
    names_list = [id_to_name[i] for i in sorted_ids]
    data_yaml = f"nc: {len(names_list)}\nnames: {names_list}\n"

    import io
    import zipfile
    try:
        import cv2
        import numpy as np
        _cv2_available = True
    except ImportError:
        _cv2_available = False

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", data_yaml)

        for root, dirs, files in os.walk(out_dir):
            for file in files:
                fl = file.lower()
                # Skip mask files and metadata
                if fl.endswith('_mask.png') or fl == '_class_meta.json':
                    continue
                if not fl.endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_full = os.path.join(root, file)
                stem = Path(file).stem
                zf.write(img_full, arcname=f"images/train/{file}")

                # Find class_id from filename prefix matching known class names
                matched_class = None
                for cls_name in class_id_map:
                    if stem.startswith(cls_name + '_'):
                        matched_class = cls_name
                        break
                class_id = class_id_map.get(matched_class, 0)

                # Generate YOLO polygon label from saved mask
                mask_full = os.path.join(root, f"{stem}_mask.png")
                if _cv2_available and os.path.isfile(mask_full):
                    mask_arr = cv2.imread(mask_full, cv2.IMREAD_GRAYSCALE)
                    if mask_arr is not None:
                        _, thresh = cv2.threshold(mask_arr, 127, 255, cv2.THRESH_BINARY)
                        polygons = _mask_to_yolo_polygons(thresh)
                        lines = []
                        for poly in polygons:
                            coords = ' '.join(f'{v:.6f}' for v in poly)
                            lines.append(f"{class_id} {coords}")
                        zf.writestr(f"labels/train/{stem}.txt", '\n'.join(lines))

    memory_file.seek(0)

    # Clean up mask sidecar files and metadata after ZIP is built
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith('_mask.png') or f == '_class_meta.json':
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass

    return send_file(memory_file, download_name=f"dataset_{job_id}.zip", as_attachment=True, mimetype='application/zip')

@gen_bp.post('/api/generate/preview')
def api_preview_generate():
    data = request.get_json(force=True)
    class_name = data.get('class_name', '')
    good_image = data.get('good_image', '')
    mask_path  = data.get('mask_path', '')
    
    material = data.get('material', 'metal')
    intensity = float(data.get('intensity', 0.5))
    naturalness = float(data.get('naturalness', 0.5))
    position_jitter = float(data.get('position_jitter', 0.0))
    engine_override = data.get('engine_override')

    cls = next((c for c in state.classes if c.name == class_name), None)
    if cls is None:
        return jsonify(ok=False, error='Defect class does not exist')
        
    has_custom_base = bool(data.get('base_image'))
    has_custom_mask = bool(data.get('mask'))
    
    _ensure_mask_dir(cls)
    print(f"[PREVIEW] class={class_name} mask_dir={getattr(cls,'mask_dir','')} ref_dir={getattr(cls,'ref_dir','')}", flush=True)

    if not has_custom_mask:
        if not mask_path and getattr(cls, 'mask_dir', ''):
            mask_files = list(Path(cls.mask_dir).rglob('mask_*.png'))
            if mask_files:
                mask_path_obj = random.choice(mask_files)
                mask_path = str(mask_path_obj.relative_to(cls.mask_dir)).replace('\\', '/')
                print(f"[PREVIEW] selected mask={mask_path_obj}", flush=True)
                
                # Auto-derive good_image from the exact mask subfolder if not explicitly asked
                if not good_image and not has_custom_base and getattr(state, 'good_images_path', ''):
                    subfolder_name = mask_path_obj.parent.name
                    for f in os.listdir(state.good_images_path):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and Path(f).stem == subfolder_name:
                            good_image = f
                            break

    if not has_custom_base:
        if not good_image and getattr(state, 'good_images_path', ''):
            all_goods = [f for f in os.listdir(state.good_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if all_goods: good_image = all_goods[0]
            
        if not good_image or not getattr(state, 'good_images_path', ''):
            return jsonify(ok=False, error='Good images not loaded')

    if has_custom_base:
        base_b64 = data.get('base_image')
    else:
        good_full = os.path.join(state.good_images_path, good_image)
        if not os.path.isfile(good_full):
            return jsonify(ok=False, error='Good image not found')
        with open(good_full, 'rb') as f:
            base_b64 = base64.b64encode(f.read()).decode()

    if has_custom_mask:
        mask_b64 = data.get('mask')
    else:
        if not mask_path or not getattr(cls, 'mask_dir', ''):
            return jsonify(ok=False, error='Mask not found'), 400
        full_mask = os.path.normpath(os.path.join(cls.mask_dir, mask_path))
        if not os.path.isfile(full_mask):
            return jsonify(ok=False, error='Mask file not found'), 400
        with open(full_mask, 'rb') as f:
            mask_b64 = base64.b64encode(f.read()).decode()

    ref_b64 = None
    # Build candidate ref dirs: configured ref_dir first, then session upload fallback
    ref_search_dirs = []
    if getattr(cls, 'ref_dir', None):
        cropped_dir = Path(cls.ref_dir).parent / 'cropped'
        ref_search_dirs += [cropped_dir, Path(cls.ref_dir)]
    # Fallback: auto-find bad images from session upload folder
    sid = _get_session_id()
    session_bad = Path(f'/tmp/hondaplus_uploads/{sid}/classes/{cls.name}/bad')
    if session_bad not in ref_search_dirs:
        ref_search_dirs.append(session_bad)

    for search_dir in ref_search_dirs:
        if search_dir.exists():
            refs = [f for f in search_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
            if refs:
                with open(refs[0], 'rb') as f:
                    ref_b64 = base64.b64encode(f.read()).decode()
                break

    # Apply GenAI defaults when engine is genai; allow frontend override
    is_genai = (engine_override == 'genai')
    _d = GENAI_DEFAULTS if is_genai else {}
    payload = {
        "base_image": base_b64,
        "mask": mask_b64,
        "defect_type": class_name,
        "material": material,
        "intensity": intensity,
        "naturalness": naturalness,
        "position_jitter": position_jitter,
        "engine_override": engine_override,
        "ref_image_b64": ref_b64,
        "prompts": getattr(cls, 'prompts', None),
        "negative_prompt": getattr(cls, 'negative_prompt', None),
        # Frontend can override via data keys; fall back to genai defaults then class defaults
        "strength":         float(data.get('strength',         getattr(cls, 'strength',         _d.get('strength', 0.5)))),
        "guidance_scale":   float(data.get('guidance_scale',   getattr(cls, 'guidance_scale',   _d.get('guidance_scale', 7.0)))),
        "steps":            int(data.get('steps',              getattr(cls, 'steps',            _d.get('steps', 30)))),
        "ip_scale":         float(data.get('ip_scale',         getattr(cls, 'ip_scale',         _d.get('ip_scale', 0.5)))),
        "controlnet_scale": float(data.get('controlnet_scale', getattr(cls, 'controlnet_scale', _d.get('controlnet_scale', 0.5)))),
        "inject_alpha":     float(data.get('inject_alpha',     getattr(cls, 'inject_alpha',     _d.get('inject_alpha', 0.5)))),
        "epsilon_factor":   float(data.get('epsilon_factor',   0.03)),
    }


    preview_id = str(uuid.uuid4())[:8]
    with _preview_lock:
        _preview_jobs[preview_id] = {
            'status': 'running',
            'image_b64': '',
            'base_image_b64': f"data:image/jpeg;base64,{base_b64}" if base_b64 else None,
            'error': '',
            'progress': {'status': 'queued', 'queued': 0, 'step': 0, 'total_steps': 0},
        }

    # Start generation in background thread to avoid Cloudflare 524 timeout
    t = threading.Thread(target=_run_preview_thread, args=(preview_id, payload, state.server_url, state.api_key))
    t.daemon = True
    t.start()

    return jsonify(ok=True, preview_id=preview_id)

def _run_preview_thread(preview_id, payload, server_url, api_key):
    _t_start = time.time()
    _MAX_POST_RETRIES = 3
    _MAX_POLL_TIME = 300  # 5 minutes max polling

    def _local_server_url(path):
        base = server_url.rstrip('/')
        url = f'{base}{path}'
        if api_key and 'runpod.net' in base:
            sep = '&' if '?' in url else '?'
            url = f'{url}{sep}runpodApiKey={api_key}'
        return url

    def _local_auth_headers():
        if api_key:
            return {'Authorization': f'Bearer {api_key}'}
        return {}

    # Poll engine progress in background while waiting for main response
    _stop_poll = threading.Event()
    _latest_progress = {}

    def _poll_progress():
        prog_url = _local_server_url('/api/generate/progress')
        while not _stop_poll.is_set():
            try:
                pr = req_lib.get(prog_url, headers=_local_auth_headers(), timeout=5, verify=False)
                if pr.status_code == 200:
                    prog_data = pr.json()
                    _latest_progress.update(prog_data)
                    with _preview_lock:
                        if preview_id in _preview_jobs:
                            _preview_jobs[preview_id]['progress'] = prog_data
            except Exception:
                pass
            _stop_poll.wait(1.5)
    _poll_t = threading.Thread(target=_poll_progress, daemon=True)
    _poll_t.start()

    try:
        url = _local_server_url('/api/generate/preview')
        engine_job_id = None

        # Retry POST to handle proxy/Cloudflare transient failures
        for attempt in range(1, _MAX_POST_RETRIES + 1):
            try:
                print(f"[PREVIEW → ENGINE] POST {url} (attempt {attempt})", flush=True)
                resp = req_lib.post(url, json=payload, headers=_local_auth_headers(),
                                    timeout=(15, 300), verify=False)

                try:
                    resp_data = resp.json()
                except Exception as json_err:
                    body_preview = resp.text[:500] if resp.text else '(empty)'
                    print(f"[PREVIEW ← ENGINE] JSON parse failed: {json_err} | body={body_preview}", flush=True)
                    resp_data = {}

                print(f"[PREVIEW ← ENGINE] HTTP {resp.status_code} | {_safe_log(resp_data)}", flush=True)

                if resp.status_code != 200:
                    err = resp_data.get('detail', f'Engine HTTP {resp.status_code}')
                    if attempt < _MAX_POST_RETRIES:
                        print(f"[PREVIEW] Non-200 response, retrying in 3s...", flush=True)
                        time.sleep(3)
                        continue
                    with _preview_lock:
                        _preview_jobs[preview_id].update({'status': 'error', 'error': err})
                    return

                engine_job_id = resp_data.get('job_id')
                if engine_job_id:
                    break  # success — proceed to polling

                # Legacy sync response (result_image returned directly)
                img_b64 = resp_data.get('result_image') or resp_data.get('image_b64')
                if img_b64:
                    pre_b64 = resp_data.get('result_pre_refine')
                    meta    = resp_data.get('metadata', {})
                    with _preview_lock:
                        _preview_jobs[preview_id].update({
                            'status':          'done',
                            'image_b64':       f"data:image/jpeg;base64,{img_b64}",
                            'pre_refine_b64':  f"data:image/jpeg;base64,{pre_b64}" if pre_b64 else None,
                            'engine_used':     resp_data.get('engine_used', ''),
                            'method_used':     meta.get('method', ''),
                            'processing_time': round(time.time() - _t_start, 2),
                            'error': '',
                        })
                    return

                # No job_id and no image — likely proxy interference
                if attempt < _MAX_POST_RETRIES:
                    print(f"[PREVIEW] No job_id in response (proxy issue?), retrying in 3s...", flush=True)
                    time.sleep(3)
                    continue

                with _preview_lock:
                    _preview_jobs[preview_id].update({'status': 'error', 'error': 'No image from engine (POST failed after retries)'})
                return

            except req_lib.exceptions.Timeout as te:
                print(f"[PREVIEW] POST timeout (attempt {attempt}): {te}", flush=True)
                if attempt < _MAX_POST_RETRIES:
                    time.sleep(3)
                    continue
                with _preview_lock:
                    _preview_jobs[preview_id].update({'status': 'error', 'error': f'Engine timeout after {_MAX_POST_RETRIES} attempts'})
                return
            except Exception as post_e:
                print(f"[PREVIEW] POST error (attempt {attempt}): {post_e}", flush=True)
                if attempt < _MAX_POST_RETRIES:
                    time.sleep(3)
                    continue
                with _preview_lock:
                    _preview_jobs[preview_id].update({'status': 'error', 'error': str(post_e)})
                return

        if not engine_job_id:
            with _preview_lock:
                _preview_jobs[preview_id].update({'status': 'error', 'error': 'Failed to get job_id from engine'})
            return

        # Poll engine job status until done (timeout-aware, matches batch behavior)
        poll_url = _local_server_url(f'/api/generate/preview/status/{engine_job_id}')
        print(f"[PREVIEW] Polling engine job {engine_job_id}", flush=True)
        poll_start = time.time()

        while (time.time() - poll_start) < _MAX_POLL_TIME:
            time.sleep(2)
            try:
                sr = req_lib.get(poll_url, headers=_local_auth_headers(),
                                 timeout=(10, 30), verify=False)
                if sr.status_code != 200:
                    print(f"[PREVIEW] Poll HTTP {sr.status_code}, retrying...", flush=True)
                    continue

                sdata = sr.json()

                # Update progress from engine status
                with _preview_lock:
                    if preview_id in _preview_jobs:
                        _preview_jobs[preview_id]['progress'] = _latest_progress or {}

                if sdata.get('status') == 'done':
                    img_b64 = sdata.get('result_image')
                    if not img_b64:
                        # Response was truncated or image missing — retry poll once more
                        print(f"[PREVIEW] status=done but result_image empty, retrying poll...", flush=True)
                        time.sleep(2)
                        try:
                            sr2 = req_lib.get(poll_url, headers=_local_auth_headers(),
                                              timeout=(10, 60), verify=False)
                            if sr2.status_code == 200:
                                sdata2 = sr2.json()
                                img_b64 = sdata2.get('result_image')
                        except Exception as retry_e:
                            print(f"[PREVIEW] Retry poll failed: {retry_e}", flush=True)

                    if not img_b64:
                        with _preview_lock:
                            _preview_jobs[preview_id].update({'status': 'error', 'error': 'No image from engine'})
                        return

                    pre_b64 = sdata.get('result_pre_refine')
                    meta    = sdata.get('metadata', {})
                    with _preview_lock:
                        _preview_jobs[preview_id].update({
                            'status':          'done',
                            'image_b64':       f"data:image/jpeg;base64,{img_b64}",
                            'pre_refine_b64':  f"data:image/jpeg;base64,{pre_b64}" if pre_b64 else None,
                            'engine_used':     sdata.get('engine_used', ''),
                            'method_used':     meta.get('method', ''),
                            'processing_time': round(time.time() - _t_start, 2),
                            'error': '',
                        })
                    print(f"[PREVIEW] job={engine_job_id} done in {round(time.time()-_t_start,1)}s", flush=True)
                    return
                elif sdata.get('status') == 'error':
                    with _preview_lock:
                        _preview_jobs[preview_id].update({'status': 'error', 'error': sdata.get('error', 'Engine error')})
                    return

            except Exception as poll_e:
                print(f"[PREVIEW] Poll error: {poll_e}", flush=True)

        # Exceeded max poll time
        with _preview_lock:
            _preview_jobs[preview_id].update({'status': 'error', 'error': f'Preview timed out after {_MAX_POLL_TIME}s'})

    except Exception as e:
        print(f"[PREVIEW THREAD] Exception: {e}", flush=True)
        with _preview_lock:
            _preview_jobs[preview_id].update({'status': 'error', 'error': str(e)})
    finally:
        _stop_poll.set()

@gen_bp.get('/api/generate/preview/status/<preview_id>')
def api_preview_status(preview_id):
    with _preview_lock:
        job = _preview_jobs.get(preview_id)
    if job is None:
        return jsonify(status='not_found', image_b64='', error='')
    return jsonify(
        status=job['status'],
        image_b64=job['image_b64'],
        base_image_b64=job.get('base_image_b64'),
        engine_used=job.get('engine_used', ''),
        method_used=job.get('method_used', ''),
        processing_time=job.get('processing_time'),
        progress=job.get('progress', {}),
        error=job['error'],
    )

@gen_bp.post('/api/generate/batch')
def api_start_job():
    if not state.connected:
        return jsonify(ok=False, error='Not connected to Server API')
    if not getattr(state, 'good_images_path', ''):
        return jsonify(ok=False, error='Good images not loaded')
    if not getattr(state, 'output_root', ''):
        sid = _get_session_id()
        state.output_root = f'/tmp/hondaplus_uploads/sessions/{sid}/output'

    req_data = request.get_json(force=True)
    class_name_ui = req_data.get('class_name')
    material = req_data.get('material', 'metal')
    intensity = float(req_data.get('intensity', 0.5))
    naturalness = float(req_data.get('naturalness', 0.5))
    position_jitter = float(req_data.get('position_jitter', 0.0))
    engine_override = req_data.get('engine_override')
    num_images_per_class = int(req_data.get('num_images', 100))
    ssim_threshold = float(req_data.get('ssim_threshold', 0.85))
    qa_enabled = bool(req_data.get('qa_enabled', True))

    target_classes = [c for c in state.classes if c.name == class_name_ui] if class_name_ui else state.classes

    missing_masks = [c.name for c in target_classes if not _has_masks(c)]
    if missing_masks:
        return jsonify(ok=False,
                       error=f'Following classes lack masks: {", ".join(missing_masks)}',
                       missing_masks=missing_masks)

    os.makedirs(state.output_root, exist_ok=True)

    valid_classes = []
    
    good_imgs = []
    for f in os.listdir(state.good_images_path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            good_imgs.append(os.path.join(state.good_images_path, f))
    
    if not good_imgs:
        return jsonify(ok=False, error='No valid good images.')

    for cls in target_classes:
        _ensure_mask_dir(cls)
        mask_imgs = []
        if getattr(cls, 'mask_dir', '') and os.path.isdir(cls.mask_dir):
            for mp in Path(cls.mask_dir).rglob('mask_*.png'):
                mask_imgs.append(str(mp))
        if not mask_imgs:
            continue
            
        ref_imgs = []
        ref_search_dirs = []
        if getattr(cls, 'ref_dir', ''):
            cropped_dir = Path(cls.ref_dir).parent / 'cropped'
            ref_search_dirs += [cropped_dir, Path(cls.ref_dir)]
        # Fallback: session upload bad folder
        sid = _get_session_id()
        session_bad = Path(f'/tmp/hondaplus_uploads/{sid}/classes/{cls.name}/bad')
        if session_bad not in ref_search_dirs:
            ref_search_dirs.append(session_bad)
        for search_dir in ref_search_dirs:
            if search_dir.exists():
                found = [str(rp) for rp in search_dir.glob('*.*') if rp.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
                if found:
                    ref_imgs.extend(found)
                    break
                    
        # Apply GenAI defaults when genai engine selected
        is_genai_batch = (engine_override == 'genai')
        _bd = GENAI_DEFAULTS if is_genai_batch else {}
        valid_classes.append({
            "defect_type": cls.name,
            "material": material,
            "intensity": intensity,
            "naturalness": naturalness,
            "position_jitter": position_jitter,
            "num_images": num_images_per_class,
            "ssim_threshold": ssim_threshold,
            "qa_enabled": qa_enabled,
            "good_imgs": good_imgs,
            "mask_imgs": mask_imgs,
            "ref_imgs": ref_imgs,
            "prompts": getattr(cls, 'prompts', []),
            "negative_prompt": getattr(cls, 'negative_prompt', ''),
            # Frontend override keys from req_data take priority
            "strength":         float(req_data.get('strength',         getattr(cls, 'strength',         _bd.get('strength', 0.5)))),
            "guidance_scale":   float(req_data.get('guidance_scale',   getattr(cls, 'guidance_scale',   _bd.get('guidance_scale', 7.0)))),
            "steps":            int(req_data.get('steps',              getattr(cls, 'steps',            _bd.get('steps', 30)))),
            "ip_scale":         float(req_data.get('ip_scale',         getattr(cls, 'ip_scale',         _bd.get('ip_scale', 0.5)))),
            "controlnet_scale": float(req_data.get('controlnet_scale', getattr(cls, 'controlnet_scale', _bd.get('controlnet_scale', 0.5)))),
            "inject_alpha":     float(req_data.get('inject_alpha',     getattr(cls, 'inject_alpha',     _bd.get('inject_alpha', 0.5)))),
            "epsilon_factor":   float(req_data.get('epsilon_factor',   0.03)),
        })


    if not valid_classes:
        return jsonify(ok=False, error='No classes meet generation criteria.')

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(state.output_root, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save class metadata for YOLO export
    class_meta = {c.name: c.class_id for c in state.classes}
    with open(os.path.join(job_dir, '_class_meta.json'), 'w') as _f:
        json.dump(class_meta, _f)

    with _batch_lock:
        _batch_jobs[job_id] = {
            'status': 'running',
            'progress': 0,
            'discard_count': 0,
            'output_dir': job_dir,
            'error': '',
            'message': 'Đang tiến hành generation...'
        }
        
    t = threading.Thread(target=_run_batch_thread, args=(job_id, valid_classes, job_dir, engine_override, state.server_url, state.api_key))
    t.daemon = True
    t.start()
    
    return jsonify(ok=True, job_id=job_id)

def _run_batch_thread(job_id, payload_classes, output_dir, engine_override, server_url, api_key):
    total_images = sum(c['num_images'] for c in payload_classes)
    completed = 0
    discarded = 0
    
    def _local_server_url(path):
        base = server_url.rstrip('/')
        url = f'{base}{path}'
        if api_key and 'runpod.net' in base:
            sep = '&' if '?' in url else '?'
            url = f'{url}{sep}runpodApiKey={api_key}'
        return url

    def _local_auth_headers():
        if api_key:
            return {'Authorization': f'Bearer {api_key}'}
        return {}
    
    try:
        for pcls in payload_classes:
            defect_type = pcls['defect_type']
            images_to_gen = pcls['num_images']
            generated_for_class = 0
            
            while generated_for_class < images_to_gen:
                success_this_iteration = False
                mask_path = random.choice(pcls['mask_imgs'])
                
                # Derive the original good_image base name from the mask's parent folder
                subfolder_name = Path(mask_path).parent.name
                good_path = None
                for gp in pcls['good_imgs']:
                    if Path(gp).stem == subfolder_name:
                        good_path = gp
                        break
                
                if not good_path:
                    # Fallback just in case the original good image was deleted
                    good_path = random.choice(pcls['good_imgs'])
                    
                ref_path = random.choice(pcls['ref_imgs']) if pcls.get('ref_imgs') else None
                
                with open(good_path, 'rb') as f:
                    base_b64 = base64.b64encode(f.read()).decode()
                with open(mask_path, 'rb') as f:
                    mask_b64 = base64.b64encode(f.read()).decode()
                ref_b64 = None
                if ref_path:
                    with open(ref_path, 'rb') as f:
                        ref_b64 = base64.b64encode(f.read()).decode()
                
                payload = {
                    "base_image": base_b64,
                    "mask": mask_b64,
                    "defect_type": defect_type,
                    "material": pcls['material'],
                    "intensity": pcls['intensity'],
                    "naturalness": pcls['naturalness'],
                    "position_jitter": pcls.get('position_jitter', 0.0),
                    "engine_override": engine_override,
                    "ref_image_b64": ref_b64,
                    "prompts": pcls.get('prompts') if pcls.get('prompts') else None,
                    "negative_prompt": pcls.get('negative_prompt') if pcls.get('negative_prompt') else None,
                    "strength": pcls.get('strength'),
                    "guidance_scale": pcls.get('guidance_scale'),
                    "ip_scale": pcls.get('ip_scale'),
                    "controlnet_scale": pcls.get('controlnet_scale'),
                    "inject_alpha": pcls.get('inject_alpha'),
                    "epsilon_factor": pcls.get('epsilon_factor', 0.03),
                    "count": 1
                }

                
                try:
                    start_url = _local_server_url('/api/generate/batch')
                    print(f"[BATCH → ENGINE] POST {start_url} | defect={pcls['defect_type']}", flush=True)
                    resp = req_lib.post(start_url, json=payload, timeout=300, headers=_local_auth_headers(), verify=False)
                    if resp.status_code == 200:
                        remote_job_id = resp.json().get('job_id')
                        print(f"[BATCH ← ENGINE] Job started | remote_job_id={remote_job_id}", flush=True)
                        while True:
                            time.sleep(1)
                            poll_url = _local_server_url(f'/api/generate/status/{remote_job_id}')
                            sr = req_lib.get(poll_url, headers=_local_auth_headers(), timeout=30, verify=False)
                            if sr.status_code == 200:
                                sdata = sr.json()
                                print(f"[BATCH POLL] status={sdata.get('status')} progress={sdata.get('progress')} results={len(sdata.get('results', []))}", flush=True)
                                if sdata.get('status') == 'done':
                                    results = sdata.get('results', [])
                                    if len(results) > 0:
                                        img_data = base64.b64decode(results[0])
                                        passed_ssim = True
                                        if pcls.get('qa_enabled', True):
                                            try:
                                                import cv2
                                                import numpy as np
                                                from skimage.metrics import structural_similarity as ssim

                                                ref_img = cv2.imread(good_path, cv2.IMREAD_GRAYSCALE)
                                                np_arr = np.frombuffer(img_data, np.uint8)
                                                gen_img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
                                                if ref_img is not None and gen_img is not None:
                                                    if ref_img.shape != gen_img.shape:
                                                        gen_img = cv2.resize(gen_img, (ref_img.shape[1], ref_img.shape[0]))
                                                    score, _ = ssim(ref_img, gen_img, full=True)
                                                    print(f"[BATCH SSIM] {defect_type} | score={score:.4f} threshold={pcls.get('ssim_threshold', 0.85)}", flush=True)
                                                    if score < pcls.get('ssim_threshold', 0.85):
                                                        passed_ssim = False
                                            except Exception as e:
                                                print(f"[BATCH SSIM] Warning: {e}", flush=True)
                                        else:
                                            print(f"[BATCH SSIM] QA disabled — skipping filter", flush=True)
                                            
                                        if passed_ssim:
                                            out_path = os.path.join(output_dir, f"{defect_type}_{job_id}_{completed}.png")
                                            with open(out_path, 'wb') as f:
                                                f.write(img_data)
                                            # Save mask alongside image for YOLO export
                                            mask_out = os.path.join(output_dir, f"{defect_type}_{job_id}_{completed}_mask.png")
                                            shutil.copy2(mask_path, mask_out)

                                            # Save metadata JSON (filter out bulky base64 blobs)
                                            meta_out = os.path.join(output_dir, f"{defect_type}_{job_id}_{completed}.json")
                                            json_meta = {
                                                k: v for k, v in payload.items() 
                                                if not k.endswith('_b64') and k not in ('base_image', 'mask')
                                            }
                                            if 'score' in locals():
                                                json_meta['ssim_score'] = float(score)
                                            with open(meta_out, 'w') as mf:
                                                json.dump(json_meta, mf, indent=2)

                                            success_this_iteration = True

                                        else:
                                            discarded += 1
                                    else:
                                        discarded += 1
                                    break
                                elif sdata.get('status') == 'error':
                                    err_msg = sdata.get('error', 'Unknown error from engine')
                                    print(f"[BATCH POLL] ERROR: {err_msg}", flush=True)
                                    with _batch_lock:
                                        _batch_jobs[job_id]['message'] = f"Engine Error: {err_msg}"
                                    break
                            else:
                                print(f"[BATCH POLL] HTTP error: {sr.status_code}", flush=True)
                                with _batch_lock:
                                    _batch_jobs[job_id]['message'] = f"Poll status error: HTTP {sr.status_code}"
                                break
                    else:
                        print(f"[BATCH → ENGINE] ERROR HTTP {resp.status_code}: {resp.text[:200]}", flush=True)
                        with _batch_lock:
                            _batch_jobs[job_id]['message'] = f"Job send failed: HTTP {resp.status_code}"
                except Exception as e:
                    print(f"[BATCH LOOP] Exception: {e}", flush=True)
                    with _batch_lock:
                        _batch_jobs[job_id]['message'] = f"Internal loop error: {str(e)}"
                    pass  # ignore small network flakes, let loop continue
                
                if success_this_iteration:
                    generated_for_class += 1
                    completed += 1
                
                with _batch_lock:
                    _batch_jobs[job_id]['progress'] = int(min(99, (completed / total_images) * 100))
                    _batch_jobs[job_id]['discard_count'] = discarded
        
        with _batch_lock:
            _batch_jobs[job_id]['status'] = 'done'
            _batch_jobs[job_id]['progress'] = 100
            
    except Exception as e:
        with _batch_lock:
            _batch_jobs[job_id]['status'] = 'error'
            _batch_jobs[job_id]['error'] = str(e)


@gen_bp.get('/api/generate/status/<job_id>')
def api_job_status(job_id):
    with _batch_lock:
        job = _batch_jobs.get(job_id)
        if not job:
            return jsonify(status="error", error="Job ID not found in local pool")
        return jsonify(job)

@gen_bp.get('/api/peek-results')
def api_peek_results():
    job_id = request.args.get('job_id')
    search_dir = state.output_root
    if job_id:
        search_dir = os.path.join(state.output_root, job_id)
        
    if not os.path.isdir(search_dir):
        return jsonify(images=[])
        
    imgs = []
    try:
        files = []
        # If job_id specified, we can just look in that folder (non-recursive)
        # or still walk if there are sub-sub folders. Let's walk to be safe.
        for root, dirs, f_names in os.walk(search_dir):
            for f in f_names:
                fl = f.lower()
                if fl.endswith(('.png', '.jpg', '.jpeg')) and not fl.endswith('_mask.png'):
                    full = os.path.join(root, f)
                    files.append((full, os.path.getmtime(full)))
        files.sort(key=lambda x: x[1], reverse=True)
        files = files[:12]
        
        for full, _mtime in files:
            with open(full, 'rb') as fp:
                b64 = base64.b64encode(fp.read()).decode()
                imgs.append(f"data:image/jpeg;base64,{b64}")
    except Exception as e:
        pass
        
    return jsonify(images=imgs)


# ─────── QA Review APIs ───────────────────────────────────────────────────────

@gen_bp.get('/api/review/batches')
def api_review_batches():
    """List all batch folders in the output root."""
    out = getattr(state, 'output_root', '')
    if not out or not os.path.isdir(out):
        return jsonify(batches=[])
    batches = []
    for name in sorted(os.listdir(out), reverse=True):
        full = os.path.join(out, name)
        if os.path.isdir(full):
            imgs = [f for f in os.listdir(full) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            mtime = os.path.getmtime(full)
            import datetime
            date_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            batches.append({'id': name, 'count': len(imgs), 'date': date_str})
    return jsonify(batches=batches)


@gen_bp.get('/api/review/images/<batch_id>')
def api_review_images(batch_id):
    """Return all images in a batch with optional SSIM scores vs good images."""
    out = getattr(state, 'output_root', '')
    batch_dir = os.path.join(out, batch_id) if out else ''
    if not os.path.isdir(batch_dir):
        return jsonify(images=[], error='Batch folder not found')

    # Try to compute SSIM against good images
    good_path = getattr(state, 'good_images_path', '')
    good_gray = None
    if good_path and os.path.isdir(good_path):
        try:
            import cv2
            good_files = sorted([f for f in os.listdir(good_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            if good_files:
                good_gray = cv2.imread(os.path.join(good_path, good_files[0]), cv2.IMREAD_GRAYSCALE)
        except ImportError:
            good_gray = None

    images = []
    for fname in sorted(os.listdir(batch_dir)):
        fl = fname.lower()
        if not fl.endswith(('.png', '.jpg', '.jpeg')) or fl.endswith('_mask.png'):
            continue
        full = os.path.join(batch_dir, fname)
        with open(full, 'rb') as f:
            b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

        # Compute SSIM
        ssim_score = None
        if good_gray is not None:
            try:
                import cv2
                import numpy as np
                from skimage.metrics import structural_similarity as ssim_fn
                gen_img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
                if gen_img is not None and good_gray is not None:
                    if gen_img.shape != good_gray.shape:
                        gen_img = cv2.resize(gen_img, (good_gray.shape[1], good_gray.shape[0]))
                    score, _ = ssim_fn(good_gray, gen_img, full=True)
                    ssim_score = round(float(score), 4)
            except Exception:
                pass

        images.append({'filename': fname, 'b64': b64, 'ssim': ssim_score})

    return jsonify(images=images)


@gen_bp.post('/api/review/delete')
def api_review_delete():
    """Delete specified files from a batch folder."""
    data = request.get_json(force=True)
    batch_id = data.get('batch_id', '')
    filenames = data.get('filenames', [])
    out = getattr(state, 'output_root', '')
    batch_dir = os.path.join(out, batch_id) if out else ''
    if not os.path.isdir(batch_dir):
        return jsonify(ok=False, error='Batch folder not found')

    deleted = 0
    for fname in filenames:
        full = os.path.join(batch_dir, fname)
        # Security: ensure path stays inside batch_dir
        if not os.path.normpath(full).startswith(os.path.normpath(batch_dir)):
            continue
        if os.path.isfile(full):
            os.remove(full)
            deleted += 1
            print(f"[QA REVIEW] Deleted: {full}", flush=True)

    remaining = len([f for f in os.listdir(batch_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return jsonify(ok=True, deleted=deleted, remaining=remaining)
