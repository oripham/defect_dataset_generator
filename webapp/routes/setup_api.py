import os
import shutil
import io
import yaml
from flask import Blueprint, jsonify, request, send_file
import requests as req_lib

from utils import (
    state, T, _get_session_id, _classes_list, _has_masks, 
    _auth_headers, _server_url, _auto_mask_dir
)
from gui.app_state import ClassConfig
from app_settings import save_settings # I need to define app_settings somewhere or in utils

setup_bp = Blueprint('setup_api', __name__)

@setup_bp.post('/api/set-language')
def api_set_language():
    data = request.get_json(force=True)
    state.language = data.get('language', data.get('lang', 'en'))
    return jsonify(ok=True)

@setup_bp.post('/api/test-connection')
def api_test_connection():
    data = request.get_json(force=True)
    url = data.get('url', '').rstrip('/')
    state.server_url = url
    if 'api_key' in data:
        state.api_key = data['api_key']
    try:
        resp = req_lib.get(_server_url('/health'), timeout=15, verify=False, headers=_auth_headers())
        if resp.status_code == 403:
            state.connected = False
            return jsonify(ok=False, detail='403 Forbidden — Pod起動直後はプロキシが準備中です。2〜3分待ってから再試行してください。')
        resp.raise_for_status()
        info = resp.json()
        state.connected = True
        from app_settings import save_settings
        save_settings(state)
        gpu = info.get('gpu_name', '?')
        mem = info.get('gpu_memory', '?')
        detail = T('page1.gpu_yes', name=gpu, mem=mem) if info.get('gpu_available') else T('page1.gpu_no')

        # ---- Capability probe (avoid "Connected but still CV-only") ----
        caps = {"metal_cap": False, "generate": False, "jobs": False}
        suggested_url = None
        try:
            oai = req_lib.get(_server_url('/openapi.json'), timeout=15, verify=False, headers=_auth_headers())
            if oai.status_code == 200:
                spec = oai.json() or {}
                paths = spec.get("paths") or {}
                caps["metal_cap"] = "/api/metal_cap/preview" in paths
                caps["generate"] = "/api/generate/preview" in paths
                caps["jobs"] = "/jobs" in paths
        except Exception:
            pass

        # Heuristic: user connected to the job server (has /jobs + /api/generate/*) but lacks metal_cap endpoints.
        if caps["jobs"] and caps["generate"] and not caps["metal_cap"]:
            # Suggest common local engine API port
            if url.endswith(":8005"):
                suggested_url = url[:-5] + ":8001"
            elif "localhost" in url or "127.0.0.1" in url:
                suggested_url = "http://127.0.0.1:8001"

        return jsonify(ok=True, detail=detail, caps=caps, suggested_url=suggested_url)
    except Exception as e:
        state.connected = False
        err = str(e)
        if '403' in err:
            err = '403 Forbidden — Pod起動直後はプロキシが準備中です。2〜3分待ってから再試行してください。'
        return jsonify(ok=False, detail=err)

@setup_bp.post('/api/save-step2')
def api_save_step2():
    data = request.get_json(force=True)
    state.job_name = data.get('job_name', state.job_name)
    state.good_images_path = data.get('good_images_path', '')
    state.output_root = data.get('output_root', '')
    state.image_width = int(data.get('image_width', 1024))
    state.image_height = int(data.get('image_height', 1024))
    return jsonify(ok=True)

_DEFECT_DEFAULT_PROMPTS = {
    "scratch":    (
        ["industrial metal surface scratch mark, fine linear groove, directional scratch, realistic inspection photo"],
        "smooth, clean, no defect, blur, cartoon",
    ),
    "dent":       (
        ["physical dent depression with shadow shading, surface deformation, concentric ring distortion, directional light shadow, realistic inspection photo"],
        "smooth, clean, perfect, no defect, blur, cartoon",
    ),
    "bulge":      (
        ["surface bulge protrusion, raised bump with highlight shading, realistic metal surface inspection"],
        "smooth, flat, clean, no defect, blur, cartoon",
    ),
    "chip":       (
        ["chipped edge defect, missing material fragment, sharp broken edge, realistic inspection photo"],
        "smooth, clean, no defect, blur, cartoon",
    ),
    "foreign":    (
        ["foreign particle contamination, debris on surface, realistic inspection photo"],
        "clean, no defect, blur, cartoon",
    ),
}

@setup_bp.post('/api/add-class')
def api_add_class():
    data = request.get_json(force=True)
    name = data.get('name', '').strip()
    if not name:
        return jsonify(ok=False, error='Class name is empty')
    if any(c.name == name for c in state.classes):
        return jsonify(ok=False, error='Class name already exists')
    default_p, default_neg = _DEFECT_DEFAULT_PROMPTS.get(name, (
        ['a defect on a product surface, realistic'],
        'clean, smooth, perfect, no defect, blur, cartoon',
    ))
    next_id = max((c.class_id for c in state.classes), default=-1) + 1
    c = ClassConfig(name=name, class_id=next_id, prompts=default_p, negative_prompt=default_neg)
    state.classes.append(c)
    return jsonify(ok=True, classes=_classes_list())

_DEFAULT_CLASSES = {
    'deform_mc': {
        'class_id': 1,
        'prompts': ['metal deformation following circular geometry'],
        'negative_prompt': 'clean surface',
        'strength': 0.7, 'ip_scale': 0.5, 'controlnet_scale': 0.45,
        'inject_alpha': 0.75, 'guidance_scale': 7.0,
    },
    'deform_mold': {
        'class_id': 2,
        'prompts': ['irregular contamination patch'],
        'negative_prompt': 'clean surface',
        'strength': 0.7, 'ip_scale': 0.5, 'controlnet_scale': 0.45,
        'inject_alpha': 0.75, 'guidance_scale': 7.0,
    },
}

@setup_bp.post('/api/add-default-classes')
def api_add_default_classes():
    added = []
    for name, defaults in _DEFAULT_CLASSES.items():
        if any(c.name == name for c in state.classes):
            continue
        c = ClassConfig(
            name=name, class_id=defaults['class_id'], prompts=defaults['prompts'],
            negative_prompt=defaults['negative_prompt'], strength=defaults['strength'],
            ip_scale=defaults['ip_scale'], controlnet_scale=defaults['controlnet_scale'],
            inject_alpha=defaults['inject_alpha'], guidance_scale=defaults['guidance_scale'],
        )
        state.classes.append(c)
        added.append(name)
    return jsonify(ok=True, added=added, classes=_classes_list())

@setup_bp.post('/api/update-class')
def api_update_class():
    data = request.get_json(force=True)
    name = data.get('name')
    cls = next((c for c in state.classes if c.name == name), None)
    if cls is None:
        return jsonify(ok=False, error='Class not found')
    cls.class_id     = int(data.get('class_id', cls.class_id))
    cls.ref_dir      = data.get('ref_dir', cls.ref_dir)
    cls.mask_dir     = _auto_mask_dir(cls) if cls.ref_dir else ''
    
    if 'prompts' in data: cls.prompts = data['prompts']
    if 'negative_prompt' in data: cls.negative_prompt = data['negative_prompt']
    if 'strength' in data: cls.strength = float(data['strength'])
    if 'guidance_scale' in data: cls.guidance_scale = float(data['guidance_scale'])
    if 'ip_scale' in data: cls.ip_scale = float(data['ip_scale'])
    if 'controlnet_scale' in data: cls.controlnet_scale = float(data['controlnet_scale'])
    if 'inject_alpha' in data: cls.inject_alpha = float(data['inject_alpha'])
    return jsonify(ok=True, classes=_classes_list())

@setup_bp.delete('/api/class/<name>')
def api_delete_class(name):
    before = len(state.classes)
    state.classes = [c for c in state.classes if c.name != name]
    if len(state.classes) == before:
        return jsonify(ok=False, error='Class not found'), 404
    return jsonify(ok=True, classes=_classes_list())

@setup_bp.get('/api/classes')
def api_get_classes():
    return jsonify(classes=_classes_list())

@setup_bp.get('/api/check-masks')
def api_check_masks():
    missing = [c.name for c in state.classes if not _has_masks(c)]
    return jsonify(ok=len(missing) == 0, missing=missing)

@setup_bp.post('/api/upload-dataset')
def api_upload_dataset():
    upload_type = request.form.get('type')
    class_name = request.form.get('class_name', '')
    if not upload_type: return jsonify(ok=False, error="Missing upload type")
    files = request.files.getlist('files')
    if not files: return jsonify(ok=False, error="No files loaded")
    sid = _get_session_id()
    if upload_type == 'good_images':
        save_path = os.path.join('/tmp', 'hondaplus_uploads', sid, 'good_images')
    elif upload_type == 'ref_dir' and class_name:
        save_path = os.path.join('/tmp', 'hondaplus_uploads', sid, 'classes', class_name, 'bad')
    else:
        return jsonify(ok=False, error="Invalid upload parameters")

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    count = 0
    for f in files:
        if f.filename and f.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            fname = os.path.basename(f.filename)
            f.save(os.path.join(save_path, fname))
            count += 1
            
    if upload_type == 'good_images':
        state.good_images_path = save_path
    elif upload_type == 'ref_dir' and class_name:
        cls = next((c for c in state.classes if c.name == class_name), None)
        if cls:
            cls.ref_dir = save_path
            cls.mask_dir = os.path.join('/tmp', 'hondaplus_uploads', sid, 'classes', class_name, 'mask')
    return jsonify(ok=True, path=save_path, count=count)

@setup_bp.post('/api/clear-session')
def api_clear_session():
    sid = _get_session_id()
    sess_dir = os.path.join('/tmp', 'hondaplus_uploads', sid)
    if os.path.exists(sess_dir):
        shutil.rmtree(sess_dir)
    state.good_images_path = ''
    state.output_root = ''
    state.classes = []
    return jsonify(ok=True)

@setup_bp.post('/api/update-model')
def api_update_model():
    data = request.get_json(force=True)
    state.model_name = data.get('model_name', state.model_name)
    state.device     = data.get('device', state.device)
    cfg_yaml = yaml.dump(state.build_config_dict(), allow_unicode=True, sort_keys=False)
    return jsonify(ok=True, yaml=cfg_yaml)

@setup_bp.get('/api/download-config')
def api_download_config():
    cfg_yaml = yaml.dump(state.build_config_dict(), allow_unicode=True, sort_keys=False)
    buf = io.BytesIO(cfg_yaml.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/yaml', as_attachment=True, download_name='config.yaml')

@setup_bp.post('/api/reset')
def api_reset():
    state.reset()
    return jsonify(ok=True)
