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


@setup_bp.get('/api/download-config')
def api_download_config():
    cfg_yaml = yaml.dump(state.build_config_dict(), allow_unicode=True, sort_keys=False)
    buf = io.BytesIO(cfg_yaml.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/yaml', as_attachment=True, download_name='config.yaml')

