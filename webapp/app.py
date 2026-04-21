"""
Defect Dataset Generator - Modular Flask Web App
Run:  python webapp/app.py
Open: http://localhost:5000
"""
import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, g, request
from flask_session import Session
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Suppress noisy polling requests from Flask access log
class _SilencePolling(logging.Filter):
    _SKIP = ("/api/generate/preview/status/", "/api/generate/progress")
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self._SKIP)

logging.getLogger("werkzeug").addFilter(_SilencePolling())
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.app_state import AppState
from utils import _get_session_id, SESSION_DIR

def _get_session_file(sid: str) -> Path:
    return SESSION_DIR / f'{sid}.json'

app = Flask(__name__)
app.secret_key = 'defect-gen-local-2025'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['TEMPLATES_AUTO_RELOAD'] = True
os.makedirs('/tmp/flask_session', exist_ok=True)
Session(app)

# --- Session Isolation (Disk-backed JSON) ---
GLOBAL_STATES = {}
SESSION_DIR.mkdir(parents=True, exist_ok=True)

@app.before_request
def load_state():
    sid = _get_session_id()
    if sid not in GLOBAL_STATES:
        sf = _get_session_file(sid)
        if sf.exists():
            try:
                GLOBAL_STATES[sid] = AppState.from_dict(json.loads(sf.read_text(encoding='utf-8')))
            except Exception:
                GLOBAL_STATES[sid] = AppState()
        else:
            GLOBAL_STATES[sid] = AppState()
            
    # Apply global settings if missing
    if not GLOBAL_STATES[sid].server_url:
        from app_settings import load_settings
        load_settings(GLOBAL_STATES[sid])
    g.state = GLOBAL_STATES[sid]

@app.after_request
def save_state(response):
    sid = _get_session_id()
    response.set_cookie('sid', sid)
    
    # Save the RAM state to Disk
    if hasattr(g, 'state'):
        try:
            sf = _get_session_file(sid)
            sf.write_text(json.dumps(g.state.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            pass # print(f"[WARN] Could not save session {sid}: {e}")
            
    return response

# --- Register Blueprints ---
from routes.views import views_bp
from routes.setup_api import setup_bp
from routes.pharma_api import pharma_bp
from routes.cap_api import cap_bp
from routes.metal_cap_api import metal_cap_bp
from routes.other_api import other_bp
from routes.eval_api import eval_bp

app.register_blueprint(views_bp)
app.register_blueprint(setup_bp)
app.register_blueprint(pharma_bp)
app.register_blueprint(cap_bp)
app.register_blueprint(metal_cap_bp)
app.register_blueprint(other_bp)
app.register_blueprint(eval_bp)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('===================================================')
    print(' Defect Dataset Generator Web UI (Modular)')
    print(f' Port: {port}')
    print('===================================================')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
