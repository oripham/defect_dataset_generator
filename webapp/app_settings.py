import json
from pathlib import Path

_SETTINGS_FILE = Path.home() / '.defect_gen_settings.json'

def load_settings(target_state):
    try:
        if _SETTINGS_FILE.exists():
            data = json.loads(_SETTINGS_FILE.read_text(encoding='utf-8'))
            if 'server_url' in data:
                target_state.server_url = data['server_url']
            if 'api_key' in data:
                target_state.api_key = data['api_key']
    except Exception:
        pass

def save_settings(target_state=None):
    if not target_state:
        from utils import state
        target_state = state
    try:
        _SETTINGS_FILE.write_text(
            json.dumps({'server_url': target_state.server_url, 'api_key': target_state.api_key},
                       ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
    except Exception:
        pass
