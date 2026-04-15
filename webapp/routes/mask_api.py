import os
import base64
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file

from utils import state, _ensure_mask_dir

mask_bp = Blueprint('mask_api', __name__)

@mask_bp.post('/api/save-ref-crop')
def api_save_ref_crop():
    data       = request.get_json(force=True)
    class_name = data.get('class_name', '')
    filename   = data.get('filename', '')
    img_data   = data.get('image_data', '')

    cls = next((c for c in state.classes if c.name == class_name), None)
    if cls is None or not cls.ref_dir:
        return jsonify(ok=False, error='Class or defect image folder not configured')

    if ',' in img_data:
        img_data = img_data.split(',', 1)[1]
    try:
        png_bytes = base64.b64decode(img_data)
    except Exception as e:
        return jsonify(ok=False, error=f'Image decode error: {e}')

    save_dir = Path(cls.ref_dir).parent / 'cropped'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    save_path.write_bytes(png_bytes)

    total = len([f for f in save_dir.iterdir() if f.is_file()])
    return jsonify(ok=True, path=str(save_path), total=total)

@mask_bp.get('/api/list-good-images')
def api_list_good_images():
    gdir = state.good_images_path
    if not gdir or not os.path.isdir(gdir):
        return jsonify(images=[], dir='')
    imgs = [f for f in sorted(os.listdir(gdir))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    return jsonify(images=imgs, dir=gdir)

@mask_bp.get('/api/good-image/<path:filename>')
def api_good_image(filename):
    full = os.path.join(state.good_images_path, filename)
    if not os.path.isfile(full):
        return '', 404
    return send_file(full)

@mask_bp.get('/api/list-ref-images/<class_name>')
def api_list_ref_images(class_name):
    cls = next((c for c in state.classes if c.name == class_name), None)
    if cls is None or not getattr(cls, 'ref_dir', None) or not os.path.isdir(cls.ref_dir):
        return jsonify(images=[], dir='')
    imgs = [f for f in sorted(os.listdir(cls.ref_dir))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    return jsonify(images=imgs, dir=cls.ref_dir)

@mask_bp.get('/api/ref-image/<class_name>/<path:filename>')
def api_ref_image(class_name, filename):
    cls = next((c for c in state.classes if c.name == class_name), None)
    if cls is None or not getattr(cls, 'ref_dir', None):
        return '', 404
    full = os.path.join(cls.ref_dir, filename)
    if not os.path.isfile(full):
        return '', 404
    return send_file(full)

@mask_bp.get('/api/list-subfolders/<class_name>')
def api_list_subfolders(class_name):
    cls = next((c for c in state.classes if c.name == class_name), None)
    _ensure_mask_dir(cls) if cls else None
    if cls is None or not getattr(cls, 'mask_dir', None) or not os.path.isdir(cls.mask_dir):
        return jsonify(subfolders=[], mask_dir=getattr(cls, 'mask_dir', '') if cls else '')
    subs = [d for d in sorted(os.listdir(cls.mask_dir))
            if os.path.isdir(os.path.join(cls.mask_dir, d))]
    return jsonify(subfolders=subs, mask_dir=cls.mask_dir)

@mask_bp.post('/api/create-subfolder')
def api_create_subfolder():
    data = request.get_json(force=True)
    class_name  = data.get('class_name', '')
    folder_name = data.get('folder_name', '').strip()
    cls = next((c for c in state.classes if c.name == class_name), None)
    _ensure_mask_dir(cls) if cls else None
    if cls is None or not getattr(cls, 'mask_dir', None) or not folder_name:
        return jsonify(ok=False, error='Invalid class or folder name')
    path = os.path.join(cls.mask_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return jsonify(ok=True, path=path)

@mask_bp.post('/api/save-mask')
def api_save_mask():
    data       = request.get_json(force=True)
    class_name = data.get('class_name', '')
    subfolder  = data.get('subfolder', '')
    img_data   = data.get('image_data', '')

    cls = next((c for c in state.classes if c.name == class_name), None)
    _ensure_mask_dir(cls) if cls else None
    if cls is None or not getattr(cls, 'mask_dir', None):
        return jsonify(ok=False, error='Class or defect image folder not configured')
    if not subfolder:
        return jsonify(ok=False, error='Please select a subfolder')

    if ',' in img_data:
        img_data = img_data.split(',', 1)[1]
    try:
        png_bytes = base64.b64decode(img_data)
    except Exception as e:
        return jsonify(ok=False, error=f'Image decode error: {e}')

    save_dir = Path(cls.mask_dir) / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    idx = len(list(save_dir.glob('mask_*.png')))
    save_path = save_dir / f'mask_{idx:03d}.png'
    save_path.write_bytes(png_bytes)

    total = len(list(Path(cls.mask_dir).rglob('mask_*.png')))
    return jsonify(ok=True, path=str(save_path), total=total)

@mask_bp.get('/api/list-masks/<class_name>')
def api_list_masks(class_name):
    cls = next((c for c in state.classes if c.name == class_name), None)
    _ensure_mask_dir(cls) if cls else None
    if cls is None or not getattr(cls, 'mask_dir', None) or not os.path.isdir(cls.mask_dir):
        return jsonify(masks=[], mask_dir='')
    masks = []
    mask_dir = Path(cls.mask_dir)
    for subfolder in sorted(mask_dir.iterdir()):
        if subfolder.is_dir():
            for f in sorted(subfolder.iterdir()):
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                    masks.append(f'{subfolder.name}/{f.name}')
    return jsonify(masks=masks, mask_dir=cls.mask_dir)

@mask_bp.get('/api/mask-image/<class_name>/<path:filepath>')
def api_mask_image(class_name, filepath):
    cls = next((c for c in state.classes if c.name == class_name), None)
    _ensure_mask_dir(cls) if cls else None
    if cls is None or not getattr(cls, 'mask_dir', None):
        return '', 404
    full = os.path.normpath(os.path.join(cls.mask_dir, filepath))
    if not full.startswith(os.path.normpath(cls.mask_dir)):
        return '', 403
    if not os.path.isfile(full):
        return '', 404
    return send_file(full)
