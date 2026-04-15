# gui/pages/page_preview.py
import base64
import io
import json
import os
import threading
import time
import zipfile

import cv2
import numpy as np
import requests as req_lib
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr
from gui.theme import FG_DIM

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
CANVAS_W, CANVAS_H = 800, 600


def _collect_images(folder):
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    )


def _collect_masks(mask_dir):
    paths = []
    if not mask_dir or not os.path.isdir(mask_dir):
        return paths
    for sub in sorted(os.listdir(mask_dir)):
        sub_path = os.path.join(mask_dir, sub)
        if os.path.isdir(sub_path):
            for f in sorted(os.listdir(sub_path)):
                if f.lower().endswith(IMG_EXTS):
                    paths.append(os.path.join(sub_path, f))
    return paths


def _rotate_mask(mask_arr, angle_deg):
    h, w = mask_arr.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(mask_arr, M, (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _build_composite(base_pil, ref_pil, mask_arr, blend, canvas_size, show_outline):
    """Blend ref image into base image where mask is white."""
    cw, ch = canvas_size

    base = np.array(base_pil.convert('RGB').resize(canvas_size, Image.LANCZOS), dtype=np.float32)
    ref  = np.array(ref_pil.convert('RGB').resize(canvas_size, Image.LANCZOS), dtype=np.float32)
    mask = cv2.resize(mask_arr, canvas_size, interpolation=cv2.INTER_NEAREST)
    white = (mask > 127)

    result = base.copy()
    result[white] = base[white] * (1.0 - blend) + ref[white] * blend
    result = np.clip(result, 0, 255).astype(np.uint8)

    if show_outline:
        # Cyan outline at mask boundary
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded  = cv2.erode(mask, kernel, iterations=1)
        edge    = (dilated > 127) & ~(eroded > 127)
        result[edge] = [0, 229, 255]

    return Image.fromarray(result)


class PreviewPage(QWidget):
    _sig_result = Signal(str)   # emits base64 data-URL when test generation completes
    _sig_error  = Signal(str)   # emits error message string

    def __init__(self, state, app):
        super().__init__()
        self.state = state
        self.app = app
        self._good_images = []
        self._good_idx = 0
        self._ref_images = []
        self._ref_idx = 0
        self._masks = []
        self._mask_idx = 0
        self._build_ui()
        self.retranslate_ui()
        self._sig_result.connect(self._show_result)
        self._sig_error.connect(self._show_gen_error)

    def t(self, key, **kwargs):
        return tr(self.state, key, **kwargs)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(10)

        self._title = QLabel()
        self._title.setStyleSheet('font-size: 16pt; font-weight: bold;')
        self._sub = QLabel()
        self._sub.setWordWrap(True)
        self._sub.setStyleSheet(f'color: {FG_DIM};')
        root.addWidget(self._title)
        root.addWidget(self._sub)

        # Class selector
        cls_row = QHBoxLayout()
        self._cls_label = QLabel()
        self._cls_combo = QComboBox()
        self._cls_combo.currentIndexChanged.connect(self._on_class_change)
        cls_row.addWidget(self._cls_label)
        cls_row.addWidget(self._cls_combo)
        cls_row.addStretch()
        root.addLayout(cls_row)

        # Canvas
        self._canvas = QLabel()
        self._canvas.setFixedSize(CANVAS_W, CANVAS_H)
        self._canvas.setAlignment(Qt.AlignCenter)
        self._canvas.setStyleSheet(
            'background: #0b1220; border: 2px solid #7c3aed; border-radius: 8px;'
        )
        root.addWidget(self._canvas, 0, Qt.AlignHCenter)

        # Good image nav
        img_nav = QHBoxLayout()
        self._good_nav_lbl = QLabel()
        self._good_nav_lbl.setStyleSheet('color: #38bdf8; font-weight: bold;')
        self._prev_good_btn = QPushButton()
        self._prev_good_btn.setObjectName('btn_secondary')
        self._prev_good_btn.clicked.connect(self._prev_good)
        self._next_good_btn = QPushButton()
        self._next_good_btn.setObjectName('btn_secondary')
        self._next_good_btn.clicked.connect(self._next_good)
        self._good_count_lbl = QLabel()
        img_nav.addWidget(self._good_nav_lbl)
        img_nav.addWidget(self._prev_good_btn)
        img_nav.addWidget(self._next_good_btn)
        img_nav.addWidget(self._good_count_lbl)
        img_nav.addStretch()
        root.addLayout(img_nav)

        # Ref image nav
        ref_nav = QHBoxLayout()
        self._ref_nav_lbl = QLabel()
        self._ref_nav_lbl.setStyleSheet('color: #f87171; font-weight: bold;')
        self._prev_ref_btn = QPushButton()
        self._prev_ref_btn.setObjectName('btn_secondary')
        self._prev_ref_btn.clicked.connect(self._prev_ref)
        self._next_ref_btn = QPushButton()
        self._next_ref_btn.setObjectName('btn_secondary')
        self._next_ref_btn.clicked.connect(self._next_ref)
        self._ref_count_lbl = QLabel()
        ref_nav.addWidget(self._ref_nav_lbl)
        ref_nav.addWidget(self._prev_ref_btn)
        ref_nav.addWidget(self._next_ref_btn)
        ref_nav.addWidget(self._ref_count_lbl)
        ref_nav.addStretch()
        root.addLayout(ref_nav)

        # Mask nav
        mask_nav = QHBoxLayout()
        self._mask_nav_lbl = QLabel()
        self._mask_nav_lbl.setStyleSheet('color: #a78bfa; font-weight: bold;')
        self._prev_mask_btn = QPushButton()
        self._prev_mask_btn.setObjectName('btn_secondary')
        self._prev_mask_btn.clicked.connect(self._prev_mask)
        self._next_mask_btn = QPushButton()
        self._next_mask_btn.setObjectName('btn_secondary')
        self._next_mask_btn.clicked.connect(self._next_mask)
        self._mask_count_lbl = QLabel()
        mask_nav.addWidget(self._mask_nav_lbl)
        mask_nav.addWidget(self._prev_mask_btn)
        mask_nav.addWidget(self._next_mask_btn)
        mask_nav.addWidget(self._mask_count_lbl)
        mask_nav.addStretch()
        root.addLayout(mask_nav)

        # Rotation
        rot_row = QHBoxLayout()
        self._rot_label = QLabel()
        self._rot_spin = QDoubleSpinBox()
        self._rot_spin.setRange(-360.0, 360.0)
        self._rot_spin.setSingleStep(5.0)
        self._rot_spin.setValue(0.0)
        self._rot_spin.setSuffix('°')
        self._rot_spin.valueChanged.connect(self._redraw)
        self._rot_hint = QLabel()
        self._rot_hint.setStyleSheet(f'color: {FG_DIM}; font-size: 9pt;')
        rot_row.addWidget(self._rot_label)
        rot_row.addWidget(self._rot_spin)
        rot_row.addWidget(self._rot_hint)
        rot_row.addStretch()
        root.addLayout(rot_row)

        # Blend slider
        blend_row = QHBoxLayout()
        self._blend_label = QLabel()
        self._blend_slider = QSlider(Qt.Horizontal)
        self._blend_slider.setRange(10, 100)
        self._blend_slider.setValue(70)
        self._blend_slider.setFixedWidth(200)
        self._blend_slider.valueChanged.connect(self._redraw)
        blend_row.addWidget(self._blend_label)
        blend_row.addWidget(self._blend_slider)
        blend_row.addStretch()
        root.addLayout(blend_row)

        # Outline checkbox
        self._outline_chk = QCheckBox()
        self._outline_chk.setChecked(True)
        self._outline_chk.stateChanged.connect(self._redraw)
        root.addWidget(self._outline_chk)

        # Test generation
        gen_row = QHBoxLayout()
        self._gen_btn = QPushButton()
        self._gen_btn.setObjectName('btn_success')
        self._gen_btn.clicked.connect(self._start_test_generate)
        self._gen_status_lbl = QLabel()
        self._gen_status_lbl.setStyleSheet(f'color: {FG_DIM}; font-size: 9pt;')
        gen_row.addWidget(self._gen_btn)
        gen_row.addWidget(self._gen_status_lbl)
        gen_row.addStretch()
        root.addLayout(gen_row)

        # Result canvas (AI generated image)
        self._result_canvas = QLabel()
        self._result_canvas.setFixedSize(CANVAS_W, CANVAS_H)
        self._result_canvas.setAlignment(Qt.AlignCenter)
        self._result_canvas.setStyleSheet(
            'background: #0b1220; border: 2px solid #f87171; border-radius: 8px;'
        )
        self._result_canvas.hide()
        root.addWidget(self._result_canvas, 0, Qt.AlignHCenter)

        # Nav buttons
        nav = QHBoxLayout()
        self._back_btn = QPushButton()
        self._back_btn.clicked.connect(self.app.go_prev)
        self._next_btn = QPushButton()
        self._next_btn.setObjectName('btn_success')
        self._next_btn.clicked.connect(self.app.go_next)
        nav.addWidget(self._back_btn)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        root.addLayout(nav)

    def retranslate_ui(self):
        self._title.setText(self.t('page3b.title'))
        self._sub.setText(self.t('page3b.sub'))
        self._cls_label.setText(self.t('page3b.class_select'))
        self._good_nav_lbl.setText(self.t('page3b.good_label'))
        self._prev_good_btn.setText(self.t('page3b.prev'))
        self._next_good_btn.setText(self.t('page3b.next'))
        self._ref_nav_lbl.setText(self.t('page3b.ref_label'))
        self._prev_ref_btn.setText(self.t('page3b.prev'))
        self._next_ref_btn.setText(self.t('page3b.next'))
        self._mask_nav_lbl.setText(self.t('page3b.mask_label'))
        self._prev_mask_btn.setText(self.t('page3b.prev_mask'))
        self._next_mask_btn.setText(self.t('page3b.next_mask'))
        self._rot_label.setText(self.t('page3b.rotation'))
        self._blend_label.setText(self.t('page3b.blend'))
        self._outline_chk.setText(self.t('page3b.show_outline'))
        self._back_btn.setText(self.t('page3b.back'))
        self._next_btn.setText(self.t('page3b.next_step'))
        self._gen_btn.setText(self.t('page3b.test_generate'))

    def on_show(self):
        self._refresh_class_combo()

    def _refresh_class_combo(self):
        self._cls_combo.blockSignals(True)
        self._cls_combo.clear()
        for c in self.state.classes:
            self._cls_combo.addItem(c.name)
        self._cls_combo.blockSignals(False)
        self._on_class_change(self._cls_combo.currentIndex())

    def _on_class_change(self, idx):
        if idx < 0 or idx >= len(self.state.classes):
            self._good_images = []
            self._ref_images = []
            self._masks = []
            self._redraw()
            return

        cls = self.state.classes[idx]
        self._good_images = _collect_images(self.state.good_images_path)
        self._good_idx = 0
        self._ref_images = _collect_images(cls.ref_dir)
        self._ref_idx = 0
        self._masks = _collect_masks(cls.mask_dir)
        self._mask_idx = 0

        rot_mid = (cls.mask_rotation_min + cls.mask_rotation_max) / 2.0
        self._rot_spin.setValue(rot_mid)
        self._rot_hint.setText(
            f'[{cls.mask_rotation_min}° ~ {cls.mask_rotation_max}°]'
        )
        self._update_counts()
        self._redraw()

    def _update_counts(self):
        ti = len(self._good_images)
        tr_ = len(self._ref_images)
        tm = len(self._masks)
        self._good_count_lbl.setText(
            self.t('page3b.image_count', index=self._good_idx + 1 if ti else 0, total=ti))
        self._ref_count_lbl.setText(
            self.t('page3b.image_count', index=self._ref_idx + 1 if tr_ else 0, total=tr_))
        self._mask_count_lbl.setText(
            self.t('page3b.image_count', index=self._mask_idx + 1 if tm else 0, total=tm))

    def _prev_good(self):
        if self._good_images:
            self._good_idx = (self._good_idx - 1) % len(self._good_images)
            self._update_counts(); self._redraw()

    def _next_good(self):
        if self._good_images:
            self._good_idx = (self._good_idx + 1) % len(self._good_images)
            self._update_counts(); self._redraw()

    def _prev_ref(self):
        if self._ref_images:
            self._ref_idx = (self._ref_idx - 1) % len(self._ref_images)
            self._update_counts(); self._redraw()

    def _next_ref(self):
        if self._ref_images:
            self._ref_idx = (self._ref_idx + 1) % len(self._ref_images)
            self._update_counts(); self._redraw()

    def _prev_mask(self):
        if self._masks:
            self._mask_idx = (self._mask_idx - 1) % len(self._masks)
            self._update_counts(); self._redraw()

    def _next_mask(self):
        if self._masks:
            self._mask_idx = (self._mask_idx + 1) % len(self._masks)
            self._update_counts(); self._redraw()

    def _redraw(self):
        if not self._good_images:
            self._canvas.setText(self.t('page3b.no_good_images'))
            return
        if not self._ref_images:
            self._canvas.setText(self.t('page3b.no_ref_images'))
            return
        if not self._masks:
            self._canvas.setText(self.t('page3b.no_masks'))
            return

        try:
            base_pil = Image.open(self._good_images[self._good_idx]).convert('RGB')
            ref_pil  = Image.open(self._ref_images[self._ref_idx]).convert('RGB')
        except Exception:
            return

        try:
            mask_arr = cv2.imread(self._masks[self._mask_idx], cv2.IMREAD_GRAYSCALE)
            if mask_arr is None:
                return
            mask_arr = cv2.resize(mask_arr, (base_pil.width, base_pil.height),
                                  interpolation=cv2.INTER_NEAREST)
            mask_arr = (mask_arr > 127).astype(np.uint8) * 255
        except Exception:
            return

        angle = self._rot_spin.value()
        if angle != 0.0:
            mask_arr = _rotate_mask(mask_arr, angle)

        blend = self._blend_slider.value() / 100.0
        show_outline = self._outline_chk.isChecked()

        preview = _build_composite(base_pil, ref_pil, mask_arr, blend,
                                   (CANVAS_W, CANVAS_H), show_outline)
        pix = QPixmap.fromImage(ImageQt(preview))
        self._canvas.setPixmap(pix)

    # ------------------------------------------------------------------
    # Test generation (1-image AI result)
    # ------------------------------------------------------------------

    def _start_test_generate(self):
        if not self.state.connected:
            QMessageBox.warning(self, '', self.t('page3b.not_connected'))
            return
        if not self._good_images:
            QMessageBox.warning(self, '', self.t('page3b.no_good_images'))
            return
        if not self._masks:
            QMessageBox.warning(self, '', self.t('page3b.no_masks'))
            return

        idx = self._cls_combo.currentIndex()
        if idx < 0 or idx >= len(self.state.classes):
            return
        cls = self.state.classes[idx]

        good_image_path = self._good_images[self._good_idx]
        mask_path_full  = self._masks[self._mask_idx]
        rot_angle       = self._rot_spin.value()

        self._gen_btn.setEnabled(False)
        self._gen_status_lbl.setText(self.t('page3b.gen_running'))
        self._result_canvas.hide()

        def _run():
            try:
                server_url = self.state.server_url.rstrip('/')
                api_key    = self.state.api_key

                def _url(path):
                    url = f'{server_url}{path}'
                    if api_key and 'runpod.net' in server_url:
                        sep = '&' if '?' in url else '?'
                        url = f'{url}{sep}runpodApiKey={api_key}'
                    return url

                headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}

                # Build ZIP
                buf = io.BytesIO()
                good_fname = os.path.basename(good_image_path)
                with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(good_image_path, f'good_images/{good_fname}')

                    # mask: mask_path_full is an absolute path; archive relative to mask_dir
                    if cls.mask_dir and os.path.isfile(mask_path_full):
                        rel = os.path.relpath(mask_path_full, cls.mask_dir)
                        parts = rel.replace('\\', '/').split('/', 1)
                        if len(parts) == 2:
                            subfolder, fname = parts
                            zf.write(mask_path_full,
                                     f'mask_root/{cls.name}/{subfolder}/{fname}')

                    # ref images
                    if cls.ref_dir and os.path.isdir(cls.ref_dir):
                        cropped_dir = os.path.join(cls.ref_dir, 'cropped')
                        use_dir = (cropped_dir
                                   if os.path.isdir(cropped_dir)
                                   and any(True for _ in os.scandir(cropped_dir))
                                   else cls.ref_dir)
                        img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
                        for rf in os.listdir(use_dir):
                            if rf.lower().endswith(img_exts):
                                rp = os.path.join(use_dir, rf)
                                zf.write(rp, f'defect_refs/{cls.name}/{rf}')

                # Build config
                api_cfg = self.state.build_api_config()
                api_cfg['classes'] = [c for c in api_cfg['classes'] if c['name'] == cls.name]
                for c in api_cfg['classes']:
                    c['generation']['num_images'] = 1
                    c['generation']['mask_rotation_min'] = rot_angle
                    c['generation']['mask_rotation_max'] = rot_angle

                buf.seek(0)
                resp = req_lib.post(
                    _url('/jobs'),
                    files={'dataset_zip': ('dataset.zip', buf, 'application/zip')},
                    data={'config_json': json.dumps(api_cfg)},
                    timeout=300, verify=False, headers=headers,
                )
                resp.raise_for_status()
                job_id = resp.json()['job_id']

                # Poll
                for _ in range(160):
                    time.sleep(2.0)
                    try:
                        sr = req_lib.get(_url(f'/jobs/{job_id}/status'),
                                         timeout=10, verify=False, headers=headers)
                        sr.raise_for_status()
                        status = sr.json().get('status', '')
                        if status == 'done':
                            break
                        if status == 'error':
                            raise RuntimeError('サーバーで生成エラーが発生しました')
                    except RuntimeError:
                        raise
                    except Exception:
                        pass
                else:
                    raise TimeoutError('生成タイムアウト（5分超過）')

                # Download result
                dr = req_lib.get(_url(f'/jobs/{job_id}/results'),
                                 timeout=120, verify=False, headers=headers)
                dr.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(dr.content)) as zf:
                    img_files = sorted([
                        n for n in zf.namelist()
                        if n.lower().endswith(('.jpg', '.jpeg', '.png'))
                        and not n.startswith('__MACOSX')
                    ])
                    if not img_files:
                        raise RuntimeError('生成画像がZIPに含まれていません')
                    img_bytes = zf.read(img_files[0])

                ext  = img_files[0].rsplit('.', 1)[-1].lower()
                mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else 'image/png'
                b64  = base64.b64encode(img_bytes).decode()
                self._sig_result.emit(f'data:{mime};base64,{b64}')

            except Exception as e:
                self._sig_error.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def _show_result(self, data_url: str):
        self._gen_btn.setEnabled(True)
        self._gen_status_lbl.setText(self.t('page3b.gen_done'))
        try:
            header, b64data = data_url.split(',', 1)
            img_bytes = base64.b64decode(b64data)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            pix = QPixmap.fromImage(ImageQt(pil_img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)))
            self._result_canvas.setPixmap(pix)
            self._result_canvas.show()
        except Exception as e:
            self._gen_status_lbl.setText(f'表示エラー: {e}')

    def _show_gen_error(self, msg: str):
        self._gen_btn.setEnabled(True)
        self._gen_status_lbl.setText(f'❌ {msg}')
