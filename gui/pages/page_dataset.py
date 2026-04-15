import os
import shutil
from typing import Optional

from PIL import Image, ImageDraw, ImageOps
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from gui.app_state import ClassConfig
from gui.i18n import tr
from gui.theme import FG_DIM


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


class RegionAnnotatorDialog(QDialog):
    def __init__(self, image_path: str, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 720)
        self._image_path = image_path
        self._base = Image.open(image_path).convert('RGB')
        self._mask = Image.new('L', self._base.size, 0)
        self._display_size = self._fit_size(self._base.size, (760, 520))
        self._start = None
        self._current_rect = None

        root = QVBoxLayout(self)
        self._hint = QLabel()
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f'color: {FG_DIM};')
        root.addWidget(self._hint)

        self._canvas = QLabel()
        self._canvas.setFixedSize(*self._display_size)
        self._canvas.setAlignment(Qt.AlignCenter)
        self._canvas.setStyleSheet('background: #0b1220; border: 2px solid #7c3aed;')
        self._canvas.mousePressEvent = self._mouse_press
        self._canvas.mouseMoveEvent = self._mouse_move
        self._canvas.mouseReleaseEvent = self._mouse_release
        root.addWidget(self._canvas, 0, Qt.AlignCenter)

        self._note = QTextEdit()
        self._note.setPlaceholderText('例: 左上に線傷 / Example: scratch on the upper-left area')
        self._note.setMaximumHeight(90)
        root.addWidget(self._note)

        btns = QHBoxLayout()
        self._clear_btn = QPushButton('Clear Region')
        self._clear_btn.setObjectName('btn_secondary')
        self._clear_btn.clicked.connect(self._clear)
        self._ok_btn = QPushButton('Use This Annotation')
        self._ok_btn.setObjectName('btn_success')
        self._ok_btn.clicked.connect(self.accept)
        self._cancel_btn = QPushButton('Cancel')
        self._cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self._clear_btn)
        btns.addStretch()
        btns.addWidget(self._cancel_btn)
        btns.addWidget(self._ok_btn)
        root.addLayout(btns)

        self._hint.setText('ドラッグして不良領域を矩形で指定します。白い領域が不良マスクとして保存されます。 / Drag to mark the defect area. The white region will be saved as the defect mask.')
        self._redraw()

    def _fit_size(self, size, max_size):
        w, h = size
        mw, mh = max_size
        scale = min(mw / w, mh / h, 1.0)
        return max(1, int(w * scale)), max(1, int(h * scale))

    def _to_image_xy(self, pos):
        x = min(max(int(pos.position().x()), 0), self._display_size[0] - 1)
        y = min(max(int(pos.position().y()), 0), self._display_size[1] - 1)
        ix = int(x * self._base.size[0] / self._display_size[0])
        iy = int(y * self._base.size[1] / self._display_size[1])
        return ix, iy

    def _mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self._start = self._to_image_xy(event)
            self._current_rect = (*self._start, *self._start)
            self._redraw()

    def _mouse_move(self, event):
        if self._start is not None and event.buttons() & Qt.LeftButton:
            x, y = self._to_image_xy(event)
            self._current_rect = (self._start[0], self._start[1], x, y)
            self._redraw()

    def _mouse_release(self, event):
        if self._start is not None and event.button() == Qt.LeftButton:
            x, y = self._to_image_xy(event)
            x0, y0 = self._start
            draw = ImageDraw.Draw(self._mask)
            draw.rectangle((min(x0, x), min(y0, y), max(x0, x), max(y0, y)), fill=255)
            self._start = None
            self._current_rect = None
            self._redraw()

    def _clear(self):
        self._mask = Image.new('L', self._base.size, 0)
        self._start = None
        self._current_rect = None
        self._redraw()

    def _redraw(self):
        base = self._base.copy().convert('RGBA')
        red = Image.new('RGBA', base.size, (220, 50, 50, 160))
        base.paste(red, mask=self._mask)
        if self._current_rect is not None:
            draw = ImageDraw.Draw(base)
            x0, y0, x1, y1 = self._current_rect
            draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0, 255), width=max(2, base.size[0] // 220))
        pix = QPixmap.fromImage(ImageQt(base.resize(self._display_size).convert('RGB')))
        self._canvas.setPixmap(pix)

    def get_mask_and_note(self):
        return self._mask, self._note.toPlainText().strip()


class ReferencePrepDialog(QDialog):
    def __init__(self, state, cls: ClassConfig, tfunc, parent=None):
        super().__init__(parent)
        self.state = state
        self.cls = cls
        self.t = tfunc
        self.images = []
        self.index = -1
        self.current_mask = None
        self.resize(1180, 760)
        self._build_ui()
        self._load_images()
        self._refresh()

    def _build_ui(self):
        self.setWindowTitle(self.t('page2.ref_helper_title', name=self.cls.name))
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        self._title = QLabel()
        self._title.setStyleSheet('font-size: 14pt; font-weight: bold;')
        self._title.setText(self.t('page2.ref_helper_title', name=self.cls.name))
        left_v.addWidget(self._title)

        self._hint = QLabel(self.t('page2.ref_helper_hint'))
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f'color: {FG_DIM};')
        left_v.addWidget(self._hint)

        self._image_label = QLabel()
        self._image_label.setFixedSize(760, 560)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet('background: #0b1220; border: 2px solid #7c3aed; border-radius: 8px;')
        left_v.addWidget(self._image_label)

        nav = QHBoxLayout()
        self._prev_btn = QPushButton(self.t('page2.prev_image'))
        self._prev_btn.setObjectName('btn_secondary')
        self._prev_btn.clicked.connect(self._prev)
        self._next_btn = QPushButton(self.t('page2.next_image'))
        self._next_btn.setObjectName('btn_secondary')
        self._next_btn.clicked.connect(self._next)
        self._progress_label = QLabel()
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._next_btn)
        nav.addWidget(self._progress_label)
        nav.addStretch()
        left_v.addLayout(nav)

        root.addWidget(left, 1)

        right = QWidget()
        right.setFixedWidth(300)
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        self._source_label = QLabel(self.t('page2.mixed_dir'))
        self._source_edit = QLineEdit(self.cls.mixed_images_dir)
        self._source_btn = QPushButton(self.t('common.browse'))
        self._source_btn.setObjectName('btn_secondary')
        self._source_btn.clicked.connect(self._pick_source)
        row = QHBoxLayout(); row.addWidget(self._source_edit); row.addWidget(self._source_btn)
        right_v.addWidget(self._source_label)
        right_v.addLayout(row)

        self._stats = QLabel()
        self._stats.setWordWrap(True)
        self._stats.setStyleSheet('background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px;')
        right_v.addWidget(self._stats)

        self._good_btn = QPushButton(self.t('page2.mark_good'))
        self._good_btn.setObjectName('btn_success')
        self._good_btn.clicked.connect(self._mark_good)
        self._defect_btn = QPushButton(self.t('page2.mark_defect'))
        self._defect_btn.setObjectName('btn_primary')
        self._defect_btn.clicked.connect(self._mark_defect)
        self._skip_btn = QPushButton(self.t('page2.skip_image'))
        self._skip_btn.setObjectName('btn_secondary')
        self._skip_btn.clicked.connect(self._next)
        self._reload_btn = QPushButton(self.t('page2.reload_images'))
        self._reload_btn.setObjectName('btn_secondary')
        self._reload_btn.clicked.connect(self._reload)
        for btn in [self._good_btn, self._defect_btn, self._skip_btn, self._reload_btn]:
            btn.setMinimumHeight(46)
            right_v.addWidget(btn)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(260)
        right_v.addWidget(self._log, 1)

        self._close_btn = QPushButton(self.t('common.close'))
        self._close_btn.clicked.connect(self.accept)
        right_v.addWidget(self._close_btn)
        root.addWidget(right)

    def _pick_source(self):
        path = QFileDialog.getExistingDirectory(self, self.t('common.browse'), self._source_edit.text() or '')
        if path:
            self._source_edit.setText(path)
            self.cls.mixed_images_dir = path
            self._reload()

    def _ensure_dirs(self):
        dirs = [self.cls.good_pool_dir or self.state.good_images_path, self.cls.ref_dir, self.cls.defect_mask_dir]
        for path in dirs:
            if path:
                os.makedirs(path, exist_ok=True)

    def _load_images(self):
        source = self._source_edit.text().strip()
        self.images = []
        if source and os.path.isdir(source):
            self.images = [os.path.join(source, n) for n in sorted(os.listdir(source)) if n.lower().endswith(IMAGE_EXTS)]
        self.index = 0 if self.images else -1

    def _reload(self):
        self._load_images()
        self._refresh()

    def _current_path(self) -> Optional[str]:
        if 0 <= self.index < len(self.images):
            return self.images[self.index]
        return None

    def _refresh(self):
        path = self._current_path()
        if not path:
            self._image_label.setText(self.t('page2.no_images_found'))
            self._progress_label.setText('')
        else:
            img = Image.open(path).convert('RGB')
            disp = img.copy()
            disp.thumbnail((760, 560))
            self._image_label.setPixmap(QPixmap.fromImage(ImageQt(disp)))
            self._progress_label.setText(self.t('page2.image_progress', index=self.index + 1, total=len(self.images)))
        good_count = len(os.listdir(self.cls.good_pool_dir)) if self.cls.good_pool_dir and os.path.isdir(self.cls.good_pool_dir) else 0
        ref_count = len([n for n in os.listdir(self.cls.ref_dir)]) if self.cls.ref_dir and os.path.isdir(self.cls.ref_dir) else 0
        mask_count = len([n for n in os.listdir(self.cls.defect_mask_dir)]) if self.cls.defect_mask_dir and os.path.isdir(self.cls.defect_mask_dir) else 0
        self._stats.setText(self.t('page2.ref_stats', good=good_count, defect=ref_count, mask=mask_count))

    def _copy_unique(self, src, dst_dir, suffix=''):
        os.makedirs(dst_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        ext = os.path.splitext(src)[1] or '.png'
        i = 0
        while True:
            name = f'{base}{suffix}_{i:03d}{ext}' if suffix else f'{base}_{i:03d}{ext}'
            path = os.path.join(dst_dir, name)
            if not os.path.exists(path):
                shutil.copy2(src, path)
                return path
            i += 1

    def _next(self):
        if self.images and self.index < len(self.images) - 1:
            self.index += 1
        self._refresh()

    def _prev(self):
        if self.images and self.index > 0:
            self.index -= 1
        self._refresh()

    def _mark_good(self):
        src = self._current_path()
        if not src:
            return
        if not self.cls.good_pool_dir:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page2.good_pool_missing'))
            return
        self._ensure_dirs()
        out = self._copy_unique(src, self.cls.good_pool_dir, '_good')
        self._log.append(self.t('page2.good_saved', path=out))
        self._next()

    def _mark_defect(self):
        src = self._current_path()
        if not src:
            return
        if not self.cls.ref_dir or not self.cls.defect_mask_dir:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page2.ref_paths_missing'))
            return
        dlg = RegionAnnotatorDialog(src, self.t('page2.annotate_title'), self)
        if dlg.exec() != QDialog.Accepted:
            return
        mask, note = dlg.get_mask_and_note()
        if mask.getbbox() is None:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page2.annotation_required'))
            return
        self._ensure_dirs()
        ref_path = self._copy_unique(src, self.cls.ref_dir, '_defect')
        base = os.path.splitext(os.path.basename(ref_path))[0]
        mask_path = os.path.join(self.cls.defect_mask_dir, f'{base}_mask.png')
        mask.save(mask_path)
        if note:
            with open(os.path.join(self.cls.defect_mask_dir, f'{base}_note.txt'), 'w', encoding='utf-8') as f:
                f.write(note)
        self._log.append(self.t('page2.defect_saved', ref=ref_path, mask=mask_path))
        self._next()


class DatasetPage(QWidget):
    def __init__(self, state, app):
        super().__init__()
        self.state = state
        self.app = app
        self._selected_idx = None
        self._field_labels = {}
        self._build_ui()
        self.retranslate_ui()

    def t(self, key, **kwargs):
        return tr(self.state, key, **kwargs)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        self._title = QLabel(); self._title.setStyleSheet('font-size: 16pt; font-weight: bold;')
        self._sub = QLabel(); self._sub.setWordWrap(True); self._sub.setStyleSheet(f'color: {FG_DIM};')
        root.addWidget(self._title); root.addWidget(self._sub)

        top_card = QWidget()
        top_card.setStyleSheet('background: rgba(255,255,255,0.03); border-radius: 8px;')
        form = QGridLayout(top_card)
        form.setContentsMargins(16, 16, 16, 16)
        root.addWidget(top_card)

        self._job_label = QLabel(); self._job_edit = QLineEdit(self.state.job_name)
        self._good_label = QLabel(); self._good_edit = QLineEdit(self.state.good_images_path)
        self._good_btn = QPushButton(); self._good_btn.setObjectName('btn_secondary'); self._good_btn.clicked.connect(lambda: self._pick_dir(self._good_edit))
        self._out_label = QLabel(); self._out_edit = QLineEdit(self.state.output_root)
        self._out_btn = QPushButton(); self._out_btn.setObjectName('btn_secondary'); self._out_btn.clicked.connect(lambda: self._pick_dir(self._out_edit))
        self._size_label = QLabel(); self._w_spin = QSpinBox(); self._w_spin.setRange(32, 8192); self._w_spin.setValue(self.state.image_width)
        self._h_spin = QSpinBox(); self._h_spin.setRange(32, 8192); self._h_spin.setValue(self.state.image_height)
        form.addWidget(self._job_label, 0, 0); form.addWidget(self._job_edit, 0, 1, 1, 2)
        form.addWidget(self._good_label, 1, 0); form.addWidget(self._good_edit, 1, 1); form.addWidget(self._good_btn, 1, 2)
        form.addWidget(self._out_label, 2, 0); form.addWidget(self._out_edit, 2, 1); form.addWidget(self._out_btn, 2, 2)
        size_row = QHBoxLayout(); size_row.addWidget(self._w_spin); size_row.addWidget(QLabel('×')); size_row.addWidget(self._h_spin); size_row.addStretch()
        form.addWidget(self._size_label, 3, 0); form.addLayout(size_row, 3, 1, 1, 2)

        body = QHBoxLayout(); root.addLayout(body, 1)
        left = QWidget(); left.setFixedWidth(270); left.setStyleSheet('background: rgba(255,255,255,0.03); border-radius: 8px;')
        left_v = QVBoxLayout(left); left_v.setContentsMargins(12, 12, 12, 12)
        self._class_title = QLabel(); self._class_title.setStyleSheet('font-size: 12pt; font-weight: bold;')
        self._class_list = QListWidget(); self._class_list.currentRowChanged.connect(self._on_class_select)
        self._add_btn = QPushButton(); self._add_btn.setObjectName('btn_primary'); self._add_btn.clicked.connect(self._add_class)
        self._remove_btn = QPushButton(); self._remove_btn.setObjectName('btn_danger'); self._remove_btn.clicked.connect(self._remove_class)
        self._create_defaults_btn = QPushButton(); self._create_defaults_btn.setObjectName('btn_secondary'); self._create_defaults_btn.clicked.connect(self._create_default_folders_for_current)
        left_v.addWidget(self._class_title); left_v.addWidget(self._class_list, 1)
        r = QHBoxLayout(); r.addWidget(self._add_btn); r.addWidget(self._remove_btn); left_v.addLayout(r)
        left_v.addWidget(self._create_defaults_btn)
        body.addWidget(left)

        self._detail_card = QWidget(); self._detail_card.setStyleSheet('background: rgba(255,255,255,0.03); border-radius: 8px;')
        self._detail_layout = QVBoxLayout(self._detail_card); self._detail_layout.setContentsMargins(16, 16, 16, 16)
        body.addWidget(self._detail_card, 1)
        self._detail_title = QLabel(); self._detail_title.setStyleSheet('font-size: 12pt; font-weight: bold;')
        self._empty_label = QLabel(); self._empty_label.setStyleSheet(f'color: {FG_DIM};'); self._empty_label.setAlignment(Qt.AlignCenter)
        self._detail_layout.addWidget(self._detail_title); self._detail_layout.addWidget(self._empty_label, 1)

        self._form_wrap = QWidget(); self._form_layout = QFormLayout(self._form_wrap); self._form_layout.setSpacing(10); self._form_wrap.hide(); self._detail_layout.addWidget(self._form_wrap)
        self._name_edit = QLineEdit(); self._id_spin = QSpinBox(); self._id_spin.setRange(0, 999999)
        self._mask_edit = QLineEdit(); self._mask_btn = QPushButton(); self._mask_btn.setObjectName('btn_secondary'); self._mask_btn.clicked.connect(lambda: self._pick_dir(self._mask_edit))
        self._ref_edit = QLineEdit(); self._ref_btn = QPushButton(); self._ref_btn.setObjectName('btn_secondary'); self._ref_btn.clicked.connect(lambda: self._pick_dir(self._ref_edit))
        self._num_spin = QSpinBox(); self._num_spin.setRange(1, 100000)
        self._neg_edit = QLineEdit()
        self._gs_spin = QDoubleSpinBox(); self._gs_spin.setRange(0.0, 100.0); self._gs_spin.setValue(7.5)
        self._steps_spin = QSpinBox(); self._steps_spin.setRange(1, 1000); self._steps_spin.setValue(30)
        self._strength_spin = QDoubleSpinBox(); self._strength_spin.setRange(0.0, 1.0); self._strength_spin.setSingleStep(0.05); self._strength_spin.setValue(0.75)
        self._ip_spin = QDoubleSpinBox(); self._ip_spin.setRange(0.0, 5.0); self._ip_spin.setSingleStep(0.05); self._ip_spin.setValue(0.35)

        # Defect type selector
        self._defect_type_combo = QComboBox()
        self._defect_type_combo.addItem('-- 種類を選んでプロンプトを自動設定 --', '')
        self._defect_type_combo.addItem('整形不良（成形不良・ウェルドライン等）', '整形不良')
        self._defect_type_combo.addItem('バリ（フラッシュ）', 'バリ')
        self._defect_type_combo.addItem('ヒケ（収縮痕）', 'ヒケ')
        self._defect_type_combo.addItem('気泡・ボイド', '気泡・ボイド')
        self._defect_type_combo.addItem('傷・スクラッチ', '傷・スクラッチ')
        self._defect_type_combo.addItem('変色・焼け', '変色・焼け')
        self._defect_type_combo.addItem('欠け・クラック', '欠け・クラック')
        self._defect_type_combo.addItem('異物（コンタミ）', '異物')
        self._defect_type_combo.addItem('すべての種類（全39候補）', 'すべて')
        self._defect_type_combo.currentIndexChanged.connect(self._apply_defect_type_prompts)

        # Prompts multi-line editor
        self._prompts_edit = QTextEdit()
        self._prompts_edit.setPlaceholderText('1行に1つのプロンプトを入力してください')
        self._prompts_edit.setMinimumHeight(120)
        self._prompts_hint = QLabel()
        self._prompts_hint.setStyleSheet(f'color: {FG_DIM}; font-size: 9pt;')
        self._load_default_prompts_btn = QPushButton()
        self._load_default_prompts_btn.setObjectName('btn_secondary')
        self._load_default_prompts_btn.clicked.connect(self._load_default_prompts)
        prompts_widget = QWidget()
        prompts_layout = QVBoxLayout(prompts_widget)
        prompts_layout.setContentsMargins(0, 0, 0, 0)
        prompts_layout.addWidget(self._defect_type_combo)
        prompts_layout.addWidget(self._prompts_edit)
        prompts_layout.addWidget(self._prompts_hint)
        prompts_layout.addWidget(self._load_default_prompts_btn)

        # Rotation fields
        self._rot_min_spin = QDoubleSpinBox(); self._rot_min_spin.setRange(-360.0, 360.0); self._rot_min_spin.setSingleStep(10.0); self._rot_min_spin.setSuffix('°')
        self._rot_max_spin = QDoubleSpinBox(); self._rot_max_spin.setRange(-360.0, 360.0); self._rot_max_spin.setSingleStep(10.0); self._rot_max_spin.setSuffix('°')
        self._rot_hint_lbl = QLabel()
        self._rot_hint_lbl.setStyleSheet(f'color: {FG_DIM}; font-size: 9pt;')
        rot_widget = QWidget()
        rot_layout = QHBoxLayout(rot_widget)
        rot_layout.setContentsMargins(0, 0, 0, 0)
        rot_layout.addWidget(self._rot_min_spin)
        rot_layout.addWidget(QLabel('~'))
        rot_layout.addWidget(self._rot_max_spin)
        rot_layout.addWidget(self._rot_hint_lbl)
        rot_layout.addStretch()

        self._add_form_row('page2.class_name', self._name_edit)
        self._add_form_row('page2.class_id', self._id_spin)
        self._add_form_row('page2.mask_dir', self._with_button(self._mask_edit, self._mask_btn))
        self._add_form_row('page2.ref_dir', self._with_button(self._ref_edit, self._ref_btn))
        self._add_form_row('page2.num_images', self._num_spin)
        self._add_form_row('page2.prompt', prompts_widget)
        self._add_form_row('page2.neg_prompt', self._neg_edit)
        self._add_form_row('page2.guidance', self._gs_spin)
        self._add_form_row('page2.steps', self._steps_spin)
        self._add_form_row('page2.strength', self._strength_spin)
        self._add_form_row('page2.ip_scale', self._ip_spin)
        self._add_form_row('page2.rotation_min', rot_widget)

        self._ref_group = QGroupBox(); self._ref_group.setStyleSheet('margin-top: 8px;')
        ref_layout = QVBoxLayout(self._ref_group)
        self._ref_desc = QLabel(); self._ref_desc.setWordWrap(True); self._ref_desc.setStyleSheet(f'color: {FG_DIM};')
        ref_layout.addWidget(self._ref_desc)
        mode_row = QHBoxLayout()
        self._has_ref_yes = QRadioButton(); self._has_ref_no = QRadioButton()
        self._ref_mode_group = QButtonGroup(self); self._ref_mode_group.addButton(self._has_ref_yes); self._ref_mode_group.addButton(self._has_ref_no)
        self._has_ref_yes.toggled.connect(self._update_ref_mode_ui); self._has_ref_no.toggled.connect(self._update_ref_mode_ui)
        mode_row.addWidget(self._has_ref_yes); mode_row.addWidget(self._has_ref_no); mode_row.addStretch(); ref_layout.addLayout(mode_row)

        self._mixed_edit = QLineEdit(); self._mixed_btn = QPushButton(); self._mixed_btn.setObjectName('btn_secondary'); self._mixed_btn.clicked.connect(lambda: self._pick_dir(self._mixed_edit))
        self._good_pool_edit = QLineEdit(); self._good_pool_btn = QPushButton(); self._good_pool_btn.setObjectName('btn_secondary'); self._good_pool_btn.clicked.connect(lambda: self._pick_dir(self._good_pool_edit))
        self._defect_mask_edit = QLineEdit(); self._defect_mask_btn = QPushButton(); self._defect_mask_btn.setObjectName('btn_secondary'); self._defect_mask_btn.clicked.connect(lambda: self._pick_dir(self._defect_mask_edit))
        self._ref_fields = QWidget(); ref_form = QFormLayout(self._ref_fields)
        self._mixed_label = QLabel(); self._good_pool_label = QLabel(); self._defect_mask_label = QLabel()
        ref_form.addRow(self._mixed_label, self._with_button(self._mixed_edit, self._mixed_btn))
        ref_form.addRow(self._good_pool_label, self._with_button(self._good_pool_edit, self._good_pool_btn))
        ref_form.addRow(self._defect_mask_label, self._with_button(self._defect_mask_edit, self._defect_mask_btn))
        ref_layout.addWidget(self._ref_fields)

        helper_row = QHBoxLayout()
        self._open_helper_btn = QPushButton(); self._open_helper_btn.setObjectName('btn_primary'); self._open_helper_btn.clicked.connect(self._open_reference_helper)
        self._save_btn = QPushButton(); self._save_btn.setObjectName('btn_success'); self._save_btn.clicked.connect(self._save_class)
        helper_row.addWidget(self._open_helper_btn); helper_row.addStretch(); helper_row.addWidget(self._save_btn)
        ref_layout.addLayout(helper_row)
        self._form_layout.addRow('', self._ref_group)

        nav = QHBoxLayout(); self._back_btn = QPushButton(); self._back_btn.clicked.connect(self.app.go_prev); self._next_btn = QPushButton(); self._next_btn.setObjectName('btn_success'); self._next_btn.clicked.connect(self._save_and_next)
        nav.addWidget(self._back_btn); nav.addStretch(); nav.addWidget(self._next_btn); root.addLayout(nav)

    def _with_button(self, edit, button):
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.addWidget(edit); l.addWidget(button); return w

    def _add_form_row(self, key, widget):
        label = QLabel(); self._field_labels[key] = label; self._form_layout.addRow(label, widget)

    def retranslate_ui(self):
        self._title.setText(self.t('page2.title')); self._sub.setText(self.t('page2.sub')); self._job_label.setText(self.t('page2.job_name'))
        self._good_label.setText(self.t('page2.good_dir')); self._good_btn.setText(self.t('common.browse')); self._out_label.setText(self.t('page2.out_dir')); self._out_btn.setText(self.t('common.browse'))
        self._size_label.setText(self.t('page2.image_size')); self._class_title.setText(self.t('page2.classes')); self._add_btn.setText(self.t('page2.add')); self._remove_btn.setText(self.t('page2.remove'))
        self._create_defaults_btn.setText(self.t('page2.create_defaults')); self._empty_label.setText(self.t('page2.no_class_selected')); self._detail_title.setText(self.t('page2.class_detail'))
        for key, label in self._field_labels.items(): label.setText(self.t(key))
        self._mask_btn.setText(self.t('common.browse')); self._ref_btn.setText(self.t('common.browse')); self._back_btn.setText(self.t('page2.back')); self._next_btn.setText(self.t('page2.next'))
        self._prompts_hint.setText(self.t('page2.prompt_hint'))
        self._load_default_prompts_btn.setText(self.t('page2.load_default_prompts'))
        self._rot_hint_lbl.setText(self.t('page2.rotation_hint'))
        self._ref_group.setTitle(self.t('page2.ref_group')); self._ref_desc.setText(self.t('page2.ref_question')); self._has_ref_yes.setText(self.t('page2.ref_yes')); self._has_ref_no.setText(self.t('page2.ref_no'))
        self._mixed_label.setText(self.t('page2.mixed_dir')); self._mixed_btn.setText(self.t('common.browse')); self._good_pool_label.setText(self.t('page2.good_pool_dir')); self._good_pool_btn.setText(self.t('common.browse')); self._defect_mask_label.setText(self.t('page2.defect_mask_dir')); self._defect_mask_btn.setText(self.t('common.browse'))
        self._open_helper_btn.setText(self.t('page2.open_helper')); self._save_btn.setText(self.t('page2.save_class'))
        self._refresh_class_list(); self._update_detail()

    def _project_base_dir(self):
        good = self._good_edit.text().strip()
        if good:
            return os.path.dirname(good)
        return ''

    def _default_dirs(self, class_name: str):
        base = self._project_base_dir()
        if not base:
            good_pool = self.state.good_images_path or ''
            return '', '', '', good_pool
        good_pool = self.state.good_images_path or os.path.join(base, 'good_pool', class_name)
        return (
            os.path.join(base, 'mask_root', class_name),
            os.path.join(base, 'defect_refs', class_name),
            os.path.join(base, 'defect_masks', class_name),
            good_pool,
        )

    def _pick_dir(self, edit):
        path = QFileDialog.getExistingDirectory(self, self.t('common.browse'), edit.text() or '')
        if path:
            edit.setText(path)

    def _add_class(self):
        text, ok = QInputDialog.getText(self, self.t('page2.add'), self.t('page2.new_class'))
        if not ok or not text.strip():
            return
        name = text.strip()
        if any(c.name == name for c in self.state.classes):
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page2.duplicate'))
            return
        mask_dir, ref_dir, defect_mask_dir, good_pool_dir = self._default_dirs(name)
        cls = ClassConfig(name=name, class_id=len(self.state.classes), mask_dir=mask_dir, ref_dir=ref_dir, defect_mask_dir=defect_mask_dir, good_pool_dir=good_pool_dir)
        self.state.classes.append(cls)
        self._refresh_class_list()
        self._class_list.setCurrentRow(len(self.state.classes) - 1)

    def _remove_class(self):
        row = self._class_list.currentRow()
        if row < 0:
            return
        cls = self.state.classes[row]
        if QMessageBox.question(self, self.t('msg.confirm'), self.t('page2.remove_confirm', name=cls.name)) != QMessageBox.Yes:
            return
        self.state.classes.pop(row)
        self._selected_idx = None
        self._refresh_class_list(); self._update_detail()

    def _refresh_class_list(self):
        current = self._class_list.currentRow()
        self._class_list.blockSignals(True)
        self._class_list.clear()
        for c in self.state.classes:
            self._class_list.addItem(c.name)
        self._class_list.blockSignals(False)
        if self.state.classes:
            if 0 <= current < len(self.state.classes):
                self._class_list.setCurrentRow(current)
            elif self._selected_idx is not None and 0 <= self._selected_idx < len(self.state.classes):
                self._class_list.setCurrentRow(self._selected_idx)

    def _on_class_select(self, row):
        self._selected_idx = row if row >= 0 else None
        self._update_detail()

    def _update_ref_mode_ui(self):
        has_ref = self._has_ref_yes.isChecked()
        self._ref_fields.setVisible(not has_ref)
        self._open_helper_btn.setVisible(not has_ref)

    def _update_detail(self):
        cls = self.state.classes[self._selected_idx] if self._selected_idx is not None and self._selected_idx < len(self.state.classes) else None
        has_cls = cls is not None
        self._form_wrap.setVisible(has_cls); self._empty_label.setVisible(not has_cls)
        if not has_cls:
            return
        self._detail_title.setText(self.t('page2.class_detail_name', name=cls.name))
        self._name_edit.setText(cls.name); self._id_spin.setValue(cls.class_id); self._mask_edit.setText(cls.mask_dir); self._ref_edit.setText(cls.ref_dir); self._num_spin.setValue(cls.num_images)
        self._defect_type_combo.blockSignals(True)
        self._defect_type_combo.setCurrentIndex(0)
        self._defect_type_combo.blockSignals(False)
        self._prompts_edit.setPlainText('\n'.join(cls.prompts))
        self._neg_edit.setText(cls.negative_prompt); self._gs_spin.setValue(cls.guidance_scale); self._steps_spin.setValue(cls.steps); self._strength_spin.setValue(cls.strength); self._ip_spin.setValue(cls.ip_scale)
        self._rot_min_spin.setValue(cls.mask_rotation_min); self._rot_max_spin.setValue(cls.mask_rotation_max)
        self._mixed_edit.setText(cls.mixed_images_dir); self._defect_mask_edit.setText(cls.defect_mask_dir); self._good_pool_edit.setText(cls.good_pool_dir or self.state.good_images_path)
        self._has_ref_yes.setChecked(cls.has_reference_images); self._has_ref_no.setChecked(not cls.has_reference_images); self._update_ref_mode_ui()

    def _ensure_default_dirs(self, cls: ClassConfig):
        mask_dir, ref_dir, defect_mask_dir, good_pool_dir = self._default_dirs(cls.name)
        if not cls.mask_dir: cls.mask_dir = mask_dir
        if not cls.ref_dir: cls.ref_dir = ref_dir
        if not cls.defect_mask_dir: cls.defect_mask_dir = defect_mask_dir
        if not cls.good_pool_dir: cls.good_pool_dir = good_pool_dir or self.state.good_images_path

    def _create_default_folders_for_current(self):
        cls = self.state.classes[self._selected_idx] if self._selected_idx is not None and self._selected_idx < len(self.state.classes) else None
        if not cls:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page2.no_class_selected'))
            return
        self._save_class(silent=True)
        for p in [cls.mask_dir, cls.ref_dir, cls.defect_mask_dir, cls.good_pool_dir]:
            if p:
                os.makedirs(p, exist_ok=True)
        QMessageBox.information(self, self.t('msg.info'), self.t('page2.default_created'))
        self._update_detail()

    def _save_class(self, silent=False):
        if self._selected_idx is None or self._selected_idx >= len(self.state.classes):
            return
        cls = self.state.classes[self._selected_idx]
        old_name = cls.name
        cls.name = self._name_edit.text().strip() or old_name
        cls.class_id = self._id_spin.value(); cls.mask_dir = self._mask_edit.text().strip(); cls.ref_dir = self._ref_edit.text().strip(); cls.num_images = self._num_spin.value()
        raw_prompts = [p.strip() for p in self._prompts_edit.toPlainText().splitlines() if p.strip()]
        cls.prompts = raw_prompts if raw_prompts else ['a defect on a product surface, realistic']
        cls.negative_prompt = self._neg_edit.text().strip(); cls.guidance_scale = self._gs_spin.value(); cls.steps = self._steps_spin.value(); cls.strength = self._strength_spin.value(); cls.ip_scale = self._ip_spin.value()
        cls.mask_rotation_min = self._rot_min_spin.value(); cls.mask_rotation_max = self._rot_max_spin.value()
        cls.has_reference_images = self._has_ref_yes.isChecked(); cls.mixed_images_dir = self._mixed_edit.text().strip(); cls.defect_mask_dir = self._defect_mask_edit.text().strip(); cls.good_pool_dir = self._good_pool_edit.text().strip() or self.state.good_images_path
        self._ensure_default_dirs(cls)
        if not silent:
            self._refresh_class_list(); self._class_list.setCurrentRow(self._selected_idx); QMessageBox.information(self, self.t('msg.info'), self.t('page2.saved', name=cls.name))

    def _apply_defect_type_prompts(self):
        from gui.app_state import DEFAULT_PROMPTS
        defect_type_map = {
            '整形不良': DEFAULT_PROMPTS[0:9],
            'バリ': DEFAULT_PROMPTS[9:12],
            'ヒケ': DEFAULT_PROMPTS[12:15],
            '気泡・ボイド': DEFAULT_PROMPTS[15:18],
            '傷・スクラッチ': DEFAULT_PROMPTS[18:22],
            '変色・焼け': DEFAULT_PROMPTS[22:26],
            '欠け・クラック': DEFAULT_PROMPTS[26:29],
            '異物': DEFAULT_PROMPTS[29:],
            'すべて': DEFAULT_PROMPTS,
        }
        key = self._defect_type_combo.currentData()
        if not key:
            return
        prompts = defect_type_map.get(key, [])
        if prompts:
            self._prompts_edit.setPlainText('\n'.join(prompts))

    def _load_default_prompts(self):
        from gui.app_state import DEFAULT_PROMPTS
        self._prompts_edit.setPlainText('\n'.join(DEFAULT_PROMPTS))

    def _open_reference_helper(self):
        if self._selected_idx is None or self._selected_idx >= len(self.state.classes):
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page2.no_class_selected'))
            return
        self._save_class(silent=True)
        cls = self.state.classes[self._selected_idx]
        dlg = ReferencePrepDialog(self.state, cls, self.t, self)
        dlg.exec()
        self._update_detail()

    def _save_global(self):
        self.state.job_name = self._job_edit.text().strip() or self.state.job_name
        self.state.good_images_path = self._good_edit.text().strip()
        self.state.output_root = self._out_edit.text().strip()
        self.state.image_width = self._w_spin.value(); self.state.image_height = self._h_spin.value()

    def _save_and_next(self):
        self._save_global()
        if self._selected_idx is not None:
            self._save_class(silent=True)
        self.app.go_next()


    def on_hide(self):
        self._save_global()
        if self._selected_idx is not None:
            self._save_class(silent=True)

    def on_show(self):
        self._job_edit.setText(self.state.job_name)
        self._good_edit.setText(self.state.good_images_path)
        self._out_edit.setText(self.state.output_root)
        self._w_spin.setValue(self.state.image_width); self._h_spin.setValue(self.state.image_height)
        self._refresh_class_list(); self._update_detail()
