import os
import shutil

from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr
from gui.theme import FG_DIM

CANVAS_W = 640
CANVAS_H = 480


class MaskCanvas(QLabel):
    pressed = Signal(int, int)
    moved = Signal(int, int)
    released = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setFixedSize(CANVAS_W, CANVAS_H)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pressed.emit(int(event.position().x()), int(event.position().y()))

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.moved.emit(int(event.position().x()), int(event.position().y()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.released.emit(int(event.position().x()), int(event.position().y()))


class MaskEditorPage(QWidget):
    def __init__(self, state, app):
        super().__init__()
        self.state = state
        self.app = app

        self._mask_img = Image.new('L', (CANVAS_W, CANVAS_H), 0)
        self._bg_img = None
        self._undo_stack = []
        self._tool = 'brush'
        self._brush_size = 25
        self._drawing = False
        self._last_xy = None
        self._rect_start = None
        self._selected_class = ''
        self._preview_rect = None

        self._build_ui()
        self.retranslate_ui()
        self._redraw()

    def t(self, key, **kwargs):
        return tr(self.state, key, **kwargs)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)

        self._title = QLabel()
        self._title.setStyleSheet('font-size: 16pt; font-weight: bold;')
        root.addWidget(self._title)

        self._sub = QLabel()
        self._sub.setWordWrap(True)
        self._sub.setStyleSheet(f'color: {FG_DIM};')
        root.addWidget(self._sub)

        self._help = QLabel()
        self._help.setWordWrap(True)
        self._help.setStyleSheet('background: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px;')
        root.addWidget(self._help)

        body = QHBoxLayout()
        body.setSpacing(10)
        root.addLayout(body)

        left = QWidget()
        left.setFixedWidth(250)
        left.setStyleSheet('background: rgba(255,255,255,0.03); border-radius: 8px;')
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(12, 12, 12, 12)
        self._tools_title = QLabel(); left_v.addWidget(self._tools_title)
        self._rb_brush = QRadioButton(); self._rb_brush.setChecked(True)
        self._rb_rect = QRadioButton()
        self._rb_ellipse = QRadioButton()
        self._rb_eraser = QRadioButton()
        for btn, name in [
            (self._rb_brush, 'brush'),
            (self._rb_rect, 'rect'),
            (self._rb_ellipse, 'ellipse'),
            (self._rb_eraser, 'eraser'),
        ]:
            btn.toggled.connect(lambda checked, n=name: checked and self._set_tool(n))
            left_v.addWidget(btn)
        self._brush_title = QLabel(); left_v.addWidget(self._brush_title)
        self._brush_label = QLabel('25')
        self._brush_slider = QSlider(Qt.Horizontal)
        self._brush_slider.setRange(5, 120)
        self._brush_slider.setValue(25)
        self._brush_slider.valueChanged.connect(self._on_brush_changed)
        left_v.addWidget(self._brush_slider)
        left_v.addWidget(self._brush_label)
        self._actions_title = QLabel(); left_v.addWidget(self._actions_title)
        self._undo_btn = QPushButton(); self._undo_btn.clicked.connect(self._undo)
        self._clear_btn = QPushButton(); self._clear_btn.setObjectName('btn_danger'); self._clear_btn.clicked.connect(self._clear)
        self._invert_btn = QPushButton(); self._invert_btn.clicked.connect(self._invert)
        self._load_bg_btn = QPushButton(); self._load_bg_btn.clicked.connect(self._load_bg)
        self._auto_bg_btn = QPushButton(); self._auto_bg_btn.clicked.connect(self._load_first_good_image)
        self._remove_bg_btn = QPushButton(); self._remove_bg_btn.clicked.connect(self._remove_bg)
        for w in [self._undo_btn, self._clear_btn, self._invert_btn, self._load_bg_btn, self._auto_bg_btn, self._remove_bg_btn]:
            w.setObjectName('btn_secondary' if w is not self._clear_btn else 'btn_danger')
            w.setMinimumHeight(42)
            left_v.addWidget(w)
        left_v.addStretch()
        body.addWidget(left)

        center = QWidget()
        center_v = QVBoxLayout(center)
        center_v.setContentsMargins(0, 0, 0, 0)
        self._preview_hint = QLabel()
        self._preview_hint.setStyleSheet(f'color: {FG_DIM};')
        center_v.addWidget(self._preview_hint)
        self._bg_status = QLabel()
        self._bg_status.setWordWrap(True)
        self._bg_status.setStyleSheet('background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px;')
        center_v.addWidget(self._bg_status)
        self._canvas = MaskCanvas()
        self._canvas.setStyleSheet('background: black; border: 2px solid #7c3aed;')
        self._canvas.pressed.connect(self._on_press)
        self._canvas.moved.connect(self._on_drag)
        self._canvas.released.connect(self._on_release)
        center_v.addWidget(self._canvas, 0, Qt.AlignLeft)
        body.addWidget(center, 1)

        right = QWidget()
        right.setFixedWidth(250)
        right.setStyleSheet('background: rgba(255,255,255,0.03); border-radius: 8px;')
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(12, 12, 12, 12)
        self._class_title = QLabel(); right_v.addWidget(self._class_title)
        self._class_combo = QComboBox(); self._class_combo.currentTextChanged.connect(self._on_cls_change)
        right_v.addWidget(self._class_combo)
        self._sf_title = QLabel(); right_v.addWidget(self._sf_title)
        self._sf_hint = QLabel(); self._sf_hint.setStyleSheet(f'color: {FG_DIM}; font-size: 9pt;')
        self._sf_hint.setWordWrap(True)
        right_v.addWidget(self._sf_hint)
        self._sf_list = QListWidget(); self._sf_list.currentRowChanged.connect(self._update_count)
        right_v.addWidget(self._sf_list)
        sf_row = QHBoxLayout()
        self._new_folder_btn = QPushButton(); self._new_folder_btn.setObjectName('btn_secondary'); self._new_folder_btn.clicked.connect(self._new_subfolder)
        self._del_folder_btn = QPushButton(); self._del_folder_btn.setObjectName('btn_danger'); self._del_folder_btn.clicked.connect(self._del_subfolder)
        sf_row.addWidget(self._new_folder_btn); sf_row.addWidget(self._del_folder_btn)
        right_v.addLayout(sf_row)
        self._save_load_title = QLabel(); right_v.addWidget(self._save_load_title)
        self._save_mask_btn = QPushButton(); self._save_mask_btn.setObjectName('btn_success'); self._save_mask_btn.clicked.connect(self._save_mask)
        self._load_mask_btn = QPushButton(); self._load_mask_btn.setObjectName('btn_secondary'); self._load_mask_btn.clicked.connect(self._load_mask)
        right_v.addWidget(self._save_mask_btn)
        right_v.addWidget(self._load_mask_btn)
        self._count_lbl = QLabel(); self._count_lbl.setStyleSheet(f'color: {FG_DIM};')
        self._count_lbl.setWordWrap(True)
        right_v.addWidget(self._count_lbl)
        right_v.addStretch()
        body.addWidget(right)

        nav = QHBoxLayout()
        self._back_btn = QPushButton(); self._back_btn.setObjectName('btn_secondary'); self._back_btn.clicked.connect(self.app.go_prev)
        self._next_btn = QPushButton(); self._next_btn.setObjectName('btn_success'); self._next_btn.clicked.connect(self.app.go_next)
        nav.addWidget(self._back_btn)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        root.addLayout(nav)

    def retranslate_ui(self):
        self._title.setText(self.t('page3.title'))
        self._sub.setText(self.t('page3.sub'))
        self._help.setText(self.t('page3.help'))
        self._tools_title.setText(self.t('page3.tools'))
        self._rb_brush.setText(self.t('page3.brush'))
        self._rb_rect.setText(self.t('page3.rect'))
        self._rb_ellipse.setText(self.t('page3.ellipse'))
        self._rb_eraser.setText(self.t('page3.eraser'))
        self._brush_title.setText(self.t('page3.brush_size'))
        self._actions_title.setText(self.t('page3.actions'))
        self._undo_btn.setText(self.t('page3.undo'))
        self._clear_btn.setText(self.t('page3.clear'))
        self._invert_btn.setText(self.t('page3.invert'))
        self._load_bg_btn.setText(self.t('page3.load_bg'))
        self._auto_bg_btn.setText('Good Images先頭を読込 / Load first Good Image')
        self._remove_bg_btn.setText(self.t('page3.remove_bg'))
        self._preview_hint.setText(self.t('page3.preview_hint'))
        self._update_bg_status()
        self._class_title.setText(self.t('page3.select_class'))
        self._sf_title.setText(self.t('page3.mask_subfolder'))
        self._sf_hint.setText(self.t('page3.mask_subfolder_hint'))
        self._new_folder_btn.setText(self.t('page3.new_folder'))
        self._del_folder_btn.setText(self.t('page3.del_folder'))
        self._save_load_title.setText(self.t('page3.save_load'))
        self._save_mask_btn.setText(self.t('page3.save_mask'))
        self._load_mask_btn.setText(self.t('page3.load_mask'))
        self._back_btn.setText(self.t('page3.back'))
        self._next_btn.setText(self.t('page3.next'))
        self._update_count()

    def _on_brush_changed(self, value):
        self._brush_size = value
        self._brush_label.setText(str(value))

    def _set_tool(self, name):
        self._tool = name


    def _update_bg_status(self):
        if self._bg_img is not None:
            self._bg_status.setText('背景プレビュー: 読み込み済み / Background preview: loaded')
        else:
            gd = self.state.good_images_path or '(未設定 / not set)'
            self._bg_status.setText(f'背景プレビュー未読込です。STEP2 の Good Images を保存してから 「Good Images の先頭を使う」 または 「背景画像を開く」 を押してください。\nGood Images: {gd}')

    def _redraw(self):
        if self._bg_img is not None:
            base = self._bg_img.resize((CANVAS_W, CANVAS_H)).convert('RGBA')
            base = ImageEnhance.Brightness(base).enhance(0.55)
        else:
            base = Image.new('RGBA', (CANVAS_W, CANVAS_H), (10, 10, 10, 255))
        overlay = Image.new('RGBA', (CANVAS_W, CANVAS_H), (220, 50, 50, 200))
        alpha_mask = self._mask_img.point(lambda p: int(p * 0.85))
        base.paste(overlay, mask=alpha_mask)
        pixmap = QPixmap.fromImage(ImageQt(base.convert('RGB')))
        self._canvas.setPixmap(pixmap)
        self._update_bg_status()

    def _push_undo(self):
        self._undo_stack.append(self._mask_img.copy())
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)

    def _draw_brush(self, x, y, fill):
        draw = ImageDraw.Draw(self._mask_img)
        r = max(1, self._brush_size // 2)
        if self._last_xy and self._tool in ('brush', 'eraser'):
            lx, ly = self._last_xy
            dist = max(abs(x - lx), abs(y - ly))
            steps = max(dist // max(r // 3, 1), 1)
            for i in range(steps + 1):
                t = i / steps
                ix = int(lx + (x - lx) * t)
                iy = int(ly + (y - ly) * t)
                draw.ellipse([(ix - r, iy - r), (ix + r, iy + r)], fill=fill)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=fill)

    def _clamp(self, x, y):
        return max(0, min(CANVAS_W - 1, x)), max(0, min(CANVAS_H - 1, y))

    def _on_press(self, x, y):
        x, y = self._clamp(x, y)
        self._push_undo()
        if self._tool in ('brush', 'eraser'):
            self._drawing = True
            self._last_xy = (x, y)
            self._draw_brush(x, y, 255 if self._tool == 'brush' else 0)
            self._redraw()
        else:
            self._rect_start = (x, y)

    def _on_drag(self, x, y):
        x, y = self._clamp(x, y)
        if self._tool in ('brush', 'eraser') and self._drawing:
            self._draw_brush(x, y, 255 if self._tool == 'brush' else 0)
            self._last_xy = (x, y)
            self._redraw()

    def _on_release(self, x, y):
        x, y = self._clamp(x, y)
        if self._tool in ('brush', 'eraser'):
            self._drawing = False
            self._last_xy = None
            return
        if self._rect_start:
            sx, sy = self._rect_start
            x0, x1 = min(sx, x), max(sx, x)
            y0, y1 = min(sy, y), max(sy, y)
            draw = ImageDraw.Draw(self._mask_img)
            if self._tool == 'rect':
                draw.rectangle([(x0, y0), (x1, y1)], fill=255)
            else:
                draw.ellipse([(x0, y0), (x1, y1)], fill=255)
            self._rect_start = None
            self._redraw()

    def _undo(self):
        if self._undo_stack:
            self._mask_img = self._undo_stack.pop()
            self._redraw()

    def _clear(self):
        if QMessageBox.question(self, self.t('msg.confirm'), self.t('page3.clear_confirm')) != QMessageBox.Yes:
            return
        self._push_undo()
        self._mask_img = Image.new('L', (CANVAS_W, CANVAS_H), 0)
        self._redraw()

    def _invert(self):
        self._push_undo()
        self._mask_img = ImageOps.invert(self._mask_img)
        self._redraw()

    def _load_bg(self):
        path, _ = QFileDialog.getOpenFileName(self, self.t('page3.bg_load_title'), '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if path:
            self._bg_img = Image.open(path).convert('RGB')
            self._redraw()

    def _load_first_good_image(self, silent=False):
        good_dir = self.state.good_images_path
        if not good_dir or not os.path.isdir(good_dir):
            if not silent:
                QMessageBox.warning(self, self.t('msg.warning'), self.t('page3.no_good_dir'))
            return
        for name in sorted(os.listdir(good_dir)):
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                self._bg_img = Image.open(os.path.join(good_dir, name)).convert('RGB')
                self._redraw()
                return
        if not silent:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page3.no_good_dir'))

    def _remove_bg(self):
        self._bg_img = None
        self._redraw()

    def _get_cls(self):
        name = self._class_combo.currentText().strip()
        for c in self.state.classes:
            if c.name == name:
                return c
        return None

    def _on_cls_change(self, text):
        self._selected_class = text
        self._refresh_sf_list()

    def _refresh_sf_list(self):
        current = self._sf_list.currentItem().text() if self._sf_list.currentItem() else None
        self._sf_list.clear()
        cls = self._get_cls()
        if not cls or not cls.mask_dir or not os.path.isdir(cls.mask_dir):
            self._update_count()
            return
        items = []
        for d in sorted(os.listdir(cls.mask_dir)):
            if os.path.isdir(os.path.join(cls.mask_dir, d)):
                items.append(d)
                self._sf_list.addItem(d)
        if current in items:
            self._sf_list.setCurrentRow(items.index(current))
        elif items:
            self._sf_list.setCurrentRow(0)
        self._update_count()

    def _new_subfolder(self):
        cls = self._get_cls()
        if not cls:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page3.no_class'))
            return
        if not cls.mask_dir:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page3.no_mask_dir'))
            return
        os.makedirs(cls.mask_dir, exist_ok=True)
        existing = sorted(d for d in os.listdir(cls.mask_dir) if os.path.isdir(os.path.join(cls.mask_dir, d)))
        new_name = f'{len(existing):03d}'
        os.makedirs(os.path.join(cls.mask_dir, new_name), exist_ok=True)
        self._refresh_sf_list()
        matches = self._sf_list.findItems(new_name, Qt.MatchExactly)
        if matches:
            self._sf_list.setCurrentItem(matches[0])

    def _del_subfolder(self):
        item = self._sf_list.currentItem()
        cls = self._get_cls()
        if not item or not cls:
            return
        path = os.path.join(cls.mask_dir, item.text())
        if QMessageBox.question(self, self.t('msg.confirm'), self.t('page3.delete_confirm', name=item.text(), path=path)) != QMessageBox.Yes:
            return
        shutil.rmtree(path, ignore_errors=True)
        self._refresh_sf_list()

    def _update_count(self):
        cls = self._get_cls()
        if not cls or not cls.mask_dir or not os.path.isdir(cls.mask_dir):
            self._count_lbl.setText('')
            return
        total = 0
        for d in os.listdir(cls.mask_dir):
            dp = os.path.join(cls.mask_dir, d)
            if os.path.isdir(dp):
                total += len([f for f in os.listdir(dp) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self._count_lbl.setText(self.t('page3.total_masks', count=total))

    def _save_mask(self):
        cls = self._get_cls()
        if not cls:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page3.no_class'))
            return
        if not cls.mask_dir:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page3.no_mask_dir'))
            return
        item = self._sf_list.currentItem()
        if not item:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page3.no_subfolder'))
            return
        save_dir = os.path.join(cls.mask_dir, item.text())
        os.makedirs(save_dir, exist_ok=True)
        existing = [f for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        save_path = os.path.join(save_dir, f'mask_{len(existing):03d}.png')
        self._mask_img.save(save_path)
        QMessageBox.information(self, self.t('msg.info'), self.t('page3.saved_mask', path=save_path))
        self._update_count()

    def _load_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, self.t('page3.mask_load_title'), '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if path:
            self._push_undo()
            self._mask_img = Image.open(path).convert('L').resize((CANVAS_W, CANVAS_H), Image.NEAREST)
            self._redraw()

    def on_show(self):
        current = self._class_combo.currentText()
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItems([c.name for c in self.state.classes])
        idx = self._class_combo.findText(current)
        if idx >= 0:
            self._class_combo.setCurrentIndex(idx)
        elif self._class_combo.count() > 0:
            self._class_combo.setCurrentIndex(0)
        self._class_combo.blockSignals(False)
        self._refresh_sf_list()
        if self._bg_img is None and self.state.good_images_path:
            try:
                self._load_first_good_image(silent=True)
            except Exception:
                pass
