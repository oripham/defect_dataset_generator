import os

import yaml
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr
from gui.theme import FG_DIM


class ConfigPage(QWidget):
    def __init__(self, state, app):
        super().__init__()
        self.state = state
        self.app = app
        self._build_ui()
        self.retranslate_ui()

    def t(self, key, **kwargs):
        return tr(self.state, key, **kwargs)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        self._title = QLabel()
        self._title.setStyleSheet('font-size: 16pt; font-weight: bold;')
        root.addWidget(self._title)

        self._sub = QLabel()
        self._sub.setWordWrap(True)
        self._sub.setStyleSheet(f'color: {FG_DIM};')
        root.addWidget(self._sub)

        card = QWidget()
        card.setStyleSheet('background: rgba(255,255,255,0.03); border-radius: 8px;')
        form = QFormLayout(card)
        form.setContentsMargins(16, 16, 16, 16)
        self._model_label = QLabel()
        self._model_edit = QLineEdit(self.state.model_name)
        self._device_label = QLabel()
        self._device_edit = QLineEdit(self.state.device)
        form.addRow(self._model_label, self._model_edit)
        form.addRow(self._device_label, self._device_edit)
        root.addWidget(card)

        self._yaml_label = QLabel()
        self._yaml_label.setStyleSheet(f'color: {FG_DIM};')
        root.addWidget(self._yaml_label)

        self._yaml_text = QPlainTextEdit()
        self._yaml_text.setMinimumHeight(430)
        root.addWidget(self._yaml_text, 1)

        nav = QHBoxLayout()
        self._back_btn = QPushButton()
        self._back_btn.clicked.connect(self.app.go_prev)
        self._regen_btn = QPushButton()
        self._regen_btn.clicked.connect(self._regenerate)
        self._save_btn = QPushButton()
        self._save_btn.clicked.connect(self._save_config)
        self._next_btn = QPushButton()
        self._next_btn.setObjectName('btn_success')
        self._next_btn.clicked.connect(self.app.go_next)
        nav.addWidget(self._back_btn)
        nav.addWidget(self._regen_btn)
        nav.addWidget(self._save_btn)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        root.addLayout(nav)

    def retranslate_ui(self):
        self._title.setText(self.t('page4.title'))
        self._sub.setText(self.t('page4.sub'))
        self._model_label.setText(self.t('page4.model'))
        self._device_label.setText(self.t('page4.device'))
        self._yaml_label.setText(self.t('page4.generated'))
        self._back_btn.setText(self.t('page4.back'))
        self._regen_btn.setText(self.t('page4.regen'))
        self._save_btn.setText(self.t('page4.save'))
        self._next_btn.setText(self.t('page4.next'))

    def _regenerate(self):
        self.state.model_name = self._model_edit.text().strip()
        self.state.device = self._device_edit.text().strip()
        cfg = self.state.build_config_dict()
        self._yaml_text.setPlainText(
            yaml.dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)
        )

    def _save_config(self):
        self.state.model_name = self._model_edit.text().strip()
        self.state.device = self._device_edit.text().strip()
        yaml_content = self._yaml_text.toPlainText().strip()
        if not yaml_content:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page4.empty'))
            return
        try:
            yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            QMessageBox.critical(self, self.t('msg.error'), self.t('page4.yaml_error', error=e))
            return

        default_dir = self.state.good_images_path or os.path.expanduser('~')
        default_path = os.path.join(os.path.dirname(default_dir) if default_dir else os.path.expanduser('~'), 'config.yaml')
        path, _ = QFileDialog.getSaveFileName(self, self.t('page4.save'), default_path, 'YAML Files (*.yaml *.yml);;All Files (*)')
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            self.state._config_save_path = path
            QMessageBox.information(self, self.t('msg.info'), self.t('page4.saved', path=path))

    def on_show(self):
        self._model_edit.setText(self.state.model_name)
        self._device_edit.setText(self.state.device)
        self._regenerate()
