import os
import shutil
import subprocess
import sys
import threading

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr
from gui.theme import ACCENT2, BG_SIDEBAR, DANGER, FG_DIM


class ConnectionPage(QWidget):
    _sig_log = Signal(str)
    _sig_status = Signal(str, bool)

    def __init__(self, state, app):
        super().__init__()
        self.state = state
        self.app = app
        self._sig_log.connect(self._on_log)
        self._sig_status.connect(self._on_status)
        self._build_ui()
        self.retranslate_ui()

    def t(self, key, **kwargs):
        return tr(self.state, key, **kwargs)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 22)
        root.setSpacing(14)

        self._title = QLabel()
        self._title.setStyleSheet('font-size: 16pt; font-weight: bold;')
        root.addWidget(self._title)

        self._sub = QLabel()
        self._sub.setWordWrap(True)
        self._sub.setStyleSheet(f'color: {FG_DIM}; font-size: 9pt;')
        root.addWidget(self._sub)

        self._docker_group = QGroupBox()
        dv = QVBoxLayout(self._docker_group)
        self._docker_info = QLabel()
        self._docker_info.setStyleSheet(f"color: {FG_DIM}; font-family: 'Courier New', monospace; font-size: 9pt;")
        self._docker_info.setWordWrap(True)
        dv.addWidget(self._docker_info)
        dbr = QHBoxLayout()
        self._btn_up = QPushButton('docker compose up --build -d')
        self._btn_up.setObjectName('btn_ghost')
        self._btn_up.setStyleSheet(f"font-family: 'Courier New'; background: {BG_SIDEBAR};")
        self._btn_up.clicked.connect(self._docker_start)
        self._btn_dn = QPushButton('docker compose down')
        self._btn_dn.setObjectName('btn_danger')
        self._btn_dn.clicked.connect(self._docker_stop)
        dbr.addWidget(self._btn_up)
        dbr.addWidget(self._btn_dn)
        dbr.addStretch()
        dv.addLayout(dbr)
        root.addWidget(self._docker_group)

        self._remote_group = QGroupBox()
        rv = QVBoxLayout(self._remote_group)
        self._remote_info = QLabel()
        self._remote_info.setWordWrap(True)
        rv.addWidget(self._remote_info)
        root.addWidget(self._remote_group)

        self._url_group = QGroupBox()
        self._url_form = QFormLayout(self._url_group)
        url_row = QHBoxLayout()
        self._url_edit = QLineEdit(self.state.server_url)
        url_row.addWidget(self._url_edit)
        for lbl, url in [('localhost:8005', 'http://localhost:8005'), ('localhost:8080', 'http://localhost:8080')]:
            b = QPushButton(lbl)
            b.setObjectName('btn_ghost')
            b.clicked.connect(lambda _=False, u=url: self._url_edit.setText(u))
            url_row.addWidget(b)
        self._url_form.addRow('', url_row)
        root.addWidget(self._url_group)

        cb = QHBoxLayout()
        self._conn_btn = QPushButton()
        self._conn_btn.setObjectName('btn_primary')
        self._conn_btn.clicked.connect(self._test)
        cb.addWidget(self._conn_btn)
        self._status_lbl = QLabel('')
        self._status_lbl.setStyleSheet('font-size: 11pt;')
        cb.addWidget(self._status_lbl)
        cb.addStretch()
        root.addLayout(cb)

        self._response_label = QLabel()
        root.addWidget(self._response_label)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMaximumHeight(160)
        root.addWidget(self._log_box)

        root.addStretch()
        nav = QHBoxLayout()
        nav.addStretch()
        self._next_btn = QPushButton()
        self._next_btn.setObjectName('btn_success')
        self._next_btn.clicked.connect(self.app.go_next)
        nav.addWidget(self._next_btn)
        root.addLayout(nav)

    def retranslate_ui(self):
        self._title.setText(self.t('page1.title'))
        self._sub.setText(self.t('page1.sub'))
        self._docker_group.setTitle(self.t('page1.docker'))
        self._docker_info.setText(self.t('page1.docker_info'))
        self._remote_group.setTitle(self.t('page1.remote'))
        self._remote_info.setText(self.t('page1.remote_info'))
        self._url_group.setTitle(self.t('page1.server_url'))
        self._url_form.setWidget(0, QFormLayout.LabelRole, QLabel(self.t('page1.url')))
        self._conn_btn.setText(self.t('page1.test'))
        self._response_label.setText(self.t('page1.response'))
        self._next_btn.setText(self.t('page1.next'))

    @Slot(str)
    def _on_log(self, msg: str):
        self._log_box.append(msg)

    @Slot(str, bool)
    def _on_status(self, text: str, ok: bool):
        self._status_lbl.setText(text)
        color = ACCENT2 if ok else DANGER
        self._status_lbl.setStyleSheet(f'font-size: 11pt; color: {color};')

    def _test(self):
        self.state.server_url = self._url_edit.text().strip().rstrip('/')
        self._sig_status.emit(self.t('page1.connecting'), True)
        self._log_box.clear()
        threading.Thread(target=self._do_test, daemon=True).start()

    def _do_test(self):
        try:
            import requests
        except ImportError:
            self._sig_log.emit(self.t('page1.requests_missing'))
            self._sig_status.emit(self.t('page1.failed'), False)
            return

        url = self.state.server_url
        self._sig_log.emit(self.t('page1.health', url=url))
        try:
            r = requests.get(f'{url}/health', timeout=8)
            r.raise_for_status()
            d = r.json()
            gpu = d.get('gpu_available', False)
            info = self.t('page1.gpu_yes', name=d.get('gpu_name'), mem=d.get('gpu_memory')) if gpu else self.t('page1.gpu_no')
            self.state.connected = True
            self._sig_log.emit(self.t('page1.connect_success', info=info))
            self._sig_status.emit(self.t('page1.connected'), True)
        except Exception as e:
            self.state.connected = False
            self._sig_log.emit(f'❌ {e}')
            self._sig_status.emit(self.t('page1.failed'), False)

    def _project_dir(self):
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _run_compose(self, args):
        pdir = self._project_dir()
        cmd = ['docker', 'compose'] + args
        try:
            if sys.platform == 'win32':
                subprocess.Popen(['cmd', '/k', ' '.join(cmd)], cwd=pdir, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                for term in ('gnome-terminal', 'xterm', 'konsole'):
                    if shutil.which(term):
                        subprocess.Popen([term, '--'] + cmd, cwd=pdir)
                        return
                subprocess.Popen(cmd, cwd=pdir)
        except Exception as e:
            self._sig_log.emit(self.t('page1.docker_error', error=e))

    def _docker_start(self):
        self._run_compose(['up', '--build', '-d'])

    def _docker_stop(self):
        self._run_compose(['down'])

    def on_show(self):
        self._url_edit.setText(self.state.server_url)
