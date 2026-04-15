import io
import json
import os
import threading
import time
import zipfile

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)

from gui.i18n import tr
from gui.theme import ACCENT2, DANGER, FG, FG_DIM


class RunPage(QWidget):
    _sig_log = Signal(str)
    _sig_icon = Signal(str, str)
    _sig_finish = Signal(bool)
    _sig_job = Signal(str)
    _sig_download_done = Signal(str)
    _sig_download_fail = Signal(str)

    def __init__(self, state, app):
        super().__init__()
        self.state = state
        self.app = app
        self._job_id = ''
        self._step_labels = {}
        self._sig_log.connect(self._append_log)
        self._sig_icon.connect(self._set_icon)
        self._sig_finish.connect(self._finish)
        self._sig_job.connect(self._set_job)
        self._sig_download_done.connect(self._download_done)
        self._sig_download_fail.connect(self._download_fail)
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

        self._step_rows = {}
        for key in ('zip', 'upload', 'gen'):
            row_widget = QWidget()
            row_widget.setStyleSheet('background: rgba(255,255,255,0.03); border-radius: 8px;')
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(12, 10, 12, 10)
            icon = QLabel('⬜')
            icon.setFixedWidth(26)
            icon.setStyleSheet('font-size: 14pt;')
            text = QLabel()
            row.addWidget(icon)
            row.addWidget(text)
            row.addStretch()
            self._step_rows[key] = (icon, text)
            root.addWidget(row_widget)

        self._progress = QProgressBar()
        self._progress.setMaximum(0)
        self._progress.hide()
        root.addWidget(self._progress)

        out_row = QHBoxLayout()
        self._out_label = QLabel()
        self._out_edit = QLineEdit(self.state.output_root)
        self._out_btn = QPushButton()
        self._out_btn.clicked.connect(self._browse_out)
        out_row.addWidget(self._out_label)
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(self._out_btn)
        root.addLayout(out_row)

        btn_row = QHBoxLayout()
        self._back_btn = QPushButton()
        self._back_btn.clicked.connect(self.app.go_prev)
        self._run_btn = QPushButton()
        self._run_btn.setObjectName('btn_primary')
        self._run_btn.clicked.connect(self._start)
        self._dl_btn = QPushButton()
        self._dl_btn.setObjectName('btn_success')
        self._dl_btn.setEnabled(False)
        self._dl_btn.clicked.connect(self._download_results)
        self._job_lbl = QLabel()
        self._job_lbl.setStyleSheet(f'color: {FG_DIM};')
        self._new_project_btn = QPushButton()
        self._new_project_btn.setObjectName('btn_secondary')
        self._new_project_btn.clicked.connect(self._new_project)
        btn_row.addWidget(self._back_btn)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._dl_btn)
        btn_row.addWidget(self._job_lbl)
        btn_row.addStretch()
        btn_row.addWidget(self._new_project_btn)
        root.addLayout(btn_row)

        self._log_label = QLabel()
        self._log_label.setStyleSheet(f'color: {FG_DIM};')
        root.addWidget(self._log_label)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(320)
        root.addWidget(self._log, 1)

    def retranslate_ui(self):
        self._title.setText(self.t('page5.title'))
        self._sub.setText(self.t('page5.sub'))
        self._step_rows['zip'][1].setText(self.t('page5.zip'))
        self._step_rows['upload'][1].setText(self.t('page5.upload'))
        self._step_rows['gen'][1].setText(self.t('page5.generate'))
        self._out_label.setText(self.t('page5.out_dir'))
        self._out_btn.setText(self.t('common.browse'))
        self._back_btn.setText(self.t('page5.back'))
        self._run_btn.setText(self.t('page5.run'))
        self._dl_btn.setText(self.t('page5.download'))
        self._new_project_btn.setText(self.t('page5.new_project'))
        self._log_label.setText(self.t('page5.log'))
        if self._job_id:
            self._job_lbl.setText(self.t('page5.job_label', job_id=self._job_id))

    @Slot(str)
    def _append_log(self, msg):
        self._log.append(msg)

    @Slot(str, str)
    def _set_icon(self, key, state):
        icons = {'ok': ('✅', ACCENT2), 'err': ('❌', DANGER), 'run': ('🔄', FG), 'wait': ('⬜', FG_DIM)}
        txt, color = icons.get(state, ('⬜', FG_DIM))
        icon = self._step_rows[key][0]
        icon.setText(txt)
        icon.setStyleSheet(f'font-size: 14pt; color: {color};')

    @Slot(bool)
    def _finish(self, ok):
        self._progress.hide()
        self._run_btn.setEnabled(True)
        if ok:
            self._dl_btn.setEnabled(True)
            self._append_log(self.t('page5.done'))
        else:
            self._append_log(self.t('page5.failed'))

    @Slot(str)
    def _set_job(self, job_id):
        self._job_id = job_id
        self.state._current_job_id = job_id
        self._job_lbl.setText(self.t('page5.job_label', job_id=job_id))

    @Slot(str)
    def _download_done(self, path):
        self._append_log(self.t('page5.saved_results', path=path))
        QMessageBox.information(self, self.t('msg.info'), self.t('page5.download_complete', path=path))

    @Slot(str)
    def _download_fail(self, error):
        self._append_log(self.t('page5.download_failed', error=error))

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, self.t('common.browse'), self._out_edit.text() or '')
        if path:
            self._out_edit.setText(path)
            self.state.output_root = path

    def _start(self):
        if not self.state.connected:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page5.not_connected'))
            return
        if not self.state.good_images_path:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page5.no_dataset'))
            return
        if not self.state.classes:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page5.no_classes'))
            return
        self.state.output_root = self._out_edit.text().strip()
        self._run_btn.setEnabled(False)
        self._dl_btn.setEnabled(False)
        self._progress.show()
        self._log.clear()
        for key in self._step_rows:
            self._set_icon(key, 'wait')
        threading.Thread(target=self._workflow, daemon=True).start()

    def _workflow(self):
        try:
            import requests
        except ImportError:
            self._sig_log.emit(self.t('page5.requests_missing'))
            self._sig_finish.emit(False)
            return

        self._sig_icon.emit('zip', 'run')
        self._sig_log.emit(self.t('page5.pack'))
        try:
            zip_buf = self._build_zip()
            size_mb = len(zip_buf.getvalue()) / 1024 / 1024
            self._sig_log.emit(self.t('page5.zip_size', size=size_mb))
            self._sig_icon.emit('zip', 'ok')
        except Exception as e:
            self._sig_log.emit(self.t('page5.zip_failed', error=e))
            self._sig_icon.emit('zip', 'err')
            self._sig_finish.emit(False)
            return

        self._sig_icon.emit('upload', 'run')
        self._sig_log.emit(self.t('page5.uploading'))
        try:
            config_json = json.dumps(self.state.build_api_config())
            zip_buf.seek(0)
            resp = requests.post(
                f'{self.state.server_url}/jobs',
                files={'dataset_zip': ('dataset.zip', zip_buf, 'application/zip')},
                data={'config_json': config_json},
                timeout=300,
            )
            resp.raise_for_status()
            job_id = resp.json()['job_id']
            self._job_id = job_id
            self._sig_job.emit(job_id)
            self._sig_log.emit(self.t('page5.job_id', job_id=job_id))
            self._sig_icon.emit('upload', 'ok')
        except Exception as e:
            self._sig_log.emit(self.t('page5.upload_failed', error=e))
            self._sig_icon.emit('upload', 'err')
            self._sig_finish.emit(False)
            return

        self._sig_icon.emit('gen', 'run')
        self._sig_log.emit(self.t('page5.running'))
        ok = self._poll_until_done(requests)
        self._sig_icon.emit('gen', 'ok' if ok else 'err')
        self._sig_finish.emit(ok)

    def _build_zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            good = self.state.good_images_path
            if not os.path.isdir(good):
                raise FileNotFoundError(f'Good images folder not found: {good}')
            for fname in os.listdir(good):
                fpath = os.path.join(good, fname)
                if os.path.isfile(fpath):
                    zf.write(fpath, f'good_images/{fname}')
            for cls in self.state.classes:
                if cls.mask_dir and os.path.isdir(cls.mask_dir):
                    for sf in os.listdir(cls.mask_dir):
                        sf_path = os.path.join(cls.mask_dir, sf)
                        if os.path.isdir(sf_path):
                            for mf in os.listdir(sf_path):
                                mp = os.path.join(sf_path, mf)
                                if os.path.isfile(mp):
                                    zf.write(mp, f'mask_root/{cls.name}/{sf}/{mf}')
                if cls.ref_dir and os.path.isdir(cls.ref_dir):
                    for rf in os.listdir(cls.ref_dir):
                        rp = os.path.join(cls.ref_dir, rf)
                        if os.path.isfile(rp):
                            zf.write(rp, f'defect_refs/{cls.name}/{rf}')
        return buf

    def _poll_until_done(self, requests_module):
        log_idx = 0
        while True:
            time.sleep(1.5)
            try:
                resp = requests_module.get(
                    f'{self.state.server_url}/jobs/{self._job_id}/status',
                    params={'since': log_idx},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                self._sig_log.emit(self.t('page5.poll_error', error=e))
                continue
            for line in data.get('new_logs', []):
                self._sig_log.emit(line)
            log_idx = data.get('log_count', log_idx)
            status = data.get('status', '')
            if status == 'done':
                return True
            if status == 'error':
                return False

    def _download_results(self):
        if not self._job_id:
            QMessageBox.warning(self, self.t('msg.warning'), self.t('page5.no_job'))
            return
        save_dir = self._out_edit.text().strip()
        if not save_dir:
            save_dir = QFileDialog.getExistingDirectory(self, self.t('common.browse'), '')
            if not save_dir:
                return
            self._out_edit.setText(save_dir)
            self.state.output_root = save_dir
        threading.Thread(target=self._do_download, args=(save_dir,), daemon=True).start()

    def _do_download(self, save_dir):
        try:
            import requests
        except ImportError:
            self._sig_download_fail.emit(self.t('page5.requests_missing'))
            return
        os.makedirs(save_dir, exist_ok=True)
        self._sig_log.emit(self.t('page5.downloading'))
        try:
            resp = requests.get(f'{self.state.server_url}/jobs/{self._job_id}/results', timeout=120, stream=True)
            resp.raise_for_status()
            zip_path = os.path.join(save_dir, f'results_{self._job_id}.zip')
            with open(zip_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(save_dir)
            os.remove(zip_path)
            self._sig_download_done.emit(save_dir)
        except Exception as e:
            self._sig_download_fail.emit(str(e))

    def _new_project(self):
        if QMessageBox.question(
            self, self.t('msg.confirm'), self.t('page5.new_project_confirm')
        ) != QMessageBox.Yes:
            return
        self.state.reset()
        self._job_id = ''
        self._job_lbl.setText('')
        self._log.clear()
        self._dl_btn.setEnabled(False)
        self._run_btn.setEnabled(True)
        self._progress.hide()
        for key in self._step_rows:
            self._set_icon(key, 'wait')
        for page in self.app._pages.values():
            if hasattr(page, 'on_show'):
                page.on_show()
            if hasattr(page, 'retranslate_ui'):
                page.retranslate_ui()
        self.app._show('step2')

    def on_show(self):
        self._out_edit.setText(self.state.output_root)
        if self.state._current_job_id:
            self._set_job(self.state._current_job_id)
