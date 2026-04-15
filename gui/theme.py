# PySide6 dark theme

BG_SIDEBAR = "#1a1a2e"
BG_MAIN    = "#16213e"
BG_CARD    = "#0f3460"
BG_INPUT   = "#1e2a45"
BORDER     = "#2d3a55"
ACCENT     = "#7c3aed"
ACCENT2    = "#059669"
DANGER     = "#dc2626"
FG         = "#e2e8f0"
FG_DIM     = "#94a3b8"
FG_MUTED   = "#64748b"

STYLESHEET = f"""
* {{
    font-family: "Yu Gothic UI", "Meiryo UI", "MS UI Gothic", sans-serif;
    font-size: 10.5pt;
}}
QMainWindow, QDialog {{
    background-color: {BG_MAIN};
}}
QWidget {{
    background-color: transparent;
    color: {FG};
}}
QScrollArea {{
    background-color: {BG_MAIN};
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background-color: {BG_MAIN};
}}
QScrollBar:vertical {{
    background: {BG_CARD};
    width: 10px;
    border-radius: 5px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #4a5568;
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {BG_CARD};
    height: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background: #4a5568;
    border-radius: 5px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QWidget#sidebar {{
    background-color: {BG_SIDEBAR};
}}
QLabel#sidebar_title {{
    color: {FG};
    font-size: 13pt;
    font-weight: bold;
    background: {BG_SIDEBAR};
    padding: 4px;
}}
QPushButton#nav_btn {{
    background-color: transparent;
    color: {FG_DIM};
    border: none;
    padding: 14px 16px;
    text-align: left;
    border-radius: 10px;
    font-size: 10pt;
    min-height: 46px;
}}
QPushButton#nav_btn:hover {{
    background-color: #2d3a55;
    color: {FG};
}}
QPushButton#nav_btn:checked {{
    background-color: {ACCENT};
    color: white;
    font-weight: bold;
}}
QGroupBox {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 12px 12px 12px;
    color: {FG};
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    top: -7px;
    color: {FG_DIM};
    font-size: 9pt;
    font-weight: normal;
    background: transparent;
}}
QLabel {{
    color: {FG};
    background: transparent;
}}
QLineEdit {{
    background-color: {BG_INPUT};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 10px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QTextEdit, QPlainTextEdit {{
    background-color: {BG_CARD};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px;
    font-family: "Courier New", "Consolas", monospace;
    font-size: 9pt;
    selection-background-color: {ACCENT};
}}
QPushButton {{
    background-color: {BG_CARD};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 10.5pt;
    font-weight: 700;
    min-height: 26px;
    min-width: 108px;
}}
QPushButton:hover {{
    background-color: #1a3a6a;
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: white;
}}
QPushButton:disabled {{
    color: {FG_MUTED};
    border-color: {BG_CARD};
    background-color: {BG_INPUT};
}}
QPushButton#btn_primary {{
    background-color: {ACCENT};
    color: white;
    border: none;
    font-size: 11pt;
    font-weight: bold;
    padding: 10px 24px;
}}
QPushButton#btn_primary:hover {{ background-color: #6d28d9; }}
QPushButton#btn_primary:pressed {{ background-color: #5b21b6; }}
QPushButton#btn_success {{
    background-color: {ACCENT2};
    color: white;
    border: none;
    padding: 7px 16px;
}}
QPushButton#btn_success:hover {{ background-color: #047857; }}
QPushButton#btn_danger {{
    background-color: {DANGER};
    color: white;
    border: none;
}}
QPushButton#btn_danger:hover {{ background-color: #b91c1c; }}
QPushButton#btn_ghost {{
    background-color: {BG_INPUT};
    color: {FG};
    border: 1px solid {BORDER};
}}
QPushButton#btn_ghost:hover {{ border-color: {ACCENT}; }}
QPushButton#btn_tool {{
    background-color: {BG_INPUT};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 9pt;
    text-align: left;
}}
QPushButton#btn_tool:checked {{
    background-color: {ACCENT};
    color: white;
    border-color: {ACCENT};
}}
QPushButton#btn_tool:hover {{ border-color: {ACCENT}; }}
QComboBox {{
    background-color: {BG_INPUT};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 10px;
    min-width: 100px;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
}}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    color: {FG};
    selection-background-color: {ACCENT};
    border: 1px solid {BORDER};
    outline: none;
}}
QListWidget {{
    background-color: {BG_INPUT};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    outline: none;
}}
QListWidget::item {{ padding: 5px 8px; }}
QListWidget::item:selected {{
    background-color: {ACCENT};
    color: white;
}}
QListWidget::item:hover {{ background-color: #2d3a55; }}
QProgressBar {{
    background-color: {BG_INPUT};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}
QSlider::groove:horizontal {{
    background: {BG_INPUT};
    height: 6px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 16px;
    height: 16px;
    border-radius: 8px;
    margin: -5px 0;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 3px;
}}
QSplitter::handle {{ background: {BORDER}; width: 1px; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {BORDER};
}}
QPushButton#btn_secondary {{
    background-color: #1e293b;
    color: white;
    border: 1px solid #475569;
}}
QPushButton#btn_secondary:hover {{
    background-color: #334155;
    border-color: #7c3aed;
}}
"""
