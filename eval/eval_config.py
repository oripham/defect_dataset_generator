"""
Shared configuration for all eval scripts.
Change paths here once — all scripts pick it up automatically.
"""
from pathlib import Path

# Thư mục chứa file này: V:\HondaPlus\defect_dataset_generator\eval\
_EVAL_DIR = Path(__file__).parent

# ── INPUT ─────────────────────────────────────────────────────────────────────
SRC_DIR = Path(r"V:\defect_samples\results\cap\b6_full_100")
OK_IMG  = Path(r"V:\dataHondatPlus\Good_IMG_MKA.jpg")   # ảnh OK thật

# ── OUTPUT — nằm trong eval/output/, dễ quản lý, không cần C:\ ───────────────
OUT_DIR = _EVAL_DIR / "output"

# Sub-paths — derived automatically, không cần sửa
SPLIT_DIR    = OUT_DIR / "split_lists"
ANOMALIB_DIR = OUT_DIR / "anomalib" / "mka_defect"
RESULTS_DIR  = OUT_DIR / "results"

# ── DATASET CONFIG ─────────────────────────────────────────────────────────────
NUM_CLASSES = 6   # 0=background, 1-5=defect classes

# Pixel values khớp với mask PNG thực tế (scan từ b6_full_100/masks/)
CLASS_NAMES = {
    0: "background",
    1: "dark_spots",
    2: "dent",
    3: "thread",
    4: "plastic_flow",
    5: "scratch",
}

CV_PREFIXES = ["dark_spots", "dent", "thread", "scratch", "plastic_flow"]

VAL_RATIO = 0.2
SEED      = 42
