"""
engines/utils.py — Shared encode/decode helpers for all engines.
"""

import base64
import cv2
import numpy as np


def encode_b64(image: np.ndarray) -> str:
    """Encode uint8 RGB/Gray numpy array to base64 PNG string."""
    _, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                          if image.ndim == 3 else image)
    return base64.b64encode(buf).decode("utf-8")


def decode_b64(b64str: str) -> np.ndarray:
    """Decode base64 PNG string to uint8 RGB numpy array."""
    data = base64.b64decode(b64str)
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_b64_gray(b64str: str) -> np.ndarray:
    """Decode base64 PNG string to uint8 grayscale numpy array."""
    data = base64.b64decode(b64str)
    arr  = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
