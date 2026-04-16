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
    """Decode base64 string to uint8 RGB numpy array. Handles data URL prefixes."""
    if not b64str:
        raise ValueError("Empty base64 string provided")
    
    # Strip data URL prefix if present
    if "," in b64str:
        b64str = b64str.split(",")[-1]
    
    # Clean whitespace and handle padding
    b64str = b64str.strip()
    padding = len(b64str) % 4
    if padding:
        b64str += "=" * (4 - padding)
    
    print(f"[engine_utils] Decoding image (len={len(b64str)})")
    try:
        data = base64.b64decode(b64str)
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None: Invalid image data")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[engine_utils] Decode FAILED: {e}")
        raise

def decode_b64_gray(b64str: str) -> np.ndarray:
    """Decode base64 string to uint8 grayscale numpy array. Handles data URL prefixes."""
    if not b64str:
        return np.zeros((1,1), np.uint8)
    
    if "," in b64str:
        b64str = b64str.split(",")[-1]
        
    b64str = b64str.strip()
    padding = len(b64str) % 4
    if padding:
        b64str += "=" * (4 - padding)

    try:
        data = base64.b64decode(b64str)
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
             return np.zeros((1,1), np.uint8)
        return img
    except Exception:
        return np.zeros((1,1), np.uint8)
