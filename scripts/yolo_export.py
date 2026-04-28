# scripts/yolo_export.py
import cv2
import numpy as np

def mask_to_yolo_polygons(mask):
    h, w = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue

        poly = cnt.squeeze()
        if len(poly.shape) != 2:
            continue

        poly_norm = []
        for x, y in poly:
            poly_norm.extend([x / w, y / h])

        polygons.append(poly_norm)

    return polygons
