"""
engines/core/data_manager.py
============================
Handles persistent storage of generated defects and their evaluations.
Uses SQLite to store metadata and saves images to the filesystem.
"""

import os
import sqlite3
import json
import uuid
import base64
import shutil
import base64
from datetime import datetime

# Configure storage paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "database.db")
DATA_DIR = os.path.join(BASE_DIR, "data", "products")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            product TEXT,
            defect_type TEXT,
            status TEXT,
            reasons TEXT,
            params TEXT,
            base_image_path TEXT,
            mask_image_path TEXT,
            result_image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def _save_image(b64_data: str, folder_path: str, filename: str) -> str:
    if not b64_data:
        return ""
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return file_path

def save_evaluation(data: dict) -> str:
    init_db()
    
    eval_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    product = data.get("product", "unknown")
    defect_type = data.get("defect_type", "unknown")
    status = data.get("status", "unknown")
    reasons = json.dumps(data.get("reasons", []))
    params = json.dumps(data.get("params", {}))
    
    # Create folder structure: data/products/{product}/{YYYY-MM-DD}/{eval_id}
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join(DATA_DIR, product, date_str, eval_id)
    
    # Save images
    base_img_path = _save_image(data.get("base_image"), folder_path, "base.png")
    mask_img_path = _save_image(data.get("mask_image"), folder_path, "mask.png")
    result_img_path = _save_image(data.get("result_image"), folder_path, "result.png")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO evaluations (id, timestamp, product, defect_type, status, reasons, params, base_image_path, mask_image_path, result_image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (eval_id, timestamp, product, defect_type, status, reasons, params, base_img_path, mask_img_path, result_img_path))
    
    conn.commit()
    conn.close()
    
    return eval_id

def get_evaluations():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, product, defect_type, status, reasons, params, base_image_path, mask_image_path, result_image_path 
        FROM evaluations 
        ORDER BY timestamp DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "product": row["product"],
            "defect_type": row["defect_type"],
            "status": row["status"],
            "reasons": json.loads(row["reasons"]) if row["reasons"] else [],
            "params": json.loads(row["params"]) if row["params"] else {},
            "base_image_path": row["base_image_path"],
            "mask_image_path": row["mask_image_path"],
            "result_image_path": row["result_image_path"]
        })
    return results

def delete_evaluation(eval_id: str) -> bool:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get paths first to delete files
    cursor.execute('SELECT base_image_path FROM evaluations WHERE id = ?', (eval_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False
        
    base_img_path = row[0]
    
    # Delete from DB
    cursor.execute('DELETE FROM evaluations WHERE id = ?', (eval_id,))
    conn.commit()
    conn.close()
    
    # Delete from filesystem (the parent directory containing the 3 images)
    if base_img_path and os.path.exists(base_img_path):
        folder_path = os.path.dirname(base_img_path)
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            print(f"Failed to cleanly remove directory {folder_path}: {e}")
            
    return True

def clear_all_evaluations() -> bool:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM evaluations')
    conn.commit()
    conn.close()
    
    # Clear the data directory completely
    if os.path.exists(DATA_DIR):
        try:
            shutil.rmtree(DATA_DIR)
        except Exception as e:
            print(f"Failed to cleanly remove DATA_DIR: {e}")
            
    return True

# Initialize DB on import
if not os.path.exists(DB_PATH):
    init_db()

