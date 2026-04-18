import os
import sys
from flask import Blueprint, request, jsonify, send_file

# Ensure engines/core is accessible
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from engines.core.data_manager import save_evaluation, get_evaluations, delete_evaluation, clear_all_evaluations

eval_bp = Blueprint('eval_api', __name__, url_prefix='/api/eval')

@eval_bp.route('/save', methods=['POST'])
def save_eval():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        eval_id = save_evaluation(data)
        
        return jsonify({
            "status": "ok",
            "eval_id": eval_id
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@eval_bp.route('/list', methods=['GET'])
def list_evals():
    try:
        evals = get_evaluations()
        return jsonify({"status": "ok", "data": evals})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@eval_bp.route('/delete/<eval_id>', methods=['DELETE'])
def delete_eval(eval_id):
    print(f"[eval_api] Request to delete evaluation: {eval_id}")
    try:
        success = delete_evaluation(eval_id)
        if success:
            return jsonify({"status": "ok"})
        else:
            return jsonify({"error": "Evaluation not found or could not be deleted"}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@eval_bp.route('/delete/all', methods=['DELETE'])
def clear_all():
    print("[eval_api] Request to CLEAR ALL evaluations")
    try:
        success = clear_all_evaluations()
        if success:
            return jsonify({"status": "ok"})
        else:
            return jsonify({"error": "Failed to clear database"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@eval_bp.route('/image', methods=['GET'])
def serve_image():
    """Serves an image from an absolute path for the dashboard."""
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return "Image not found", 404
    
    # Security: Ensure the path is within the products data directory
    abs_path = os.path.abspath(path)
    data_dir = os.path.abspath(os.path.join(_BASE_DIR, "data", "products"))
    if not abs_path.startswith(data_dir):
        return "Access denied", 403
        
    return send_file(abs_path)
