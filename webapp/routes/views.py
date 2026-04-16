from flask import Blueprint, redirect, render_template, url_for
from utils import state, T

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    return redirect(url_for('views.setup'))

@views_bp.route('/setup')
def setup():
    return render_template('setup.html', state=state, T=T)

@views_bp.route('/other')
def other():
    return render_template('other_studio.html', state=state, T=T, active='other')

@views_bp.route('/studio')
def studio():
    return render_template('defect_studio.html', state=state, T=T, active='studio')

@views_bp.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', state=state, T=T, active='dashboard')

@views_bp.route('/api/health')
def health():
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        gpu_available = False
    from flask import jsonify
    return jsonify({"gpu_available": gpu_available})

