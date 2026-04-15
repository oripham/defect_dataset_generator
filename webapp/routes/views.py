from flask import Blueprint, redirect, render_template, url_for
from utils import state, T, _has_masks

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    return redirect(url_for('views.setup'))

@views_bp.route('/tuning')
def tuning():
    return render_template('tuning_station.html', state=state, T=T)

@views_bp.route('/setup')
def setup():
    return render_template('setup.html', state=state, T=T)

@views_bp.route('/masking')
def masking():
    missing = [c.name for c in state.classes if not _has_masks(c)]
    return render_template('masking.html', state=state, T=T, missing=missing)

@views_bp.route('/review')
def review():
    return render_template('qa_review.html', state=state, T=T)

@views_bp.route('/pharma')
def pharma():
    return redirect(url_for('views.studio'))

@views_bp.route('/cap')
def cap():
    return redirect(url_for('views.studio'))

@views_bp.route('/other')
def other():
    return render_template('other_studio.html', state=state, T=T, active='other')

@views_bp.route('/studio')
def studio():
    return render_template('defect_studio.html', state=state, T=T, active='studio')
