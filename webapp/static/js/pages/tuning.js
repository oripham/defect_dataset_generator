// static/js/pages/tuning.js

// Default parameters (from scripts/generator.py)
const GENAI_DEFAULTS = {
  strength: 0.28,
  guidance_scale: 6.5,
  steps: 30,
  ip_scale: 0.6,
  controlnet_scale: 0.35,
  inject_alpha: 0.80,
  epsilon: 0.03
};


function resetAIDefaults() {
  document.getElementById('ai-strength').value     = GENAI_DEFAULTS.strength;
  document.getElementById('ai-guidance').value     = GENAI_DEFAULTS.guidance_scale;
  document.getElementById('ai-steps').value        = GENAI_DEFAULTS.steps;
  document.getElementById('ai-ip-scale').value     = GENAI_DEFAULTS.ip_scale;
  document.getElementById('ai-controlnet').value   = GENAI_DEFAULTS.controlnet_scale;
  document.getElementById('ai-inject-alpha').value = GENAI_DEFAULTS.inject_alpha;
  
  const epsInput = document.getElementById('ai-epsilon');
  if (epsInput) {
    epsInput.value = GENAI_DEFAULTS.epsilon;
    document.getElementById('val-ai-epsilon').innerText = GENAI_DEFAULTS.epsilon;
  }
}


function _getEngineOverride() {
  // Returns null (auto), 'cv', or 'genai'
  if (document.getElementById('engineCV')?.checked)   return 'cv';
  if (document.getElementById('engineAI')?.checked)   return 'genai';
  return null; // Auto → server router decides
}

function onEngineChange() {
  const isAI = document.getElementById('engineAI').checked;
  if (isAI) resetAIDefaults();
}

window.addEventListener('load', () => {});

function _getAIParams() {
  return {
    strength:          parseFloat(document.getElementById('ai-strength')?.value)     || GENAI_DEFAULTS.strength,
    guidance_scale:    parseFloat(document.getElementById('ai-guidance')?.value)     || GENAI_DEFAULTS.guidance_scale,
    steps:             parseInt(document.getElementById('ai-steps')?.value)           || GENAI_DEFAULTS.steps,
    ip_scale:          parseFloat(document.getElementById('ai-ip-scale')?.value)     || GENAI_DEFAULTS.ip_scale,
    controlnet_scale:  parseFloat(document.getElementById('ai-controlnet')?.value)   || GENAI_DEFAULTS.controlnet_scale,
    inject_alpha:      parseFloat(document.getElementById('ai-inject-alpha')?.value) || GENAI_DEFAULTS.inject_alpha,
    epsilon_factor:    parseFloat(document.getElementById('ai-epsilon')?.value)      || GENAI_DEFAULTS.epsilon,
  };
}


async function runPreview() {
  const defectType  = document.getElementById('defect-select')?.value;
  const material    = document.getElementById('material-select')?.value;
  const intensity   = (parseInt(document.getElementById('slider-intensity')?.value) || 50) / 100.0;
  const naturalness = (parseInt(document.getElementById('slider-natural')?.value) || 3) / 3.0;
  const engine      = _getEngineOverride();
  const jitter      = (parseFloat(document.getElementById('slider-jitter')?.value) || 0.0) / 180.0;

  if (!defectType) {
    alert(t('Please select a defect config first'));
    return;
  }

  const previewImg    = document.getElementById('preview-img');
  const spinner       = document.getElementById('preview-spinner');
  const badgeRow      = document.getElementById('engine-badge-row');
  const engineBadge   = document.getElementById('engine-badge');
  const methodBadge   = document.getElementById('method-badge');
  const timeBadge     = document.getElementById('time-badge');
  const previewPanel    = document.getElementById('preview-panel');
  const previewSingle   = document.getElementById('preview-single');
  const previewImgOrig  = document.getElementById('preview-img-orig');
  const previewImgSingle = document.getElementById('preview-img-single');

  previewPanel.style.display  = 'none';
  previewSingle.style.display = 'none';
  badgeRow.style.display      = 'none';
  spinner.style.display       = 'block';

  // Always send AI params (applies to both CV and GenAI)
  const aiParams = _getAIParams();

  try {
    const resp = await fetch('/api/generate/preview', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        class_name: defectType,
        material: material,
        intensity: intensity,
        naturalness: naturalness,
        position_jitter: jitter,
        engine_override: engine,
        ...aiParams
      })
    });

    if (!resp.ok) {
      spinner.style.display = 'none';
      let errMsg = `HTTP ${resp.status}`;
      try { const ed = await resp.json(); errMsg = ed.error || ed.detail || errMsg; } catch(_) {}
      alert(t('❌ Server Error: ') + errMsg);
      return;
    }

    const data = await resp.json();
    if (!data.ok) {
      spinner.style.display = 'none';
      alert(t('❌ Render Error: ') + data.error);
      return;
    }

    // Async polling to avoid Cloudflare 524 timeout
    const previewId = data.preview_id;
    const pollInterval = setInterval(async () => {
      try {
        const sresp = await fetch(`/api/generate/preview/status/${previewId}`);
        if (!sresp.ok) return;
        const sdata = await sresp.json();

        if (sdata.status === 'done') {
          clearInterval(pollInterval);
          spinner.style.display = 'none';

          // Engine + method badge
          const eng = sdata.engine_used || '';
          const mth = sdata.method_used || '';
          engineBadge.textContent = eng === 'cv' ? '⚡ CV Engine' : eng === 'genai' ? '🤖 GenAI Engine' : eng;
          engineBadge.className   = 'badge fs-6 px-3 py-2 ' + (eng === 'cv' ? 'bg-info text-dark' : 'bg-warning text-dark');
          methodBadge.textContent = mth || '—';
          timeBadge.textContent   = sdata.processing_time ? `${sdata.processing_time}s` : '';
          badgeRow.style.display  = 'flex';

          // Original vs Generated panel (fallback: single image if no base)
          if (sdata.base_image_b64) {
            previewImgOrig.src          = sdata.base_image_b64;
            previewImgOrig.style.display = 'block';
            previewImg.src              = sdata.image_b64;
            previewImg.style.display    = 'block';
            previewPanel.style.display  = 'flex';
            previewSingle.style.display = 'none';
          } else {
            previewImgSingle.src         = sdata.image_b64;
            previewImgSingle.style.display = 'block';
            previewSingle.style.display  = 'flex';
            previewPanel.style.display   = 'none';
          }
        } else if (sdata.status === 'error') {
          clearInterval(pollInterval);
          spinner.style.display = 'none';
          alert(t('❌ Render Error: ') + sdata.error);
        } else {
          const pg = sdata.progress || {};
          if (pg.status === 'queued' && pg.queued > 0) {
            spinner.title = t('⏳ Queued — waiting for GPU...');
          } else if (pg.status === 'generating') {
            const step = pg.step || 0;
            const total = pg.total_steps || 0;
            spinner.title = total > 0
              ? t('🔄 Generating... step ') + step + '/' + total
              : t('🔄 Generating...');
          }
        }
      } catch(e) {
        console.error('Polling error', e);
      }
    }, 1500);

  } catch(e) {
    spinner.style.display = 'none';
    alert(t('❌ Network Error: ') + e);
    placeholder.style.display = 'block';
  }
}

async function startBatch() {
  const material    = document.getElementById('material-select')?.value;
  const intensity   = (parseInt(document.getElementById('slider-intensity')?.value) || 50) / 100.0;
  const naturalness = (parseInt(document.getElementById('slider-natural')?.value) || 3) / 3.0;
  const engine      = _getEngineOverride();
  const numImages   = parseInt(document.getElementById('batch-count-input')?.value) || 100;
  const jitter      = (parseFloat(document.getElementById('slider-jitter')?.value) || 0.0) / 180.0;
  const ssimDesc    = document.getElementById('slider-ssim')?.value;
  const ssim_threshold = parseFloat(ssimDesc) || 0.85;
  const qa_enabled  = document.getElementById('qa-enabled')?.checked ?? true;

  const area    = document.getElementById('batch-progress-area');
  const bar     = document.getElementById('batch-progress-bar');
  const percent = document.getElementById('batch-percent');
  const text    = document.getElementById('batch-status-text');
  const discard = document.getElementById('discard-count');
  const btnStart  = document.getElementById('btn-start-batch');
  const btnZip    = document.getElementById('btn-download-zip');

  if (btnStart) btnStart.disabled = true;
  if (btnZip) btnZip.style.display = 'none';

  area.style.display = 'block';
  bar.style.width = '0%';
  percent.innerText = '0%';
  text.innerText = t('🔄 Sending request to engine...');

  // Always send AI params
  const aiParams = _getAIParams();

  try {
    const resp = await fetch('/api/generate/batch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        class_name: document.getElementById('batch-class-select')?.value || null,
        material: material,
        intensity: intensity,
        naturalness: naturalness,
        position_jitter: jitter,
        engine_override: engine,
        num_images: numImages,
        ssim_threshold: ssim_threshold,
        qa_enabled: qa_enabled,
        ...aiParams
      })
    });

    if (!resp.ok) {
      let errMsg = `HTTP ${resp.status}`;
      try { const ed = await resp.json(); errMsg = ed.error || ed.detail || errMsg; } catch(_) {}
      text.innerText = t('❌ Server Error: ') + errMsg;
      if (btnStart) btnStart.disabled = false;
      return;
    }

    const data = await resp.json();
    if (!data.ok) {
      text.innerText = t('❌ JS Error: ') + data.error;
      if (btnStart) btnStart.disabled = false;
      return;
    }

    const jobId = data.job_id;
    const interval = setInterval(async () => {
      try {
        const sr    = await fetch(`/api/generate/status/${jobId}`);
        if (!sr.ok) return;
        const sdata = await sr.json();
        if (sdata.status === 'done') {
          clearInterval(interval);
          bar.style.width = '100%';
          percent.innerText = '100%';
          text.innerText = t('✅ Batch generated successfully! Download now!');
          discard.innerText = sdata.discard_count || 0;
          if (btnStart) btnStart.style.display = 'none';
          if (btnZip) btnZip.style.display = 'block';
          window._currentJobId = jobId;
          loadResultsToGallery(jobId);
          setTimeout(() => {
            alert(t('🎉 BATCH COMPLETE!\nDataset written to Output Dir: ') + (sdata.output_dir || ''));
          }, 500);
        } else if (sdata.status === 'error') {
          clearInterval(interval);
          text.innerText = t('❌ Server Error: ') + sdata.error;
          if (btnStart) btnStart.disabled = false;
        } else {
          const pct = sdata.progress || 0;
          bar.style.width = pct + '%';
          percent.innerText = pct + '%';
          text.innerText = t('🔄 Generating... QA check... (') + pct + '%)';
          discard.innerText = sdata.discard_count || 0;
        }
      } catch(ee) {}
    }, 1500);

  } catch(e) {
    text.innerText = t('❌ Flask Error: ') + e;
    if (btnStart) btnStart.disabled = false;
  }
}

async function loadResultsToGallery(jobId = null) {
  const gallery = document.getElementById('results-gallery');
  gallery.innerHTML = `<span class="text-muted small">${t('Loading images from Output...')}</span>`;
  try {
    let url = '/api/peek-results';
    if (jobId) url += `?job_id=${jobId}`;
    const resp = await fetch(url);
    if (!resp.ok) { gallery.innerHTML = '<span class="text-danger small">Load error</span>'; return; }
    const data = await resp.json();
    gallery.innerHTML = '';
    if (!data.images || data.images.length === 0) {
      gallery.innerHTML = `<span class="text-muted small">${t('No images in Output yet...')}</span>`;
      return;
    }
    data.images.forEach(b64 => {
      const img = document.createElement('img');
      img.src = b64;
      img.style.cssText = 'width:150px;height:100px;object-fit:cover;border-radius:4px;border:1px solid #3a3a5c;cursor:pointer;';
      img.onclick = () => window.open(b64, '_blank');
      gallery.appendChild(img);
    });
  } catch(e) {
    gallery.innerHTML = `<span class="text-danger small">${t('Gallery display error: ')}${e}</span>`;
  }
}

function downloadZip() {
  if (window._currentJobId) {
    window.location.href = `/api/download-batch/${window._currentJobId}`;
  }
}
