// static/js/pages/setup.js

// ---------- Server Connection ----------
async function testConnection() {
  const url     = document.getElementById('server-url').value.trim();
  const api_key = document.getElementById('api-key').value.trim();
  const el      = document.getElementById('conn-result');
  el.innerHTML = `<span class="text-info">${t('Connecting...')}</span>`;
  let resp;
  try {
    resp = await fetch('/api/test-connection', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url, api_key})
    });
  } catch(e) {
    el.innerHTML = `<span class="text-danger">${t('❌ Network Error: ')}${e}</span>`;
    return;
  }
  const text = await resp.text();
  let data;
  try { data = JSON.parse(text); } catch(e) {
    el.innerHTML = `<span class="text-danger">❌ Server Error: HTTP ${resp.status}</span>`;
    return;
  }
  if (data.ok) {
    const caps = data.caps || {};
    const okMetalCap = !!caps.metal_cap;
    const warn = (!okMetalCap)
      ? `<div class="mt-1 text-warning">
           ⚠ Metal Cap API not found on this server → <strong>/studio will fall back to CV-only</strong>.
           ${data.suggested_url ? `<div class="mt-1">Try Server URL: <code>${data.suggested_url}</code></div>` : ``}
         </div>`
      : `<div class="mt-1 text-success">✅ Metal Cap API available (hybrid CV+SDXL supported).</div>`;
    el.innerHTML = `<div class="text-success fw-bold">${t('✅ Connected! ')}${data.detail}</div>${warn}`;
  } else {
    el.innerHTML = `<span class="text-danger">${t('❌ Failed: ')}${data.detail}</span>`;
  }
}

// ---------- Dataset Upload ----------
let currentClass = null;

async function uploadDataset(uploadType, className, files, inputId, statusId) {
  if (files.length === 0) return;
  const statusEl = statusId ? document.getElementById(statusId) : null;

  const validFiles = Array.from(files).filter(f => {
    const ext = f.name.split('.').pop().toLowerCase();
    return ['png', 'jpg', 'jpeg', 'bmp'].includes(ext) && f.size <= 50 * 1024 * 1024;
  });
  if (validFiles.length === 0) {
    if (statusEl) statusEl.innerHTML = '<span class="text-danger">No valid images found</span>';
    return;
  }

  const formData = new FormData();
  formData.append('type', uploadType);
  if (className) formData.append('class_name', className);
  validFiles.forEach(f => formData.append('files', f));

  const el = document.getElementById(inputId);
  const oldVal = el.value;
  el.value = `Uploading ${validFiles.length} files...`;
  if (statusEl) statusEl.innerHTML = '<span class="text-info">⏳ Uploading...</span>';

  try {
    const resp = await fetch('/api/upload-dataset', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.ok) {
      el.value = data.path;
      if (statusEl) statusEl.innerHTML = `<span class="text-success">✅ ${data.count} images uploaded</span>`;
      if (inputId === 'good-dir') {
        await _autoSetImageDimensions(validFiles[0]);
        saveBasic();
      }
      if (inputId === 'cls-ref-dir') {
        saveClass();
        // Auto-open crop panel after NG upload so user can crop immediately
        const cropPanel = document.getElementById('crop-panel');
        if (cropPanel && cropPanel.style.display === 'none') {
          cropPanel.style.display = '';
          loadCropImages();
          setCropTool('rect');
          _initCropOverlay();
        } else {
          loadCropImages();
        }
      }
    } else {
      if (statusEl) statusEl.innerHTML = `<span class="text-danger">❌ ${data.error}</span>`;
      el.value = oldVal;
    }
  } catch(err) {
    if (statusEl) statusEl.innerHTML = `<span class="text-danger">❌ ${err.message}</span>`;
    el.value = oldVal;
  }
}

function _autoSetImageDimensions(file) {
  return new Promise(resolve => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      document.getElementById('img-w').value = img.naturalWidth;
      document.getElementById('img-h').value = img.naturalHeight;
      URL.revokeObjectURL(url);
      resolve();
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(); };
    img.src = url;
  });
}

async function saveBasic() {
  await fetch('/api/save-step2', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      job_name: 'my_job',
      good_images_path: document.getElementById('good-dir').value,
      output_root: document.getElementById('out-dir')?.value || '',
      image_width:  parseInt(document.getElementById('img-w').value) || 1024,
      image_height: parseInt(document.getElementById('img-h').value) || 1024,
    })
  });
  const el = document.getElementById('basic-saved');
  el.style.display = '';
  setTimeout(() => el.style.display = 'none', 2000);
  await refreshMaskStatus();
}

async function addAPIClass() {
  const sel = document.getElementById('new-defect-type');
  let name = sel.value;
  if (name === '__custom__') {
    name = prompt(t('Enter English defect name (e.g. paint_peel, dirt):'));
    if (!name) return;
  }
  if (!name) return;
  const resp = await fetch('/api/add-class', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: name.trim()})
  });
  const data = await resp.json();
  if (!data.ok) { alert(t(data.error)); return; }
  renderClassList(data.classes);
  selectClass(name.trim());
}

function renderClassList(classes) {
  const el = document.getElementById('class-list');
  if (!classes.length) {
    el.innerHTML = `<p class="text-muted small">${t('No classes yet. Add a defect type to begin.')}</p>`;
    return;
  }
  el.innerHTML = classes.map(c => {
    const maskBadge = c.has_masks
      ? '<span class="badge bg-success ms-1" style="font-size:9px;">Mask OK</span>'
      : '<span class="badge bg-warning text-dark ms-1" style="font-size:9px;">No Mask</span>';
    const cropBadge = c.has_cropped_refs
      ? '<span class="badge bg-info text-dark ms-1" style="font-size:9px;">✂ Crop OK</span>' : '';
    return `<button class="btn btn-outline-secondary btn-sm me-1 mb-1 class-btn${c.name === currentClass ? ' active' : ''}"
             data-name="${c.name}" onclick="selectClass('${c.name}')">${c.name}${maskBadge}${cropBadge}</button>`;
  }).join('');
  refreshMaskStatus(classes);
}

function refreshMaskStatus(classes) {
  if (!classes) { fetch('/api/classes').then(r => r.json()).then(d => refreshMaskStatus(d.classes)); return; }
  const missing = classes.filter(c => !c.has_masks);
  const bar     = document.getElementById('mask-status-bar');
  const content = document.getElementById('mask-status-content');
  if (!classes.length) { bar.style.display = 'none'; return; }
  bar.style.display = '';
  if (missing.length === 0) {
    content.innerHTML = `<span class="text-success">${t('✅ All classes have masks!')}</span>`;
  } else {
    const names = missing.map(c => `<strong>${c.name}</strong>`).join(', ');
    content.innerHTML = `<span class="text-danger">${t('⚠ Missing masks for: ')}${names}</span>
      <a href="/masking" class="btn btn-sm btn-outline-danger ms-3">${t('Go to Mask Editor →')}</a>`;
  }
}

async function selectClass(name) {
  currentClass = name;
  document.querySelectorAll('.class-btn').forEach(b => b.classList.toggle('active', b.dataset.name === name));
  const resp = await fetch('/api/classes');
  const data = await resp.json();
  const cls = data.classes.find(c => c.name === name);
  if (!cls) return;

  document.getElementById('class-detail').style.display = '';
  document.getElementById('class-detail-title').textContent = `Class Info: ${name}`;
  document.getElementById('cls-name').value    = cls.name;
  document.getElementById('cls-id').value      = cls.class_id;
  document.getElementById('cls-ref-dir').value = cls.ref_dir || '';
  document.getElementById('cls-prompts').value = (cls.prompts || []).join(', ');
  document.getElementById('cls-neg-prompt').value = cls.negative_prompt || '';
}

async function saveClass() {
  if (!currentClass) return;
  const resp = await fetch('/api/update-class', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      name: currentClass,
      class_id: parseInt(document.getElementById('cls-id').value) || 0,
      ref_dir: document.getElementById('cls-ref-dir').value,
      prompts: document.getElementById('cls-prompts').value.split(',').map(s => s.trim()).filter(Boolean),
      negative_prompt: document.getElementById('cls-neg-prompt').value,
    })
  });
  const data = await resp.json();
  if (!data.ok) { alert(t(data.error)); return; }
  renderClassList(data.classes);
  const el = document.getElementById('cls-saved');
  el.style.display = '';
  setTimeout(() => el.style.display = 'none', 2000);
}

async function deleteClass() {
  if (!currentClass || !confirm(`Delete class "${currentClass}"?`)) return;
  const resp = await fetch(`/api/class/${encodeURIComponent(currentClass)}`, {method: 'DELETE'});
  const data = await resp.json();
  currentClass = null;
  document.getElementById('class-detail').style.display = 'none';
  renderClassList(data.classes || []);
}

// ---------- Crop Tool ----------
const CROP_W = 640, CROP_H = 480;
let cropImg = null, cropFilename = null, cropRect = null;
let cropDrawing = false, cropSX = 0, cropSY = 0;
let cropScale = 1, cropOffX = 0, cropOffY = 0;
// Brush overlay state
let cropTool        = 'rect';   // 'rect' | 'brush' | 'eraser'
let cropBrushSize   = 20;
let cropBrushActive = false;
let cropMaskHistory = [];

function setCropTool(tool) {
  cropTool = tool;
  ['rect', 'brush', 'eraser'].forEach(t => {
    const btn = document.getElementById('crop-tool-' + t);
    if (btn) btn.classList.toggle('active', t === tool);
  });
  const overlay = document.getElementById('crop-mask-overlay');
  const cv      = document.getElementById('crop-canvas');
  if (overlay) overlay.style.pointerEvents = (tool !== 'rect') ? 'auto' : 'none';
  if (cv)      cv.style.pointerEvents      = (tool === 'rect') ? 'auto' : 'none';
}

function clearCropMask() {
  const overlay = document.getElementById('crop-mask-overlay');
  if (overlay) overlay.getContext('2d').clearRect(0, 0, CROP_W, CROP_H);
  cropMaskHistory = [];
  cropRect = null;
  renderCropCanvas();
}

function cropUndoLast() {
  const overlay = document.getElementById('crop-mask-overlay');
  if (!overlay || !cropMaskHistory.length) return;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, CROP_W, CROP_H);
  cropMaskHistory.pop();
  if (cropMaskHistory.length) ctx.putImageData(cropMaskHistory[cropMaskHistory.length - 1], 0, 0);
}

function _pushCropHistory(ctx) {
  const snap = ctx.getImageData(0, 0, CROP_W, CROP_H);
  cropMaskHistory.push(snap);
  if (cropMaskHistory.length > 30) cropMaskHistory.shift();
}

function _initCropOverlay() {
  const overlay = document.getElementById('crop-mask-overlay');
  if (!overlay || overlay._initDone) return;
  overlay._initDone = true;
  const getPos = e => {
    const r = overlay.getBoundingClientRect();
    return [(e.clientX - r.left) * CROP_W / r.width, (e.clientY - r.top) * CROP_H / r.height];
  };
  const ctx = overlay.getContext('2d');
  overlay.addEventListener('mousedown', e => {
    if (cropTool === 'rect') return;
    cropBrushActive = true;
    _pushCropHistory(ctx);
    const [x, y] = getPos(e); _doBrush(ctx, x, y);
  });
  overlay.addEventListener('mousemove', e => {
    if (!cropBrushActive) return;
    const [x, y] = getPos(e); _doBrush(ctx, x, y);
  });
  const stop = () => { cropBrushActive = false; };
  overlay.addEventListener('mouseup', stop);
  overlay.addEventListener('mouseleave', stop);
}

function _doBrush(ctx, x, y) {
  ctx.save();
  if (cropTool === 'eraser') {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0,0,0,1)';
  } else {
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(0, 229, 255, 0.5)';
  }
  ctx.beginPath(); ctx.arc(x, y, cropBrushSize / 2, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

async function toggleCropPanel() {
  const panel = document.getElementById('crop-panel');
  const opening = panel.style.display === 'none';
  panel.style.display = opening ? '' : 'none';
  if (opening) {
    loadCropImages();
    setCropTool('rect');
    _initCropOverlay();
  }
}

async function loadCropImages() {
  if (!currentClass) return;
  const resp = await fetch(`/api/list-ref-images/${encodeURIComponent(currentClass)}`);
  const data = await resp.json();
  const el = document.getElementById('crop-img-list');
  if (!data.images.length) {
    el.innerHTML = `<span class="small text-muted">${t('No NG images in folder')}</span>`;
    return;
  }
  el.innerHTML = data.images.map(f =>
    `<button class="btn btn-sm btn-outline-secondary w-100 text-start mb-1 text-truncate" style="font-size:11px" onclick="loadCropImage('${f}')">${f}</button>`
  ).join('');
  loadCropImage(data.images[0]);
}

function loadCropImage(fname) {
  cropFilename = fname;
  const img = new Image();
  img.onload = () => {
    cropImg = img;
    cropScale = Math.min(CROP_W / img.width, CROP_H / img.height);
    cropOffX = (CROP_W - img.width * cropScale) / 2;
    cropOffY = (CROP_H - img.height * cropScale) / 2;
    cropRect = null;
    renderCropCanvas();
    // Clear brush overlay on new image
    clearCropMask();
  };
  img.src = `/api/ref-image/${encodeURIComponent(currentClass)}/${encodeURIComponent(fname)}`;
}

function renderCropCanvas(rx, ry, rw, rh) {
  const cv = document.getElementById('crop-canvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#111'; ctx.fillRect(0, 0, CROP_W, CROP_H);
  if (cropImg) ctx.drawImage(cropImg, cropOffX, cropOffY, cropImg.width * cropScale, cropImg.height * cropScale);
  const r = (rx !== undefined) ? {x: rx, y: ry, w: rw, h: rh} : cropRect;
  if (r && r.w && r.h) {
    ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2; ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.fillStyle = 'rgba(0, 229, 255, 0.15)'; ctx.fillRect(r.x, r.y, r.w, r.h);
  }
}

window.addEventListener('load', () => {
  const cv = document.getElementById('crop-canvas');
  if (cv) {
    const getPos = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * CROP_W / r.width, (e.clientY - r.top) * CROP_H / r.height]; };
    cv.addEventListener('mousedown', e => { [cropSX, cropSY] = getPos(e); cropDrawing = true; });
    cv.addEventListener('mousemove', e => {
      if (!cropDrawing) return;
      const [x, y] = getPos(e);
      renderCropCanvas(Math.min(cropSX, x), Math.min(cropSY, y), Math.abs(x - cropSX), Math.abs(y - cropSY));
    });
    const finishDraw = e => {
      if (!cropDrawing) return;
      cropDrawing = false;
      const [x, y] = getPos(e);
      cropRect = {x: Math.min(cropSX, x), y: Math.min(cropSY, y), w: Math.abs(x - cropSX), h: Math.abs(y - cropSY)};
      renderCropCanvas();
    };
    cv.addEventListener('mouseup', finishDraw);
    cv.addEventListener('mouseleave', finishDraw);
  }
  _initCropOverlay();
});

async function saveCrop() {
  if (!cropImg || !cropFilename) return;
  let tmp;

  if (cropTool === 'brush' || cropTool === 'eraser') {
    // Extract bounding box from brush overlay mask
    const overlay = document.getElementById('crop-mask-overlay');
    if (!overlay) return;
    const overlayCtx = overlay.getContext('2d');
    const md = overlayCtx.getImageData(0, 0, CROP_W, CROP_H);
    let minX = CROP_W, minY = CROP_H, maxX = 0, maxY = 0;
    for (let py = 0; py < CROP_H; py++) {
      for (let px = 0; px < CROP_W; px++) {
        if (md.data[(py * CROP_W + px) * 4 + 3] > 20) {
          minX = Math.min(minX, px); minY = Math.min(minY, py);
          maxX = Math.max(maxX, px); maxY = Math.max(maxY, py);
        }
      }
    }
    if (maxX <= minX || maxY <= minY) { alert(t('Please select a larger region.')); return; }
    const imgX = Math.max(0, (minX - cropOffX) / cropScale);
    const imgY = Math.max(0, (minY - cropOffY) / cropScale);
    const imgW = Math.min(cropImg.width  - imgX, (maxX - minX) / cropScale);
    const imgH = Math.min(cropImg.height - imgY, (maxY - minY) / cropScale);
    tmp = document.createElement('canvas');
    tmp.width = Math.round(imgW); tmp.height = Math.round(imgH);
    tmp.getContext('2d').drawImage(cropImg,
      Math.round(imgX), Math.round(imgY), Math.round(imgW), Math.round(imgH),
      0, 0, Math.round(imgW), Math.round(imgH));
  } else {
    // Rect mode
    if (!cropRect || cropRect.w < 5 || cropRect.h < 5) { alert(t('Please select a larger region.')); return; }
    const imgX = Math.max(0, (cropRect.x - cropOffX) / cropScale);
    const imgY = Math.max(0, (cropRect.y - cropOffY) / cropScale);
    const imgW = Math.min(cropImg.width  - imgX, cropRect.w / cropScale);
    const imgH = Math.min(cropImg.height - imgY, cropRect.h / cropScale);
    tmp = document.createElement('canvas');
    tmp.width = Math.round(imgW); tmp.height = Math.round(imgH);
    tmp.getContext('2d').drawImage(cropImg,
      Math.round(imgX), Math.round(imgY), Math.round(imgW), Math.round(imgH),
      0, 0, Math.round(imgW), Math.round(imgH));
  }

  const baseName = cropFilename.replace(/\.[^.]+$/, '');
  const resp = await fetch('/api/save-ref-crop', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      class_name: currentClass,
      filename: baseName + '_crop.png',
      image_data: tmp.toDataURL('image/png')
    })
  });
  const data = await resp.json();
  if (data.ok) {
    document.getElementById('crop-result').innerHTML = `<span class="text-success">${t('✅ Saved!')}</span>`;
    fetch('/api/classes').then(r => r.json()).then(d => renderClassList(d.classes));
  }
}

// Upload pre-cropped image (bypass canvas draw)
async function uploadPreCroppedImage(input) {
  if (!input.files.length) return;
  if (!currentClass) { alert('Please select a defect class first.'); return; }
  const file = input.files[0];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['png', 'jpg', 'jpeg', 'bmp'].includes(ext)) {
    alert('Please select a PNG, JPG, or BMP image.');
    return;
  }
  const resultEl = document.getElementById('precrop-result');
  resultEl.innerHTML = '<span class="text-info">Uploading...</span>';

  const reader = new FileReader();
  reader.onload = async (e) => {
    const resp = await fetch('/api/save-ref-crop', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        class_name: currentClass,
        filename: file.name,
        image_data: e.target.result
      })
    });
    const data = await resp.json();
    if (data.ok) {
      resultEl.innerHTML = `<span class="text-success">${t('✅ Saved!')} → ${data.path || file.name}</span>`;
      fetch('/api/classes').then(r => r.json()).then(d => renderClassList(d.classes));
    } else {
      resultEl.innerHTML = `<span class="text-danger">❌ ${data.error}</span>`;
    }
  };
  reader.readAsDataURL(file);
}

// Init
window.addEventListener('load', () => {
  fetch('/api/classes').then(r => r.json()).then(d => renderClassList(d.classes));
});
