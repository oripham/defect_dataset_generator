// static/js/pages/masking.js

const editor = new MaskEditor('mask-canvas', 'bg-canvas', 800, 600);
let currentClass = '';
let currentGoodImg = '';

window.addEventListener('load', () => {
  const initMissing = document.getElementById('init-missing-class');
  if (initMissing && initMissing.value) {
    const sel = document.getElementById('cls-select');
    sel.value = initMissing.value;
    onClassChange();
  }
  loadGoodImages();
});

function setTool(toolName) {
  editor.tool = toolName;
  ['brush', 'rect', 'ellipse', 'eraser'].forEach(x =>
    document.getElementById('tool-' + x).classList.toggle('active', x === toolName)
  );
}

function showRefImgPanel() {
  const p = document.getElementById('ref-img-panel');
  p.style.display = p.style.display === 'none' ? '' : 'none';
  if (p.style.display !== 'none') loadRefImages();
}

async function onClassChange() {
  const sel = document.getElementById('cls-select');
  currentClass = sel.value;
  currentGoodImg = '';
  if (!currentClass) return;

  const opt = sel.selectedOptions[0];
  const isMissing = opt && opt.dataset.missing === '1';
  document.getElementById('class-mask-status').innerHTML = isMissing
    ? `<span class="text-warning">${t('⚠ No mask drawn')}</span>`
    : `<span class="text-success">${t('✅ Mask ready')}</span>`;

  await loadGoodImages();
  await refreshMaskStatus();
}

async function loadGoodImages() {
  const resp = await fetch('/api/list-good-images');
  const data = await resp.json();
  const el = document.getElementById('good-img-list');
  if (!data.images.length) {
    el.innerHTML = `<span class="small text-muted">${t('Please load Good Images in Setup Tab')}</span>`;
    return;
  }
  el.innerHTML = data.images.map(f =>
    `<button class="btn btn-sm btn-outline-secondary w-100 text-start mb-1 text-truncate good-img-btn"
      id="gimg-${CSS.escape(f)}" style="font-size:11px" title="${f}"
      onclick="selectGoodImageMask('${f}')">${f}</button>`
  ).join('');
  await selectGoodImageMask(data.images[0]);
}

async function selectGoodImageMask(fname) {
  currentGoodImg = fname;
  document.querySelectorAll('.good-img-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`gimg-${CSS.escape(fname)}`);
  if (btn) btn.classList.add('active');
  document.getElementById('current-img-label').textContent = `Drawing on: ${fname}`;
  editor.loadBackground(`/api/good-image/${encodeURIComponent(fname)}`);
  if (!currentClass) return;

  const subName = fname.replace(/\.[^.]+$/, '');
  await fetch('/api/create-subfolder', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({class_name: currentClass, folder_name: subName})
  });
  await refreshMaskStatus();

  const maskResp = await fetch(`/api/load-mask/${encodeURIComponent(currentClass)}/${encodeURIComponent(subName)}`);
  if (maskResp.ok) {
    const blob = await maskResp.blob();
    const reader = new FileReader();
    reader.onload = e => editor.loadMaskFromDataURL(e.target.result);
    reader.readAsDataURL(blob);
  } else {
    editor.clear();
  }
}

async function loadRefImages() {
  if (!currentClass) return;
  const resp = await fetch(`/api/list-ref-images/${encodeURIComponent(currentClass)}`);
  const data = await resp.json();
  const el = document.getElementById('ref-img-list');
  if (!data.images.length) {
    el.innerHTML = `<span class="small text-muted">${t('No ref images.')}</span>`;
    return;
  }
  el.innerHTML = data.images.map(f =>
    `<button class="btn btn-sm btn-outline-secondary w-100 text-start mb-1 text-truncate"
      style="font-size:11px" title="${f}"
      onclick="editor.loadBackground('/api/ref-image/${encodeURIComponent(currentClass)}/${encodeURIComponent(f)}')">${f}</button>`
  ).join('');
}

async function refreshMaskStatus() {
  if (!currentClass) return;
  const resp = await fetch(`/api/list-subfolders/${encodeURIComponent(currentClass)}`);
  const data = await resp.json();
  document.getElementById('mask-dir-info').textContent = data.mask_dir ? `System folder: ${data.mask_dir}` : '';
  const n = data.subfolders.length;
  document.getElementById('mask-count').textContent = n ? `Masks linked: ${n} image(s)` : 'No mask data yet.';

  data.subfolders.forEach(sub => {
    document.querySelectorAll('.good-img-btn').forEach(btn => {
      if (btn.title.replace(/\.[^.]+$/, '') === sub && !btn.querySelector('.mask-badge')) {
        const badge = document.createElement('span');
        badge.className = 'mask-badge badge bg-success ms-1';
        badge.style.fontSize = '8px'; badge.textContent = '✓';
        btn.appendChild(badge);
      }
    });
  });
}

async function saveMask() {
  if (!currentClass || !currentGoodImg) {
    alert(t('Please select a class first'));
    return;
  }
  const subName  = currentGoodImg.replace(/\.[^.]+$/, '');
  const maskData = editor.getMaskDataURL();
  const resp = await fetch('/api/save-mask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({class_name: currentClass, subfolder: subName, image_data: maskData})
  });
  const data = await resp.json();
  const el = document.getElementById('save-result');
  if (data.ok) {
    el.innerHTML = `<span class="text-success fw-bold">${t('✅ Saved for image ')}${currentGoodImg}</span>`;
    document.getElementById('class-mask-status').innerHTML = `<span class="text-success">${t('✅ Mask OK')}</span>`;
    const alertBox = document.getElementById('missing-alert-box');
    if (alertBox) alertBox.style.display = 'none';
    await refreshMaskStatus();
  } else {
    el.innerHTML = `<span class="text-danger">❌ ${data.error}</span>`;
  }
}

function loadMaskFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => editor.loadMaskFromDataURL(ev.target.result);
  reader.readAsDataURL(file);
}
