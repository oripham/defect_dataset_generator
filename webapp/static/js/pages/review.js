// static/js/pages/review.js — QA Review Tab

let reviewImages = [];  // [{filename, b64, status:'pending'|'accepted'|'rejected', ssim: number|null}]

window.addEventListener('load', () => {
  refreshBatchList();
});

async function refreshBatchList() {
  const sel = document.getElementById('review-batch-select');
  const prev = sel.value;
  try {
    const resp = await fetch('/api/review/batches');
    const data = await resp.json();
    sel.innerHTML = `<option value="">${t('-- Select Batch --')}</option>`;
    (data.batches || []).forEach(b => {
      const opt = document.createElement('option');
      opt.value = b.id;
      opt.textContent = `${b.id}  (${b.count} ${t('images, ')}${b.date})`;
      sel.appendChild(opt);
    });
    if (prev) sel.value = prev;
  } catch(e) { console.warn('Failed to load batch list', e); }
}

async function loadReviewBatch() {
  const batchId = document.getElementById('review-batch-select').value;
  const grid = document.getElementById('review-grid');
  if (!batchId) {
    grid.innerHTML = `<div class="text-muted small w-100 text-center py-5">${t('Select a batch above.')}</div>`;
    reviewImages = [];
    updateCounts();
    return;
  }
  grid.innerHTML = `<div class="text-muted small w-100 text-center py-5">${t('Loading...')}</div>`;

  try {
    const resp = await fetch(`/api/review/images/${batchId}`);
    const data = await resp.json();
    reviewImages = (data.images || []).map(img => ({
      filename: img.filename,
      b64:      img.b64,
      ssim:     img.ssim,
      status:   'pending'
    }));
    renderGrid();
    document.getElementById('review-stats-badge').textContent =
      `${reviewImages.length} ${t('images — Batch ')}${batchId}`;
  } catch(e) {
    grid.innerHTML = `<div class="text-danger small w-100 text-center py-5">Error: ${e}</div>`;
  }
}

function renderGrid() {
  const grid = document.getElementById('review-grid');
  if (!reviewImages.length) {
    grid.innerHTML = `<div class="text-muted small w-100 text-center py-5">${t('No images in this batch.')}</div>`;
    updateCounts();
    return;
  }
  grid.innerHTML = reviewImages.map((img, i) => {
    const borderColor = img.status === 'accepted' ? '#22c55e'
                      : img.status === 'rejected' ? '#ef4444'
                      : '#3a3a5c';
    const opacity = img.status === 'rejected' ? '0.4' : '1';
    const ssimBadge = img.ssim !== null && img.ssim !== undefined
      ? `<span class="badge ${img.ssim >= 0.85 ? 'bg-success' : 'bg-danger'}" style="font-size:10px;">SSIM ${img.ssim.toFixed(3)}</span>`
      : '';
    const statusIcon = img.status === 'accepted' ? '✅'
                     : img.status === 'rejected' ? '❌'
                     : '⏳';

    return `
      <div class="review-card" style="width:200px;border:2px solid ${borderColor};border-radius:8px;overflow:hidden;background:#1a1a2e;opacity:${opacity};transition:all .2s;">
        <img src="${img.b64}" alt="${img.filename}" style="width:200px;height:150px;object-fit:cover;cursor:pointer;display:block;" onclick="toggleZoom(${i})">
        <div class="p-2">
          <div class="d-flex justify-content-between align-items-center mb-1">
            <span class="small text-truncate" style="max-width:110px;color:#94a3b8;" title="${img.filename}">${img.filename}</span>
            <span style="font-size:12px;">${statusIcon}</span>
          </div>
          <div class="mb-1">${ssimBadge}</div>
          <div class="d-flex gap-1">
            <button class="btn btn-sm ${img.status==='accepted' ? 'btn-success' : 'btn-outline-success'} flex-fill" style="font-size:11px;" onclick="setStatus(${i},'accepted')">${t('Keep')}</button>
            <button class="btn btn-sm ${img.status==='rejected' ? 'btn-danger'  : 'btn-outline-danger'}  flex-fill" style="font-size:11px;" onclick="setStatus(${i},'rejected')">${t('Drop')}</button>
          </div>
        </div>
      </div>`;
  }).join('');
  updateCounts();
}

function setStatus(idx, status) {
  if (reviewImages[idx].status === status) {
    reviewImages[idx].status = 'pending';
  } else {
    reviewImages[idx].status = status;
  }
  renderGrid();
}

function acceptAll() {
  reviewImages.forEach(img => { img.status = 'accepted'; });
  renderGrid();
}

function updateCounts() {
  const accepted = reviewImages.filter(i => i.status === 'accepted').length;
  const rejected = reviewImages.filter(i => i.status === 'rejected').length;
  document.getElementById('ct-accepted').textContent = accepted;
  document.getElementById('ct-rejected').textContent = rejected;
  document.getElementById('ct-total').textContent    = reviewImages.length;
}

async function rejectSelected() {
  const toDelete = reviewImages.filter(i => i.status === 'rejected');
  if (!toDelete.length) { alert(t('No images marked for rejection.')); return; }
  if (!confirm(`${t('Delete ')}${toDelete.length}${t(' rejected image(s)? This cannot be undone.')}`)) return;

  const batchId = document.getElementById('review-batch-select').value;
  try {
    const resp = await fetch('/api/review/delete', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        batch_id: batchId,
        filenames: toDelete.map(i => i.filename)
      })
    });
    const data = await resp.json();
    if (data.ok) {
      reviewImages = reviewImages.filter(i => i.status !== 'rejected');
      renderGrid();
      document.getElementById('review-stats-badge').textContent =
        `${reviewImages.length} ${t('images — Batch ')}${batchId}`;
    } else {
      alert('Error: ' + data.error);
    }
  } catch(e) {
    alert('Network error: ' + e);
  }
}

function toggleZoom(idx) {
  const img = reviewImages[idx];
  // Simple full-screen modal
  let modal = document.getElementById('zoom-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'zoom-modal';
    modal.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.9);z-index:9999;display:flex;align-items:center;justify-content:center;cursor:pointer;';
    modal.onclick = () => modal.remove();
    document.body.appendChild(modal);
  }
  modal.innerHTML = `<img src="${img.b64}" style="max-width:90vw;max-height:90vh;object-fit:contain;border-radius:8px;">
    <div style="position:absolute;top:20px;right:30px;color:white;font-size:14px;">
      ${img.filename} ${img.ssim !== null ? '| SSIM: ' + img.ssim.toFixed(4) : ''}${t(' — Click to close')}
    </div>`;
}
