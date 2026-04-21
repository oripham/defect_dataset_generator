// static/js/pages/studio/studio_canvas.js

function _mcResetCanvas() {
  _mcBaseImg = null; _mcMaskOff = null; _mcDrawing = false;
  const panel = document.getElementById("mc-mask-panel");
  if (panel) panel.classList.add("d-none");
  document.getElementById("mc-mask-hint").textContent = "Paint the defect region, then click Set";
  const offEl = document.getElementById("mc-rim-offset");
  const offLbl = document.getElementById("mc-rim-offset-lbl");
  if (offEl) offEl.value = 0;
  if (offLbl) offLbl.textContent = "0";
}

function toggleMcMaskPanel() {
  const panel = document.getElementById("mc-mask-panel");
  if (!panel) return;
  const isHidden = panel.classList.contains("d-none");
  if (!isHidden) { 
    panel.classList.add("d-none"); 
    return; 
  }
  if (!okImageB64) { alert("Upload an OK image first."); return; }
  panel.classList.remove("d-none");
  _initMcMaskCanvas();
}

function _initMcMaskCanvas() {
  const cv = document.getElementById("mc-mask-canvas");
  const ctx = cv.getContext("2d");
  const MAX_W = 600;
  const img = new Image();
  img.onload = () => {
    _mcBaseImg = img;
    const scale = Math.min(1, MAX_W / img.width);
    cv.width = Math.round(img.width * scale);
    cv.height = Math.round(img.height * scale);
    _mcMaskOff = document.createElement("canvas");
    _mcMaskOff.width = cv.width;
    _mcMaskOff.height = cv.height;
    _mcMaskOff.getContext("2d").clearRect(0, 0, cv.width, cv.height);
    _mcRender(cv, ctx);
    cv.onmousedown = e => { _mcDrawing = true; _mcDraw(e, cv, ctx); };
    cv.onmousemove = e => { if (_mcDrawing) _mcDraw(e, cv, ctx); };
    cv.onmouseup = () => { _mcDrawing = false; };
    cv.onmouseleave = () => { _mcDrawing = false; };
  };
  img.src = "data:image/png;base64," + okImageB64;
}

function _mcRender(cv, ctx) {
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(_mcBaseImg, 0, 0, cv.width, cv.height);
  ctx.globalAlpha = 0.5;
  ctx.drawImage(_mcMaskOff, 0, 0);
  ctx.globalAlpha = 1;
}

function _mcDraw(e, cv, ctx) {
  const r = cv.getBoundingClientRect();
  const x = (e.clientX - r.left) * cv.width / r.width;
  const y = (e.clientY - r.top) * cv.height / r.height;
  const mCtx = _mcMaskOff.getContext("2d");
  mCtx.fillStyle = "#ff6400";
  mCtx.beginPath();
  mCtx.arc(x, y, _mcBrushSz / 2, 0, Math.PI * 2);
  mCtx.fill();
  _mcRender(cv, ctx);
}

function clearMcMask() {
  maskB64 = null;
  if (!_mcMaskOff || !_mcBaseImg) return;
  _mcMaskOff.getContext("2d").clearRect(0, 0, _mcMaskOff.width, _mcMaskOff.height);
  const cv = document.getElementById("mc-mask-canvas");
  _mcRender(cv, cv.getContext("2d"));
  document.getElementById("mc-mask-hint").textContent = "Paint the defect region, then click Set";
}

function confirmMcMask() {
  if (!_mcMaskOff || !_mcBaseImg) return;
  const outCv = document.createElement("canvas");
  outCv.width = _mcBaseImg.width;
  outCv.height = _mcBaseImg.height;
  const outCtx = outCv.getContext("2d");
  outCtx.drawImage(_mcMaskOff, 0, 0, outCv.width, outCv.height);
  const id = outCtx.getImageData(0, 0, outCv.width, outCv.height);
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i] > 50 ? 255 : 0;
    d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 255;
  }
  outCtx.putImageData(id, 0, 0);
  maskB64 = outCv.toDataURL("image/png").split(",")[1];
  showPanel("panel-mask", maskB64);
  
  const hintEl = document.getElementById("mc-mask-hint");
  if (hintEl) hintEl.innerHTML = '<span class="text-success fw-bold">\u2705 MASK LOCKED</span> \u2014 Position/Span now controlled by drawing';
  
  const selTheta = document.getElementById("sel-theta");
  if (selTheta) {
    if (!Array.from(selTheta.options).some(o => o.value === "mask_locked")) {
      const opt = new Option("Locked to Mask \ud83d\udd12", "mask_locked");
      selTheta.add(opt);
    }
    selTheta.value = "mask_locked";
  }

  const panel = document.getElementById("mc-mask-panel");
  if (panel) panel.classList.add("d-none");
  log("MC mask confirmed: overriding position defaults.");
}
