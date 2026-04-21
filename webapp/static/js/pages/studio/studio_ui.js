// static/js/pages/studio/studio_ui.js

function toggleAIParams() {
  const isAi = document.getElementById("chk-use-ai").checked;
  const divBlock = document.getElementById("params-diversity");
  if(divBlock) divBlock.style.display = isAi ? "flex" : "none";
  const promptBlock = document.getElementById("params-prompt");
  if(promptBlock) promptBlock.classList.toggle("d-none", !isAi);
  if(promptBlock) promptBlock.classList.toggle("d-flex", isAi);
}

function getDefaultPrompt() {
  const defectKey = document.getElementById("sel-defect").value;
  const key = currentGroup + "|" + defectKey;
  return DEFAULT_PROMPTS[key] || { pos: "", neg: "" };
}

function resetPromptDefaults() {
  const dp = getDefaultPrompt();
  document.getElementById("txt-prompt").value = dp.pos;
  document.getElementById("txt-neg-prompt").value = dp.neg;
}

function selectGroup(group) {
  currentGroup = group;
  okImageB64 = maskB64 = resultB64 = null;
  batchImagePool = [];
  document.getElementById("upload-ok").value = "";
  clearPanels(["panel-ok", "panel-mask", "panel-result", "panel-diff"]);
  document.getElementById("thumb-ok").style.display = "none";
  document.getElementById("mask-status").textContent = "";

  document.getElementById("btn-group-cap").className =
    "btn btn-sm flex-fill " + (group === "cap" ? "btn-info" : "btn-outline-info");
  document.getElementById("btn-group-pharma").className =
    "btn btn-sm flex-fill " + (group === "pharma" ? "btn-warning" : "btn-outline-warning");
  document.getElementById("btn-group-napchai").className =
    "btn btn-sm flex-fill " + (group === "metal_cap" ? "btn-success" : "btn-outline-success");

  const hints = ["group-cap-hint", "group-pharma-hint", "group-napchai-hint"];
  const activeHint = `group-${group === "metal_cap" ? "napchai" : group}-hint`;
  hints.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = (id === activeHint) ? "" : "none";
  });
  
  const showNgRef = (group === "metal_cap") || (group === "pharma");
  document.getElementById("ng-ref-wrap").style.display = showNgRef ? "" : "none";
  if (!showNgRef) clearNgRef();

  const sel = document.getElementById("sel-defect");
  sel.innerHTML = "";
  GROUPS[group].defects.forEach(d => {
    const opt = document.createElement("option");
    opt.value = d.key;
    opt.textContent = d.label;
    sel.appendChild(opt);
  });

  onDefectChange();
}

function onDefectChange() {
  const defectKey = document.getElementById("sel-defect").value;
  const entry = GROUPS[currentGroup].defects.find(d => d.key === defectKey) || {};
  const paramType = entry.params;

  ["params-crack", "params-dent", "params-hollow", "params-polar", "params-scuff",
    "params-spots", "params-napchai-ring", "params-plastic-flow-tune", "params-napchai-scratch"]
    .forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.classList.add("d-none");
        el.classList.remove("d-flex");
      }
    });

  if (paramType) {
    const pEl = document.getElementById("params-" + paramType);
    if (pEl) {
      pEl.classList.remove("d-none");
      pEl.classList.add("d-flex");
    }
  }
  
  const showNgRef = (currentGroup === "metal_cap") ||
    (currentGroup === "pharma" && (defectKey === "hollow" || defectKey === "underfill"));
  const ngRefWrap = document.getElementById("ng-ref-wrap");

  if (showNgRef) {
    ngRefWrap.style.display = "";
    const ngRefLabel = document.getElementById("ng-ref-label");
    if (ngRefLabel) {
      if (currentGroup === "pharma") {
        ngRefLabel.innerHTML = 'NG Reference (Texture/Hollow) <span class="text-muted fw-normal">optional</span>';
      } else {
        ngRefLabel.innerHTML = 'NG Reference (IP-Adapter) <span class="text-muted fw-normal">optional</span>';
      }
    }
  } else {
    ngRefWrap.style.display = "none";
    clearNgRef();
  }

  const drawWrap = document.getElementById("mc-mask-draw-wrap");
  if (drawWrap) {
    drawWrap.classList.remove("d-none");
  }

  const showRimOffset = (currentGroup === "metal_cap") &&
    (defectKey === "scratch" || defectKey === "ring_fracture");
  const rimWrap = document.getElementById("mc-rim-offset-wrap");
  if (rimWrap) {
    if (showRimOffset) rimWrap.classList.remove("d-none");
    else rimWrap.classList.add("d-none");
  }

  const hintEl = document.getElementById("mc-mask-hint");
  if (hintEl) {
    if (currentGroup === "metal_cap") {
      if (defectKey === "scratch") hintEl.textContent = "Paint scratch region (slightly inside the outer rim), then click Set";
      else if (defectKey === "mc_deform") hintEl.textContent = "Paint a position on the outer rim (just to indicate location), then click Set";
      else if (defectKey === "ring_fracture") hintEl.textContent = "Paint a thin band on the rim you want to distort, then click Set";
      else hintEl.textContent = "Paint the defect region, then click Set";
    } else {
      hintEl.textContent = "Paint the defect region on the image, then click Set";
    }
  }
  
  if (currentGroup === "metal_cap") {
    maskB64 = null;
    _mcResetCanvas();
    clearPanels(["panel-mask"]);
  }
  resetPromptDefaults();
  toggleAIParams();
}

function onUploadOk(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    okImageB64 = e.target.result.split(",")[1];
    const thumb = document.getElementById("thumb-ok");
    thumb.src = e.target.result;
    thumb.style.display = "";
    showPanel("panel-ok", okImageB64);
    maskB64 = null;
    clearPanels(["panel-mask", "panel-result", "panel-diff"]);
    document.getElementById("mask-status").textContent = "";
    document.getElementById("btn-save").disabled = true;
    document.getElementById("btn-download").disabled = true;
    _mcResetCanvas();
    log("Uploaded: " + file.name);
  };
  reader.readAsDataURL(file);
}

function onUploadNgRef(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const dataUrl = e.target.result;
    const img = new Image();
    img.onload = () => {
      _ngFullImg = img;
      ngRefB64 = dataUrl.split(",")[1];
      document.getElementById("thumb-ng-ref").src = dataUrl;
      document.getElementById("ng-ref-name").textContent = file.name;
      document.getElementById("ng-ref-dim").textContent = img.width + "\u00d7" + img.height + " (full)";
      document.getElementById("ng-ref-loaded").style.display = "";
      document.getElementById("ng-crop-panel").style.display = "none";
      log("NG ref loaded: " + file.name);
    };
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
}

function toggleNgCropPanel() {
  const panel = document.getElementById("ng-crop-panel");
  const open = panel.style.display !== "none";
  if (open) { panel.style.display = "none"; return; }
  panel.style.display = "";
  _initNgCropCanvas();
}

function _initNgCropCanvas() {
  if (!_ngFullImg) return;
  const cv = document.getElementById("ng-crop-canvas");
  _ngCropScale = Math.min(1, _NG_CVW / _ngFullImg.width);
  cv.width = Math.round(_ngFullImg.width * _ngCropScale);
  cv.height = Math.round(_ngFullImg.height * _ngCropScale);
  _ngCropRect = null;
  _renderNgCrop();
  cv.onmousedown = e => {
    const p = _ngCropPos(e); _ngCropSX = p[0]; _ngCropSY = p[1];
    _ngCropDraw = true; _ngCropRect = null;
  };
  cv.onmousemove = e => {
    if (!_ngCropDraw) return;
    const p = _ngCropPos(e);
    _ngCropRect = {
      x: Math.min(_ngCropSX, p[0]), y: Math.min(_ngCropSY, p[1]),
      w: Math.abs(p[0] - _ngCropSX), h: Math.abs(p[1] - _ngCropSY)
    };
    _renderNgCrop();
  };
  cv.onmouseup = () => { _ngCropDraw = false; };
}

function _ngCropPos(e) {
  const cv = document.getElementById("ng-crop-canvas");
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height];
}

function _renderNgCrop() {
  const cv = document.getElementById("ng-crop-canvas");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(_ngFullImg, 0, 0, cv.width, cv.height);
  const rc = _ngCropRect;
  if (rc && rc.w > 4 && rc.h > 4) {
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(0, 0, cv.width, rc.y);
    ctx.fillRect(0, rc.y + rc.h, cv.width, cv.height - rc.y - rc.h);
    ctx.fillRect(0, rc.y, rc.x, rc.h);
    ctx.fillRect(rc.x + rc.w, rc.y, cv.width - rc.x - rc.w, rc.h);
    ctx.strokeStyle = "#00d4ff"; ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]); ctx.strokeRect(rc.x, rc.y, rc.w, rc.h);
    ctx.setLineDash([]);
    const iw = Math.round(rc.w / _ngCropScale), ih = Math.round(rc.h / _ngCropScale);
    ctx.fillStyle = "#00d4ff"; ctx.font = "12px monospace";
    ctx.fillText(iw + "\u00d7" + ih, rc.x + 4, rc.y + rc.h - 4);
  }
}

function confirmNgCrop() {
  const rc = _ngCropRect;
  if (!rc || rc.w < 10 || rc.h < 10) {
    document.getElementById("ng-crop-hint").textContent = "\u26a0 Draw a region first";
    return;
  }
  const ix = Math.round(rc.x / _ngCropScale), iy = Math.round(rc.y / _ngCropScale);
  const iw = Math.round(rc.w / _ngCropScale), ih = Math.round(rc.h / _ngCropScale);
  const tmp = document.createElement("canvas");
  tmp.width = iw; tmp.height = ih;
  tmp.getContext("2d").drawImage(_ngFullImg, ix, iy, iw, ih, 0, 0, iw, ih);
  const dataUrl = tmp.toDataURL("image/png");
  ngRefB64 = dataUrl.split(",")[1];
  document.getElementById("thumb-ng-ref").src = dataUrl;
  document.getElementById("ng-ref-dim").textContent = "\u2702 " + iw + "\u00d7" + ih + " (cropped)";
  document.getElementById("ng-crop-panel").style.display = "none";
  log("NG ref cropped: " + iw + "\u00d7" + ih + "px");
}

function cancelNgCrop() {
  document.getElementById("ng-crop-panel").style.display = "none";
}

function clearNgRef() {
  ngRefB64 = null; _ngFullImg = null; _ngCropRect = null;
  const inp = document.getElementById("upload-ng-ref");
  if (inp) inp.value = "";
  document.getElementById("ng-ref-loaded").style.display = "none";
  document.getElementById("ng-crop-panel").style.display = "none";
}

function onUploadBatchFiles(input) {
  batchImagePool = [];
  const files = Array.from(input.files);
  let loaded = 0;
  files.forEach((file, idx) => {
    const reader = new FileReader();
    reader.onload = e => {
      batchImagePool[idx] = e.target.result.split(",")[1];
      loaded++;
      if (loaded === files.length) {
        document.getElementById("batch-files-count").textContent =
          `\u2705 ${files.length} images ready / ${files.length}\u679a\u6e96\u5099\u5b8c\u4e86`;
      }
    };
    reader.readAsDataURL(file);
  });
}

function openBatchModal() {
  const defectKey = document.getElementById("sel-defect").value;
  document.getElementById("batch-crack-opts").style.display = defectKey === "crack" ? "" : "none";
  document.getElementById("batch-progress-wrap").style.display = "none";
  document.getElementById("btn-batch-start").disabled = false;

  const srcInfo = document.getElementById("batch-src-info");
  if (batchImagePool.length > 0)
    srcInfo.textContent = `\u2705 ${batchImagePool.length} source images / ${batchImagePool.length}\u679a\u306e\u30bd\u30fc\u30b9\u753b\u50cf`;
  else if (okImageB64)
    srcInfo.textContent = "\u26a0 Single OK image will be reused. / 1\u679a\u306eOK\u753b\u50cf\u3092\u7e70\u308a\u8fd4\u3057\u4f7f\u7528\u3057\u307e\u3059\u3002";
  else
    srcInfo.textContent = "\u274c No OK image loaded. / OK\u54c1\u753b\u50cf\u304c\u3042\u308a\u307e\u305b\u3093\u3002";

  document.getElementById("batchModal").style.display = "flex";
}

function closeBatchModal() {
  document.getElementById("batchModal").style.display = "none";
}

function buildParams() {
  const defectKey = document.getElementById("sel-defect").value;
  const isAiChecked = document.getElementById("chk-use-ai").checked;

  const p = {
    intensity: parseInt(document.getElementById("slider-intensity")?.value || 70) / 100,
    seed: parseInt(document.getElementById("inp-seed")?.value) || 42,
    sdxl: isAiChecked,
    use_genai: isAiChecked,
    sdxl_refine: isAiChecked,
    refine_ai: isAiChecked,
    use_ai: isAiChecked,
  };
  if (isAiChecked) {
    const pp = (document.getElementById("txt-prompt")?.value || "").trim();
    const np = (document.getElementById("txt-neg-prompt")?.value || "").trim();
    if (pp) p.prompt = pp;
    if (np) p.negative_prompt = np;
  }
  if (defectKey === "crack") {
    p.break_type = document.getElementById("sel-break-type").value;
    p.depth = parseInt(document.getElementById("slider-depth").value) / 100;
    const angleRandom = document.getElementById("chk-angle-random").checked;
    p.angle = angleRandom
      ? Math.floor(Math.random() * 8) * 45
      : parseFloat(document.getElementById("slider-angle").value);
  }
  if (defectKey === "dent") {
    p.dent_strength = parseFloat(document.getElementById("slider-dent-strength").value);
    p.dent_size = parseFloat(document.getElementById("slider-dent-size").value);
  }
  if (defectKey === "hollow" || defectKey === "underfill") {
    const fixedEl = document.getElementById("chk-fixed-region");
    if (fixedEl) p.fixed_region = fixedEl.checked;
    p.refine_ai = isAiChecked;
  }
  if (ngRefB64) {
    p.ref_image_b64 = ngRefB64;
  }
  if (defectKey === "mc_deform" && currentGroup === "metal_cap") {
    let thetaIdx = document.getElementById("sel-theta").value;
    if (thetaIdx !== "random") {
      p.theta_center = parseFloat(thetaIdx);
    }
    const spanDeg = parseFloat(document.getElementById("slider-span").value);
    p.theta_span = (spanDeg * Math.PI) / 180.0;
    p.depth = parseFloat(document.getElementById("slider-deform").value);
    log(`MC Deform params: depth=${p.depth}, span_deg=${spanDeg}, intensity=${p.intensity}`);
  }
  if (defectKey === "ring_fracture" && currentGroup === "metal_cap") {
    const ampEl = document.getElementById("slider-amplitude");
    const falloffEl = document.getElementById("slider-falloff");
    if (ampEl) p.jitter_amplitude = parseFloat(ampEl.value);
    if (falloffEl) p.falloff_width = parseFloat(falloffEl.value);
  }
  if (currentGroup === "metal_cap" && (defectKey === "scratch" || defectKey === "ring_fracture")) {
    const offEl = document.getElementById("mc-rim-offset");
    if (offEl) p.rim_offset = parseInt(offEl.value) || 0;
  }
  if (defectKey === "scratch") {
    if (currentGroup === "metal_cap") {
      p.severity = document.getElementById("sel-severity").value;
      p.count = parseInt(document.getElementById("slider-count").value);
    } else {
      p.mode = document.getElementById("sel-mode").value;
    }
  }
  if (currentGroup === "cap" && defectKey === "plastic_flow") {
    p.patch_max_dim = parseInt(document.getElementById("slider-pf-maxdim").value);
    p.patch_scale = parseFloat(document.getElementById("slider-pf-scale").value);
    p.synth_positive_only = document.getElementById("chk-pf-shadowfree").checked;
  }
  if (defectKey === "dark_spots") {
    p.n_spots_min = 1;
    p.n_spots_max = 1;
    p.r_min = parseInt(document.getElementById("inp-r-min").value);
    p.r_max = parseInt(document.getElementById("inp-r-max").value);
    p.bump = true;
  }
  return p;
}

function downloadResult() {
  if (!resultB64) return;
  const defectKey = document.getElementById("sel-defect").value;
  const a = document.createElement("a");
  a.href = "data:image/png;base64," + resultB64;
  a.download = `${defectKey}_result.png`;
  a.click();
}

function showPanel(id, b64) {
  if (!b64) return;
  document.getElementById(id).src = b64.startsWith("data:") ? b64 : "data:image/png;base64," + b64;
}

function clearPanels(ids) {
  ids.forEach(id => { const el = document.getElementById(id); if (el) el.src = ""; });
}

function addHistory(b64) {
  const strip = document.getElementById("history-strip");
  const img = document.createElement("img");
  img.src = "data:image/png;base64," + b64;
  img.className = "rounded border border-secondary";
  img.style = "width:56px;height:56px;object-fit:contain;cursor:pointer;background:#111";
  img.onclick = () => { resultB64 = b64; showPanel("panel-result", b64); };
  strip.prepend(img);
  if (strip.children.length > 12) strip.removeChild(strip.lastChild);
}

function randomSeed() {
  const el = document.getElementById("inp-seed");
  if (el) el.value = Math.floor(Math.random() * 9999);
}

function onAngleRandomToggle() {
  const isRandom = document.getElementById("chk-angle-random").checked;
  const slider = document.getElementById("slider-angle");
  const label = document.getElementById("val-angle");
  slider.disabled = isRandom;
  label.textContent = isRandom ? "Random" : slider.value + "\u00b0";
}

function clearLog() {
  document.getElementById("log-box").textContent = "";
}

function log(msg) {
  const box = document.getElementById("log-box");
  const ts = new Date().toTimeString().slice(0, 8);
  box.textContent += `[${ts}] ${msg}\n`;
  box.scrollTop = box.scrollHeight;
}

function toggleFailReasons() {
  const wrap = document.getElementById("fail-reasons-wrap");
  wrap.style.display = wrap.style.display === "none" ? "block" : "none";
}
