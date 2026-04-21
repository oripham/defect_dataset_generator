// static/js/pages/studio/studio_api.js

async function checkGPUHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    if (!data.gpu_available) {
      console.warn("GPU health check failed or no GPU available, SDXL may fail.");
    }
  } catch (e) {
    console.warn("GPU health check failed", e);
  }
}

async function runAutoMask() {
  if (!okImageB64) { alert("Upload an OK image first. / OK品画像をアップロードしてください。"); return; }
  document.getElementById("mask-status").textContent = "Detecting\u2026 / \u691c\u51fa\u4e2d\u2026";
  try {
    const resp = await fetch("/api/pharma/auto-mask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_b64: okImageB64 }),
    });
    const data = await resp.json();
    if (data.error) { log("Mask error: " + data.error); return; }
    maskB64 = data.mask_b64;
    showPanel("panel-mask", maskB64);
    document.getElementById("mask-status").textContent =
      `\u2705 area=${data.mask_area}px  bbox=${JSON.stringify(data.bbox)}`;
    log(`Auto-mask: area=${data.mask_area}`);
  } catch (e) { log("Auto-mask error: " + e); }
}

async function runPreview() {
  if (!okImageB64) { alert("Upload an OK image first. / OK\u54c1\u753b\u50cf\u3092\u30a2\u30c3\u30d7\u30ed\u30fc\u30c9\u3057\u3066\u304f\u3060\u3055\u3044\u3002"); return; }
  const btnEl = document.getElementById("btn-preview");
  btnEl.disabled = true;
  btnEl.textContent = "\u23f3 Generating\u2026 / \u751f\u6210\u4e2d\u2026";
  log("Running preview\u2026");

  const grp = GROUPS[currentGroup];
  const defectKey = document.getElementById("sel-defect").value;
  console.log("[JS] runPreview started", { group: currentGroup, defectKey });
  const entry = grp.defects.find(d => d.key === defectKey) || {};
  const api = grp.api;
  const product = entry.product || grp.product || currentGroup;
  const params = buildParams();

  try {
    const resp = await fetch(`/api/${api}/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_b64: okImageB64,
        mask_b64: maskB64 || null,
        product,
        defect_type: defectKey,
        params,
        ref_image_b64: ngRefB64 || null,
      }),
    });
    console.log("[JS] Fetch response status:", resp.status);
    const data = await resp.json();
    if (data.error) { 
      log("Error: " + data.error); 
      alert("Server Error: " + data.error);
      return; 
    }

    resultB64 = data.result_image;
    maskB64 = data.mask_b64 || maskB64;

    showPanel("panel-result", data.result_image);
    if (maskB64) showPanel("panel-mask", maskB64);
    if (data.debug_panel) showPanel("panel-diff", data.debug_panel);

    log(`Done \u2014 engine=${data.engine || api}  QC=${data.qc?.verdict || "\u2014"}`);
    addHistory(data.result_image);
    const bSave = document.getElementById("btn-save");
    if (bSave) bSave.disabled = false;
    const bDown = document.getElementById("btn-download");
    if (bDown) bDown.disabled = false;
    const bPrev = document.getElementById("btn-preview");
    if (bPrev) bPrev.disabled = false;

    // Reset and show evaluation panel
    const evPl = document.getElementById("eval-placeholder");
    if (evPl) evPl.style.display = "none";
    const evPa = document.getElementById("eval-panel");
    if (evPa) evPa.style.display = "flex";
    const evSt = document.getElementById("eval-status");
    if (evSt) evSt.textContent = "";
    const frW = document.getElementById("fail-reasons-wrap");
    if (frW) frW.style.display = "none";
    document.getElementById("btn-eval-success").disabled = false;
    document.getElementById("btn-eval-failed").disabled = false;
    document.querySelectorAll(".eval-reason-chk").forEach(el => el.checked = false);
    document.getElementById("fail-reason-other").value = "";
  } catch (e) {
    log("Preview error: " + e);
  } finally {
    btnEl.disabled = false;
    btnEl.textContent = "\u25b6 Preview / \u30d7\u30ec\u30d3\u30e5\u30fc";
  }
}

async function saveResult() {
  if (!resultB64) return;
  const grp = GROUPS[currentGroup];
  const defectKey = document.getElementById("sel-defect").value;
  const entry = grp.defects.find(d => d.key === defectKey) || {};
  const api = grp.api;
  const product = entry.product || grp.product || currentGroup;
  try {
    const resp = await fetch(`/api/${api}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        result_b64: resultB64,
        mask_b64: maskB64,
        product,
        defect_type: defectKey,
        params: buildParams(),
      }),
    });
    const d = await resp.json();
    log("Saved \u2192 " + (d.path || "?"));
  } catch (e) { log("Save error: " + e); }
}

async function startBatch() {
  if (!okImageB64 && batchImagePool.length === 0) {
    alert("Upload at least one OK image first. / OK\u54c1\u753b\u50cf\u3092\u30a2\u30c3\u30d7\u30ed\u30fc\u30c9\u3057\u3066\u304f\u3060\u3055\u3044\u3002");
    return;
  }
  const grp = GROUPS[currentGroup];
  const defectKey = document.getElementById("sel-defect").value;
  const entry = grp.defects.find(d => d.key === defectKey) || {};
  const api = grp.api;
  const product = entry.product || grp.product || currentGroup;
  const nImages = parseInt(document.getElementById("batch-n").value) || 20;
  const params = buildParams();
  const breakTypes = [];
  document.querySelectorAll("#batch-crack-opts input[type=checkbox]:checked")
    .forEach(cb => breakTypes.push(cb.value));

  const imagePool = batchImagePool.length > 0 ? batchImagePool : [okImageB64];

  document.getElementById("batch-progress-wrap").style.display = "";
  document.getElementById("btn-batch-start").disabled = true;
  log(`Batch start: ${nImages} images, source pool=${imagePool.length}`);

  const resp = await fetch(`/api/${api}/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      product,
      defect_type: defectKey,
      n_images: nImages,
      params,
      break_types: breakTypes,
      image_b64: imagePool[0],
      images_b64: imagePool,
    }),
  });
  const d = await resp.json();
  if (d.error) { log("Batch error: " + d.error); return; }
  batchJobId = d.job_id;
  log(`Batch queued \u2014 job=${batchJobId}`);
  batchTimer = setInterval(() => pollBatch(api), 1500);
}

async function pollBatch(api) {
  if (!batchJobId) return;
  const resp = await fetch(`/api/${api}/batch/${batchJobId}`);
  const d = await resp.json();
  document.getElementById("batch-bar").style.width = (d.progress || 0) + "%";
  document.getElementById("batch-status-text").textContent =
    `${d.generated || 0} / ${d.total || 0} \u2014 ${d.status}`;
  if (d.status === "done" || d.status === "error") {
    clearInterval(batchTimer);
    document.getElementById("btn-batch-start").disabled = false;
    log(`Batch done \u2192 ${d.out_dir || d.error || ""}`);
  }
}

async function evaluateResult(status) {
  if (!resultB64) {
    alert("No result to evaluate. / \u8a50\u4fa1\u3059\u308b\u7d50\u679c\u304c\u3042\u308a\u307e\u305b\u3093\u3002");
    return;
  }

  let reasons = [];
  if (status === 'failed') {
    document.querySelectorAll(".eval-reason-chk:checked").forEach(chk => {
      reasons.push(chk.value);
    });
    const otherReason = document.getElementById("fail-reason-other").value.trim();
    if (otherReason) reasons.push("other: " + otherReason);

    if (reasons.length === 0) {
      alert("Please select at least one reason for failure. / \u5931\u6557\u306e\u7406\u7531\u3092\u5c11\u306a\u304f\u3068\u30821\u3064\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044\u3002");
      return;
    }
  }

  const btnSuccess = document.getElementById("btn-eval-success");
  const btnFailed = document.getElementById("btn-eval-failed");
  const btnSubmit = document.getElementById("btn-submit-fail");
  btnSuccess.disabled = true;
  btnFailed.disabled = true;
  if (btnSubmit) btnSubmit.disabled = true;

  const statusEl = document.getElementById("eval-status");
  if (statusEl) statusEl.innerHTML = "Saving... \u23f3";

  try {
    const defectKey = document.getElementById("sel-defect").value;
    const grp = GROUPS[currentGroup];
    const product = grp.product || currentGroup;

    const payload = {
      base_image: okImageB64,
      result_image: resultB64,
      mask_image: maskB64,
      ref_image_b64: ngRefB64,
      defect_type: defectKey,
      product: product,
      params: buildParams(),
      status: status,
      reasons: reasons
    };

    const resp = await fetch("/api/eval/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) throw new Error("Server error " + resp.status);
    const data = await resp.json();

    document.getElementById("eval-status").innerHTML = `<span class="text-success">\u2705 Saved to database</span>`;
    if (status === 'failed') {
      document.getElementById("fail-reasons-wrap").style.display = "none";
    }
  } catch (e) {
    log("Evaluation error: " + e);
    document.getElementById("eval-status").innerHTML = `<span class="text-danger">\u274c Failed to save</span>`;
    btnSuccess.disabled = false;
    btnFailed.disabled = false;
    if (btnSubmit) btnSubmit.disabled = false;
  }
}
