// static/js/pages/studio/studio_canvas.js

function _mcResetCanvas() {
  _mcBaseImg = null; _mcMaskOff = null; _mcDrawing = false;
  const panel = document.getElementById("mc-mask-panel");
  if (panel) panel.classList.add("d-none");
  document.getElementById("mc-mask-hint").textContent = "Paint the defect region, then click Set / 欠陥領域を塗り、確定をクリック";
  const offEl = document.getElementById("mc-rim-offset");
  const offLbl = document.getElementById("mc-rim-offset-lbl");
  if (offEl) offEl.value = 0;
  if (offLbl) offLbl.textContent = "0";
}
