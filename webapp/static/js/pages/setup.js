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


// ---------- Clear Session ----------
async function clearSession() {
  if(!confirm(t('Clear all session data and local uploads?'))) return;
  const resp = await fetch('/api/clear-session', {method: 'POST'});
  const data = await resp.json();
  if(data.ok) {
    location.reload();
  }
}

// Init
window.addEventListener('load', () => {
  // Connection already initialized by server-side templates or session
});
