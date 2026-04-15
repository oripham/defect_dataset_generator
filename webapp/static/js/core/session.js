// static/js/core/session.js

// --- UUID Session Isolation per Tab ---
if (!sessionStorage.getItem('sessionId')) {
    sessionStorage.setItem('sessionId', Math.random().toString(16).slice(2, 10) + Math.random().toString(16).slice(2, 10));
}
window.SESSION_ID = sessionStorage.getItem('sessionId');

// Append ?sid= to sidebar links to persist session across page loads
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.sidebar-link').forEach(link => {
        try {
            const url = new URL(link.href, window.location.origin);
            url.searchParams.set('sid', window.SESSION_ID);
            link.href = url.pathname + url.search;
        } catch (e) {}
    });
});

// Auto-attach X-Session-ID to every fetch
const _originalFetch = window.fetch;
window.fetch = function(url, options = {}) {
    options.headers = options.headers || {};
    options.headers['X-Session-ID'] = window.SESSION_ID;
    return _originalFetch(url, options);
};

// --- Language Switcher (Global util) ---
window.setLanguage = function(lang) {
    fetch('/api/set-language', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({lang: lang})
    }).then(() => location.reload());
};
