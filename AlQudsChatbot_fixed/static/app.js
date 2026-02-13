// ============ THEME ============
const themeBtn = document.getElementById('themeToggle');
const THEME_KEY = 'aq_theme';

function setTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  localStorage.setItem(THEME_KEY, mode);
  if (themeBtn) themeBtn.textContent = mode === 'dark' ? '🌞' : '🌓';
}

setTheme(localStorage.getItem(THEME_KEY) || 'light');

if (themeBtn) {
  themeBtn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    setTheme(current === 'dark' ? 'light' : 'dark');
  });
}

// ============ CHAT ELEMENTS ============
const chat    = document.getElementById('chat');
const form    = document.getElementById('askForm');
const ta      = document.getElementById('q');
const sendBtn = document.getElementById('sendBtn');
const chips   = document.getElementById('quickChips');

// ============ HELPERS ============
function autoresize() {
  if (!ta) return;
  ta.style.height = 'auto';
  ta.style.height = Math.min(200, ta.scrollHeight) + 'px';
}

function scrollBottom() {
  if (!chat) return;
  chat.scrollTo({ top: chat.scrollHeight, behavior: 'smooth' });
}

function renderSources(sources) {
  if (!Array.isArray(sources) || sources.length === 0) return null;

  const wrap = document.createElement('div');
  wrap.className = 'sources';

  const label = document.createElement('span');
  label.textContent = 'Sources: ';
  wrap.appendChild(label);

  sources.forEach((s, i) => {
    const a = document.createElement('a');
    a.href = s.href;
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = s.label;
    wrap.appendChild(a);

    if (i !== sources.length - 1) {
      const sep = document.createElement('span');
      sep.textContent = ' • ';
      wrap.appendChild(sep);
    }
  });

  return wrap;
}

function addRow(role, text, sources = []) {
  if (!chat) return;

  const row = document.createElement('div');
  row.className = `row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === 'user' ? 'U' : 'AQ';

  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;

  // add sources only for bot
  if (role === 'bot') {
    const src = renderSources(sources);
    if (src) {
      bubble.appendChild(document.createElement('br'));
      bubble.appendChild(src);
    }
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chat.appendChild(row);
  scrollBottom();
}

autoresize();
scrollBottom();

// ============ CORE: sendQuestion ============
async function sendQuestion(q) {
  const text = (q || '').trim();
  if (!text) return;

  addRow('user', text);

  if (ta) {
    ta.value = '';
    autoresize();
  }
  if (sendBtn) sendBtn.disabled = true;

  // typing indicator
  const typing = document.createElement('div');
  typing.className = 'row bot';
  typing.innerHTML = `<div class="avatar bot">AQ</div><div class="bubble bot">…</div>`;
  chat.appendChild(typing);
  scrollBottom();

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text }),
    });

    if (!res.ok) throw new Error('HTTP ' + res.status);

    const data = await res.json();
    console.log('API response:', data);

    typing.remove();

    const botText =
      data.answer ||
      data.reply ||
      data.message ||
      (typeof data === 'string' ? data : null) ||
      'No reply from server.';

    addRow('bot', botText, data.sources || []);
  } catch (err) {
    console.error('sendQuestion error:', err);
    typing.remove();
    addRow('bot', '⚠️ Error reaching server. Trying again using classic submit…');

    if (form && ta) {
      ta.value = text;
      form.submit();
    }
  } finally {
    if (sendBtn) sendBtn.disabled = false;
  }
}

window.sendQuestion = sendQuestion;

// ============ FORM SUBMIT ============
if (form) {
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    sendQuestion(ta ? ta.value : '');
  });
}

// ============ QUICK CHIPS ============
if (chips) {
  chips.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-q]');
    if (!btn) return;
    const q = btn.getAttribute('data-q') || '';
    sendQuestion(q);
  });
}
