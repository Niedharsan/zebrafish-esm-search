const form = document.querySelector('#searchForm');
const queryInput = document.querySelector('#query');
const kSelect = document.querySelector('#k');
const message = document.querySelector('#message');
const tableWrap = document.querySelector('#tableWrap');
const resultsBody = document.querySelector('#resultsBody');
const resultCount = document.querySelector('#resultCount');
const matchPanel = document.querySelector('#matchPanel');
const suggestions = document.querySelector('#suggestions');

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function showMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle('error', isError);
  message.classList.remove('hidden');
  tableWrap.classList.add('hidden');
  resultCount.textContent = '';
}

function renderMatch(data) {
  const p = data.matched_protein;
  matchPanel.innerHTML = `
    <div class="match-grid">
      <div>
        <div class="kicker">Matched query protein</div>
        <strong>${escapeHtml(p.name || p.protein_id)}</strong>
        <p class="mono">${escapeHtml(p.protein_id)}</p>
      </div>
      <div>
        <div class="kicker">Match method</div>
        <strong>${escapeHtml(data.match_method)}</strong>
        <p>score ${escapeHtml(data.match_score)}</p>
      </div>
      <div>
        <div class="kicker">Sequence length</div>
        <strong>${p.sequence_length ? escapeHtml(p.sequence_length) + ' aa' : 'not stored'}</strong>
      </div>
    </div>
    ${p.description ? `<p class="desc">${escapeHtml(p.description)}</p>` : ''}
  `;
  matchPanel.classList.remove('hidden');
}

function renderResults(results) {
  resultsBody.innerHTML = results.map(r => `
    <tr>
      <td>${r.rank}</td>
      <td class="sim">${r.similarity.toFixed(5)}</td>
      <td class="mono">${escapeHtml(r.protein_id)}</td>
      <td>${escapeHtml(r.name || '—')}</td>
      <td class="desc">${escapeHtml(r.description || '—')}</td>
      <td>${r.sequence_length ? escapeHtml(r.sequence_length) : '—'}</td>
    </tr>
  `).join('');
  message.classList.add('hidden');
  tableWrap.classList.remove('hidden');
  resultCount.textContent = `${results.length} results`;
}

async function runSearch() {
  const q = queryInput.value.trim();
  const k = kSelect.value;
  if (!q) {
    showMessage('Enter a protein name to begin.');
    matchPanel.classList.add('hidden');
    return;
  }
  showMessage('Searching your local ESM database...');
  form.querySelector('button').disabled = true;
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&k=${encodeURIComponent(k)}`);
    const data = await res.json();
    if (!data.ok) {
      matchPanel.classList.add('hidden');
      showMessage(data.message || 'No match found.', true);
      return;
    }
    renderMatch(data);
    renderResults(data.results);
  } catch (err) {
    showMessage(`Search failed: ${err.message}`, true);
  } finally {
    form.querySelector('button').disabled = false;
  }
}

let suggestTimer = null;
queryInput.addEventListener('input', () => {
  clearTimeout(suggestTimer);
  const q = queryInput.value.trim();
  suggestions.innerHTML = '';
  if (q.length < 2) return;
  suggestTimer = setTimeout(async () => {
    try {
      const res = await fetch(`/api/suggest?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      suggestions.innerHTML = data.map(p => `
        <button class="suggestion" type="button" data-value="${escapeHtml(p.name || p.protein_id)}">
          ${escapeHtml(p.name || p.protein_id)} <span class="mono">${escapeHtml(p.protein_id)}</span>
        </button>
      `).join('');
    } catch (_) {
      suggestions.innerHTML = '';
    }
  }, 160);
});

suggestions.addEventListener('click', (event) => {
  const btn = event.target.closest('.suggestion');
  if (!btn) return;
  queryInput.value = btn.dataset.value;
  suggestions.innerHTML = '';
  runSearch();
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  suggestions.innerHTML = '';
  runSearch();
});
