const form = document.querySelector('#searchForm');
const queryInput = document.querySelector('#query');
const queryLabel = document.querySelector('#queryLabel');
const kSelect = document.querySelector('#k');
const message = document.querySelector('#message');
const tableWrap = document.querySelector('#tableWrap');
const resultsBody = document.querySelector('#resultsBody');
const resultCount = document.querySelector('#resultCount');
const matchPanel = document.querySelector('#matchPanel');
const discoveryPanel = document.querySelector('#discoveryPanel');
const suggestions = document.querySelector('#suggestions');
const resultsTitle = document.querySelector('#resultsTitle');
const seedHeader = document.querySelector('#seedHeader');
const modeHelp = document.querySelector('#modeHelp');
const searchButton = document.querySelector('#searchButton');
const aiStatus = document.querySelector('#aiStatus');
const modeTabs = [...document.querySelectorAll('.mode-tab')];

let currentMode = 'protein';

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

function setMode(mode) {
  currentMode = mode;
  modeTabs.forEach(tab => tab.classList.toggle('active', tab.dataset.mode === mode));
  suggestions.innerHTML = '';
  matchPanel.classList.add('hidden');
  discoveryPanel.classList.add('hidden');
  tableWrap.classList.add('hidden');
  seedHeader.classList.add('hidden');
  resultsBody.innerHTML = '';
  resultCount.textContent = '';

  if (mode === 'protein') {
    queryLabel.textContent = 'Protein / gene name / ID';
    queryInput.placeholder = 'Example: gata1a, mpx, Q7SXE0...';
    modeHelp.textContent = 'Find a protein and view its closest matches in ESM embedding space.';
    resultsTitle.textContent = 'ESM-similar proteins';
    searchButton.textContent = 'Search';
    showMessage('Enter a protein name to begin.');
  } else {
    queryLabel.textContent = 'Biological question';
    queryInput.placeholder = 'Example: proteins involved in erythropoiesis';
    modeHelp.textContent = 'Gemini uses UniProt and Ensembl APIs to identify zebrafish proteins, then ESM searches the whole-proteome embedding space.';
    resultsTitle.textContent = 'ESM-similar proteins';
    searchButton.textContent = 'Discover';
    showMessage('Ask a biological question to begin.');
  }
}

function renderMatch(data) {
  const p = data.matched_protein;
  matchPanel.innerHTML = `
    <div class="match-grid">
      <div>
        <div class="kicker">Matched protein</div>
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

function renderDiscovery(data) {
  const plan = data.plan || {};
  const terms = (plan.retrieval_terms || []).map(term => `<span class="chip">${escapeHtml(term)}</span>`).join('');
  const seeds = (data.seeds || []).map(seed => `
    <div class="seed-card">
      <strong>${escapeHtml(seed.name || seed.protein_id)}</strong>
      <span class="mono">${escapeHtml(seed.protein_id)}</span>
      <small>${escapeHtml(seed.description || 'No description')}</small>
      <em>${escapeHtml(seed.source)} · ${escapeHtml(seed.resolved_by)}</em>
    </div>
  `).join('');

  discoveryPanel.innerHTML = `
    <div class="discovery-grid">
      <div>
        <div class="kicker">Biological question</div>
        <strong>${escapeHtml(plan.normalized_question || data.query)}</strong>
        <p>${escapeHtml(plan.rationale || '')}</p>
        <div class="chips">${terms}</div>
      </div>
      <div>
        <div class="kicker">AI-selected zebrafish proteins</div>
        <strong>Gemini + biological APIs</strong>
        <p>${(data.seeds || []).length} proteins used as starting points for the ESM search</p>
      </div>
    </div>
    <div class="seed-list">${seeds}</div>
    ${data.ai_explanation ? `<div class="ai-explanation"><div class="kicker">AI interpretation</div><p>${escapeHtml(data.ai_explanation)}</p></div>` : ''}
    ${data.retrieval_warning ? `<p class="warning">UniProt lookup warning: ${escapeHtml(data.retrieval_warning)}</p>` : ''}
    <p class="privacy-note">${escapeHtml(data.privacy || '')}</p>
  `;
  discoveryPanel.classList.remove('hidden');
}

function renderResults(results, mode) {
  const discovery = mode === 'discovery';
  seedHeader.classList.toggle('hidden', !discovery);
  resultsBody.innerHTML = results.map(r => `
    <tr>
      <td>${r.rank}</td>
      <td class="sim">${Number(r.similarity).toFixed(5)}</td>
      <td class="mono">${escapeHtml(r.protein_id)}</td>
      <td>${escapeHtml(r.name || '—')}</td>
      <td class="desc">${escapeHtml(r.description || '—')}</td>
      <td>${r.sequence_length ? escapeHtml(r.sequence_length) : '—'}</td>
      ${discovery ? `<td>${escapeHtml(r.closest_seed || '—')}</td>` : ''}
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
    showMessage(currentMode === 'protein' ? 'Enter a protein name to begin.' : 'Enter a biological question.', true);
    return;
  }

  matchPanel.classList.add('hidden');
  discoveryPanel.classList.add('hidden');
  const status = currentMode === 'protein'
    ? 'Searching the local ESM embedding database...'
    : 'Using Gemini and biological APIs to identify zebrafish proteins, then searching the ESM embedding space...';
  showMessage(status);
  searchButton.disabled = true;

  try {
    const endpoint = currentMode === 'protein' ? '/api/search' : '/api/discover';
    const res = await fetch(`${endpoint}?q=${encodeURIComponent(q)}&k=${encodeURIComponent(k)}`);
    const data = await res.json();
    if (!data.ok) {
      if (data.plan && currentMode === 'discovery') {
        discoveryPanel.innerHTML = `
          <div class="kicker">AI search details</div>
          <strong>${escapeHtml(data.plan.normalized_question || q)}</strong>
          <p>${escapeHtml(data.plan.rationale || '')}</p>
        `;
        discoveryPanel.classList.remove('hidden');
      }
      showMessage(data.message || 'No match found.', true);
      return;
    }

    if (currentMode === 'protein') renderMatch(data);
    else renderDiscovery(data);
    renderResults(data.results, currentMode);
  } catch (err) {
    showMessage(`Search failed: ${err.message}`, true);
  } finally {
    searchButton.disabled = false;
  }
}

let suggestTimer = null;
queryInput.addEventListener('input', () => {
  clearTimeout(suggestTimer);
  suggestions.innerHTML = '';
  if (currentMode !== 'protein') return;
  const q = queryInput.value.trim();
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

suggestions.addEventListener('click', event => {
  const btn = event.target.closest('.suggestion');
  if (!btn) return;
  queryInput.value = btn.dataset.value;
  suggestions.innerHTML = '';
  runSearch();
});

modeTabs.forEach(tab => tab.addEventListener('click', () => setMode(tab.dataset.mode)));

form.addEventListener('submit', event => {
  event.preventDefault();
  suggestions.innerHTML = '';
  runSearch();
});

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.ai_available) {
      aiStatus.innerHTML = `<span class="mini-dot good"></span> AI-assisted search enabled · ${escapeHtml(data.ai_model)}`;
    } else {
      aiStatus.innerHTML = '<span class="mini-dot"></span> AI not configured · ESM protein similarity still available';
    }
  } catch (_) {
    aiStatus.textContent = 'AI status unavailable';
  }
}

loadStatus();
