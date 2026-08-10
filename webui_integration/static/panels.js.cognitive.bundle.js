// ── Cognitive memory (hermes-cognitive-memory plugin) ────────────────────────

let _cognitiveData = null;
let _cognitiveBusy = false;
let _cognitiveQuery = '';
let _cognitiveFilter = 'all';
let _cognitiveAddOpen = false;
let _cognitiveAddDraft = '';

async function _loadCognitiveData(force) {
  try {
    _cognitiveData = await api('/api/memory/cognitive', {cache:'no-store', timeoutMs:15000});
  } catch (e) {
    _cognitiveData = {available:false, reason:(e && e.message) ? e.message : String(e)};
  }
  _renderCognitiveMemoryDetail();
}

function _renderCognitiveMemoryDetail() {
  const title = $('memoryDetailTitle');
  const body = $('memoryDetailBody');
  const empty = $('memoryDetailEmpty');
  if (!title || !body) return;
  title.textContent = 'Cognitive Memory';
  const data = _cognitiveData;
  if (!data) {
    body.innerHTML = '<div class="main-view-content"><div class="memory-empty">Loading…</div></div>';
    body.style.display = '';
    if (empty) empty.style.display = 'none';
    _memoryMode = 'read';
    _setMemoryHeaderButtons('read');
    return;
  }
  if (data.available === false) {
    body.innerHTML = `<div class="main-view-content"><div class="memory-empty">${esc(data.reason || 'Cognitive memory store unavailable.')}</div></div>`;
    body.style.display = '';
    if (empty) empty.style.display = 'none';
    _memoryMode = 'read';
    _setMemoryHeaderButtons('read');
    return;
  }
  const stats = data.stats || {};
  const memories = Array.isArray(data.memories) ? data.memories : [];
  const chip = (label, value) => `<span class="cognitive-chip"><strong>${esc(value)}</strong> ${esc(label)}</span>`;
  const chips = [
    chip('total', stats.total || 0),
    chip('pinned', stats.pinned || 0),
    chip('hard to find', stats.hard_to_find || 0),
    chip('prunable', stats.prunable || 0),
    chip('superseded', stats.superseded || 0),
  ].join('');
  const filtered = memories.filter(m => {
    if (_cognitiveFilter === 'pinned' && !m.pinned) return false;
    if (_cognitiveFilter === 'research' && m.origin !== 'research_finding') return false;
    if (_cognitiveFilter === 'hard' && !m.hard_to_find) return false;
    if (_cognitiveFilter === 'timeless' && m.temporal !== 'timeless') return false;
    if (_cognitiveFilter === 'ephemeral' && m.temporal !== 'ephemeral') return false;
    if (_cognitiveQuery) {
      const q = _cognitiveQuery.toLowerCase();
      if (!(m.content || '').toLowerCase().includes(q)) return false;
    }
    return true;
  });
  const listHtml = filtered.length
    ? filtered.map(m => _cognitiveCardHtml(m)).join('')
    : `<div class="memory-empty">${memories.length ? 'No memories match the current filter.' : 'No cognitive memories yet. New memory writes will appear here.'}</div>`;
  const pruneLog = (Array.isArray(data.prune_log) && data.prune_log.length)
    ? `<details class="cognitive-prune-log"><summary>Prune log (${data.prune_log.length} recent entries)</summary><pre>${esc(data.prune_log.join('\n'))}</pre></details>`
    : '';
  body.innerHTML = `<div class="main-view-content">
    <div class="cognitive-controls">
      <input id="cognitiveSearch" type="search" placeholder="Filter memories…" value="${esc(_cognitiveQuery)}" oninput="cognitiveSetQuery(this.value)" />
      <select id="cognitiveFilter" onchange="cognitiveSetFilter(this.value)" aria-label="Filter">
        <option value="all" ${_cognitiveFilter==='all'?'selected':''}>All</option>
        <option value="pinned" ${_cognitiveFilter==='pinned'?'selected':''}>Pinned</option>
        <option value="research" ${_cognitiveFilter==='research'?'selected':''}>Research findings</option>
        <option value="hard" ${_cognitiveFilter==='hard'?'selected':''}>Hard to find</option>
        <option value="timeless" ${_cognitiveFilter==='timeless'?'selected':''}>Timeless</option>
        <option value="ephemeral" ${_cognitiveFilter==='ephemeral'?'selected':''}>Ephemeral</option>
      </select>
      <button type="button" class="btn-secondary" onclick="_loadCognitiveData(true)">Refresh</button>
      <button type="button" class="btn-secondary" onclick="cognitiveToggleAdd()">${_cognitiveAddOpen ? 'Close add' : '+ Add memory'}</button>
    </div>
    <div class="cognitive-stats">${chips}</div>
    <div id="cognitiveAddForm" style="${_cognitiveAddOpen ? '' : 'display:none'}"></div>
    ${listHtml}
    ${pruneLog}
  </div>`;
  if (_cognitiveAddOpen) _renderCognitiveAddForm();
  body.style.display = '';
  if (empty) empty.style.display = 'none';
  _memoryMode = 'read';
  _setMemoryHeaderButtons('read');
}

function _cognitiveCardHtml(m) {
  const id = esc(m.id || '');
  const pinned = m.pinned ? '<span class="cognitive-badge cognitive-badge-pinned">PINNED</span>' : '';
  const htf = m.hard_to_find ? '<span class="cognitive-badge cognitive-badge-hard">HARD TO FIND</span>' : '';
  const eff = (typeof m.effective_importance === 'number') ? m.effective_importance : (m.importance || 0);
  const pct = Math.max(0, Math.min(100, Math.round(eff * 100)));
  const content = (m.content || '').length > 600 ? esc((m.content || '').slice(0, 600)) + '…' : esc(m.content || '');
  return `<section class="cognitive-card${m.pinned ? ' cognitive-card-pinned' : ''}">
    <div class="cognitive-card-head">
      <span class="detail-badge">${esc(m.origin || 'unknown')}</span>
      ${pinned}${htf}
      <span class="cognitive-meta">${esc(m.temporal || 'stable')} · rel ${esc(m.reliability)} · ${esc(m.access_count)} accesses · ${esc(_cognitiveAge(m.last_access))} ago</span>
      <span class="cognitive-actions">
        <button type="button" class="btn-secondary cognitive-btn" onclick="cognitiveAction('${m.pinned ? 'unpin' : 'pin'}','${id}')">${m.pinned ? 'Unpin' : 'Pin'}</button>
        <button type="button" class="btn-secondary cognitive-btn cognitive-btn-danger" onclick="cognitiveAction('delete','${id}')">Delete</button>
      </span>
    </div>
    <div class="cognitive-bar"><span style="width:${pct}%"></span></div>
    <div class="memory-content cognitive-content">${content}</div>
  </section>`;
}

function _cognitiveAge(ts) {
  if (!ts) return 'never';
  const s = Math.max(0, (Date.now() / 1000) - Number(ts));
  if (s < 60) return Math.floor(s) + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm';
  if (s < 86400) return Math.floor(s / 3600) + 'h';
  return Math.floor(s / 86400) + 'd';
}

async function cognitiveAction(action, id) {
  if (_cognitiveBusy) return;
  _cognitiveBusy = true;
  try {
    const res = await api('/api/memory/cognitive', {method:'POST', body:JSON.stringify({action: action, id: id})});
    if (res && res.ok) {
      await _loadCognitiveData(true);
    } else {
      alert((res && res.error) ? res.error : 'Action failed');
    }
  } catch (e) {
    alert((e && e.message) ? e.message : String(e));
  } finally {
    _cognitiveBusy = false;
  }
}

function cognitiveSetQuery(q) {
  _cognitiveQuery = q;
  _renderCognitiveMemoryDetail();
}

function cognitiveSetFilter(f) {
  _cognitiveFilter = f;
  _renderCognitiveMemoryDetail();
}

function cognitiveToggleAdd() {
  _cognitiveAddOpen = !_cognitiveAddOpen;
  if (!_cognitiveAddOpen) _cognitiveAddDraft = '';
  _renderCognitiveMemoryDetail();
}

function cognitiveAddContentTyped(v) {
  _cognitiveAddDraft = v;
}

function _renderCognitiveAddForm() {
  const form = $('cognitiveAddForm');
  if (!form) return;
  form.innerHTML = `<form class="detail-form cognitive-add-form" onsubmit="event.preventDefault(); cognitiveSubmitAdd();">
    <div class="detail-form-row">
      <label for="cognitiveAddContent">Content</label>
      <textarea id="cognitiveAddContent" rows="3" spellcheck="false" placeholder="Memory content…" oninput="cognitiveAddContentTyped(this.value)">${esc(_cognitiveAddDraft)}</textarea>
    </div>
    <div class="cognitive-add-row">
      <label>Target <select id="cognitiveAddTarget"><option value="memory">memory</option><option value="user">user</option></select></label>
      <label>Origin <select id="cognitiveAddOrigin">
        <option value="unknown">unknown</option>
        <option value="user_correction">user_correction</option>
        <option value="user_preference">user_preference</option>
        <option value="research_finding">research_finding</option>
        <option value="environment_fact">environment_fact</option>
        <option value="agent_inference">agent_inference</option>
      </select></label>
      <label>Temporal <select id="cognitiveAddTemporal">
        <option value="stable">stable</option>
        <option value="timeless">timeless</option>
        <option value="ephemeral">ephemeral</option>
      </select></label>
      <label>Reliability <input id="cognitiveAddReliability" type="number" min="0" max="1" step="0.05" value="1.0" style="width:70px"></label>
    </div>
    <div class="cognitive-add-row">
      <label><input id="cognitiveAddPinned" type="checkbox"> Pinned (never pruned)</label>
      <label><input id="cognitiveAddHard" type="checkbox"> Hard to find</label>
    </div>
    <div id="cognitiveAddError" class="detail-form-error" style="display:none"></div>
    <button type="submit" class="btn-secondary">Save memory</button>
  </form>`;
}

async function cognitiveSubmitAdd() {
  const contentEl = $('cognitiveAddContent');
  const content = (contentEl ? contentEl.value : '').trim();
  if (!content) {
    _showCognitiveAddError('Content is required');
    return;
  }
  const val = id => { const el = $(id); return el ? el.value : ''; };
  const chk = id => { const el = $(id); return el ? el.checked : false; };
  const payload = {
    action: 'add',
    content: content,
    target: val('cognitiveAddTarget') || 'memory',
    origin: val('cognitiveAddOrigin') || 'unknown',
    temporal: val('cognitiveAddTemporal') || 'stable',
    reliability: parseFloat(val('cognitiveAddReliability')) || 1,
    pinned: chk('cognitiveAddPinned'),
    hard_to_find: chk('cognitiveAddHard'),
  };
  if (_cognitiveBusy) return;
  _cognitiveBusy = true;
  try {
    const res = await api('/api/memory/cognitive', {method:'POST', body:JSON.stringify(payload)});
    if (res && res.ok) {
      _cognitiveAddOpen = false;
      _cognitiveAddDraft = '';
      _cognitiveQuery = '';
      await _loadCognitiveData(true);
    } else {
      _showCognitiveAddError((res && res.error) ? res.error : 'Save failed');
    }
  } catch (e) {
    _showCognitiveAddError((e && e.message) ? e.message : String(e));
  } finally {
    _cognitiveBusy = false;
  }
}

function _showCognitiveAddError(msg) {
  const el = $('cognitiveAddError');
  if (el) {
    el.textContent = msg;
    el.style.display = '';
  }
}
