// Session-level case history. Nothing is persisted beyond this browser tab, and
// no file names or images are stored.
const KEY = 'tumorguard.session-history.v1';
export const PAGE_SIZE = 5;

export function newCaseId() {
  const bytes = new Uint8Array(4);
  (globalThis.crypto || { getRandomValues: (a) => a.fill(Math.floor(Math.random() * 255)) }).getRandomValues(bytes);
  return `TG-${Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('').toUpperCase()}`;
}

export function makeRecord(caseId, result, now = new Date()) {
  return {
    id: caseId,
    createdAt: now.toISOString(),
    sliceIndex: result.slice_index,
    numSlices: result.num_slices ?? null,
    verdict: result.verdict,
    reliabilityScore: result.reliability_score,
    recommendation: result.reliability_evidence_layer ? result.reliability_evidence_layer.recommendation : null,
    result, // full result incl. images; kept in memory only
  };
}

export function loadStored() {
  try {
    const raw = JSON.parse(sessionStorage.getItem(KEY) || '[]');
    return Array.isArray(raw) ? raw.map((r) => ({ ...r, result: null })) : [];
  } catch { return []; }
}

export function saveStored(records) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(records.map(({ result: _r, ...rest }) => rest)));
  } catch { /* storage unavailable: history stays in memory only */ }
}

export function filterSortPage(records, { query = '', sort = 'newest', page = 1, pageSize = PAGE_SIZE } = {}) {
  const q = query.trim().toLowerCase();
  let rows = records.filter((r) => !q || r.id.toLowerCase().includes(q) || String(r.verdict).toLowerCase().includes(q));
  const by = {
    newest: (a, b) => b.createdAt.localeCompare(a.createdAt),
    oldest: (a, b) => a.createdAt.localeCompare(b.createdAt),
    score_desc: (a, b) => (b.reliabilityScore ?? -1) - (a.reliabilityScore ?? -1),
    score_asc: (a, b) => (a.reliabilityScore ?? 2) - (b.reliabilityScore ?? 2),
  }[sort] || ((a, b) => b.createdAt.localeCompare(a.createdAt));
  rows = [...rows].sort(by);
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  const p = Math.min(Math.max(1, page), pages);
  return { rows: rows.slice((p - 1) * pageSize, p * pageSize), total: rows.length, page: p, pages };
}
