// Builds downloadable reports from the actual API result. No file names or
// patient identifiers are included.
import { isKnownVerdict, verdictLabel } from './verdict.js';

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const num = (x, d = 4) => (Number.isFinite(x) ? Number(x).toFixed(d) : 'n/a');

export function buildReportJSON(result, caseId, now = new Date()) {
  const { images: _images, ...rest } = result;
  return {
    report_type: 'TumorGuard AI research analysis report',
    generated_utc: now.toISOString(),
    case_id: caseId,
    note: 'Research prototype output. Not clinically validated; not a diagnosis. Images are omitted from this JSON.',
    result: rest,
  };
}

export function buildReportHTML(result, caseId, now = new Date()) {
  const ev = result.reliability_evidence_layer;
  const img = (uri, alt) => (uri ? `<figure><img alt="${esc(alt)}" src="${uri}" style="width:256px;image-rendering:pixelated"><figcaption>${esc(alt)}</figcaption></figure>` : `<figure><p>${esc(alt)}: image unavailable</p></figure>`);
  const verdict = verdictLabel(result.verdict);
  const rows = [
    ['Slice index', `${result.slice_index}${Number.isFinite(result.num_slices) ? ` of ${result.num_slices}` : ''} (${esc(result.slice_selection_mode || 'unknown')} selection)`],
    ['Reliability score (heuristic)', num(result.reliability_score)],
    ['Familiarity score', num(result.familiarity_score)],
    ['Uncertainty (MC variance)', num(result.uncertainty_score, 6)],
    ['Predictive entropy', num(result.predictive_entropy)],
    ['Max pass divergence', num(result.max_probability_difference)],
  ];
  const evHtml = ev ? `<h2>Reliability Evidence Layer</h2>
<p><strong>${esc(ev.recommendation)}</strong> (workflow recommendation, not a clinical statement)</p>
<ul>${(ev.reasons || []).map((r) => `<li>${esc(r.text)}</li>`).join('')}</ul>
<table>${(ev.signals || []).map((s) => `<tr><td>${esc(s.label)}</td><td>${esc(s.display ?? 'n/a')}</td><td>${esc(s.state)}</td><td>${esc(s.origin)}</td></tr>`).join('')}</table>
<h3>Limitations</h3><ul>${(ev.limitations || []).map((l) => `<li>${esc(l)}</li>`).join('')}</ul>` : '<p>Evidence layer unavailable for this result.</p>';
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>TumorGuard AI report ${esc(caseId)}</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#111}td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}table{border-collapse:collapse;margin:1rem 0}figure{display:inline-block;margin:0 1rem 1rem 0}</style></head><body>
<h1>TumorGuard AI &mdash; research analysis report</h1>
<p>Case ${esc(caseId)} &middot; generated ${esc(now.toISOString())}</p>
<p><strong>Research prototype. Not clinically validated. Not a diagnosis.</strong></p>
<h2>Reliability verdict: ${esc(verdict)}</h2>
${isKnownVerdict(result.verdict) ? '' : '<p>The verdict was missing or unrecognised and is shown as manual review.</p>'}
<table>${rows.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${v}</td></tr>`).join('')}</table>
${img(result.images?.mri, 'FLAIR MRI slice')}${img(result.images?.segmentation_overlay, 'U-Net segmentation overlay')}${img(result.images?.uncertainty_heatmap, 'MC Dropout uncertainty heatmap')}
${evHtml}
<p>${esc(result.disclaimer || 'This is a research prototype. It does not diagnose cancer.')}</p></body></html>`;
}

export function downloadText(filename, text, mime) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
