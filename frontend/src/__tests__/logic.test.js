import { describe, it, expect, vi, beforeEach } from 'vitest';
import { normalizeResult, predict, ApiError } from '../api.js';
import { filterSortPage, makeRecord, saveStored, loadStored, newCaseId, PAGE_SIZE } from '../history.js';
import { buildReportHTML, buildReportJSON } from '../report.js';
import { parseHash } from '../router.js';
import { recommendationOf, stateOf } from '../evidence.js';

const PNG = 'data:image/png;base64,AAAA';
export const okBody = (over = {}) => ({
  status: 'success', slice_index: 7, num_slices: 40, slice_selection_mode: 'automatic', verdict: 'MODERATE RELIABILITY',
  reliability_score: 0.55, familiarity_score: 0.1, uncertainty_score: 0.0002, predictive_entropy: 0.03, max_probability_difference: 0.7,
  mask_pixels: 100, images: { mri: PNG, segmentation_overlay: PNG, uncertainty_heatmap: PNG }, disclaimer: 'd', ...over,
});

describe('normalizeResult', () => {
  it('accepts a valid body and preserves fields', () => {
    const r = normalizeResult(okBody());
    expect(r.verdict).toBe('MODERATE RELIABILITY');
    expect(r.images.mri).toBe(PNG);
    expect(r.reliability_evidence_layer).toBeNull();
  });
  it('nulls invalid images instead of crashing', () => {
    const r = normalizeResult(okBody({ images: { mri: 'http://evil/x.png', segmentation_overlay: 5 } }));
    expect(r.images).toEqual({ mri: null, segmentation_overlay: null, uncertainty_heatmap: null });
  });
  it('rejects non-success and malformed bodies', () => {
    for (const b of [null, 'x', {}, { status: 'error' }]) expect(() => normalizeResult(b)).toThrow(ApiError);
  });
});

// A fake XMLHttpRequest: this unit test exercises predict()'s own logic only (mock, not the real server).
class FakeXHR {
  static last = null;
  constructor() { this.upload = {}; FakeXHR.last = this; }
  open(m, u) { this.method = m; this.url = u; }
  send(form) { this.form = form; }
  abort() { this.onabort(); }
}
describe('predict (fake XHR)', () => {
  beforeEach(() => { vi.stubGlobal('XMLHttpRequest', FakeXHR); });
  const files = () => ({ flair: new File(['a'], 'f.nii'), t1ce: new File(['b'], 't.nii'), t2: new File(['c'], 't2.nii') });
  it('posts the three modalities and optional slice index', async () => {
    const p = predict({ files: files(), sliceIndex: 12, baseUrl: 'http://x' });
    const x = FakeXHR.last;
    expect(x.method).toBe('POST'); expect(x.url).toBe('http://x/predict');
    expect([...x.form.keys()].sort()).toEqual(['flair', 'slice_index', 't1ce', 't2'].sort());
    expect(x.form.get('slice_index')).toBe('12');
    x.status = 200; x.responseText = JSON.stringify(okBody()); x.onload();
    expect((await p).slice_index).toBe(7);
  });
  it('omits slice_index when automatic', async () => {
    const p = predict({ files: files(), baseUrl: 'http://x' });
    expect(FakeXHR.last.form.has('slice_index')).toBe(false);
    FakeXHR.last.status = 200; FakeXHR.last.responseText = JSON.stringify(okBody()); FakeXHR.last.onload(); await p;
  });
  it('maps server errors, unreadable bodies and network failures', async () => {
    let p = predict({ files: files(), baseUrl: 'http://x' });
    FakeXHR.last.status = 400; FakeXHR.last.responseText = JSON.stringify({ error: 'bad', error_code: 'SPATIAL_MISMATCH' }); FakeXHR.last.onload();
    await expect(p).rejects.toMatchObject({ message: 'bad', code: 'SPATIAL_MISMATCH', status: 400 });
    p = predict({ files: files(), baseUrl: 'http://x' });
    FakeXHR.last.status = 500; FakeXHR.last.responseText = '<html>'; FakeXHR.last.onload();
    await expect(p).rejects.toMatchObject({ code: 'HTTP_ERROR' });
    p = predict({ files: files(), baseUrl: 'http://x' });
    FakeXHR.last.status = 200; FakeXHR.last.responseText = 'not json'; FakeXHR.last.onload();
    await expect(p).rejects.toMatchObject({ code: 'BAD_RESPONSE' });
    p = predict({ files: files(), baseUrl: 'http://x' }); FakeXHR.last.onerror();
    await expect(p).rejects.toMatchObject({ code: 'NETWORK' });
  });
  it('supports cancellation and missing files', async () => {
    const ctrl = new AbortController();
    const p = predict({ files: files(), baseUrl: 'http://x', signal: ctrl.signal });
    ctrl.abort();
    await expect(p).rejects.toMatchObject({ code: 'ABORTED' });
    await expect(predict({ files: { flair: null }, baseUrl: 'http://x' })).rejects.toMatchObject({ code: 'MISSING_MODALITY' });
  });
});

describe('history', () => {
  const rec = (i, score = i / 10) => ({ ...makeRecord(`TG-${i}`, okBody({ reliability_score: score }), new Date(2026, 0, i + 1)) });
  it('paginates, filters and sorts', () => {
    const rows = Array.from({ length: 12 }, (_, i) => rec(i));
    const p1 = filterSortPage(rows, {});
    expect(p1.rows).toHaveLength(PAGE_SIZE); expect(p1.pages).toBe(3); expect(p1.rows[0].id).toBe('TG-11');
    expect(filterSortPage(rows, { page: 3 }).rows).toHaveLength(2);
    expect(filterSortPage(rows, { page: 99 }).page).toBe(3);
    expect(filterSortPage(rows, { query: 'tg-3' }).total).toBe(1);
    expect(filterSortPage(rows, { sort: 'oldest' }).rows[0].id).toBe('TG-0');
    expect(filterSortPage(rows, { sort: 'score_asc' }).rows[0].id).toBe('TG-0');
    expect(filterSortPage(rows, { sort: 'score_desc' }).rows[0].id).toBe('TG-11');
    expect(filterSortPage([], {}).pages).toBe(1);
  });
  it('stores scores only (no images) and restores them', () => {
    saveStored([rec(1)]);
    const raw = sessionStorage.getItem('tumorguard.session-history.v1');
    expect(raw).not.toContain('data:image'); expect(raw).not.toContain('images');
    const back = loadStored();
    expect(back[0].id).toBe('TG-1'); expect(back[0].result).toBeNull();
  });
  it('generates distinct, anonymous case ids', () => {
    const a = newCaseId(); const b = newCaseId();
    expect(a).toMatch(/^TG-[0-9A-F]{8}$/); expect(a).not.toBe(b);
  });
});

describe('reports', () => {
  it('JSON report omits images and file names', () => {
    const j = buildReportJSON(okBody(), 'TG-1');
    expect(j.result.images).toBeUndefined(); expect(JSON.stringify(j)).not.toMatch(/\.nii/);
  });
  it('HTML report escapes content, embeds images and flags unknown verdicts', () => {
    const html = buildReportHTML(okBody({ verdict: '<script>alert(1)</script>', disclaimer: '<b>x</b>' }), 'TG-<1>');
    expect(html).not.toContain('<script>alert'); expect(html).toContain('MANUAL REVIEW REQUIRED');
    expect(html).toContain('unrecognised'); expect(html).toContain('data:image/png;base64');
    expect(html).toContain('Not clinically validated');
  });
  it('HTML report copes with missing images and evidence', () => {
    const html = buildReportHTML(okBody({ images: { mri: null, segmentation_overlay: null, uncertainty_heatmap: null }, reliability_score: NaN }), 'TG-1');
    expect(html).toContain('image unavailable'); expect(html).toContain('n/a'); expect(html).toContain('Evidence layer unavailable');
  });
});

describe('router and evidence helpers', () => {
  it('parses hashes safely', () => {
    expect(parseHash('#/upload')).toBe('upload'); expect(parseHash('#/evidence?x=1')).toBe('evidence');
    expect(parseHash('#/__proto__')).toBe('home'); expect(parseHash('')).toBe('home');
  });
  it('unknown evidence values fall back to the conservative state', () => {
    expect(recommendationOf({ recommendation: 'SAFE' })).toBe('ASSESSMENT_UNAVAILABLE');
    expect(recommendationOf(null)).toBe('ASSESSMENT_UNAVAILABLE');
    expect(stateOf({ state: 'VALID' })).toBe('MISSING_OR_UNAVAILABLE');
    expect(recommendationOf({ recommendation: 'HUMAN_REVIEW_RECOMMENDED' })).toBe('HUMAN_REVIEW_RECOMMENDED');
  });
});
