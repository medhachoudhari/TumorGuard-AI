// Browser functional audit against the REAL Flask/PyTorch backend.
// Inputs are SYNTHETIC NIfTI volumes unless FILES_DIR points at real data; the audit checks
// software behaviour only, never segmentation quality.
//   BASE=http://127.0.0.1:4173 FILES_DIR=/path/with/flair.nii.gz,t1ce.nii.gz,t2.nii.gz,... \
//   OUT_DIR=/tmp/shots PLAYWRIGHT_PATH=playwright node e2e/functional_audit.mjs
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const BASE = process.env.BASE || 'http://127.0.0.1:4173';
const DIR = process.env.FILES_DIR;
const OUT = process.env.OUT_DIR || './e2e-out';
fs.mkdirSync(OUT, { recursive: true });
const F = (n) => path.join(DIR, n);

const results = [];
const consoleErrors = [];
const failedRequests = [];
let injectedFailure = false;

async function check(group, name, fn) {
  try { await fn(); results.push({ group, name, ok: true }); }
  catch (e) { results.push({ group, name, ok: false, error: String(e.message).split('\n')[0] }); }
}
const expect = (cond, msg) => { if (!cond) throw new Error(msg); };

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1360, height: 900 }, acceptDownloads: true });
const page = await ctx.newPage();
page.on('console', (m) => { if (m.type() === 'error' && !injectedFailure) consoleErrors.push(m.text()); });
page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));
page.on('requestfailed', (r) => { if (!injectedFailure) failedRequests.push(`${r.method()} ${r.url()} ${r.failure()?.errorText}`); });
const inventory = {};
async function inv(label) {
  inventory[label] = await page.evaluate(() => {
    const sel = 'button, a[href], input, select, textarea, summary, [role="tab"]';
    const els = [...document.querySelectorAll(sel)].filter((e) => { const r = e.getBoundingClientRect(); const cs = getComputedStyle(e); return cs.visibility !== 'hidden' && cs.display !== 'none' && (r.width > 0 || e.classList.contains('tg-visually-hidden')); });
    const name = (e) => (e.getAttribute('aria-label') || e.innerText || e.value || e.title || (e.id && document.querySelector(`label[for="${e.id}"]`)?.innerText) || '').trim();
    const unnamed = els.filter((e) => !name(e)).map((e) => e.outerHTML.slice(0, 80));
    const badLinks = els.filter((e) => e.tagName === 'A' && (!e.getAttribute('href') || e.getAttribute('href') === '#')).map((e) => e.outerHTML.slice(0, 80));
    return { count: els.length, unnamed, badLinks };
  });
}
const shot = (n) => page.screenshot({ path: path.join(OUT, `${n}.png`), fullPage: true });
const hash = () => new URL(page.url()).hash;

await page.goto(BASE);
await page.waitForSelector('text=Server online', { timeout: 15000 });

// ---------------- Landing and navigation ----------------
await check('Navigation', 'Landing renders heading', async () => { await page.getByRole('heading', { level: 1 }).first().waitFor(); await inv('landing'); await shot('01_landing'); });
for (const [label, route, heading] of [['Upload', 'upload', 'Upload brain MRI'], ['Results', 'results', 'No analysis yet'], ['History', 'history', 'Session history'], ['Research', 'research', 'Research and validation'], ['About', 'about', 'About TumorGuard AI'], ['Home', 'home', null]]) {
  await check('Navigation', `Nav link: ${label}`, async () => {
    await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: label, exact: true }).click();
    expect(hash() === `#/${route}`, `hash ${hash()}`);
    if (heading) await page.getByRole('heading', { name: heading }).first().waitFor({ timeout: 3000 });
  });
}
await check('Navigation', 'Logo link goes home', async () => { await page.getByRole('link', { name: 'TumorGuard AI home' }).click(); expect(hash() === '#/home', hash()); });
await check('Navigation', 'Nav "New analysis" opens upload', async () => { await page.getByRole('button', { name: 'New analysis' }).click(); expect(hash() === '#/upload', hash()); });
await page.goto(`${BASE}/#/home`);
await check('Landing', 'Get started opens upload', async () => { await page.getByRole('button', { name: 'Get started' }).click(); expect(hash() === '#/upload', hash()); });
await page.goto(`${BASE}/#/home`);
await check('Landing', 'Learn more scrolls to How it works', async () => {
  await page.getByRole('button', { name: 'Learn more' }).click();
  await page.waitForTimeout(900);
  const top = await page.getByRole('heading', { name: 'How it works' }).evaluate((el) => el.getBoundingClientRect().top);
  expect(top >= 0 && top < 400, `heading top=${top}`);
});

// ---------------- Upload and validation ----------------
await page.goto(`${BASE}/#/upload`);
const start = page.getByRole('button', { name: 'Start analysis' });
await check('Upload', 'Start disabled with reason before any file', async () => { expect(await start.isDisabled(), 'enabled'); await page.getByText('Add all three valid volumes').waitFor(); });
await check('Upload', 'Wrong file type rejected', async () => { await page.locator('#file-flair').setInputFiles(F('notes.txt')); await page.getByText('Unsupported file type').first().waitFor(); });
await check('Upload', 'Replace button reopens picker (file chooser)', async () => {
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.getByRole('button', { name: 'Replace' }).first().click()]);
  await chooser.setFiles(F('broken.nii.gz'));
  await page.getByText(/Not a NIfTI-1 file|could not be read/).first().waitFor();
});
await check('Upload', 'Remove FLAIR file', async () => { await page.getByRole('button', { name: 'Remove FLAIR file' }).click(); await page.getByRole('button', { name: 'Choose FLAIR file' }).waitFor(); });
await check('Upload', 'Choose file button opens real file chooser', async () => {
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.getByRole('button', { name: 'Choose FLAIR file' }).click()]);
  await chooser.setFiles(F('flair.nii.gz'));
  await page.getByText(/Valid NIfTI volume 96 × 96 × 40/).first().waitFor();
});
await check('Upload', 'Drag and drop accepts a file', async () => {
  const buf = fs.readFileSync(F('t1ce.nii.gz')).toString('base64');
  const dt = await page.evaluateHandle(async (b64) => {
    const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const d = new DataTransfer(); d.items.add(new File([bin], 't1ce.nii.gz')); return d;
  }, buf);
  await page.locator('.tg-slot').nth(1).dispatchEvent('drop', { dataTransfer: dt });
  await page.getByText('t1ce.nii.gz').waitFor();
});
await check('Upload', 'Spatial mismatch detected client-side and blocks start', async () => {
  await page.locator('#file-t2').setInputFiles(F('t2_mismatch.nii.gz'));
  await page.getByText(/Spatial mismatch/).first().waitFor();
  expect(await start.isDisabled(), 'start enabled despite mismatch');
});
await check('Upload', 'Valid trio enables Start', async () => {
  await page.locator('#file-t2').setInputFiles(F('t2.nii.gz'));
  await page.getByText(/All checks passed/).waitFor();
  expect(await start.isEnabled(), 'start disabled');
  await inv('upload (3 valid files)');
  await shot('02_upload_valid');
});
await check('Upload', 'Clear all empties the form', async () => {
  await page.getByRole('button', { name: 'Clear all' }).click();
  await page.getByRole('button', { name: 'Choose T2 file' }).waitFor();
  expect(await start.isDisabled(), 'start enabled');
});
for (const [k, n] of [['flair', 'flair'], ['t1ce', 't1ce'], ['t2', 't2']]) await page.locator(`#file-${k}`).setInputFiles(F(`${n}.nii.gz`));
await page.getByText(/All checks passed/).waitFor();

// ---------------- Analysis: injected network failure, retry, cancel ----------------
await check('Analysis', 'Cancel aborts a slow request and returns to upload (fault injection: delayed response)', async () => {
  await page.route('**/predict', async (r) => { await new Promise((x) => setTimeout(x, 6000)); await r.continue().catch(() => {}); }, { times: 1 });
  await start.click();
  await page.getByRole('button', { name: 'Cancel' }).click();
  await page.getByRole('heading', { name: 'Upload brain MRI' }).waitFor({ timeout: 5000 });
  expect((await page.getByRole('alert').count()) === 0, 'error shown after cancel');
});
await check('Analysis', 'Network failure shows error and Retry works', async () => {
  injectedFailure = true;
  await page.route('**/predict', (r) => r.abort('failed'), { times: 1 });
  await start.click();
  await page.getByText('Analysis failed.').waitFor({ timeout: 10000 });
  await shot('03_analysis_error');
  injectedFailure = false;
  await page.getByRole('button', { name: 'Retry' }).click();
  await page.waitForURL(/#\/results/, { timeout: 90000 });
});
await check('Results', 'Three real images decode at native 128x128', async () => {
  await page.getByRole('img', { name: /FLAIR MRI slice/ }).waitFor();
  const dims = await page.locator('.tg-panel img').evaluateAll((els) => els.map((e) => [e.naturalWidth, e.naturalHeight]));
  expect(dims.length === 3 && dims.every(([w, h]) => w === 128 && h === 128), JSON.stringify(dims));
  await inv('results'); await shot('04_results');
});
const sliceText = () => page.locator('.tg-page-head .tg-fine').innerText();
const curSlice = async () => Number((await sliceText()).match(/slice (\d+)/)[1]);
let s0;
await check('Results', 'Verdict is one of the canonical labels', async () => {
  const t = await page.locator('.tg-verdict-value').innerText();
  expect(['HIGH RELIABILITY', 'MODERATE RELIABILITY', 'LOW RELIABILITY', 'MANUAL REVIEW REQUIRED'].includes(t), t);
  s0 = await curSlice();
});
await check('Results', 'Next slice re-runs inference (+1)', async () => { await page.getByRole('button', { name: 'Next slice' }).click(); await page.waitForFunction((n) => /slice (\d+)/.exec(document.querySelector('.tg-page-head .tg-fine').innerText)[1] == n, String(s0 + 1), { timeout: 90000 }); });
await check('Results', 'Previous slice (-1)', async () => { await page.getByRole('button', { name: 'Previous slice' }).click(); await page.waitForFunction((n) => /slice (\d+)/.exec(document.querySelector('.tg-page-head .tg-fine').innerText)[1] == n, String(s0), { timeout: 90000 }); });
await check('Results', 'Go to slice 5 (explicit)', async () => {
  await page.locator('#slice-jump').fill('5'); await page.getByRole('button', { name: 'Go', exact: true }).click();
  await page.waitForFunction(() => /slice 5 /.test(document.querySelector('.tg-page-head .tg-fine').innerText) && /explicit/.test(document.querySelector('.tg-page-head .tg-fine').innerText), null, { timeout: 90000 });
});
await check('Results', 'Out-of-range slice cannot be submitted', async () => { await page.locator('#slice-jump').fill('999'); expect(await page.getByRole('button', { name: 'Go', exact: true }).isDisabled(), 'Go enabled for 999'); await page.locator('#slice-jump').fill(''); });
await check('Results', 'Previous disabled at slice 0', async () => {
  await page.locator('#slice-jump').fill('0'); await page.getByRole('button', { name: 'Go', exact: true }).click();
  await page.waitForFunction(() => /slice 0 /.test(document.querySelector('.tg-page-head .tg-fine').innerText), null, { timeout: 90000 });
  expect(await page.getByRole('button', { name: 'Previous slice' }).isDisabled(), 'previous enabled at 0');
});
await check('Results', 'Auto slice restores automatic selection', async () => { await page.getByRole('button', { name: 'Auto slice' }).click(); await page.waitForFunction(() => /automatic/.test(document.querySelector('.tg-page-head .tg-fine').innerText), null, { timeout: 90000 }); });
await check('Results', 'Zoom in / out / reset', async () => {
  const w = () => page.locator('.tg-panel img').first().evaluate((e) => e.style.width);
  await page.getByRole('button', { name: 'Zoom in' }).click(); expect((await w()) === '150%', await w());
  await page.getByRole('button', { name: 'Zoom out' }).click(); expect((await w()) === '100%', await w());
  await page.getByRole('button', { name: 'Zoom in' }).click(); await page.getByRole('button', { name: 'Reset zoom' }).click(); expect((await w()) === '100%', await w());
});
await check('Results', 'Download PNG links carry real image data', async () => {
  const links = await page.getByRole('link', { name: 'Download PNG' }).evaluateAll((as) => as.map((a) => [a.getAttribute('download'), a.href.slice(0, 22)]));
  expect(links.length === 3 && links.every(([d, h]) => d && h === 'data:image/png;base64,'), JSON.stringify(links));
});
await check('Results', 'Download report (HTML)', async () => {
  const [d] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download report (HTML)' }).click()]);
  const txt = fs.readFileSync(await d.path(), 'utf8'); expect(/Not a diagnosis/i.test(txt) && txt.includes('data:image/png;base64'), 'report content');
});
await check('Results', 'Download data (JSON) is valid and omits images', async () => {
  const [d] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download data (JSON)' }).click()]);
  const j = JSON.parse(fs.readFileSync(await d.path(), 'utf8')); expect(j.result.verdict && !j.result.images, 'json shape');
});
await check('Results', 'Score details expands', async () => { await page.getByText('Score details').click(); await page.getByRole('columnheader', { name: 'Contribution' }).waitFor(); });
await check('Results', 'View evidence layer opens evidence page', async () => { await page.getByRole('button', { name: 'View evidence layer' }).click(); expect(hash() === '#/evidence', hash()); });

// ---------------- Evidence ----------------
await check('Evidence', 'Recommendation and 6 signals shown', async () => {
  const rec = await page.getByTestId('recommendation').innerText();
  expect(['Human review recommended', 'No additional review flagged', 'Assessment unavailable'].includes(rec), rec);
  expect((await page.locator('.tg-signal').count()) === 6, 'signal count');
  await inv('evidence'); await shot('05_evidence');
});
await check('Evidence', 'Every signal "Details" expands', async () => {
  const sums = page.locator('.tg-signal summary'); const n = await sums.count();
  for (let i = 0; i < n; i++) { await sums.nth(i).click(); expect(await page.locator('.tg-signal details').nth(i).evaluate((d) => d.open), `signal ${i} not open`); }
});
await check('Evidence', 'Limitations / thresholds / provenance collapsibles', async () => {
  for (const t of ['Thresholds and score breakdown', 'Model and evaluation provenance']) { await page.getByText(t).click(); }
  await page.getByText(/SHA-256/).waitFor(); expect(/[0-9a-f]{64}/.test(await page.locator('.tg-wrap').first().innerText()), 'sha');
  expect(await page.getByText('Research limitations').isVisible(), 'limitations');
});
await check('Evidence', 'Back to results', async () => { await page.getByRole('button', { name: 'Back to results' }).click(); expect(hash() === '#/results', hash()); });

// ---------------- History ----------------
await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'History' }).click();
await check('History', 'Session record listed', async () => { await page.locator('tbody tr').first().waitFor(); expect((await page.locator('tbody tr').count()) === 1, 'rows'); await inv('history (1 record)'); await shot('06_history'); });
await check('History', 'Search filters (no match, then match)', async () => {
  await page.locator('#hist-q').fill('zzzz'); await page.getByText('No cases match your search.').waitFor();
  await page.locator('#hist-q').fill('tg-'); expect((await page.locator('tbody tr').count()) === 1, 'match'); await page.locator('#hist-q').fill('');
});
await check('History', 'Sort select works', async () => { await page.locator('#hist-sort').selectOption('score_desc'); expect((await page.locator('tbody tr').count()) === 1, 'rows'); });
await check('History', 'Pagination controls disabled on a single page', async () => { expect(await page.getByRole('button', { name: 'Previous page' }).isDisabled() && await page.getByRole('button', { name: 'Next page' }).isDisabled(), 'pager'); });
await check('History', 'View opens the case', async () => { await page.getByRole('button', { name: /View case/ }).click(); expect(hash() === '#/results', hash()); await page.getByRole('img', { name: /FLAIR MRI slice/ }).waitFor(); });
await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'History' }).click();
await check('History', 'Delete asks for confirmation; Cancel keeps record', async () => {
  await page.getByRole('button', { name: /Delete case/ }).click(); await page.getByRole('alertdialog').waitFor();
  await page.getByRole('button', { name: 'Cancel' }).click(); expect((await page.locator('tbody tr').count()) === 1, 'record removed on cancel');
});
await check('History', 'Escape closes the dialog', async () => { await page.getByRole('button', { name: /Delete case/ }).click(); await page.keyboard.press('Escape'); expect((await page.getByRole('alertdialog').count()) === 0, 'dialog open'); });
await check('History', 'Delete confirms and removes', async () => { await page.getByRole('button', { name: /Delete case/ }).click(); await page.getByRole('alertdialog').getByRole('button', { name: 'Delete' }).click(); await page.getByText('No analyses in this session').waitFor(); });

// ---------------- Research and About ----------------
await page.goto(`${BASE}/#/research`);
for (const [label, text] of [['Methodology', 'Methodology'], ['Limitations', 'Known limitations'], ['Citations', 'References'], ['Model performance', 'Historical evaluation']]) {
  await check('Research', `Tab: ${label}`, async () => { await page.getByRole('tab', { name: label }).click(); await page.getByRole('heading', { name: new RegExp(text) }).waitFor(); });
}
await check('Research', 'Arrow-key tab navigation', async () => { await page.getByRole('tab', { name: 'Model performance' }).focus(); await page.keyboard.press('ArrowRight'); expect((await page.getByRole('tab', { name: 'Methodology' }).getAttribute('aria-selected')) === 'true', 'tab'); });
await check('Research', 'Download metrics JSON matches artifact values', async () => {
  await page.getByRole('tab', { name: 'Model performance' }).click();
  const [d] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download metrics (JSON)' }).click()]);
  const j = JSON.parse(fs.readFileSync(await d.path(), 'utf8')); expect(j.metrics.validation_samples === 398, 'samples');
  await inv('research'); await shot('07_research');
});
await check('Research', 'Citation links are https, new-tab, noopener', async () => {
  await page.getByRole('tab', { name: 'Citations' }).click();
  const a = await page.locator('a.tg-link').evaluateAll((as) => as.map((x) => [x.href, x.target, x.rel]));
  expect(a.length === 4 && a.every(([h, t, r]) => h.startsWith('https://') && t === '_blank' && /noopener/.test(r)), JSON.stringify(a));
});
await page.goto(`${BASE}/#/about`);
await check('About', 'GitHub link, research and start buttons', async () => {
  expect((await page.getByRole('link', { name: 'GitHub' }).getAttribute('href')) === 'https://github.com/medhachoudhari/TumorGuard-AI', 'href');
  await page.getByRole('button', { name: 'Research and validation' }).click(); expect(hash() === '#/research', hash());
  await page.goto(`${BASE}/#/about`); await page.getByRole('button', { name: 'Start an analysis' }).click(); expect(hash() === '#/upload', hash());
  await page.goto(`${BASE}/#/about`); await inv('about'); await shot('08_about');
});

// ---------------- Backend-error surfaces ----------------
await page.goto(`${BASE}/#/upload`);
await check('Analysis', 'Direct visit to #/analysis without a run falls back to upload', async () => { await page.goto(`${BASE}/#/analysis`); await page.getByRole('heading', { name: 'Upload brain MRI' }).waitFor(); });

// ---------------- Mobile navigation ----------------
await page.setViewportSize({ width: 390, height: 800 });
await page.goto(`${BASE}/#/home`);
await check('Mobile', 'Menu opens, navigates and closes', async () => {
  const t = page.getByRole('button', { name: 'Open menu' }); await t.click();
  expect((await page.getByRole('button', { name: 'Close menu' }).getAttribute('aria-expanded')) === 'true', 'expanded');
  await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Research' }).click();
  expect(hash() === '#/research', hash());
  expect((await page.getByRole('button', { name: 'Open menu' }).getAttribute('aria-expanded')) === 'false', 'menu stayed open');
});
await check('Mobile', 'No horizontal page overflow', async () => {
  for (const r of ['home', 'upload', 'research', 'about', 'history']) { await page.goto(`${BASE}/#/${r}`); const o = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth); expect(o <= 1, `${r} overflows by ${o}px`); }
  await shot('09_mobile_home');
});

await browser.close();
const bad = results.filter((r) => !r.ok);
const byGroup = {};
for (const r of results) { byGroup[r.group] ??= { pass: 0, fail: 0 }; byGroup[r.group][r.ok ? 'pass' : 'fail']++; }
const unnamed = Object.entries(inventory).filter(([, v]) => v.unnamed.length || v.badLinks.length);
if (unnamed.length) bad.push({ group: 'Inventory', name: 'controls without accessible names or valid hrefs', error: JSON.stringify(unnamed) });
const summary = { inventory, total: results.length, passed: results.length - bad.length, failed: bad.length, byGroup, failures: bad, consoleErrors, failedRequests };
fs.writeFileSync(path.join(OUT, 'audit.json'), JSON.stringify({ results, ...summary }, null, 2));
console.log(JSON.stringify(summary, null, 2));
process.exit(bad.length ? 1 : 0);
