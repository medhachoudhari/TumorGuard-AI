# Functionality audit (UI controls)

Scope: every visible interactive element of the redesigned frontend. Evidence comes from three sources, kept separate:

1. **Real-browser audit** (`frontend/e2e/functional_audit.mjs`, Chromium via Playwright) against the **real Flask/PyTorch backend**, using **synthetic NIfTI volumes** (no BraTS data was available). It verifies software behaviour, not segmentation quality.
2. **Component and flow tests** (Vitest + Testing Library, jsdom): 54 tests. `app.test.jsx` uses a **mocked API module** and `logic.test.js` a fake XHR; they are not evidence about the real server.
3. **Backend tests** (pytest): API behaviour with the real model on synthetic volumes.

## Real-browser results (59 checks: 59 passed, 0 failed)

| Group | Checks | Passed | Failed |
|---|---|---|---|
| Navigation | 9 | 9 | 0 |
| Landing | 2 | 2 | 0 |
| Upload | 9 | 9 | 0 |
| Analysis | 3 | 3 | 0 |
| Results | 14 | 14 | 0 |
| Evidence | 4 | 4 | 0 |
| History | 8 | 8 | 0 |
| Research | 7 | 7 | 0 |
| About | 1 | 1 | 0 |
| Mobile | 2 | 2 | 0 |

Console errors during the run: 0. Failed network requests: 1 (the single aborted `/predict` is the deliberate user-cancel test).
Fault injection (clearly artificial): one aborted request to test the error + Retry path, one delayed request to test Cancel.

## Interactive-element inventory (visible controls per page, including the shared navigation)

| Page | Controls counted | Without accessible name | Links without a real href |
|---|---|---|---|
| landing | 11 | 0 | 0 |
| upload (3 valid files) | 20 | 0 | 0 |
| results | 25 | 0 | 0 |
| evidence | 19 | 0 | 0 |
| history (1 record) | 16 | 0 | 0 |
| research | 14 | 0 | 0 |
| about | 12 | 0 | 0 |

Counted: `button, a[href], input, select, textarea, summary, [role=tab]`. All have an accessible name; no placeholder links. Controls disabled by design show their reason (upload Start, slice controls, zoom limits).

## Checks performed

| Group | Check | Result |
|---|---|---|
| Navigation | Landing renders heading | PASS |
| Navigation | Nav link: Upload | PASS |
| Navigation | Nav link: Results | PASS |
| Navigation | Nav link: History | PASS |
| Navigation | Nav link: Research | PASS |
| Navigation | Nav link: About | PASS |
| Navigation | Nav link: Home | PASS |
| Navigation | Logo link goes home | PASS |
| Navigation | Nav "New analysis" opens upload | PASS |
| Landing | Get started opens upload | PASS |
| Landing | Learn more scrolls to How it works | PASS |
| Upload | Start disabled with reason before any file | PASS |
| Upload | Wrong file type rejected | PASS |
| Upload | Replace button reopens picker (file chooser) | PASS |
| Upload | Remove FLAIR file | PASS |
| Upload | Choose file button opens real file chooser | PASS |
| Upload | Drag and drop accepts a file | PASS |
| Upload | Spatial mismatch detected client-side and blocks start | PASS |
| Upload | Valid trio enables Start | PASS |
| Upload | Clear all empties the form | PASS |
| Analysis | Cancel aborts a slow request and returns to upload (fault injection: delayed response) | PASS |
| Analysis | Network failure shows error and Retry works | PASS |
| Results | Three real images decode at native 128x128 | PASS |
| Results | Verdict is one of the canonical labels | PASS |
| Results | Next slice re-runs inference (+1) | PASS |
| Results | Previous slice (-1) | PASS |
| Results | Go to slice 5 (explicit) | PASS |
| Results | Out-of-range slice cannot be submitted | PASS |
| Results | Previous disabled at slice 0 | PASS |
| Results | Auto slice restores automatic selection | PASS |
| Results | Zoom in / out / reset | PASS |
| Results | Download PNG links carry real image data | PASS |
| Results | Download report (HTML) | PASS |
| Results | Download data (JSON) is valid and omits images | PASS |
| Results | Score details expands | PASS |
| Results | View evidence layer opens evidence page | PASS |
| Evidence | Recommendation and 6 signals shown | PASS |
| Evidence | Every signal "Details" expands | PASS |
| Evidence | Limitations / thresholds / provenance collapsibles | PASS |
| Evidence | Back to results | PASS |
| History | Session record listed | PASS |
| History | Search filters (no match, then match) | PASS |
| History | Sort select works | PASS |
| History | Pagination controls disabled on a single page | PASS |
| History | View opens the case | PASS |
| History | Delete asks for confirmation; Cancel keeps record | PASS |
| History | Escape closes the dialog | PASS |
| History | Delete confirms and removes | PASS |
| Research | Tab: Methodology | PASS |
| Research | Tab: Limitations | PASS |
| Research | Tab: Citations | PASS |
| Research | Tab: Model performance | PASS |
| Research | Arrow-key tab navigation | PASS |
| Research | Download metrics JSON matches artifact values | PASS |
| Research | Citation links are https, new-tab, noopener | PASS |
| About | GitHub link, research and start buttons | PASS |
| Analysis | Direct visit to #/analysis without a run falls back to upload | PASS |
| Mobile | Menu opens, navigates and closes | PASS |
| Mobile | No horizontal page overflow | PASS |

## Controls intentionally not implemented (removed rather than faked)

* **Compare slices / Compare cases:** the design shows them; they were not implemented (no stored multi-case image data) and no control is shown.
* **Persistent history, server-side records, history of patient names:** not implemented. History is session-only (sessionStorage, scores only, no images or file names) and says so on the page.
* **Landing-page and dashboard numbers from the concept image** (e.g. AUROC 0.842, 369/125/125 patient splits, "~45 s" estimates, per-stage progress percentages): not used; they are not verified. Only metrics from `models/phase9_evaluation.json` are shown, labelled historical.
* **Accounts / profile icon / "Download Report" PDF:** not implemented. Reports are HTML and JSON generated from the actual result.
* **Progress percentages:** only the real browser upload progress is shown; the server pipeline is shown as one running stage because the server does not report stages.

## Not verified

Real BraTS data in the browser; Firefox and Safari; screen readers; true OS-level drag-and-drop (a synthetic drop event was used); very large (>100 MB) uploads in the browser; browsers without `DecompressionStream` (the header check then falls back to server validation, covered by a unit test only); touch gestures.
