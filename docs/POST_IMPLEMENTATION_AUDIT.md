# Post-implementation verification and demo-readiness audit

Audited: branch `release-fix/ui-evidence-layer`, 2026-10-20, Linux container (4 CPU, no GPU), Python 3.13.16, torch 2.14.1 (CPU), Node 22.22.0.
Method: read-only inspection plus test runs. No tracked file was modified, nothing was committed or pushed. Two new **untracked** files were created: this document and `tests/test_mc_variation_and_contract.py`.

**Evidence classes used below** (never mixed): `UNIT` = pytest without a server; `API-SYN` = pytest Flask test client with real model on **synthetic** volumes; `MOCK-UI` = Vitest with mocked network; `BROWSER-SYN` = Chromium against the real backend with **synthetic** volumes; `REAL` = genuine BraTS data. **No REAL evidence was obtained in this audit.**

## Overall status

**READY WITH CLEARLY DOCUMENTED LIMITATIONS**, conditional on three things: (1) demo from `release-fix/ui-evidence-layer`, **not** `main` (see F-1); (2) one real-case dry run (section 4) before any public demo, because the stripe and all-manual-review symptoms are still unconfirmed; (3) single-user, one request at a time, until the concurrency defect F-3 is fixed. It is **not ready** for any statement about segmentation quality on real MRI.

## Task 1: repository verification

| Check | Evidence | Result |
|---|---|---|
| Branch / tree | `release-fix/ui-evidence-layer`; `git status -sb` clean (only ignored `frontend/dist/`) | OK |
| Local vs remote HEAD | local `6e1f973ebef0eac42f184d389c792c2c711d80ac`; `git ls-remote` shows the same hash | OK |
| Reported commit exists | `git cat-file -t 6e1f973…` = `commit` | OK |
| Phase 2B ancestor | `git merge-base --is-ancestor 42e6933… HEAD` = true | OK |
| Other branches | `integration/phase-2a-reconciled` = `42e6933`; `fix/phase-2a-independent` = `b4c5928`; unchanged by this work | OK |
| Artifacts | `models/tumorguard_model.pt` sha256 `eb50ced634beb064e10d44d8586f555a0207a2b1915157819137c3b268c636fe`, `phase9_evaluation.json` `acecc7f3…`, `training_stats.json` `fa13f42e…`; `git diff ceca4f1 HEAD -- <each>` empty | Preserved |
| Original backend/frontend | Flask app, `tumorguard_core.py`, evaluation and calibration scripts present; frontend was replaced by the redesign (intended) | OK |

### F-1 (critical, not caused by this work): `main` no longer contains the real application

`origin/main` is `048d668` "feat: initialize project structure", authored by the repository owner on 2026-10-20 15:17 +0530, parent `ceca4f1`. It touches 44 paths: **27 deleted** (Flask server, `tumorguard_core.py`, evaluation scripts, the React app, **`models/tumorguard_model.pt`**), 12 added, 4 renamed, 1 modified (README). The deleted files are **not lost**: they remain in `ceca4f1` and every later commit, including this release branch. The commit adds a TypeScript project (`src/App.tsx`, `src/tumorguard/engine.ts`, `samples.ts`, `nifti.ts`, `trainingStats.ts`): the browser-only AI Studio prototype whose engine simulates MC Dropout. Anyone cloning `main` now gets simulated predictions and no real model. This work did not touch `main`. The release branch shares only `ceca4f1` with it, so a merge would conflict (modify/delete). **Decision needed from the owner**; nothing was changed.

## Task 2: real inference contract

Verified by reading `server.py` and by `API-SYN` test `test_api_outputs_equal_direct_pipeline_with_same_seed`:

* Checkpoint: `server.py:52` `load_model_checkpoint(model, CKPT_PATH)`, strict, `weights_only=True`; the served checkpoint SHA-256 is returned in `reliability_evidence_layer.provenance`.
* Preprocessing: `load_and_preprocess_mri(flair, t1ce, t2, slice_idx)` (unchanged; modality order FLAIR, T1CE, T2).
* Mask: `pred_binary = (mean_pred > 0.5)` from the model's MC passes. **Proof:** with the same seed, the same file and the same slice, the API's `mask_pixels`, `uncertainty_score`, `predictive_entropy`, `max_probability_difference`, `familiarity_score`, `reliability_score` and `verdict` equal a direct call of the core functions (test passes). Synthetic or heuristic substitution would break this equality.
* MC uncertainty: from 10 stochastic passes of the real model (`run_mc_dropout_inference`).
* Phase 2B fields present and unchanged: `slice_selection_mode` (`explicit` | `automatic_flair_heuristic`), `prediction_method` (`mc_dropout_consensus_10_passes`), `mc_prediction`, `deterministic_prediction`, `status: "error"` bodies, 150 MB limit, `/health` fields.
* Evidence layer: pure function of already-computed floats; test `test_evidence_layer_does_not_change_inputs` shows its arguments are not mutated, and the response's `legacy_verdict` / `legacy_reliability_score` equal the top-level fields. It never feeds back into the mask, score or verdict.
* Frontend: the only hard-coded numbers are the **labelled historical** baseline copy (`frontend/src/data/historical_baseline.json`, checked equal to `models/phase9_evaluation.json` by a test) and literature citations. A scan found no `Math.random`, simulated progress, or placeholder metrics in live views.

**Contract finding F-2 (UX regression risk):** if the familiarity reference is missing, the legacy score uses a neutral 0.5 (D-16, unchanged by design) and can still reach HIGH (0.8 with zero uncertainty). The evidence layer then says `ASSESSMENT_UNAVAILABLE`, but the Results page still shows the legacy verdict in the green HIGH style with the evidence badge only in the Signals card. The shipped `training_stats.json` contains the reference, so this path is latent. A frontend banner next to the verdict when the recommendation is `ASSESSMENT_UNAVAILABLE` would close it without touching research semantics.

## Task 3: verdict non-determinism

**Conclusion:** the run-to-run variation is **expected from MC Dropout** (unseeded stochastic passes), amplified by the D-15 statistic; it is **not** an uncontrolled-randomness bug in preprocessing. A separate, real defect (F-3) can make variation worse under concurrent requests.

Evidence (`UNIT`/`API-SYN`, `tests/test_mc_variation_and_contract.py`, 8 tests passing):

* Dropout is enabled only on `nn.Dropout` modules; all other leaf modules (BatchNorm, Conv, ReLU) stay in eval mode during MC passes.
* After MC inference no module is left in train mode and weights and BatchNorm buffers are bit-identical to before (Phase 2B's `model.eval()` reset works).
* The deterministic pass is bitwise repeatable and unaffected by a prior MC run; preprocessing is bitwise deterministic. So **dropout is the only randomness source**.
* The MC result is reproducible with `torch.manual_seed` and differs with another seed.
* The API separates the two predictions: `deterministic_prediction` (single eval pass, `>=`) and `mc_prediction` (MC mean, `>`); the returned mask and overlay are the MC-mean mask.

Recorded variation (200 unseeded repeats, same synthetic input, 10 passes; command: `TUMORGUARD_VARIATION_OUT=out.json TUMORGUARD_VARIATION_RUNS=200 pytest tests/test_mc_variation_and_contract.py`):

| Quantity | Min | Max |
|---|---|---|
| Reliability score | 0.344 | 0.575 (mean 0.435, std 0.045) |
| Verdict counts | LOW 182 (91%) | MODERATE 18 (9%) |
| `max_probability_difference` | 0.672 | 0.979 |
| MC variance | 0.00039 | 0.00064 |
| Mask pixels | 291 | 314 |

The score range crosses the 0.50 cut-off, so the verdict flips. Back-of-envelope from the table: the single first-vs-last pass difference alone swings the score by about 0.18 (its confidence term goes from about 0.94 to 0.06 at weight 0.20), the variance term by about 0.08. That makes D-15 the main noise source. Legitimately, repeated runs on the same case can give different verdicts whenever the mean score is within roughly 0.05 of a cut-off.

**Limitations:** one synthetic input, not MRI; the model's behaviour and the spread on real cases are unknown; the decomposition is arithmetic from ranges, not a measured attribution.

**UI explanation gap (F-4):** the Results page explains that the score is a heuristic but never says that it is stochastic and can change between runs. No change to thresholds, formulas or seeding is proposed; add one sentence and, as a research decision, consider reporting a spread over repeated runs.

### F-3 (high): shared mutable model state is not thread-safe

`server.py` uses one global model, and `enable_dropout()` / `model.eval()` mutate its module modes. `app.run()` is threaded by default. Measured (`UNIT`, two threads, one running MC, one running deterministic passes): **MC uncertainty collapsed** (max 7.7e-7 versus 6.8e-6 to 1.1e-5 sequentially, and 2 of 15 MC runs had **zero variance**), and **12 of 161 "deterministic" passes were not equal to the reference**. Overlapping requests happen in the UI: Cancel only aborts the browser request, the server keeps computing; a user can then re-submit or re-slice. Silent under-estimation of uncertainty biases verdicts toward "more reliable". Smallest fix (not applied): a process-wide lock around the inference section of `predict()` (or `threaded=False`), plus a test with two concurrent requests.

## Task 4: real BraTS verification

**Blocked, not substituted.** A filesystem search found no BraTS data (only synthetic files created by tests). Phase 2B's real-data tests (`tests/test_real_data_e2e.py`, `tests/test_spatial_validation.py`) were **skipped** (12 skips in total, listed under "Tests"). No real-case results (dimensions, spacing, affine, slice, verdict, signals, stripes) can be reported. Needed: the three files of one case, for example `BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_001/BraTS20_Validation_001_{flair,t1ce,t2}.nii` (the path the Phase 2B tests expect), run on your machine (section 4 of the demo list below).

### F-5 (high): baseline reproduction is tied to a different checkpoint hash than the committed one

`docs/RECONCILIATION.md` (Phase 2B, from your machine) reports a reproduction of Mean Dice 0.7863 with "Verified Checkpoint `models/tumorguard_model.pt` (SHA-256 `785d1e2e69736c929dc8870183e20e8b2ea1322fa01ffb0faef4c29ea338db71`)". The checkpoint committed in this repository hashes to `eb50ced6…c636fe`. Either the Windows file differs from the committed one, or one hash was mis-transcribed. Until `Get-FileHash` on both files is compared, **the reported reproduction is not demonstrated for the checkpoint this repository (and the UI provenance panel) names.** That report is a claim from another environment; it was not verified here. The same file also says a second checkpoint `tumorguard_dev.pth` was tested; it is not in the repository (test skipped).

## Task 5: security and functionality

| Area | Finding | Evidence class |
|---|---|---|
| CORS | `CORS(app)` allows every origin. Server binds `127.0.0.1:5000`; any website opened in the same browser can call `/health` and `/predict` and read results. Impact is low (no stored data, local model) but it should be narrowed. **Narrowest safe option for the actual deployment (local Flask + Vite preview/dev):** `CORS(app, resources={r"/(health\|predict)": {"origins": [<exact origins>]}}, methods=["GET","POST"])` with origins `http://127.0.0.1:4173`, `http://localhost:4173`, `http://127.0.0.1:3000`, `http://localhost:3000`, read from an env var; or serve the built frontend from Flask (same origin, no CORS). Not applied: needs your decision on the hosting model, and a wrong list would break the UI. | code inspection |
| Upload limit | 150 MB total (`TUMORGUARD_MAX_CONTENT_LENGTH`); 413 returns the Phase 2B message, `status: error`, no paths. Three uncompressed BraTS volumes are well below this. | API-SYN |
| Error leakage | 400/413/422/500 bodies contain no paths or tracebacks; an injected `RuntimeError("secret /home/…")` is replaced by the generic message; tracebacks go to the server log only. | API-SYN |
| Validation | missing modality, `seg`, bad extension, corrupt NIfTI, 2D/4D, spatial mismatch, non-finite, empty, out-of-range slice all rejected with 4xx. | API-SYN, UNIT |
| Retry / cancel | Retry succeeds after an injected network failure; Cancel returns to Upload without an error; duplicate submit is ignored. **Cancel does not stop server-side computation** (see F-3). | BROWSER-SYN, MOCK-UI |
| Controls | 59 browser checks passed; every visible control has an accessible name and a real href (inventory in `docs/FUNCTIONALITY_AUDIT.md`). | BROWSER-SYN |

### Tests run in this audit (fresh)

| Command | Class | Result |
|---|---|---|
| `python -m pytest tests -rs` (clean venv from `requirements-dev.txt`) | UNIT + API-SYN | **140 passed, 12 skipped**. Skips: 1 CUDA, 1 `tumorguard_dev.pth` absent, 10 Phase 2B real-BraTS tests (data absent) |
| `npm test` (Vitest) | MOCK-UI | 54 passed |
| `node frontend/test_verdict.mjs` | MOCK-UI | passed |
| `npm run lint` | static | clean |
| `npm run build` | build | succeeded |
| `node e2e/functional_audit.mjs` against the real backend | BROWSER-SYN | 59 / 59 (one aborted request = deliberate Cancel test), 0 console errors |
| Real BraTS tests | REAL | **not run** (no data) |

## Task 6: demo readiness

**1. Verified working (synthetic data, real checkpoint):** strict checkpoint load; upload with client header checks; server validation and safe errors; automatic and explicit slice selection; MC and deterministic prediction fields; three aligned 128 x 128 panels; evidence layer with four states; downloads (HTML/JSON/PNG); session history; navigation, responsive menu, retry/cancel.

**2. Unverified:** everything on real MRI (stripes, fragmented overlay, heatmap, verdict distribution, orientation/left-right); the 0.7863 baseline for the committed checkpoint (F-5); behaviour with real file sizes; Firefox/Safari; screen readers; GPU.

**3. Known defects:** F-1 `main` replaced by the simulated prototype; F-2 unavailable-evidence banner; F-3 inference race; F-4 no run-to-run caveat; F-5 hash mismatch; open research items D-15, D-16, D-21, D-23, D-24; open CORS; Cancel does not stop the server.

**4. Manual steps for one real BraTS case (PowerShell, nothing leaves your machine):**
1. Use a clean copy: `git fetch origin; git worktree add ..\TumorGuard-release origin/release-fix/ui-evidence-layer; cd ..\TumorGuard-release`.
2. Hashes: `Get-FileHash models\tumorguard_model.pt -Algorithm SHA256` (expect `eb50ced6…`) and `Get-FileHash "C:\Medha\TumorGuard AI\models\tumorguard_model.pt" -Algorithm SHA256`. Write both down (F-5).
3. `python -m venv .venv; .\.venv\Scripts\pip install -r requirements-dev.txt`.
4. Put one case where the tests expect it, e.g. `BraTS2020_ValidationData\MICCAI_BraTS2020_ValidationData\BraTS20_Validation_001\BraTS20_Validation_001_{flair,t1ce,t2}.nii`, then `.\.venv\Scripts\python -m pytest tests\test_real_data_e2e.py tests\test_spatial_validation.py -v -rs`. Record pass/skip counts.
5. Header facts: `.\.venv\Scripts\python -c "import nibabel as nib,sys;[print(f, (i:=nib.load(f)).shape, i.header.get_zooms()[:3], nib.aff2axcodes(i.affine)) for f in sys.argv[1:]]" <flair> <t1ce> <t2>`.
6. Start the API: `$env:MPLBACKEND='Agg'; .\.venv\Scripts\python server.py`. In a second terminal: `cd frontend; npm ci; npm run build; npx vite preview --port 4173`.
7. API record: `curl.exe -s -X POST -F "flair=@<flair>" -F "t1ce=@<t1ce>" -F "t2=@<t2>" http://127.0.0.1:5000/predict -o response.json`. Keep `response.json` (it contains images; do not share it).
8. UI: open `http://127.0.0.1:4173/#/upload`, upload the same three files, note the slice, verdict, score, each evidence signal and the "Score details" table. Screenshot the three panels.
9. Check for the two reported symptoms: horizontal stripes, fragmented overlay, streaky heatmap; and whether the verdict is MANUAL REVIEW REQUIRED. If yes, record `reliability_breakdown` from `response.json`.
10. Orientation: open the same FLAIR slice in 3D Slicer or ITK-SNAP at the same slice index and compare the panel (rotation, flip).
11. Repeat step 8 five times on the same slice and record the verdicts (expected non-determinism, section 3).
12. Send back the header line from step 5, the two hashes, the pass/skip counts, `reliability_breakdown`, and the screenshots.

**5. Evidence layer as an experimental explanation feature:** **yes, with its labels intact.** It is derived only from real pipeline outputs, never alters the prediction, marks familiarity as not validated, never reports "safe", and lists limitations. Say aloud that its rules are untuned heuristics, that no signal has been shown to predict errors, and that "No additional review flagged" is not evidence of correctness. Fix F-2 first if the reference could be absent.

**6. Unsupported or potentially misleading UI statements:** (a) the Research page lists "training patients 001-030, validation 031-040" as fact; it comes from the code, not from a verified record (and see F-5); (b) the landing-page Mean Dice / IoU tiles are labelled historical and not reproduced, but the evidence panel shows the served checkpoint's hash `eb50ced6…` next to "historical evaluation", which F-5 says may not be the evaluated file; (c) "Available and valid" on MC signals means "a finite measurement", not "validated"; the card adds "Not validated as an error predictor", which is correct but easy to miss; (d) no statement that verdicts vary between runs (F-4). No clinical-readiness or diagnostic claims were found.

## Recommended next action (nothing modified yet)

1. You: decide what to do about `main` (F-1) and run the real-case steps above, including the two `Get-FileHash` commands (F-5).
2. Then, with your approval, one small branch off `release-fix/ui-evidence-layer` containing only: (a) a lock around inference in `predict()` with a two-request test (F-3); (b) the ASSESSMENT_UNAVAILABLE banner and the run-to-run sentence in the UI (F-2, F-4); (c) the audit test file and this document; (d) a CORS origin allow-list behind an env variable defaulting to the current behaviour. No thresholds, formulas, seeds or model changes.

## Follow-up verification (read-only, later the same day)

* F-1 confirmed with the corrected counts above; main's `engine.ts` states it "Simulates multiple stochastic forward passes" and `samples.ts` generates synthetic brains; it contains no PyTorch, ONNX or backend calls.
* F-5 remains unresolved: the only checkpoint blob in all Git history hashes to `eb50ced6...`, and `785d1e2e...` matches no committed file. The checkpoint's epoch (20), validation loss and sample count equal those in `models/phase9_evaluation.json`, which is circumstantial support only.
* F-3 is confirmed at code level (global model, no lock, Flask threaded by default); the two-thread reproduction was run once at core level and its script was not kept.
