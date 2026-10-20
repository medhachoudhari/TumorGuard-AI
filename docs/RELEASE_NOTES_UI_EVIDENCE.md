# Release notes: redesigned UI, Reliability Evidence Layer, API hardening

Base: `integration/phase-2a-reconciled` (`e3a13b6`). Branch: `release-fix/ui-evidence-layer`.

## Important context

* The prompt referred to an integration commit `42e6933` and to Phase 2B features (spatial validation, upload limits, explicit slice selection, `mc_prediction`, `deterministic_prediction`). **None of these exist in any repository reachable from this environment.** The GitHub integration branch is `e3a13b6`. The overlapping behaviour (validation, upload limit, safe errors, explicit slice selection, prediction-method metadata) was therefore implemented here independently, and the fields `mc_prediction` / `deterministic_prediction` were **not** created. Expect merge conflicts with any separate Phase 2B work; the API additions are documented in `docs/INFERENCE_CONTRACT.md`.
* The separate Google AI Studio prototype was not available; its concepts (four evidence states, structured report, review reasons, missing-reference safeguard, limitations/provenance) were implemented from the description, with no simulated inference.
* The uploaded eight-screen concept image was used as a visual reference only. None of its numbers were copied into the application.

## What changed

* **Backend:** `input_validation.py` (shape/affine/finite/empty checks), `rendering.py` (aligned native-resolution PNGs), `reliability_evidence.py` (evidence layer), `server.py` (limits, explicit slice, safe JSON errors, new additive fields), `tumorguard_core.py` (`reliability_components`; score proven identical to the original formula on a grid).
* **Frontend:** new React app (hash-routed): landing, upload with client-side NIfTI header validation, analysis progress (real upload progress), results (three aligned panels, slice controls, zoom, downloads), evidence layer, session history, research and validation, about.
* **Tests:** pytest 105 passed / 1 skipped; Vitest 54 passed; browser audit 59 / 59 (see `docs/FUNCTIONALITY_AUDIT.md`).

## Root causes

* **Verdict bug:** the original frontend mapped `SAFE` / `NEEDS` strings that the backend never sends (fixed earlier as D-01). "Every output is MANUAL REVIEW REQUIRED" was **not reproduced**; with the real checkpoint and synthetic input the backend returned MODERATE or LOW. The score is structurally capped near 0.6 by the degenerate familiarity reference (D-24), so HIGH is nearly unreachable. Evidence needed: one real BraTS case; the per-signal breakdown is now returned and displayed for exactly this purpose.
* **Visualization bug (stripes, fragmented overlay, streaky heatmap):** **not reproduced** with the old or the new renderer on well-formed synthetic volumes. Found and fixed real defects (grey background, edge artifact, inconsistent per-panel figures, as-stored orientation). The most probable real-data trigger, volumes whose third axis is not axial, now produces a visible warning. Evidence needed: one real BraTS case, before/after.

## Not changed (research-sensitive, per instructions)

Thresholds (0.70 / 0.50 / 0.30), weights, D-15 (first-vs-last pass), D-16 (0.5 familiarity fallback, now flagged by the evidence layer as ASSESSMENT_UNAVAILABLE), D-21 (whole-slice averaging), D-10 (`>` vs `>=` operator), MC Dropout seeding (D-23), the model, checkpoint and `models/phase9_evaluation.json`.

## How to verify

```
pip install -r requirements-dev.txt && python -m pytest tests
cd frontend && npm ci && npm test && npm run lint && npm run build
# real-browser audit (needs the backend on :5000 and `npm run preview --port 4173`)
BASE=http://127.0.0.1:4173 FILES_DIR=<dir with flair/t1ce/t2.nii.gz, t2_mismatch.nii.gz, notes.txt, broken.nii.gz> node e2e/functional_audit.mjs
```
