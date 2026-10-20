# Release notes: redesigned UI, Reliability Evidence Layer, API hardening

Base: `integration/phase-2a-reconciled` (`e3a13b6`). Branch: `release-fix/ui-evidence-layer`.

## Important context

* The prompt referred to integration commit `42e6933` (Phase 2B). When this work started that commit was **not** visible: the remote integration branch was `e3a13b6`, so the first version of this branch (`d168390`) re-implemented validation, limits and explicit slices independently. By the time of the first push `42e6933` had appeared on `integration/phase-2a-reconciled`. It has since been **merged into this branch** (normal merge commit, no history rewrite) and Phase 2B is treated as authoritative: its `validate_modality_volumes`, `run_deterministic_inference`, error messages, `status: "error"` bodies, 150 MB default limit (`TUMORGUARD_MAX_CONTENT_LENGTH`), `slice_selection_mode`, `prediction_method`, `mc_prediction` and `deterministic_prediction` are preserved, and the overlapping code from this branch was removed or adapted. Phase 2B's real-data tests (`tests/test_real_data_e2e.py`) were **skipped** here because the BraTS data is not available in this environment.
* The separate Google AI Studio prototype was not available; its concepts (four evidence states, structured report, review reasons, missing-reference safeguard, limitations/provenance) were implemented from the description, with no simulated inference.
* The uploaded eight-screen concept image was used as a visual reference only. None of its numbers were copied into the application.

## What changed

* **Backend:** `input_validation.py` (summary, warnings and stable error codes on top of Phase 2B validation), `rendering.py` (aligned native-resolution PNGs), `reliability_evidence.py` (evidence layer), `server.py` (limits, explicit slice, safe JSON errors, new additive fields), `tumorguard_core.py` (`reliability_components`; score proven identical to the original formula on a grid).
* **Frontend:** new React app (hash-routed): landing, upload with client-side NIfTI header validation, analysis progress (real upload progress), results (three aligned panels, slice controls, zoom, downloads), evidence layer, session history, research and validation, about.
* **Tests:** pytest 132 passed / 12 skipped (real-data and CUDA tests); Vitest 54 passed; browser audit 59 / 59 (see `docs/FUNCTIONALITY_AUDIT.md`).

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
