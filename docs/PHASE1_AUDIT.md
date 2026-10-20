# Phase 1 audit: repository stabilization and baseline recovery

Scope: evidence available in the review container. Antigravity's private task history was not available and is not relied upon.

## Provenance of this work

The Phase 2A changes reported by Antigravity exist only in the author's Windows working tree and were **not available** to this reviewer. Phase 2A was therefore **re-implemented independently** on top of `main` (`ceca4f1`) in a clean clone. It must be reconciled with the Windows working tree before anything is pushed (see "Reconciliation" below). This audit does not review Antigravity's code.

## Acceptance matrix

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Repo structure, Git state, recovery point | PARTIALLY COMPLETE | `main` = `ceca4f1` (3 commits), pushed. No tag. |
| 2 | Dependency/runtime documentation | PARTIALLY COMPLETE | `requirements.txt` lists packages (now with `flask-cors`) but unpinned; environment used for these checks recorded in `BASELINE.md` section 5. |
| 3 | Architecture and checkpoint compatibility | VERIFIED COMPLETE for `models/tumorguard_model.pt` | Strict load, 66 tensors, forward pass output `(1,1,128,128)`; `tests/test_checkpoint_loader.py::test_repository_checkpoint_loads_strictly`. Other checkpoints are absent. |
| 4 | Preprocessing/inference contract | NOT COMPLETED | Deferred to Phase 2 (`docs/INFERENCE_CONTRACT.md`). Code comments describe z-score per slice with background zeroed, modality order FLAIR, T1CE, T2, 128x128. |
| 5 | Evaluation procedure and provenance | PARTIALLY COMPLETE | `evaluate_phase9.py` and `phase9_evaluation.json` exist; run date/commit/environment unknown. |
| 6 | Historical vs verified metrics separated | VERIFIED COMPLETE (documentation) | `BASELINE.md` sections 1-2; reproduction itself NOT VERIFIED. |
| 7 | Defect register | VERIFIED COMPLETE (initial) | `DEFECT_REGISTER.md`. |
| 8 | Partitions, leakage, checkpoint provenance | PARTIALLY COMPLETE | Code defines train 001-030 / val 031-040 with no overlap (`train_cached.py:35-36`, `prepare_dataset_fast.py:23-24`). Checkpoint metadata (1200 / 398 samples) is consistent but stores no patient IDs. Selection on validation data and the 4-patient familiarity reference are recorded as D-08 / D-07. Cached tensors unavailable, so patient-level leakage in the cache was not inspected. |
| 9 | Reproducible baseline procedure | PARTIALLY COMPLETE | Procedure written (`BASELINE.md` section 7); never executed. |
| 10 | Rollback/recovery | PARTIALLY COMPLETE | `main` at `ceca4f1`. No tag created (needs authorization). |
| 11 | Tests and end-to-end evidence | PARTIALLY COMPLETE | Unit tests and a **synthetic-volume** API smoke test below. No real MRI case, no browser/UI test. |

## Verification actually performed (2026-10-20, container described in BASELINE.md)

| Command | Result |
|---|---|
| `python -m pytest tests -q` | 47 passed, 1 skipped (CUDA test; no GPU) |
| `node frontend/test_verdict.mjs` | `verdict tests passed` (assertions in script: 4 known verdicts x 3 checks, 17 invalid values x 3 checks) |
| `npm run lint` (oxlint) | exit 0 |
| `npm run build` (vite) | built successfully, 1796 modules |
| `python server.py` + `curl /health` | `{"model_loaded":true,"status":"ok","training_stats_loaded":true}` |
| `POST /predict` with three **synthetic** NIfTI volumes (ellipsoid with noise, not real MRI) | HTTP 200; keys `verdict, reliability_score, familiarity_score, predictive_entropy, uncertainty_score, max_probability_difference, slice_index, images, disclaimer`; verdict `MODERATE RELIABILITY` (score 0.552). Values are meaningless scientifically; this only shows the pipeline runs with the strict loader. |
| `POST /predict` with only `flair` | HTTP 400 `Missing required modalities...` |

Not done: `evaluate_phase9.py` (no data), real BraTS upload, browser/UI test, GPU/device-mapping test, `train_cached.py` / calibration scripts (no data; only the changed load lines were edited).

## Verdicts

* **Phase 1: FAIL (not yet acceptable).** Baseline reproduction (item 5/6/9) and the real end-to-end run (item 11) have not been performed, and no recovery tag exists. Documentation gaps (items 1, 2, 7) are now filled.
* **Phase 2A: code delivered; engineering acceptance PASS WITH CAVEATS** for the independently re-implemented version, pending reconciliation with the Windows working tree and a run on real data.

## Remaining blockers for closing Step 1

1. Reconcile this implementation with the Windows working tree (Antigravity's version).
2. On a machine with data: run `evaluate_phase9.py` to a new output path, record results in `BASELINE.md`.
3. One real BraTS case through `/predict` and the UI.
4. Authorize a recovery tag on `main`.
