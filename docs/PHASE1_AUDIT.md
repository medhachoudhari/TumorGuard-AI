# Phase 1 audit: repository stabilization and baseline recovery

Scope: evidence available in the review container. Antigravity's private task history was not available and is not relied upon.

## Provenance of this work

The Phase 2A changes reported by Antigravity exist only in the author's Windows working tree and were **not available** to this reviewer. Phase 2A was therefore **re-implemented independently** on top of `main` (`ceca4f1`) and published as `fix/phase-2a-independent`. See `docs/RECONCILIATION.md`. This audit does not review Antigravity's code.

## Acceptance matrix

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Repo structure, Git state, recovery point | PARTIALLY COMPLETE | `main` = `ceca4f1` (3 commits), pushed. No tag. |
| 2 | Dependency/runtime documentation | PARTIALLY COMPLETE | `requirements.txt` (now with `flask-cors`) and `requirements-dev.txt` install cleanly in a fresh venv; versions recorded but not pinned; original training environment unknown. |
| 3 | Architecture and checkpoint compatibility | VERIFIED COMPLETE for `models/tumorguard_model.pt` | Strict load, 66 tensors, forward pass output `(1,1,128,128)`; `tests/test_checkpoint_loader.py::test_repository_checkpoint_loads_strictly`. Other checkpoints are absent. |
| 4 | Preprocessing/inference contract | PARTIALLY COMPLETE | `docs/INFERENCE_CONTRACT.md` written from the source code; not validated against real BraTS data; several mismatches recorded (D-10, D-13, D-15, D-17, D-20). |
| 5 | Evaluation procedure and provenance | PARTIALLY COMPLETE | `evaluate_phase9.py` and `phase9_evaluation.json` exist; run date/commit/environment unknown. |
| 6 | Historical vs verified metrics separated | VERIFIED COMPLETE (documentation) | `BASELINE.md` sections 1-2; reproduction itself NOT VERIFIED. |
| 7 | Defect register | VERIFIED COMPLETE (initial) | `DEFECT_REGISTER.md`. |
| 8 | Partitions, leakage, checkpoint provenance | PARTIALLY COMPLETE | Code defines train 001-030 / val 031-040 with no overlap (`train_cached.py:35-36`, `prepare_dataset_fast.py:23-24`). Checkpoint metadata (1200 / 398 samples) is consistent but stores no patient IDs. Selection on validation data and the 4-patient familiarity reference are recorded as D-08 / D-07. Cached tensors unavailable, so patient-level leakage in the cache was not inspected. |
| 9 | Reproducible baseline procedure | PARTIALLY COMPLETE | Procedure written (`BASELINE.md` section 7); never executed. |
| 10 | Rollback/recovery | PARTIALLY COMPLETE | `main` at `ceca4f1`. No tag created (needs authorization). |
| 11 | Tests and end-to-end evidence | PARTIALLY COMPLETE | 53 passing automated tests and synthetic-volume API smoke tests. No real MRI case, no browser/UI test. |

## Verification actually performed (2026-10-20, container: Linux, 4 CPU, no GPU, Python 3.13.16)

Environment for the commands below: a fresh venv created from `requirements-dev.txt` (torch 2.14.1, numpy 2.5.4, nibabel 5.4.2, scikit-image 0.26.0, scipy 1.18.1, matplotlib 3.11.2, Flask 3.1.3, flask-cors 6.0.5, pytest 9.1.1) plus Node 22.22.0 / npm 10.9.4.

| Command | Result |
|---|---|
| `python -m pytest tests -q` | 53 passed, 1 skipped (CUDA device test; no GPU) |
| `node frontend/test_verdict.mjs` | `verdict tests passed` |
| `npm run lint` (oxlint) | exit 0 |
| `npm run build` (vite) | succeeded |
| `git diff --check` | clean |
| `tests/test_api_smoke.py` (Flask test client, **synthetic** volumes) | health ok; missing modality, `seg` upload, bad extension and corrupt NIfTI -> HTTP 400; valid synthetic upload -> HTTP 200 with canonical verdict and the documented JSON keys |
| Real server process earlier (`python server.py` + `curl`), synthetic volumes | `/health` ok; `/predict` 200 (`MODERATE RELIABILITY`, score 0.552); missing modality 400 |
| `evaluate_phase9.py` on **synthetic random tensors** (16 slices) | runs end to end with the strict loader; wrote a run-specific JSON including `checkpoint_sha256`; second run refused to overwrite; `models/phase9_evaluation.json` hash unchanged (`acecc7f3...`). The metrics from this run are meaningless. |

Not done: evaluation on real data (no BraTS or cached tensors), real BraTS upload, browser/UI test, GPU device-mapping test, training and calibration scripts (only their load lines changed). No log-redaction fix was made (D-18).

## Verdicts

* **Phase 1: FAIL (not yet acceptable).** Baseline reproduction (items 5, 6, 9) and a real end-to-end run (item 11) have not been performed, and no recovery tag exists. Documentation gaps (items 1, 2, 4, 7) are largely filled.
* **Phase 2A: PASS WITH CAVEATS** for the independent implementation on this branch: code and tests pass in a clean environment. Caveats: no comparison with Antigravity's version was possible, no real-data run, no browser test.

## Remaining blockers for closing Step 1

1. Provide Antigravity's changes (patch or files) and compare them with this branch (`docs/RECONCILIATION.md`).
2. On a machine with the cached tensors, run the command in `docs/BASELINE.md` section 7 and record the result as a new, run-specific file.
3. Run one authorized real BraTS case through `/predict` and the UI.
4. Authorize a recovery tag on `main`.
