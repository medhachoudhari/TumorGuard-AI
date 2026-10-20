# Phase 2A reconciliation report

## Inputs

| Source | Available? | Notes |
|---|---|---|
| `main` @ `ceca4f1` | Yes | Baseline. Unchanged. |
| `fix/phase-2a-independent` @ `b4c5928` | Yes (remote) | Independent implementation. Preserved; not modified. |
| Antigravity implementation (`fix/phase-2a-verdict-checkpoint-loader`) | **No** | Exists only in the author's Windows working tree. It is not on the remote and no patch was supplied. |

## Limitation

**No comparison with the Antigravity implementation was possible**, so nothing here is a file-by-file reconciliation of two versions. This branch is the independent implementation, re-reviewed critically, plus the fixes below. When the Antigravity files or a `git diff` patch are supplied, the overlapping files should be compared against this branch: `tumorguard_core.py`, `server.py`, `evaluate_phase9.py`, `calibrate_reliability.py`, `calibrate_new_model.py`, `train_cached.py`, `frontend/src/App.jsx`, `frontend/src/verdict.js`, the three test files and `requirements.txt`.

Antigravity's reported results (16 Python tests, 16 JavaScript assertions, lint, build, historical evaluation 0.7863) are **not verified** by this work.

## Branch base

`integration/phase-2a-reconciled` was created from `fix/phase-2a-independent` (the only available Phase 2A implementation) and adds commits on top. `main` and both source branches were not modified.

## Review of the independent implementation (changes made on this branch)

| Area | Finding | Decision |
|---|---|---|
| Checkpoint loader | Verified from code: `weights_only=True`, wrapper-key detection, ambiguity check, all-tensor check, strict key and shape validation before touching the model, finite-weight check, basename-only messages. No `weights_only=False` remains in the repository. | Keep. |
| Loader trust model | `weights_only=True` narrows what unpickling can execute; it does not make an untrusted file safe, and the extension is not checked. Documented in the code comment. `module.`-prefixed (DataParallel) checkpoints fail strict loading with a clear message rather than being silently remapped. | Keep. |
| Verdict logic | Thresholds unchanged. Invalid inputs -> `MANUAL REVIEW REQUIRED`. | Keep. |
| Frontend | `verdict.js` + conservative fallback; `LOW RELIABILITY` uses the red style (design choice). CSS classes `v-safe`, `v-attention`, `v-manual` exist in `App.css`. | Keep. |
| `evaluate_phase9.py` | Overwrote the historical `models/phase9_evaluation.json` and hard-coded Windows paths. | **Changed**: run-specific output, refuses to overwrite, records checkpoint SHA-256 and library versions, env-var path overrides. |
| `requirements.txt` | `server.py` imports `flask_cors` at module level, so it is a genuine dependency. | Keep `flask-cors`; add `requirements-dev.txt` (pytest). A fresh venv built from `requirements-dev.txt` runs the full test suite. |
| Tests | Loader and verdict tests call production functions. The contract test parses `verdict.js` and compares it with `core.VERDICTS`. | Added `tests/test_api_smoke.py` (Flask test client, synthetic volumes). |
| Docs | `PHASE_8_VALIDATION.md` is stale (obsolete thresholds and checkpoint). | Left unchanged (historical); recorded as D-05. |
| New docs | `INFERENCE_CONTRACT.md` written from the source, with newly found defects D-15 to D-22 recorded. | Added. |

## Not changed (deliberately)

Model architecture, weights, preprocessing, MC Dropout, reliability weights and thresholds, `server.py` request handling, and the defects D-15 to D-18 and D-20 to D-22, which need a research decision rather than a stabilization fix.
