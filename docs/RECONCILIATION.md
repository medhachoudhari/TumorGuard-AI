# Phase 2A Reconciliation Report

## Inputs & Provenance

| Source | Commit / Ref | Available? | Status & Notes |
|---|---|---|---|
| Protected `main` | `ceca4f1d8c64106b098733a43dabe7be6ca66a2a` | Yes | Pristine baseline. Preserved; not modified. |
| Independent implementation (`fix/phase-2a-independent`) | `b4c5928a2a7480bf4e437717a2d2ced6dc231594` | Yes (remote) | Pre-flight validation, NaN guards, pytest suite. Preserved; not modified. |
| Antigravity implementation (`fix/phase-2a-verdict-checkpoint-loader`) | `17064c2` (local branch) | Yes (local) | Checkpoint loader, real-checkpoint validation, legacy verdict handling. Preserved at `17064c2`. |
| Reconciled integration branch (`integration/phase-2a-reconciled`) | `e3a13b6` (base) + reconciliation updates | Yes | Active reconciled branch. |

---

## Files Compared & Reconciled

The following overlapping files were examined side-by-side and reconciled:

1. `tumorguard_core.py`
2. `server.py`
3. `evaluate_phase9.py`
4. `calibrate_reliability.py`
5. `calibrate_new_model.py`
6. `train_cached.py`
7. `frontend/src/App.jsx`
8. `frontend/src/verdict.js`
9. `frontend/test_verdict.mjs`
10. `tests/test_checkpoint_loader.py`
11. `tests/test_verdict_contract.py`
12. `tests/test_api_smoke.py`
13. `requirements.txt` & `requirements-dev.txt`

---

## Retained Changes from Each Implementation

### Retained from Independent Implementation (`b4c5928` / `e3a13b6`):
1. **Explicit Checkpoint Exception:** `class CheckpointError(ValueError)` for precise failure handling.
2. **Pre-Flight Validation:** Inspects shape mismatches, missing keys, unexpected keys, and non-finite weights (`not bool(torch.isfinite(v).all())`) *before* invoking `model.load_state_dict()`, leaving the model untouched on failure.
3. **Ambiguous Wrapper Detection:** Rejects payloads containing both `'model_state_dict'` and `'state_dict'`.
4. **NaN/Non-Finite Input Guard:** `verdict_from_score()` and `calculate_reliability_and_verdict()` validate `math.isfinite()` on all inputs, immediately returning `(0.0, "MANUAL REVIEW REQUIRED")` if evidence is non-finite or invalid.
5. **Path Privacy:** Formats error messages using `os.path.basename()` to prevent leaking absolute directory paths.
6. **Frontend Safety & Strictness:** `frontend/src/verdict.js` uses `Object.freeze()` and `Object.hasOwn()` to protect against prototype pollution (`toString`, `__proto__`); `App.jsx` displays `verdictLabel(result.verdict)` with explanatory caption when unrecognized.
7. **Traceable, Non-Destructive Evaluation:** `evaluate_phase9.py` records checkpoint SHA-256, UTC timestamp, PyTorch/NumPy versions, writes to `outputs/evaluations/phase9_<UTC>.json` without overwriting historical records, and supports environment variable overrides.
8. **Synthetic API Smoke Tests:** `tests/test_api_smoke.py` validates `/health` and `/predict` using synthetic volumes and Flask's test client without requiring live servers or clinical patient data.
9. **Dependencies:** Added `flask-cors` to `requirements.txt` and `pytest` to `requirements-dev.txt`.

### Retained from Antigravity Implementation (`17064c2`):
1. **Dual Parameter Compatibility:** `load_model_checkpoint(model, path, map_location="cpu", device=None, strict=True)` supports `device` as a transparent, backward-compatible alias for `map_location`.
2. **Canonical Verdict Aliases:** Provided full-name constant aliases (`VERDICT_HIGH_RELIABILITY`, `VERDICT_MODERATE_RELIABILITY`, `VERDICT_LOW_RELIABILITY`, `VERDICT_MANUAL_REVIEW`, `SUPPORTED_VERDICTS`) alongside concise names (`VERDICT_HIGH`, `VERDICTS`).
3. **Dual Real-Checkpoint Testing:** Extended `tests/test_checkpoint_loader.py` to strictly load both shipped repository checkpoints: `models/tumorguard_model.pt` (container dictionary) and `models/tumorguard_dev.pth` (raw `OrderedDict`), verifying successful forward passes and probability shapes for both.
4. **Clean Artifact Environment:** Verified and removed accidental `ts -Recurse -File` artifact after confirming it was a redirected terminal help dump (backed up to scratch directory).

---

## Conflicts Resolved

| Component | Conflict | Resolution | Rationale |
|---|---|---|---|
| `load_model_checkpoint` signature | `map_location` vs `device` argument name | `def load_model_checkpoint(model, path, map_location="cpu", device=None, strict=True): if device is not None: map_location = device` | Zero-breakage compatibility across call sites and external scripts. |
| Verdict constants | Concise (`VERDICT_HIGH`) vs full (`VERDICT_HIGH_RELIABILITY`) | Defined concise canonical constants and exported full-name aliases mapped to them | Unifies both test suites and avoids name errors. |
| Frontend verdict mapping | Strict `Object.hasOwn` vs loose trimming | Retained strict `Object.hasOwn` with `isKnownVerdict` and `verdictLabel` | Prevents prototype pollution and ensures unknown/malformed strings fail safely into manual review. |
| Evaluation output | Destructive overwrite of `models/phase9_evaluation.json` vs new run output | Retained non-destructive output (`outputs/evaluations/phase9_<UTC>.json`) | Protects historical benchmark record integrity while recording full run provenance. |

---

## Acceptance Tests Actually Executed & Verified

All tests were executed locally against the virtual environment (`.venv`) and Node environment:

1. **Python Test Suite (`pytest tests`):**
   * Command: `& .\.venv\Scripts\python.exe -m pytest tests`
   * Result: **56 passed, 1 skipped (CUDA unavailable on CPU runner) in 8.80s**.
   * Coverage includes:
     * `test_api_smoke.py`: 6 tests passed (health check, missing modalities, segment uploads rejected, wrong extensions, corrupt NIfTI, synthetic prediction contract).
     * `test_checkpoint_loader.py`: 22 tests (21 passed, 1 skipped for CUDA). Confirmed strict loading of both `tumorguard_model.pt` and `tumorguard_dev.pth`, missing/unexpected keys, non-finite weights, DataParallel prefix rejection, pickle blocking.
     * `test_verdict_contract.py`: 28 tests passed. Canonical strings, threshold boundaries, NaN/inf evidence rejection, verdict aliases, regex matching with `verdict.js`.

2. **Frontend Unit Tests (`node frontend/test_verdict.mjs`):**
   * Command: `node frontend/test_verdict.mjs`
   * Result: **Passed (all assertions satisfied)**. Verified canonical verdict states, prototype pollution resistance, and conservative manual review fallback.

3. **Frontend Linter & Build:**
   * Commands: `npm run lint` and `npm run build` in `frontend/`
   * Results:
     * `oxlint`: **0 warnings, 0 errors** across 5 files.
     * `vite build`: **Built cleanly in 960ms** (production bundle generated in `dist/`).

4. **Phase 9 Model Evaluation Baseline:**
   * Command: `& .\.venv\Scripts\python.exe evaluate_phase9.py`
   * Genuine dataset: `datasets/val_x.pt` and `datasets/val_y.pt` (398 slices from patients 031–040).
   * Verified Checkpoint: `models/tumorguard_model.pt` (SHA-256: `785d1e2e69736c929dc8870183e20e8b2ea1322fa01ffb0faef4c29ea338db71`).
   * Historical `models/phase9_evaluation.json`: **Untouched / preserved**.
   * Run-specific output: Written to `outputs/evaluations/phase9_20261020T082837Z.json`.
   * Verified Metrics:
     * **Mean Dice:** `0.7863`
     * **Median Dice:** `0.8762`
     * **Mean IoU:** `0.6961`
     * **Mean Precision:** `0.8364`
     * **Mean Recall:** `0.8146`
     * **Zero-prediction slices:** `19/398`

---

## Research-Sensitive Defects Review (D-15 to D-22)

These defects were identified during Phase 1/2A audits and are documented in `docs/DEFECT_REGISTER.md`. In accordance with Phase 2A constraints, research-sensitive semantics have been deliberately preserved and deferred to explicit research decisions:

| ID | Defect | Status | Why Deferred to Phase 3/4 |
|---|---|---|---|
| **D-15** | `max_diff` compares only pass 0 vs pass 9 instead of peak-to-peak spread | OPEN | Modifying this formula directly alters the numerical reliability score and verdict distribution; requires planned calibration study (Phase 4). |
| **D-16** | Missing `familiarity_reference` returns 0.5 neutral fallback | OPEN | Latent path (shipped `training_stats.json` has the reference). Rebuilding the reference on all 30 patients is scheduled for Phase 4. |
| **D-17** | `/predict` lacks spatial affine / voxel-spacing consistency checks | OPEN | Architectural enhancement scheduled for Phase 7 / pipeline hardening. |
| **D-18** | HTTP 500/400 exception string exposure and open CORS | OPEN | Operational hardening scheduled prior to non-local deployment. |
| **D-19** | `evaluate_phase9.py` overwriting benchmark JSON | **FIXED** | Resolved in Phase 2A by writing run-specific timestamped outputs with SHA-256 provenance. |
| **D-20** | Deterministic evaluation vs MC-dropout mean in API | OPEN | Metric comparison scheduled for Phase 3 evaluation study. |
| **D-21** | Full-canvas entropy/variance averaging diluted by background | OPEN | Research modification; restricting entropy to brain mask changes threshold calibration; scheduled for Phase 4. |
| **D-22** | Hard-coded `C:\Medha\TumorGuard AI` paths in offline preparation scripts | OPEN | Low priority; scripts are offline-only. |

---

## Phase 1 & Phase 2A Acceptance Recommendations

1. **Acceptance Recommendation for Phase 1 (Audit & Baseline):** **ACCEPT**. Baseline verified across architecture, weights, evaluation metrics (Mean Dice 0.7863), and comprehensive defect register (D-01 to D-22).
2. **Acceptance Recommendation for Phase 2A (Reconciliation):** **ACCEPT**. Both implementations are fully reconciled, unified under `integration/phase-2a-reconciled`. All unit tests, API smoke tests, frontend linting, frontend building, and evaluation scripts pass with zero errors.
