# Defect register

Severity: P0 integrity/correctness, P1 required before phase acceptance, P2 required before Phase 5 comparisons, P3 later.
Status: FIXED (in working tree, tested), OPEN, or OPEN-UNVERIFIED.

| ID | Sev | Status | Defect | Evidence | Action |
|---|---|---|---|---|---|
| D-01 | P0 | FIXED | Frontend verdict mapping never matched backend verdicts: `verdictClass` tested for `SAFE` / `NEEDS`, so every backend verdict, including `HIGH RELIABILITY`, rendered with the manual-review (red) style. | `App.jsx` on `main` lines 13-17 vs `tumorguard_core.py` verdict strings. | `frontend/src/verdict.js`; unknown/missing -> manual review. Tests: `frontend/test_verdict.mjs`, `tests/test_verdict_contract.py`. |
| D-02 | P0 | FIXED | Inconsistent, partly unsafe checkpoint loading: `evaluate_phase9.py` used `weights_only=False`; scripts differed on raw vs wrapped formats; no shape/key diagnostics. | `main`: `server.py:42`, `evaluate_phase9.py:128`, `calibrate_reliability.py:99`, `calibrate_new_model.py:99`, `train_cached.py:279`. | `load_model_checkpoint` in `tumorguard_core.py`, used at all five sites. |
| D-03 | P1 | FIXED | Invalid numeric evidence (NaN/None/inf) could produce a normal score and verdict; with NaN inputs the result depended on `min`/`max` argument order. | `calculate_reliability_and_verdict` before change. | Non-finite/unparsable input -> `(0.0, "MANUAL REVIEW REQUIRED")`. Thresholds unchanged. |
| D-04 | P1 | FIXED | `requirements.txt` omitted `flask-cors`, which `server.py` imports; a clean install cannot start the server. | `ModuleNotFoundError: flask_cors` in a clean venv. | Added `flask-cors`. Still unpinned (see D-09). |
| D-05 | P1 | OPEN | `PHASE_8_VALIDATION.md` is stale and contradicts the code: names `tumorguard_dev.pth`, thresholds `SAFE`/`NEEDS ATTENTION`, a 50/50 score; the code uses 0.40/0.20/0.20/0.20 weights and 0.70/0.50/0.30 cut-offs. | File vs `tumorguard_core.py`. | Mark as historical or rewrite after Phase 2. |
| D-06 | P1 | OPEN-UNVERIFIED | Historical metrics are not reproduced; the evaluation script overwrites its own historical JSON path. | `docs/BASELINE.md` section 2. | Rerun with a new output path where data exists. |
| D-07 | P1 | OPEN | Familiarity reference built from 4 patients (`training_stats.json`), not the 30 training patients. | `training_stats.json: training_patients`. | Rebuild in Phase 4; document as limitation until then. |
| D-08 | P1 | OPEN | Validation patients 031-040 were used for model selection and are not an independent test set. | `train_cached.py` selects best checkpoint on them. | Hold out a new test set (Phase 3). |
| D-09 | P2 | OPEN | Dependencies unpinned; no lockfile; no recorded versions for the original environment. | `requirements.txt`. | Freeze lockfile. |
| D-10 | P2 | OPEN | Server and evaluator use different threshold comparisons (`> 0.5` on probability in `server.py:163`, `>= 0.5` in `evaluate_phase9.py:29`). | Source. | Align in Phase 2 after reviewing the contract. |
| D-11 | P2 | OPEN | Checkpoint `models/tumorguard_model.pt` is tracked in Git although `.gitignore` lists `*.pt`; provenance (patient IDs, seed, command) is not stored in it. | `git ls-files`. | Decide on LFS/release asset; record provenance. |
| D-12 | P2 | OPEN | Reliability weights/cut-offs (`VAR_LOW/HIGH`, `DIFF_LOW`, 0.70/0.50/0.30) are heuristics; calibration against actual segmentation error is not demonstrated. | `calculate_reliability_and_verdict`; `calibrate_new_model.py` uses patients 041-053. | Phase 4 study. The score must not be described as a probability. |
| D-13 | P2 | OPEN | Live inference slice selection uses a FLAIR brain-voxel heuristic, while evaluation/training uses a ground-truth tumour-slice heuristic; the live path is therefore not what was evaluated. | `select_inference_slice` vs `select_best_tumor_slice`. | Phase 2 contract. |
| D-14 | P3 | OPEN | Mixed line endings on the author's Windows checkout (CRLF warnings); no `.gitattributes`. The repository content is LF. | `git ls-files --eol`. | Add `.gitattributes` (`* text=auto eol=lf`) in a separate commit. |
