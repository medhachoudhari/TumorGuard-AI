# Baseline record

Only facts actually obtained are recorded here. Anything not obtained is marked **NOT VERIFIED**.

## 1. Historical results (reported, NOT independently reproduced)

Source: `models/phase9_evaluation.json` (committed in `ceca4f1`, produced by `evaluate_phase9.py` in an earlier session; date, commit and environment of that run are not recorded).

| Metric | Value |
|---|---|
| Validation slices | 398 |
| Mean Dice | 0.7863 |
| Median Dice | 0.8762 |
| Mean IoU | 0.6961 |
| Mean precision | 0.8364 |
| Mean recall | 0.8146 |
| Zero-prediction slices | 19 |
| Checkpoint epoch / val loss | 20 / 0.3475 |

Antigravity reported rerunning `evaluate_phase9.py` and obtaining the same values. That claim is **unverified**: the script writes its output to the same JSON path, so a matching file does not prove an independent rerun.

Caveats on what these numbers mean:
* Slices are the unit of aggregation (slice-level, not patient-level).
* Slices were chosen from tumour-containing slices only (`MAX_*_SLICES_PER_PATIENT = 40`), so the metrics do not describe whole-volume or tumour-free performance.
* The validation patients (031-040) were used for checkpoint selection. They are **not** an untouched test set.

## 2. Reproduction status in this review environment

**NOT VERIFIED.** The cached validation tensors and BraTS data are not available in the review container, so `evaluate_phase9.py` was not run and no metric above was reproduced.

## 3. Checkpoint inventory (obtained 2026-10-20, repository commit `ceca4f1`)

| File | SHA-256 | Notes |
|---|---|---|
| `models/tumorguard_model.pt` | `eb50ced634beb064e10d44d8586f555a0207a2b1915157819137c3b268c636fe` | 1,896,485 bytes; wrapped format: `model_state_dict` (66 tensors) + `epoch=20`, `train_loss=0.3526`, `val_loss=0.3475`, `device=cpu`, `train_samples=1200`, `val_samples=398`. Loads with `weights_only=True` and strict key/shape matching; a zero input of shape `(1,3,128,128)` gives output `(1,1,128,128)`. |
| `models/tumorguard_dev.pth` | - | Referenced by `calibrate_reliability.py` and `PHASE_8_VALIDATION.md`; **not in the repository**. |
| `models/tumorguard_30patient_best.pth` | - | Written by `train_cached.py`; **not in the repository**. |

Other files:

| File | SHA-256 |
|---|---|
| `models/training_stats.json` | `fa13f42e2a1ca8d374d95eb29fa17fa34f6ff0a9a386619edbb9c8f6df06e3d1` |
| `models/training_history.json` | `3fda7a54ed92b8aee1ce23b52062684c1f7a3045f1a7a219aa2e7b48ee9cc998` |
| `models/phase9_evaluation.json` | `acecc7f371b431d987407dfbcc9a8b85d1c9435ffb79b3232552e87879c89ca8` |

The checkpoint does not store patient IDs. `train_samples=1200` and `val_samples=398` are consistent with `train_cached.py` (30 patients x up to 40 slices; 10 validation patients), which is indirect evidence for the 001-030 / 031-040 split, not proof.

## 4. Familiarity reference limitation

`models/training_stats.json` lists `training_patients` with **4** entries, while the checkpoint was trained on a much larger set. The familiarity reference therefore does not describe the full training population.

## 5. Environment used for the checks in this document

Linux container (4 CPU, 15 GB RAM, no GPU), Python 3.13.16, torch 2.14.1 (`+cu130` build, run on CPU), numpy 2.5.4, nibabel 5.4.2, scikit-image 0.26.0, scipy 1.18.1, matplotlib 3.11.2, Flask 3.1.3, flask-cors 6.0.5, pytest 9.1.1, Node 22.22.0 / npm 10.9.4. This is **not** the environment of the historical evaluation.

## 6. Commands run (and results)

See `docs/PHASE1_AUDIT.md`, section "Verification actually performed".

## 7. Command to reproduce the baseline (run on the Windows machine that has the cached tensors; NOT yet run)

`evaluate_phase9.py` no longer overwrites `models/phase9_evaluation.json`; it writes `outputs/evaluations/phase9_<UTC>.json` and records the checkpoint SHA-256 and library versions.

```
cd "C:\Medha\TumorGuard AI"
git rev-parse HEAD
Get-FileHash models\tumorguard_model.pt -Algorithm SHA256    # expect eb50ced6...c636fe
python -c "import sys, torch; print(sys.version, torch.__version__)"
python evaluate_phase9.py
```

Requirements for a valid reproduction: `datasets\val_x.pt` and `val_y.pt` must be the original 398-slice validation cache (patients 031-040). Compare the new JSON with section 1; record commit, versions, device, date, command and output path here. A match shows the pipeline reproduces the historical numbers on the same cache; it does not make 031-040 an independent test set.

If the cache must be rebuilt, run `prepare_dataset_fast.py` (BraTS2020 patients 001-040) and note that the regenerated cache may differ from the original.
