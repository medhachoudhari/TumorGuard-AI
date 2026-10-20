# TumorGuard AI

**Uncertainty-Aware Brain MRI Tumor Segmentation and Error Identification**

TumorGuard AI is a research prototype. It segments brain tumours on MRI slices with a PyTorch 2D U-Net, estimates uncertainty with Monte Carlo (MC) Dropout, and adds an experimental **Reliability Evidence Layer**. The layer shows which evidence is available for a prediction and when closer human review may be appropriate.

> **Research prototype.** TumorGuard AI is not clinically validated, does not diagnose disease, and must not be used for clinical decisions. Its uncertainty and reliability outputs are experimental and are not proof that a prediction is correct.

---

## Problem statement

Brain tumour segmentation from MRI is a complex task. AI models can help identify tumour regions, but their predictions may contain errors. A segmentation mask can look convincing without necessarily being reliable. Many segmentation tools show the predicted result but give limited insight into uncertainty or into when additional review may be appropriate.

The research question behind this project is: **how can AI segmentation outputs be made easier to inspect, and their limitations more transparent?**

## Proposed solution

TumorGuard AI combines three things:

1. **Segmentation.** A 2D U-Net predicts a binary whole-tumour mask from FLAIR, T1CE and T2 slices.
2. **Uncertainty.** Ten stochastic forward passes with dropout enabled give a per-pixel variance map and summary uncertainty values.
3. **Reliability Evidence Layer.** Each evidence signal is listed with an explicit validity state. A separate, conservative review recommendation explains why a case may deserve a closer look.

## Key features

- Upload of co-registered FLAIR, T1CE and T2 NIfTI volumes (`.nii` / `.nii.gz`). Headers are checked in the browser, and the server validates shape, voxel spacing, affine alignment and finite values.
- Automatic slice selection, or an explicit slice chosen by the user.
- Segmentation from the mean of 10 MC Dropout passes, plus the mask size of a single deterministic pass for comparison.
- Three aligned panels: FLAIR slice, segmentation overlay and MC Dropout uncertainty heatmap.
- A heuristic reliability score and verdict (`HIGH RELIABILITY`, `MODERATE RELIABILITY`, `LOW RELIABILITY`, `MANUAL REVIEW REQUIRED`), with a per-signal breakdown.
- A **Reliability Evidence Layer** with four validity states:
  - `AVAILABLE_AND_VALID`
  - `AVAILABLE_BUT_INVALID`
  - `MISSING_OR_UNAVAILABLE`
  - `NOT_YET_SCIENTIFICALLY_VALIDATED`

  It gives a review recommendation (`HUMAN_REVIEW_RECOMMENDED`, `NO_ADDITIONAL_REVIEW_FLAGGED` or `ASSESSMENT_UNAVAILABLE`) with reasons. Missing or invalid evidence is never reported as safe.
- Downloadable HTML and JSON reports, and session-only case history (no images or file names are stored).
- Safe JSON error responses with no tracebacks or file paths, and a configurable upload size limit (default 150 MB).

## How TumorGuard AI works

```text
FLAIR, T1CE, T2 (.nii / .nii.gz)
        │  browser header check (frontend/src/nifti.js)
        ▼
POST /predict  ─────────────────────────────── server.py (Flask)
        │  validate_modality_volumes   shape, spacing, affine, finite values
        │  slice selection             automatic (FLAIR heuristic) or explicit slice_index
        │  load_and_preprocess_mri     resize to 128×128, z-score per slice, channels [FLAIR, T1CE, T2]
        │  run_deterministic_inference single eval pass (reported as deterministic_prediction)
        │  run_mc_dropout_inference    10 passes → mean mask, variance, entropy, pass difference
        │  calculate_similarity_score  familiarity vs. training reference statistics
        │  reliability score + verdict heuristic weights and cut-offs (tumorguard_core.py)
        │  rendering.py                aligned PNG panels
        │  reliability_evidence.py     evidence report (does not change mask, score or verdict)
        ▼
JSON response → React frontend (Results and Evidence pages)
```

The full request and response contract is documented in [`docs/INFERENCE_CONTRACT.md`](docs/INFERENCE_CONTRACT.md).

## Technology stack

| Layer | Technology |
|---|---|
| Model | Python, PyTorch, 2D U-Net with dropout (p = 0.3) |
| MRI I/O and processing | nibabel, NumPy, scikit-image, SciPy |
| Backend | Flask, flask-cors |
| Frontend | React 19, Vite |
| Tests | pytest (backend), Vitest + Testing Library (frontend), Playwright script for browser checks |
| Data | BraTS 2020 (not included in this repository) |

## Project structure

```text
TumorGuard-AI/
├── server.py                 Flask API: /health, /predict
├── tumorguard_core.py        model, preprocessing, MC Dropout, familiarity, reliability score, checkpoint loader
├── input_validation.py       validation summary, warnings and error codes for the API
├── rendering.py              aligned PNG rendering of the three result panels
├── reliability_evidence.py   Reliability Evidence Layer
├── models/
│   ├── tumorguard_model.pt     trained checkpoint (committed, about 1.9 MB)
│   ├── training_stats.json     familiarity reference statistics
│   ├── training_history.json   historical training log
│   └── phase9_evaluation.json  historical evaluation record
├── frontend/                 React + Vite application (package.json is here, not at the root)
├── tests/                    pytest suites
├── docs/                     audit, defect register, baseline, inference contract, release notes
├── train.py, train_cached.py, prepare_dataset_fast.py, evaluate_phase9.py
│                             training, data preparation and evaluation scripts
└── requirements.txt, requirements-dev.txt
```

## Installation

**Requirements**

- Python 3. The test suite was run with Python 3.13; `requirements.txt` requires `torch>=2.0.0`. No exact minimum version is pinned.
- Node.js with npm. The frontend was built and tested with Node 22.
- PyTorch is a large download. A GPU is not required; the server loads the model on the CPU.

**1. Clone and choose the branch**

```bash
git clone https://github.com/medhachoudhari/TumorGuard-AI.git
cd TumorGuard-AI
git checkout release-fix/ui-evidence-layer
```

**2. Backend environment**

```bash
python -m venv .venv
# Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
pip install -r requirements.txt          # or requirements-dev.txt to include pytest
```

**3. Frontend dependencies**

```bash
cd frontend
npm ci
cd ..
```

**Model checkpoint.** The server loads `models/tumorguard_model.pt` and `models/training_stats.json` at startup, and fails with an error if either is missing. Both are committed on this branch. The checkpoint's SHA-256 is `eb50ced634beb064e10d44d8586f555a0207a2b1915157819137c3b268c636fe`. Note that `.gitignore` lists `*.pt`, so a newly trained checkpoint will not be committed automatically.

## Running the application

Open two terminals from the repository root.

**Backend (Flask, http://127.0.0.1:5000)**

```bash
python server.py
```

Check it with `http://127.0.0.1:5000/health`, which should report `"model_loaded": true`.

**Frontend (Vite)**

```bash
cd frontend
npm run dev          # development server on http://localhost:3000
```

Or build and preview the production bundle:

```bash
cd frontend
npm run build
npm run preview      # http://localhost:4173 by default
```

**Configuration (optional)**

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `VITE_BACKEND_URL` | frontend, at build/dev time | `http://127.0.0.1:5000` | API address used by the browser |
| `TUMORGUARD_MAX_CONTENT_LENGTH` | backend | `157286400` (150 MB) | maximum upload size in bytes |

The backend currently allows cross-origin requests from any origin (`CORS(app)`) and binds to `127.0.0.1`. It is intended for local use only.

## Using the application

1. Open the frontend and choose **Get started**.
2. Add one FLAIR, one T1CE and one T2 volume from the same case. Do not upload the segmentation (`*_seg.nii`); the API rejects it. **Start analysis** is enabled once all checks pass.
3. Wait for the real inference to finish. The page shows genuine upload progress; the server does not report separate stages.
4. On **Results**, compare the FLAIR slice, segmentation overlay and uncertainty heatmap. Use the slice controls to re-run inference on another slice, and open **Score details** for the per-signal breakdown.
5. Open **View evidence layer** to see the review recommendation, its reasons, each signal's validity state, research limitations and model provenance.
6. Optionally download the HTML or JSON report.

MC Dropout is random, so running the same case again can give a slightly different score and, near a cut-off, a different verdict. Send one request at a time: concurrent requests share one model object and can interfere with each other's dropout state (a known open issue, not yet fixed).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests                 # backend: unit tests and API tests on synthetic volumes
cd frontend
npm test                               # Vitest (network calls are mocked)
npm run lint
npm run build
```

Tests that need genuine BraTS data (`tests/test_real_data_e2e.py` and parts of `tests/test_spatial_validation.py`) are skipped unless the case is placed at `BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_001/`. Synthetic test volumes only check software behaviour, not segmentation quality.

## Model evaluation (historical)

The following values come from `models/phase9_evaluation.json`. They are **historical, reported results** and have **not been independently reproduced** in a documented, hash-linked evaluation.

| Metric | Value |
|---|---|
| Validation slices | 398 (tumour-containing slices, patients 031–040) |
| Mean Dice | 0.7863 |
| Median Dice | 0.8762 |
| Mean IoU | 0.6961 |
| Mean precision | 0.8364 |
| Mean recall | 0.8146 |

Limitations of these numbers:

- They are slice-level metrics over tumour-containing slices only, not patient-level or whole-volume results.
- They come from a single deterministic pass. The API mask is the MC Dropout mean.
- The validation patients were also used for model selection, so this is not an independent test set.
- The record contains no checkpoint hash. Its epoch (20) and validation loss match the committed checkpoint's metadata, but a byte-level link has not been demonstrated (see `docs/BASELINE.md`).
- No error-detection metrics (for example AUROC for flagging wrong segmentations) have been produced yet.

## Research limitations and responsible use

- **Not clinically validated.** Not a medical device, and not for diagnosis or treatment decisions.
- **Uncertainty is an estimate.** It does not prove that a pixel or a case is right or wrong.
- **The reliability score is a heuristic.** Its weights and cut-offs have not been calibrated against segmentation errors.
- **The Reliability Evidence Layer is experimental.** No signal has yet been shown to predict segmentation errors, and "No additional review flagged" is not evidence of correctness.
- **The familiarity reference is limited.** It was built from only four training patients.
- **Known open issues** are listed in [`docs/DEFECT_REGISTER.md`](docs/DEFECT_REGISTER.md), for example the first-versus-last-pass difference statistic (D-15), whole-slice averaging of uncertainty (D-21) and open CORS (D-18b). Concurrent requests can also interfere through the shared model object; this is not yet fixed.
- **Real-MRI behaviour of this version is not yet verified.** It has been tested on synthetic volumes only.
- **Data use.** BraTS data is not distributed here. Obtain it from its official source and follow its terms of use.

## Repository information

- Repository: <https://github.com/medhachoudhari/TumorGuard-AI>
- Branch documented here: `release-fix/ui-evidence-layer`, which contains the Flask/PyTorch application described above.
- At the time of writing, the `main` branch contains a different, browser-only prototype that simulates inference and does not include the PyTorch model. Use the branch above for the real application.
- Detailed engineering notes are in [`docs/`](docs/).

## Demo video

**Demo video:** [TODO: add demo video URL](https://example.com/replace-with-demo-video-url) *(placeholder – the real-MRI demo video has not been published yet)*

---

TumorGuard AI is a student research project. Feedback and issue reports are welcome through GitHub.
