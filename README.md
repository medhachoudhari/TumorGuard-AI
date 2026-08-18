# TumorGuard AI

**One-liner:** "Know when AI should step back."

TumorGuard is a prototype AI quality-check/reliability layer for brain-tumor MRI segmentation.

**Disclaimer:** This is a research/demo prototype using public data and is NOT a clinical diagnostic device. It evaluates the reliability of the AI segmentation, it does NOT independently diagnose the patient.

## Architecture

```
BraTS MRI
    ↓
tumorguard_core.py (Shared brain: U-Net, Preprocessing, Similarity, Scoring)
    ↓
U-Net segmentation model
    ↓
MC Dropout uncertainty
    ↓
training-data similarity/familiarity
    ↓
reliability score
    ↓
SAFE / NEEDS ATTENTION / MANUAL REVIEW REQUIRED
    ↓
server.py
    ↓
web dashboard
```

## Dataset

BraTS 2020 Training + Validation dataset.
Expected Modalities (NIfTI format):
- FLAIR
- T1CE (T1Gd)
- T2

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model on a small subset:
   ```bash
   python train.py
   ```
   *(This populates `models/tumorguard_model.pth` and `models/training_stats.json`)*

3. Run the live inference server:
   ```bash
   python server.py
   ```
