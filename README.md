# TumorGuard AI

### **Know when AI should step back.**

TumorGuard AI is a research prototype designed to evaluate the **reliability and uncertainty of brain-tumor MRI segmentation models**.

Instead of treating every AI prediction as equally reliable, TumorGuard combines **segmentation, uncertainty estimation, and training-data familiarity** to identify predictions that may require additional review.

> **Disclaimer:** TumorGuard is a research/demo prototype using publicly available data. It is **not a clinical diagnostic device** and must not be used for medical diagnosis or treatment decisions.

---

## Overview

Medical AI models can produce predictions even when they encounter images that are difficult or unfamiliar.

TumorGuard explores a different approach:

**Don't just ask what the model predicted — ask how reliable that prediction is.**

The system evaluates an MRI segmentation prediction using multiple signals and produces a reliability assessment such as:

* **SAFE**
* **NEEDS ATTENTION**
* **MANUAL REVIEW REQUIRED**

---

## Architecture

```text
                    BraTS MRI
                        │
                        ▼
              tumorguard_core.py
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
      U-Net Segmentation      Image Preprocessing
            │
            ▼
       MC Dropout
    Uncertainty Estimation
            │
            ▼
 Training-data Similarity /
      Familiarity
            │
            ▼
      Reliability Score
            │
            ▼
 ┌───────────────────────────────┐
 │ SAFE                          │
 │ NEEDS ATTENTION               │
 │ MANUAL REVIEW REQUIRED        │
 └───────────────────────────────┘
            │
            ▼
        server.py
            │
            ▼
       Web Dashboard
```

---

## Key Features

* **Brain MRI segmentation** using a U-Net-based deep learning model
* **MC Dropout** for estimating model uncertainty
* **Training-data similarity/familiarity analysis**
* **Reliability scoring**
* **Human-review flagging** for potentially unreliable predictions
* Flask-based inference server
* Modular architecture for experimentation and evaluation

---

## Dataset

TumorGuard uses the **BraTS 2020 dataset** for research and experimentation.

Expected MRI modalities:

* FLAIR
* T1CE / T1Gd
* T2

The dataset itself is **not included in this repository**.

Please obtain the dataset through its official source and follow its applicable terms of use.

---

## Project Structure

```text
TumorGuard/
│
├── tumorguard_core.py
├── train.py
├── server.py
├── calibrate_reliability.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── tumorguard_model.pth
│   └── training_stats.json
│
└── ...
```

---

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd TumorGuard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download and prepare the BraTS dataset according to the expected directory structure used by the training pipeline.

### 4. Train the model

For experimentation, the model can be trained using:

```bash
python train.py
```

This generates the model and training statistics required for inference.

Expected outputs include:

```text
models/tumorguard_model.pth
models/training_stats.json
```

### 5. Run the inference server

```bash
python server.py
```

The server provides the inference functionality used by the TumorGuard interface.

---

## Reliability Approach

TumorGuard explores reliability through multiple signals rather than relying only on the segmentation output.

### 1. Segmentation

A U-Net model generates the brain-tumor segmentation.

### 2. Uncertainty Estimation

MC Dropout is used during inference to estimate prediction uncertainty.

Multiple stochastic forward passes allow the system to examine how consistently the model produces its prediction.

### 3. Training-Data Familiarity

The system also considers how similar an input is to the data encountered during experimentation.

### 4. Reliability Score

These signals are combined to produce a reliability assessment.

The goal is to identify cases where the model's output should receive additional human attention.

---

## Why TumorGuard?

A model producing a prediction does not necessarily mean that the prediction should automatically be trusted.

TumorGuard explores the idea of adding a **reliability layer around medical AI systems** so that uncertain or unfamiliar cases can be surfaced for human review.

---

## Current Status

**Research / Prototype**

TumorGuard is currently an experimental project developed for learning and research in:

* Medical AI
* Computer Vision
* Deep Learning
* Uncertainty Estimation
* AI Reliability

It is not intended for clinical deployment.

---

## Future Work

Potential future improvements include:

* More extensive model evaluation
* Improved uncertainty calibration
* Additional reliability metrics
* Evaluation across different datasets
* Better visualization of uncertainty
* More robust out-of-distribution detection
* Improved inference performance
* Expanded experimentation with medical imaging models

---

## Technologies

* Python
* PyTorch
* NumPy
* Medical Image Processing
* Deep Learning
* U-Net
* MC Dropout
* Flask
* NIfTI MRI data
* BraTS 2020

---

## Disclaimer

TumorGuard AI is a research and educational prototype.

It **does not provide medical diagnoses, treatment recommendations, or clinical decisions**. The system has not been validated or approved for clinical use.

Any results produced by this project should be treated as experimental and should not be used as a substitute for qualified medical expertise.
