# TumorGuard AI

### **Know when AI should step back.**

TumorGuard AI is a medical AI research application designed to evaluate the **reliability and uncertainty of brain-tumor MRI segmentation models**.

Instead of treating every AI prediction as equally reliable, TumorGuard combines **U-Net segmentation, Monte Carlo Dropout uncertainty estimation, and training-distribution familiarity analysis** to identify predictions that require manual human review.

> **Disclaimer:** TumorGuard AI is a research and educational prototype. It is **not a clinical diagnostic device** and must not be used for medical diagnosis or treatment decisions.

---

## Key Features

* **Brain MRI Segmentation**: Analyzes multi-parametric brain MRI scans (FLAIR, T1CE, T2).
* **MC Dropout Uncertainty Quantification**: Runs stochastic forward passes with dropout to generate per-pixel predictive variance maps.
* **Training-Distribution Familiarity Analysis**: Compares input image statistics against the 18-d BraTS 2020 training baseline using normalized L1 deviation and exponential distance decay.
* **Calibrated Multi-Signal Reliability Scoring**: Synthesizes model familiarity (40%), predictive entropy confidence (20%), variance confidence (20%), and maximum pass probability divergence (20%).
* **Actionable Reliability Verdicts**:
  * **SAFE** (Reliability score &ge; 0.70)
  * **NEEDS ATTENTION** (0.50 &le; Reliability score &lt; 0.70)
  * **MANUAL REVIEW REQUIRED** (Reliability score &lt; 0.50)
* **High-Fidelity Visualizations**:
  * Original FLAIR MRI Slice
  * AI Segmentation Overlay (50% opacity red mask)
  * MC Dropout Uncertainty Heatmap (Hot colormap)
* **NIfTI Format Support**: Parses volumetric `.nii` and `.nii.gz` files in real time.
* **Built-in Clinical Benchmark Scenarios**: Includes instant presets (Canonical Glioblastoma, Diffuse Borderline Glioma, and Out-of-Distribution Motion Artifact) for evaluation without large external datasets.

---

## Tech Stack

* **Framework**: React 19 + TypeScript + Vite
* **Styling**: Tailored Futuristic Medical UI (Glassmorphic dark design)
* **Data Processing**: In-browser NIfTI binary decompression & parsing (`pako`)
* **Icons**: `lucide-react`

---

## Development

```bash
# Install dependencies
npm install

# Start development server on port 3000
npm run dev

# Build production bundle
npm run build
```
