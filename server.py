# ==============================================================================
# TumorGuard AI - Live Inference Server (Phase 6)
# ==============================================================================
# A Flask API that serves the TumorGuard AI prototype.
# Accepts uploaded FLAIR, T1CE, and T2 scans, runs MC Dropout inference via
# tumorguard_core.py, calculates reliability, and returns the verdict to the dashboard.
# ==============================================================================

import os
import json
import base64
import tempfile
import io
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
import tumorguard_core

app = Flask(__name__)
CORS(app)

# ==============================================================================
# STARTUP INITIALIZATION
# ==============================================================================

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CKPT_PATH = os.path.join(MODELS_DIR, "tumorguard_model.pt")
STATS_PATH = os.path.join(MODELS_DIR, "training_stats.json")

print("\n" + "=" * 60)
print("Starting TumorGuard AI Inference Server...")

# 1. Load Model
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

model = tumorguard_core.TumorGuardUNet()
tumorguard_core.load_model_checkpoint(model, CKPT_PATH, map_location="cpu")
model.eval() # Base state; MC Dropout will switch Dropouts to train mode.
print(f"[*] Checkpoint loaded: {CKPT_PATH}")

# 2. Load Stats
if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(f"Training stats not found: {STATS_PATH}")

with open(STATS_PATH, "r") as f:
    training_stats = json.load(f)
print(f"[*] Training stats loaded: {STATS_PATH}")
print(f"[*] Expected modalities: {tumorguard_core.REQUIRED_MODALITIES}")

print("=" * 60 + "\n")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def array_to_base64_png(fig):
    """Converts a matplotlib figure to a base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img_bytes = buf.getvalue()
    b64_str = base64.b64encode(img_bytes).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{b64_str}"

def generate_visualizations(flair_slice, pred_mask, var_map):
    """Generates base64 PNGs for the original MRI, segmentation overlay, and uncertainty heatmap."""
    
    # 1. Original MRI (FLAIR)
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.imshow(flair_slice, cmap='gray')
    ax1.axis('off')
    mri_b64 = array_to_base64_png(fig1)

    # 2. Segmentation Overlay (Binary Mask over FLAIR)
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.imshow(flair_slice, cmap='gray')
    # Create a red overlay for the mask
    mask_rgba = np.zeros(pred_mask.shape + (4,))
    mask_rgba[pred_mask == 1] = [1, 0, 0, 0.5] # Red with 50% opacity
    ax2.imshow(mask_rgba)
    ax2.axis('off')
    seg_b64 = array_to_base64_png(fig2)

    # 3. Uncertainty Heatmap (Variance over FLAIR)
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    ax3.imshow(flair_slice, cmap='gray')
    # Plot variance using 'hot' colormap, masking out zero-variance areas
    var_masked = np.ma.masked_where(var_map == 0, var_map)
    heatmap = ax3.imshow(var_masked, cmap='hot', alpha=0.6)
    ax3.axis('off')
    unc_b64 = array_to_base64_png(fig3)

    return mri_b64, seg_b64, unc_b64

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "training_stats_loaded": training_stats is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Validate Uploaded Files
        if 'flair' not in request.files or 't1ce' not in request.files or 't2' not in request.files:
            return jsonify({"error": "Missing required modalities. Expected exactly: 'flair', 't1ce', 't2'."}), 400
        
        if 'seg' in request.files:
            return jsonify({"error": "Segmentation masks ('seg') are not allowed for live inference."}), 400

        files = {
            'flair': request.files['flair'],
            't1ce': request.files['t1ce'],
            't2': request.files['t2']
        }

        for name, f in files.items():
            if f.filename == '':
                return jsonify({"error": f"Empty file submitted for '{name}'."}), 400
            if not (f.filename.endswith('.nii') or f.filename.endswith('.nii.gz')):
                return jsonify({"error": f"Invalid file type for '{name}'. Expected .nii or .nii.gz."}), 400

        # 2. Process in Temporary Directory
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = {}
            for name, f in files.items():
                ext = '.nii.gz' if f.filename.endswith('.nii.gz') else '.nii'
                path = os.path.join(temp_dir, f"{name}{ext}")
                f.save(path)
                paths[name] = path

            # 3. Slice Selection (Inference only - no GT)
            try:
                best_slice = tumorguard_core.select_inference_slice(paths['flair'])
            except Exception as e:
                return jsonify({"error": f"Failed to read FLAIR scan for slice selection: {str(e)}"}), 400

            # 4. Preprocessing
            try:
                input_tensor = tumorguard_core.load_and_preprocess_mri(
                    paths['flair'], paths['t1ce'], paths['t2'], slice_idx=best_slice
                ).unsqueeze(0) # (1, 3, H, W)
            except Exception as e:
                return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 400

            # 5. MC Dropout Inference
            mean_pred, var_map, unc_score, max_diff, entropy = \
                tumorguard_core.run_mc_dropout_inference(model, input_tensor, num_passes=10)

            # Generate binary mask from mean prediction
            pred_binary = (mean_pred > 0.5).float()

            # 6. Familiarity Score
            familiarity = tumorguard_core.calculate_similarity_score(input_tensor, training_stats)

            # 7. Reliability Score & Verdict
            reliability_score, verdict = tumorguard_core.calculate_reliability_and_verdict(familiarity, entropy, variance=unc_score, max_diff=max_diff)

            # 8. Visualizations
            # Extract the raw normalized FLAIR slice from the input tensor for background
            flair_slice_np = input_tensor[0, 0].cpu().numpy() # Channel 0 is FLAIR
            pred_mask_np = pred_binary[0, 0].cpu().numpy()
            var_map_np = var_map[0, 0].cpu().numpy()

            mri_b64, seg_b64, unc_b64 = generate_visualizations(flair_slice_np, pred_mask_np, var_map_np)

            # 9. JSON Response
            response = {
                "status": "success",
                "slice_index": int(best_slice),
                "familiarity_score": float(familiarity),
                "uncertainty_score": float(unc_score),
                "predictive_entropy": float(entropy),
                "max_probability_difference": float(max_diff),
                "reliability_score": float(reliability_score),
                "verdict": verdict,
                "images": {
                    "mri": mri_b64,
                    "segmentation_overlay": seg_b64,
                    "uncertainty_heatmap": unc_b64
                },
                "disclaimer": "This is a research prototype for AI segmentation reliability visualization. It does NOT diagnose cancer. It does NOT replace radiologist review. The verdict describes model reliability, not patient health status."
            }
            
            return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error during inference: {str(e)}"}), 500

if __name__ == '__main__':
    # Run locally
    app.run(host='127.0.0.1', port=5000, debug=False)



