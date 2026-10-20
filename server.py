# ==============================================================================
# TumorGuard AI - Live Inference Server (Phase 6)
# ==============================================================================
# A Flask API that serves the TumorGuard AI prototype.
# Accepts uploaded FLAIR, T1CE, and T2 scans, runs MC Dropout inference via
# tumorguard_core.py, calculates reliability, and returns the verdict to the dashboard.
# ==============================================================================

import os
import json
import tempfile
import hashlib
import logging
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import tumorguard_core
import rendering
import input_validation
import reliability_evidence

app = Flask(__name__)
CORS(app)

MAX_UPLOAD_MB = int(os.environ.get("TUMORGUARD_MAX_UPLOAD_MB", "512"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
NUM_MC_PASSES = 10
SEGMENTATION_THRESHOLD = 0.5
log = logging.getLogger("tumorguard")

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
checkpoint_meta = tumorguard_core.load_model_checkpoint(model, CKPT_PATH, map_location="cpu")
with open(CKPT_PATH, "rb") as _f:
    CKPT_SHA256 = hashlib.sha256(_f.read()).hexdigest()
model.eval() # Base state; MC Dropout will switch Dropouts to train mode.
print(f"[*] Checkpoint loaded: {CKPT_PATH}")

# 2. Load Stats
if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(f"Training stats not found: {STATS_PATH}")

with open(STATS_PATH, "r") as f:
    training_stats = json.load(f)
print(f"[*] Training stats loaded: {STATS_PATH}")
MODEL_PROVENANCE = {
    "checkpoint_file": os.path.basename(CKPT_PATH),
    "checkpoint_sha256": CKPT_SHA256,
    "checkpoint_epoch": checkpoint_meta.get("epoch"),
    "checkpoint_val_loss": checkpoint_meta.get("val_loss"),
    "architecture": "2D U-Net with MC Dropout (p=0.3)",
    "familiarity_reference_patients": training_stats.get("training_patients"),
    "historical_evaluation": "models/phase9_evaluation.json - historical, reported, not independently "
                             "reproduced; validation patients 031-040 were also used for model selection",
}
print(f"[*] Expected modalities: {tumorguard_core.REQUIRED_MODALITIES}")

print("=" * 60 + "\n")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

generate_visualizations = rendering.generate_visualizations


def error_response(code, message, status=400):
    """Safe error body: a stable code and a message that never contains paths or tracebacks."""
    return jsonify({"error": message, "error_code": code}), status


def parse_slice_index(raw, num_slices):
    """Returns (slice_index or None, mode). Raises InputValidationError if invalid."""
    if raw is None or str(raw).strip() == "":
        return None, "automatic"
    try:
        idx = int(str(raw).strip())
    except ValueError:
        raise input_validation.InputValidationError("INVALID_SLICE_INDEX", "slice_index must be an integer.")
    if not 0 <= idx < num_slices:
        raise input_validation.InputValidationError(
            "INVALID_SLICE_INDEX", f"slice_index must be between 0 and {num_slices - 1}.")
    return idx, "explicit"


@app.errorhandler(413)
def too_large(_e):
    return error_response("UPLOAD_TOO_LARGE", f"Upload exceeds the {MAX_UPLOAD_MB} MB limit.", 413)


@app.errorhandler(404)
def not_found(_e):
    return error_response("NOT_FOUND", "Not found.", 404)


@app.errorhandler(405)
def method_not_allowed(_e):
    return error_response("METHOD_NOT_ALLOWED", "Method not allowed.", 405)


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "training_stats_loaded": training_stats is not None,
        "limits": {"max_upload_mb": MAX_UPLOAD_MB},
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Validate uploaded files
        if 'flair' not in request.files or 't1ce' not in request.files or 't2' not in request.files:
            return error_response("MISSING_MODALITY",
                                  "Missing required modalities. Expected exactly: 'flair', 't1ce', 't2'.")

        if 'seg' in request.files:
            return error_response("SEG_NOT_ALLOWED",
                                  "Segmentation masks ('seg') are not allowed for live inference.")

        files = {
            'flair': request.files['flair'],
            't1ce': request.files['t1ce'],
            't2': request.files['t2']
        }

        for name, f in files.items():
            fname = (f.filename or '').lower()
            if fname == '':
                return error_response("EMPTY_FILE", f"Empty file submitted for '{name}'.")
            if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
                return error_response("INVALID_FILE_TYPE",
                                      f"Invalid file type for '{name}'. Expected .nii or .nii.gz.")

        # 2. Process in a temporary directory (client file names are never used on disk)
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = {}
            for name, f in files.items():
                ext = '.nii.gz' if (f.filename or '').lower().endswith('.nii.gz') else '.nii'
                path = os.path.join(temp_dir, f"{name}{ext}")
                f.save(path)
                paths[name] = path

            # 3. Spatial / content validation
            input_summary = input_validation.inspect_volumes(paths)
            num_slices = input_summary["num_slices"]

            # 4. Slice selection (inference only - no ground truth)
            requested_slice, slice_mode = parse_slice_index(request.form.get('slice_index'), num_slices)
            if requested_slice is None:
                best_slice = tumorguard_core.select_inference_slice(paths['flair'])
            else:
                best_slice = requested_slice

            # 5. Preprocessing
            input_tensor = tumorguard_core.load_and_preprocess_mri(
                paths['flair'], paths['t1ce'], paths['t2'], slice_idx=best_slice
            ).unsqueeze(0)  # (1, 3, H, W)

            # 6. MC Dropout inference
            mean_pred, var_map, unc_score, max_diff, entropy = \
                tumorguard_core.run_mc_dropout_inference(model, input_tensor, num_passes=NUM_MC_PASSES)

            if not (torch.isfinite(mean_pred).all() and torch.isfinite(var_map).all()):
                return error_response("NON_FINITE_OUTPUT",
                                      "The model produced non-finite values for this input.", 422)

            pred_binary = (mean_pred > SEGMENTATION_THRESHOLD).float()

            # 7. Familiarity, reliability score and verdict (legacy semantics unchanged)
            familiarity = tumorguard_core.calculate_similarity_score(input_tensor, training_stats)
            reliability_score, verdict = tumorguard_core.calculate_reliability_and_verdict(
                familiarity, entropy, variance=unc_score, max_diff=max_diff)
            components = tumorguard_core.reliability_components(familiarity, entropy, unc_score, max_diff)

            # 8. Visualisations (shared display transform keeps panels aligned)
            flair_slice_np = input_tensor[0, 0].cpu().numpy()
            pred_mask_np = pred_binary[0, 0].cpu().numpy()
            var_map_np = var_map[0, 0].cpu().numpy()
            mri_b64, seg_b64, unc_b64, image_info = generate_visualizations(
                flair_slice_np, pred_mask_np, var_map_np)
            mask_pixels = int(pred_mask_np.sum())

            # 9. Reliability Evidence Layer (additive; never alters the fields above)
            evidence = reliability_evidence.build_evidence_report(
                input_summary=input_summary,
                familiarity=familiarity,
                familiarity_reference_available=training_stats.get("familiarity_reference") is not None,
                entropy=entropy, variance=unc_score, max_diff=max_diff,
                mask_pixels=mask_pixels, reliability_score=reliability_score, verdict=verdict,
                provenance=MODEL_PROVENANCE)

            # 10. JSON response (original fields preserved)
            response = {
                "status": "success",
                "slice_index": int(best_slice),
                "slice_selection_mode": slice_mode,
                "num_slices": int(num_slices),
                "familiarity_score": float(familiarity),
                "uncertainty_score": float(unc_score),
                "predictive_entropy": float(entropy),
                "max_probability_difference": float(max_diff),
                "reliability_score": float(reliability_score),
                "verdict": verdict,
                "mask_pixels": mask_pixels,
                "prediction_method": {
                    "type": "mc_dropout_mean",
                    "num_passes": NUM_MC_PASSES,
                    "threshold": SEGMENTATION_THRESHOLD,
                    "operator": ">",
                },
                "reliability_breakdown": components,
                "input_validation": input_summary,
                "image_info": image_info,
                "images": {
                    "mri": mri_b64,
                    "segmentation_overlay": seg_b64,
                    "uncertainty_heatmap": unc_b64
                },
                "reliability_evidence_layer": evidence,
                "disclaimer": "This is a research prototype for AI segmentation reliability visualization. It does NOT diagnose cancer. It does NOT replace radiologist review. The verdict describes model reliability, not patient health status."
            }

            return jsonify(response)

    except input_validation.InputValidationError as e:
        return error_response(e.code, e.message, e.status)
    except HTTPException:
        raise  # e.g. 413 from MAX_CONTENT_LENGTH; handled by the JSON error handlers
    except Exception:
        # Details go to the server log only; clients receive a generic message.
        log.exception("Unhandled error during /predict")
        return error_response("INTERNAL_ERROR", "Internal error during inference.", 500)


if __name__ == '__main__':
    # Run locally
    app.run(host='127.0.0.1', port=5000, debug=False)
