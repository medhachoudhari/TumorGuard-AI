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
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import tumorguard_core
import rendering
import input_validation
import reliability_evidence

app = Flask(__name__)
CORS(app)

DEFAULT_MAX_CONTENT_LENGTH = 150 * 1024 * 1024  # 150 MB (Phase 2B)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("TUMORGUARD_MAX_CONTENT_LENGTH", DEFAULT_MAX_CONTENT_LENGTH)
)
NUM_MC_PASSES = 10
SEGMENTATION_THRESHOLD = 0.5
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tumorguard")

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


def max_upload_mb():
    return app.config.get("MAX_CONTENT_LENGTH", DEFAULT_MAX_CONTENT_LENGTH) // (1024 * 1024)


def error_response(code, message, status=400):
    """Safe error body: status, a stable code and a message with no paths or tracebacks."""
    return jsonify({"status": "error", "error": message, "error_code": code}), status


@app.errorhandler(413)
def request_entity_too_large(_error):
    logger.warning("Upload rejected: payload exceeds configured maximum size of %d MB", max_upload_mb())
    return error_response("UPLOAD_TOO_LARGE", f"Upload payload exceeds maximum allowed size ({max_upload_mb()} MB).", 413)


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
        "limits": {"max_upload_mb": max_upload_mb()},
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

        raw_slice = request.form.get('slice_index')
        if raw_slice is None:
            raw_slice = request.args.get('slice_index')

        explicit_slice = None
        if raw_slice is not None and str(raw_slice).strip() != '':
            try:
                explicit_slice = int(str(raw_slice).strip())
            except ValueError:
                return error_response("INVALID_SLICE_INDEX",
                                      f"Invalid slice_index '{raw_slice}': slice_index must be an integer.")

        # 2. Process in a temporary directory (client file names are never used on disk)
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = {}
            for name, f in files.items():
                ext = '.nii.gz' if (f.filename or '').lower().endswith('.nii.gz') else '.nii'
                path = os.path.join(temp_dir, f"{name}{ext}")
                f.save(path)
                paths[name] = path

            # 2b. Validate 3D NIfTI, shapes, voxel spacing and affines (Phase 2B)
            try:
                images = tumorguard_core.validate_modality_volumes(paths)
            except tumorguard_core.SpatialValidationError as sve:
                logger.warning("Spatial validation rejected: %s", sve)
                return error_response(input_validation.error_code_for(sve), str(sve))
            except Exception as e:
                logger.warning("NIfTI validation failed: %s", e)
                return error_response("INVALID_NIFTI",
                                      "Uploaded scan files could not be validated as valid 3D NIfTI volumes.")
            input_summary = input_validation.summarize_volumes(images)
            num_slices = input_summary["num_slices"]

            # 3. Slice selection (inference only - no ground truth)
            if explicit_slice is not None:
                if explicit_slice < 0 or explicit_slice >= num_slices:
                    return error_response(
                        "INVALID_SLICE_INDEX",
                        f"slice_index {explicit_slice} is out of range. Must be between 0 and {num_slices - 1}.")
                best_slice = explicit_slice
                selection_mode = "explicit"
            else:
                try:
                    best_slice = tumorguard_core.select_inference_slice(paths['flair'])
                    selection_mode = "automatic_flair_heuristic"
                except Exception as e:
                    logger.error("Slice selection error: %s", e, exc_info=True)
                    return error_response("SLICE_SELECTION_FAILED",
                                          "Failed to determine an informative slice from the FLAIR scan.")

            # 4. Preprocessing
            try:
                input_tensor = tumorguard_core.load_and_preprocess_mri(
                    paths['flair'], paths['t1ce'], paths['t2'], slice_idx=best_slice
                ).unsqueeze(0)  # (1, 3, H, W)
            except Exception as e:
                logger.error("Preprocessing error: %s", e, exc_info=True)
                return error_response("PREPROCESSING_FAILED",
                                      "Failed to preprocess MRI scan slices for model inference.")

            # 5a. Deterministic baseline inference (matches evaluation semantics)
            _det_probs, det_binary = tumorguard_core.run_deterministic_inference(
                model, input_tensor, threshold=SEGMENTATION_THRESHOLD)
            det_tumor_pixels = int(det_binary.sum().item())

            # 5b. MC Dropout inference (stochastic consensus and uncertainty metrics)
            mean_pred, var_map, unc_score, max_diff, entropy = \
                tumorguard_core.run_mc_dropout_inference(model, input_tensor, num_passes=NUM_MC_PASSES)

            if not (torch.isfinite(mean_pred).all() and torch.isfinite(var_map).all()):
                return error_response("NON_FINITE_OUTPUT",
                                      "The model produced non-finite values for this input.", 422)

            pred_binary = (mean_pred > SEGMENTATION_THRESHOLD).float()
            mc_tumor_pixels = int(pred_binary.sum().item())

            # 6. Familiarity, reliability score and verdict (legacy semantics unchanged)
            familiarity = tumorguard_core.calculate_similarity_score(input_tensor, training_stats)
            reliability_score, verdict = tumorguard_core.calculate_reliability_and_verdict(
                familiarity, entropy, variance=unc_score, max_diff=max_diff)
            components = tumorguard_core.reliability_components(familiarity, entropy, unc_score, max_diff)

            # 7. Visualisations (shared display transform keeps panels aligned)
            flair_slice_np = input_tensor[0, 0].cpu().numpy()
            pred_mask_np = pred_binary[0, 0].cpu().numpy()
            var_map_np = var_map[0, 0].cpu().numpy()
            mri_b64, seg_b64, unc_b64, image_info = generate_visualizations(
                flair_slice_np, pred_mask_np, var_map_np)

            # 8. Reliability Evidence Layer (additive; never alters the fields above)
            evidence = reliability_evidence.build_evidence_report(
                input_summary=input_summary,
                familiarity=familiarity,
                familiarity_reference_available=training_stats.get("familiarity_reference") is not None,
                entropy=entropy, variance=unc_score, max_diff=max_diff,
                mask_pixels=mc_tumor_pixels, reliability_score=reliability_score, verdict=verdict,
                provenance=MODEL_PROVENANCE)

            def safe_num(v):
                if v is None:
                    return None
                try:
                    f = float(v)
                    return f if math.isfinite(f) else None
                except (TypeError, ValueError):
                    return None

            response = {
                "status": "success",
                "slice_index": int(best_slice),
                "slice_selection_mode": selection_mode,
                "prediction_method": "mc_dropout_consensus_10_passes",
                "mc_prediction": {
                    "tumor_pixels": mc_tumor_pixels,
                    "threshold": SEGMENTATION_THRESHOLD
                },
                "deterministic_prediction": {
                    "tumor_pixels": det_tumor_pixels,
                    "threshold": SEGMENTATION_THRESHOLD,
                    "method": "single_pass_eval"
                },
                "familiarity_score": safe_num(familiarity),
                "uncertainty_score": safe_num(unc_score),
                "predictive_entropy": safe_num(entropy),
                "max_probability_difference": safe_num(max_diff),
                "reliability_score": float(reliability_score),
                "verdict": verdict,
                "images": {
                    "mri": mri_b64,
                    "segmentation_overlay": seg_b64,
                    "uncertainty_heatmap": unc_b64
                },
                "disclaimer": "This is a research prototype for AI segmentation reliability visualization. It does NOT diagnose cancer. It does NOT replace radiologist review. The verdict describes model reliability, not patient health status.",
                # Additive fields (release: UI + evidence layer)
                "num_slices": int(num_slices),
                "mask_pixels": mc_tumor_pixels,
                "prediction_settings": {
                    "num_passes": NUM_MC_PASSES,
                    "threshold": SEGMENTATION_THRESHOLD,
                    "mc_operator": ">",
                    "deterministic_operator": ">=",
                },
                "reliability_breakdown": components,
                "input_validation": input_summary,
                "image_info": image_info,
                "reliability_evidence_layer": evidence,
            }

            return jsonify(response)

    except HTTPException:
        raise  # e.g. 413 from MAX_CONTENT_LENGTH; handled by the JSON error handlers
    except Exception as e:
        # Details go to the server log only; clients receive a generic message.
        logger.error("Unhandled internal server error during inference: %s", e, exc_info=True)
        return error_response("INTERNAL_ERROR",
                              "An internal error occurred during inference. Please consult server logs for diagnostic details.", 500)


if __name__ == '__main__':
    # Run locally
    app.run(host='127.0.0.1', port=5000, debug=False)
