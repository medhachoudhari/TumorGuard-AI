# Inference contract (as implemented)

Verified by reading `tumorguard_core.py` and `server.py` at the commit on this branch. Items marked **(observed, not validated)** were confirmed in code but not shown to be scientifically appropriate. Nothing here has been checked against real BraTS data in this review.

## Input

| Item | Implementation |
|---|---|
| Modalities and order | `REQUIRED_MODALITIES = ["flair", "t1ce", "t2"]` -> channels 0, 1, 2 (`load_and_preprocess_mri`) |
| Upload fields | `/predict` multipart fields `flair`, `t1ce`, `t2`; a `seg` field is rejected with HTTP 400 |
| File names | must end with `.nii` or `.nii.gz` (case-insensitive). The name check is not a content check; content is validated separately (see below) |
| Volume layout | arrays read with `nibabel` `get_fdata()` (float64), NaN -> 0. The third array axis is treated as the axial slice axis. No re-orientation, spacing or affine handling |

## Preprocessing (per modality, 2D)

1. **Slice selection (live):** `select_inference_slice(flair)` = index of the axial slice with the most non-zero FLAIR voxels (argmax over axis 2); falls back to 75 if the FLAIR cannot be read. The same index is then used for all three modalities.
2. **Slice extraction:** `img[:, :, slice_idx]`. If `slice_idx >= img.shape[2]` for a modality, the middle slice of that modality is used **silently** (observed, not validated).
3. **Resize:** `skimage.transform.resize` to `(128, 128)`, `mode="constant"`, `anti_aliasing=True`, default (bilinear) interpolation. The aspect ratio is not preserved and there is **no cropping**.
4. **Normalization (`normalize_intensity`):** brain mask = pixels `> 0` after resizing; z-score using the mean and (population) std of masked pixels; background pixels set to 0. If the mask is empty the slice is returned unchanged; if std is 0 the values are not rescaled but background is still zeroed.
5. **Stack:** `float32` tensor `(3, 128, 128)`, then `unsqueeze(0)` -> `(1, 3, 128, 128)`.

Training and evaluation caches (`prepare_dataset_fast.py`) use the same resize and z-score but select slices differently: tumour-containing slices only, evenly spaced, using the ground-truth label (`select_best_tumor_slice` uses the max-tumour slice). **The live slice-selection rule is not the one the model was evaluated on** (D-13).

## Labels (training / evaluation only)

`load_segmentation_mask`: any BraTS label `> 0` (1, 2, 4) -> 1.0; resized with `order=0`, `anti_aliasing=False`; shape `(1, 128, 128)`. Whole-tumour binary target.

## Model

`TumorGuardUNet(in_channels=3, out_channels=1, dropout_rate=0.3)`: 2D U-Net, two pooling levels (32 and 64 channels, 128-channel bottleneck), `nn.Dropout(p=0.3)` in each ConvBlock and after the bottleneck, final 1x1 conv. Output is **logits** of shape `(B, 1, 128, 128)`; sigmoid is applied outside the model. Checkpoint: see `docs/BASELINE.md`.

## Segmentation threshold and the two inference modes

| Path | Computation | Threshold |
|---|---|---|
| API (`server.py`) | 10 MC Dropout passes (dropout in train mode, BatchNorm in eval mode); `mean_pred` = mean of sigmoid probabilities | `mean_pred > 0.5` |
| Evaluation (`evaluate_phase9.py`) | single deterministic pass, `model.eval()` | `sigmoid(logits) >= 0.5` |

Consequences: the metrics in `docs/BASELINE.md` describe the deterministic path, not the MC-mean mask the API returns, and the comparison operators differ (D-10).

## Uncertainty and reliability signals (all scalars over the full 128x128 image, including background)

* `uncertainty_score`: mean over all pixels of the per-pixel variance across the 10 passes.
* `max_probability_difference`: `max |p_first_pass - p_last_pass|`. The docstring says "any two passes"; the code compares only the first and last pass (D-15).
* `predictive_entropy`: mean binary entropy of `mean_pred`, probabilities clamped to `[1e-7, 1-1e-7]`.
* `familiarity_score`: `exp(-mean(|z|))` over 18 intensity statistics (mean, std, min, max, non-zero fraction, p75 per modality) against `models/training_stats.json`; the reference was built from only 4 patients (D-07). If the reference is missing the function returns a neutral 0.5 with a warning (D-16).
* `reliability_score` = 0.40 x familiarity + 0.20 x entropy confidence + 0.20 x variance confidence + 0.20 x max-diff confidence, clipped to [0, 1]. It is a **heuristic score, not a calibrated probability**.
* Verdict cut-offs (inclusive lower bounds): `>= 0.70` HIGH RELIABILITY, `>= 0.50` MODERATE RELIABILITY, `>= 0.30` LOW RELIABILITY, otherwise MANUAL REVIEW REQUIRED. Any non-finite or unparsable input -> score 0.0, MANUAL REVIEW REQUIRED.

## Output

`/predict` returns JSON. Original fields (unchanged): `status`, `slice_index`, `familiarity_score`, `uncertainty_score`, `predictive_entropy`, `max_probability_difference`, `reliability_score`, `verdict`, `images` (`mri`, `segmentation_overlay`, `uncertainty_heatmap` as base64 PNG data URIs) and `disclaimer`.

Phase 2B fields (preserved exactly as implemented there): `slice_selection_mode` (`explicit` | `automatic_flair_heuristic`), `prediction_method` (`mc_dropout_consensus_10_passes`), `mc_prediction` and `deterministic_prediction` (`tumor_pixels`, `threshold`, `method`). Fields added in this release (additive): `num_slices`, `mask_pixels` (= MC-mean mask pixels), `prediction_settings` (10 passes, threshold 0.5, MC operator `>`, deterministic operator `>=`), `reliability_breakdown` (per-signal confidence, weight, contribution), `input_validation` (shape, slices, affine consistency, non-finite fraction, warnings), `image_info` (width, height, orientation, heatmap scale) and `reliability_evidence_layer` (see `reliability_evidence.py`). The optional form field `slice_index` selects an explicit slice (0 to num_slices-1). `/health` additionally returns `limits.max_upload_mb`.

**Rendering (`rendering.py`):** all three panels use one display transform (`np.rot90`, 90 degrees counter-clockwise) and are written at the native 128 x 128 model resolution with no resampling or smoothing. FLAIR is windowed to the 1st-99th percentile of non-zero pixels (background black). The overlay is the binary mask in red at 50% alpha. The heatmap is the per-pixel variance scaled linearly from 0 to this slice's maximum (returned as `image_info.heatmap_max_variance`) with the `inferno` colour map, alpha proportional to the scaled value. Left/right convention assumes standard BraTS orientation and is not verified from the affine.

**Alignment:** the mask, overlay and heatmap are in the 128 x 128 resized-slice space. They are not mapped back to the original in-plane resolution.

## Empty mask

If no pixel exceeds 0.5 the response is still HTTP 200 with an all-zero overlay and `mask_pixels = 0`. The evidence layer treats this as a review trigger. 19 of 398 historical validation slices were predicted empty.

## Input validation (`input_validation.py`, server side)

| Case | Behaviour |
|---|---|
| Missing modality / `seg` supplied / empty name / wrong extension (case-insensitive) | HTTP 400 with `error_code` (`MISSING_MODALITY`, `SEG_NOT_ALLOWED`, `EMPTY_FILE`, `INVALID_FILE_TYPE`) |
| Unreadable NIfTI, not 3D, non-positive dimensions or voxel spacing (Phase 2B `validate_modality_volumes`) | HTTP 400 (`INVALID_NIFTI`, `INVALID_SHAPE`) |
| Different shapes, voxel spacings or affines across modalities | HTTP 400 `SPATIAL_MISMATCH` |
| All-zero volume; any non-finite voxel | HTTP 400 `EMPTY_VOLUME`, `NON_FINITE_DATA` |
| Invalid `slice_index` | HTTP 400 `INVALID_SLICE_INDEX` |
| Upload larger than `TUMORGUARD_MAX_CONTENT_LENGTH` bytes (default 150 MB total) | HTTP 413 `UPLOAD_TOO_LARGE` |
| Non-finite model output | HTTP 422 `NON_FINITE_OUTPUT` |
| Unexpected error | HTTP 500 `INTERNAL_ERROR` with a generic message; details only in the server log |
| Third axis not smaller than the in-plane axes, or strongly non-square in-plane | accepted, with `input_validation.warnings` (possible non-axial storage; resizing would distort the image) |

Every error body is `{status: "error", error, error_code}` (`error_code` is additive to the Phase 2B body). Error bodies never contain file-system paths, exception text or tracebacks (tested). Client file names are never used on disk. CORS is enabled for all origins (`CORS(app)`); the server binds to `127.0.0.1:5000`.

Not validated: voxel spacing, orientation labels in the affine, intensity ranges, and whether the files are actually FLAIR/T1CE/T2 in that order.
