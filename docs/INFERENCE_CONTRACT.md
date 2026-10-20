# Inference contract (as implemented)

Verified by reading `tumorguard_core.py` and `server.py` at the commit on this branch. Items marked **(observed, not validated)** were confirmed in code but not shown to be scientifically appropriate. Nothing here has been checked against real BraTS data in this review.

## Input

| Item | Implementation |
|---|---|
| Modalities and order | `REQUIRED_MODALITIES = ["flair", "t1ce", "t2"]` -> channels 0, 1, 2 (`load_and_preprocess_mri`) |
| Upload fields | `/predict` multipart fields `flair`, `t1ce`, `t2`; a `seg` field is rejected with HTTP 400 |
| File names | must end with `.nii` or `.nii.gz` (case-sensitive; `.NII` is rejected). The check is on the name only, not file content |
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

`/predict` returns JSON: `status`, `slice_index`, `familiarity_score`, `uncertainty_score`, `predictive_entropy`, `max_probability_difference`, `reliability_score`, `verdict`, `images` (`mri`, `segmentation_overlay`, `uncertainty_heatmap` as base64 PNG data URIs) and `disclaimer`.

**Alignment:** the mask, overlay and heatmap are in the 128x128 resized-slice space. They are not mapped back to the original in-plane resolution. Arrays are plotted as stored (no transpose or flip), so display orientation relative to a radiological view is unverified.

## Empty mask

If no pixel exceeds 0.5 the response is still HTTP 200 with an all-zero overlay; there is no explicit "no tumour predicted" flag or special reliability handling. Dice-style metrics treat empty-vs-empty as perfect in `calculate_dice_score` (returns 1.0) but `evaluate_phase9.py` uses `smooth=1e-6` (also near 1.0 for empty-vs-empty). 19 of 398 historical validation slices were predicted empty.

## Invalid input handling

| Case | Behaviour |
|---|---|
| Missing modality | HTTP 400 |
| `seg` supplied | HTTP 400 |
| Empty file name / wrong extension | HTTP 400 |
| Unreadable FLAIR | slice selection falls back to 75; preprocessing then fails -> HTTP 400 |
| Non-3D, mismatched shapes across modalities, or very large files | **not validated** (shapes are resized independently; no `MAX_CONTENT_LENGTH`) (D-17) |
| Unexpected error | HTTP 500 with `str(exception)` in the body and a traceback on stderr; exception text may include local paths (D-18) |

CORS is enabled for all origins (`CORS(app)`); the server binds to `127.0.0.1:5000`.
