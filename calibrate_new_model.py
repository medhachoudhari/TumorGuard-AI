"""
calibrate_reliability.py — Phase 5: Reliability Calibration Experiment
=======================================================================

PURPOSE
-------
Offline evaluation ONLY.  This script:
  1. Loads the trained checkpoint (tumorguard_model.pt).
  2. Loads training_stats.json (which contains the familiarity reference).
  3. Runs the existing preprocessing + MC Dropout pipeline on a set of
     calibration patients drawn from BraTS2020 Training Data.
  4. For each patient collects:
       - familiarity score        (no ground-truth used)
       - MC Dropout uncertainty   (mean pixel variance, raw)
       - predictive entropy       (better-scaled uncertainty proxy)
       - max_diff                 (max prob diff across passes)
       - segmentation Dice        (ground-truth used OFFLINE ONLY)
  5. Saves results to outputs/reliability_calibration.json and .csv
  6. Computes Pearson correlations between signals and Dice.
  7. Prints a summary report.

IMPORTANT
---------
Dice is used here ONLY for offline calibration.
It must NEVER be used during live inference (no ground-truth available).

Run with:
    C:\\Medha\\TumorGuard AI\\.venv\\Scripts\\python.exe calibrate_reliability.py
"""

import os
import json
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import tumorguard_core

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR   = r"C:\Medha\TumorGuard AI\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
MODELS_DIR = r"C:\Medha\TumorGuard AI\models"
OUTPUT_DIR = r"C:\Medha\TumorGuard AI\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_PATH  = os.path.join(MODELS_DIR, "tumorguard_model.pt")
STATS_PATH = os.path.join(MODELS_DIR, "training_stats.json")

# Calibration patients: exclude training (001-004) and primary val (005).
# Using 13 patients (041-053) — large enough to observe signal range,
# small enough to run quickly on CPU.
CALIB_PATIENTS = [f"{i:03d}" for i in range(41, 54)]   # 041 .. 053

MC_PASSES = 20   # more passes than training eval for better uncertainty estimate


# ==============================================================================
# HELPERS
# ==============================================================================

def get_patient_paths(pid):
    pdir  = os.path.join(DATA_DIR, f"BraTS20_Training_{pid}")
    flair = os.path.join(pdir, f"BraTS20_Training_{pid}_flair.nii")
    t1ce  = os.path.join(pdir, f"BraTS20_Training_{pid}_t1ce.nii")
    t2    = os.path.join(pdir, f"BraTS20_Training_{pid}_t2.nii")
    seg   = os.path.join(pdir, f"BraTS20_Training_{pid}_seg.nii")
    return flair, t1ce, t2, seg


def pearson_r(xs, ys):
    """Simple Pearson r between two equal-length lists.  Returns NaN if
    either list is constant."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) *
                      sum((y - my) ** 2 for y in ys))
    return float("nan") if denom < 1e-12 else num / denom


# ==============================================================================
# MAIN
# ==============================================================================

def run_calibration():
    print("=" * 60)
    print("TumorGuard AI - Phase 5 Reliability Calibration")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────
    print(f"\n[1/4] Loading checkpoint: {CKPT_PATH}")
    model = tumorguard_core.TumorGuardUNet()
    tumorguard_core.load_model_checkpoint(model, CKPT_PATH, map_location="cpu")
    model.eval()
    print("  Checkpoint loaded OK.")

    # ── Load familiarity reference ────────────────────────────────
    print(f"\n[2/4] Loading training stats: {STATS_PATH}")
    with open(STATS_PATH) as f:
        training_stats = json.load(f)
    ref = training_stats.get("familiarity_reference")
    if ref is None:
        raise RuntimeError("familiarity_reference missing from training_stats.json. "
                           "Re-run train.py first.")
    print(f"  Familiarity reference present: {len(ref['feature_names'])} features.")

    # ── Evaluate each calibration patient ────────────────────────
    print(f"\n[3/4] Evaluating {len(CALIB_PATIENTS)} calibration patients "
          f"({CALIB_PATIENTS[0]} to {CALIB_PATIENTS[-1]}), {MC_PASSES} MC passes each...")
    print(f"  {'PID':<6} {'Slice':>5} {'Familiarity':>12} {'Entropy':>10} "
          f"{'Variance':>12} {'MaxDiff':>10} {'Dice':>8}")
    print("  " + "-" * 62)

    results = []
    skipped = []

    for pid in CALIB_PATIENTS:
        flair_p, t1ce_p, t2_p, seg_p = get_patient_paths(pid)

        # Skip if patient directory is missing
        if not os.path.exists(flair_p):
            skipped.append(pid)
            continue

        try:
            # Slice selection uses ground-truth — OFFLINE CALIBRATION ONLY
            best_slice = tumorguard_core.select_best_tumor_slice(seg_p)

            # Preprocess — identical pipeline to training
            x = tumorguard_core.load_and_preprocess_mri(
                flair_p, t1ce_p, t2_p, slice_idx=best_slice
            ).unsqueeze(0)   # (1, 3, 128, 128)

            # Ground-truth mask — used ONLY for offline Dice computation
            y = tumorguard_core.load_segmentation_mask(seg_p, slice_idx=best_slice)

            # Familiarity score — no ground-truth used
            familiarity = tumorguard_core.calculate_similarity_score(x, training_stats)

            # MC Dropout
            mean_pred, var_map, unc_score, max_diff, entropy = \
                tumorguard_core.run_mc_dropout_inference(model, x, num_passes=MC_PASSES)

            # Offline Dice (threshold at 0.5 on the mean MC prediction)
            mean_logits_approx = mean_pred  # already probabilities from sigmoid
            pred_binary = (mean_pred > 0.5).float()
            intersection = (pred_binary * y.unsqueeze(0)).sum().item()
            union        = pred_binary.sum().item() + y.sum().item()
            dice = (2.0 * intersection / union) if union > 0 else 1.0

            row = {
                "patient_id":   pid,
                "slice_idx":    int(best_slice),
                "familiarity":  round(familiarity, 6),
                "entropy":      round(entropy, 6),
                "variance":     round(unc_score, 8),
                "max_diff":     round(max_diff, 6),
                "dice":         round(dice, 6),
            }
            results.append(row)

            print(f"  {pid:<6} {best_slice:>5} {familiarity:>12.6f} {entropy:>10.6f} "
                  f"{unc_score:>12.8f} {max_diff:>10.6f} {dice:>8.4f}")

        except Exception as e:
            skipped.append(pid)
            print(f"  {pid:<6} -- ERROR: {e}")

    # ── Compute summary statistics and correlations ───────────────
    print(f"\n[4/4] Summary and Correlations")
    print("=" * 60)

    n = len(results)
    if n == 0:
        print("  [ERROR] No patients evaluated successfully.")
        return

    fam_vals  = [r["familiarity"] for r in results]
    ent_vals  = [r["entropy"]     for r in results]
    var_vals  = [r["variance"]    for r in results]
    diff_vals = [r["max_diff"]    for r in results]
    dice_vals = [r["dice"]        for r in results]

    def _rng(vals):
        return min(vals), max(vals)

    fam_min,  fam_max  = _rng(fam_vals)
    ent_min,  ent_max  = _rng(ent_vals)
    var_min,  var_max  = _rng(var_vals)
    dice_min, dice_max = _rng(dice_vals)

    r_fam_dice  = pearson_r(fam_vals,  dice_vals)
    r_ent_dice  = pearson_r(ent_vals,  dice_vals)
    r_var_dice  = pearson_r(var_vals,  dice_vals)
    r_diff_dice = pearson_r(diff_vals, dice_vals)

    print(f"  Patients evaluated:         {n}  (skipped: {len(skipped)})")
    print(f"  Familiarity range:          {fam_min:.4f} - {fam_max:.4f}")
    print(f"  Entropy range:              {ent_min:.6f} - {ent_max:.6f}")
    print(f"  Variance range:             {var_min:.8f} - {var_max:.8f}")
    print(f"  Dice range:                 {dice_min:.4f} - {dice_max:.4f}")
    print()
    print(f"  Pearson r (familiarity, Dice):   {r_fam_dice:+.4f}")
    print(f"  Pearson r (entropy,     Dice):   {r_ent_dice:+.4f}")
    print(f"  Pearson r (variance,    Dice):   {r_var_dice:+.4f}")
    print(f"  Pearson r (max_diff,    Dice):   {r_diff_dice:+.4f}")
    print()

    # Interpretation
    # Entropy ≈ 0.693 = maximum binary entropy (model is 50/50 everywhere).
    # Entropy much below that = confident model, which can correlate with better Dice.
    print("  NOTE: Entropy near 0.693 means the model is maximally uncertain")
    print("  (predicts ~0.5 probability everywhere). Lower entropy = more confident.")
    print()
    print("  DISCLAIMER: n={} is too small for statistical or clinical conclusions.".format(n))
    print("  This is a prototype calibration to inform formula design only.")

    # ── Save results ──────────────────────────────────────────────
    summary = {
        "experiment": "Phase 5 Reliability Calibration",
        "disclaimer": (
            "Dice is used OFFLINE ONLY for calibration. "
            "Live inference never uses ground-truth labels. "
            "This is a research prototype — n={} patients, "
            "not clinically validated.".format(n)
        ),
        "calibration_patients": CALIB_PATIENTS,
        "mc_passes": MC_PASSES,
        "n_evaluated": n,
        "n_skipped": len(skipped),
        "skipped_ids": skipped,
        "ranges": {
            "familiarity": {"min": fam_min,  "max": fam_max},
            "entropy":     {"min": ent_min,  "max": ent_max},
            "variance":    {"min": var_min,  "max": var_max},
            "dice":        {"min": dice_min, "max": dice_max},
        },
        "pearson_correlations_with_dice": {
            "familiarity": round(r_fam_dice,  4),
            "entropy":     round(r_ent_dice,  4),
            "variance":    round(r_var_dice,  4),
            "max_diff":    round(r_diff_dice, 4),
        },
        "per_patient": results,
    }

    json_path = os.path.join(OUTPUT_DIR, "reliability_calibration.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"  Saved -> {json_path}")

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "reliability_calibration.csv")
    with open(csv_path, "w") as f:
        f.write("patient_id,slice_idx,familiarity,entropy,variance,max_diff,dice\n")
        for r in results:
            f.write(
                f"{r['patient_id']},{r['slice_idx']},"
                f"{r['familiarity']},{r['entropy']},"
                f"{r['variance']},{r['max_diff']},{r['dice']}\n"
            )
    print(f"  Saved -> {csv_path}")

    print("\n" + "=" * 60)
    print("Phase 5 Calibration COMPLETE")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_calibration()



