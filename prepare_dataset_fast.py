# ==============================================================================
# TumorGuard AI — Fast Dataset Preparation
# Loads each patient ONCE, selects tumor slices, saves .pt cache files.
# ==============================================================================

import os
import sys
import json
import time
import traceback

import numpy as np
import nibabel as nib
import torch

import tumorguard_core

# --- Config ---
DATA_DIR    = r"C:\Medha\TumorGuard AI\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
DATASET_DIR = r"C:\Medha\TumorGuard AI\datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_PATIENTS = [f"{i:03d}" for i in range(1, 31)]
VAL_PATIENTS   = [f"{i:03d}" for i in range(31, 41)]
MAX_SLICES     = 40


def p(msg):
    print(msg, flush=True)


def get_paths(pid):
    d = os.path.join(DATA_DIR, f"BraTS20_Training_{pid}")
    def fp(mod):
        path = os.path.join(d, f"BraTS20_Training_{pid}_{mod}.nii")
        if not os.path.exists(path):
            path += ".gz"
        return path
    return fp("flair"), fp("t1ce"), fp("t2"), fp("seg")


def process_patients(patient_ids, label):
    """
    Loads each patient ONCE. Extracts all modalities and seg in one pass.
    Returns lists of input tensors (3, H, W) and target tensors (1, H, W).
    """
    all_x = []
    all_y = []
    total = len(patient_ids)

    for idx, pid in enumerate(patient_ids, 1):
        global_idx = idx if label == "train" else idx + len(TRAIN_PATIENTS)
        p(f"[{global_idx}/{len(TRAIN_PATIENTS)+len(VAL_PATIENTS)}] Loading Patient {pid}...")
        t_patient = time.time()

        try:
            flair_p, t1ce_p, t2_p, seg_p = get_paths(pid)

            # Load all 4 volumes ONCE
            flair_vol = np.nan_to_num(nib.load(flair_p).get_fdata()).astype(np.float32)
            t1ce_vol  = np.nan_to_num(nib.load(t1ce_p).get_fdata()).astype(np.float32)
            t2_vol    = np.nan_to_num(nib.load(t2_p).get_fdata()).astype(np.float32)
            seg_vol   = np.nan_to_num(nib.load(seg_p).get_fdata()).astype(np.float32)
            p(f"[{global_idx}/{len(TRAIN_PATIENTS)+len(VAL_PATIENTS)}] Loaded volumes — shape: {flair_vol.shape}")

            # Find tumor-containing slices from seg
            binary = (seg_vol > 0).astype(np.float32)
            tumor_per_slice = binary.sum(axis=(0, 1))
            tumor_slices = np.where(tumor_per_slice > 0)[0]

            n_tumor = len(tumor_slices)

            # Deterministic evenly-spaced selection
            if n_tumor > MAX_SLICES:
                idx_sel = np.unique(np.linspace(0, n_tumor - 1, MAX_SLICES).astype(int))
                selected = tumor_slices[idx_sel]
            else:
                selected = tumor_slices

            n_selected = len(selected)
            p(f"[{global_idx}/{len(TRAIN_PATIENTS)+len(VAL_PATIENTS)}] Tumor slices found: {n_tumor} → Selected: {n_selected}")

            # Extract and preprocess each selected slice
            from skimage.transform import resize
            IMG_SIZE = tumorguard_core.IMG_SIZE  # 128

            for s in selected:
                s = int(s)

                # Extract axial slices
                def extract(vol):
                    sl = vol[:, :, s]
                    sl = resize(sl, (IMG_SIZE, IMG_SIZE), mode='constant', anti_aliasing=True)
                    return sl.astype(np.float32)

                def z_norm(sl):
                    mask = sl > 0
                    if mask.sum() == 0:
                        return sl
                    mean = sl[mask].mean()
                    std  = sl[mask].std()
                    if std > 0:
                        sl = (sl - mean) / std
                    sl[~mask] = 0.0
                    return sl

                flair_sl = z_norm(extract(flair_vol))
                t1ce_sl  = z_norm(extract(t1ce_vol))
                t2_sl    = z_norm(extract(t2_vol))

                # Stack modalities: (3, H, W)
                x = torch.tensor(np.stack([flair_sl, t1ce_sl, t2_sl], axis=0), dtype=torch.float32)

                # Binary seg mask: (1, H, W)
                seg_sl = seg_vol[:, :, s]
                seg_bin = (seg_sl > 0).astype(np.float32)
                seg_res = resize(seg_bin, (IMG_SIZE, IMG_SIZE), order=0, mode='constant', anti_aliasing=False)
                y = torch.tensor(seg_res, dtype=torch.float32).unsqueeze(0)

                all_x.append(x)
                all_y.append(y)

            elapsed = time.time() - t_patient
            p(f"[{global_idx}/{len(TRAIN_PATIENTS)+len(VAL_PATIENTS)}] Patient {pid} complete — {n_selected} slices — {elapsed:.1f}s")

        except Exception as e:
            p(f"\n[ERROR] Patient {pid} failed:")
            traceback.print_exc()
            sys.exit(1)

    batch_x = torch.stack(all_x)
    batch_y = torch.stack(all_y)
    return batch_x, batch_y


def main():
    t_total = time.time()

    p("=" * 60)
    p("TumorGuard AI — Fast Dataset Preparation")
    p("=" * 60)
    p(f"Training patients:   {TRAIN_PATIENTS[0]}–{TRAIN_PATIENTS[-1]}  ({len(TRAIN_PATIENTS)} patients)")
    p(f"Validation patients: {VAL_PATIENTS[0]}–{VAL_PATIENTS[-1]}  ({len(VAL_PATIENTS)} patients)")
    p(f"Max slices/patient:  {MAX_SLICES}")
    p(f"Output dir:          {DATASET_DIR}")
    p("")

    # --- Training ---
    p("--- Building TRAINING tensors ---")
    train_x, train_y = process_patients(TRAIN_PATIENTS, "train")

    p(f"\nTraining tensor shapes:")
    p(f"  train_x: {tuple(train_x.shape)}")
    p(f"  train_y: {tuple(train_y.shape)}")

    # --- Validation ---
    p("\n--- Building VALIDATION tensors ---")
    val_x, val_y = process_patients(VAL_PATIENTS, "val")

    p(f"\nValidation tensor shapes:")
    p(f"  val_x: {tuple(val_x.shape)}")
    p(f"  val_y: {tuple(val_y.shape)}")

    # --- Save ---
    p("\n--- Saving to disk ---")
    paths = {
        "train_x": os.path.join(DATASET_DIR, "train_x.pt"),
        "train_y": os.path.join(DATASET_DIR, "train_y.pt"),
        "val_x":   os.path.join(DATASET_DIR, "val_x.pt"),
        "val_y":   os.path.join(DATASET_DIR, "val_y.pt"),
    }
    tensors = {"train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y}

    for name, path in paths.items():
        torch.save(tensors[name], path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        p(f"  Saved {name}: {path}  ({size_mb:.1f} MB)")

    # --- Metadata ---
    meta = {
        "training_patients":   TRAIN_PATIENTS,
        "validation_patients": VAL_PATIENTS,
        "max_slices_per_patient": MAX_SLICES,
        "total_train_slices":  int(train_x.shape[0]),
        "total_val_slices":    int(val_x.shape[0]),
        "train_x_shape":       list(train_x.shape),
        "train_y_shape":       list(train_y.shape),
        "val_x_shape":         list(val_x.shape),
        "val_y_shape":         list(val_y.shape),
        "img_size":            tumorguard_core.IMG_SIZE,
        "modality_order":      tumorguard_core.REQUIRED_MODALITIES,
        "normalization":       "z_score_per_slice_brain_mask",
        "slice_selection":     "evenly_spaced_tumor_containing",
        "disclaimer":          "Research prototype — not clinically validated.",
    }
    meta_path = os.path.join(DATASET_DIR, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    p(f"  Saved metadata: {meta_path}")

    # --- Verify reload ---
    p("\n--- Verifying cache files ---")
    for name, path in paths.items():
        t = torch.load(path, weights_only=True)
        p(f"  {name}: loaded OK — shape {tuple(t.shape)}")
    p("  All cache files verified.")

    total_time = (time.time() - t_total) / 60
    p(f"\n{'=' * 60}")
    p(f"DATASET PREPARATION COMPLETE")
    p(f"{'=' * 60}")
    p(f"  Total training slices:   {train_x.shape[0]}")
    p(f"  Total validation slices: {val_x.shape[0]}")
    p(f"  train_x shape:           {tuple(train_x.shape)}")
    p(f"  train_y shape:           {tuple(train_y.shape)}")
    p(f"  val_x shape:             {tuple(val_x.shape)}")
    p(f"  val_y shape:             {tuple(val_y.shape)}")
    p(f"  Total elapsed time:      {total_time:.1f} min")
    p(f"{'=' * 60}")


if __name__ == "__main__":
    main()
