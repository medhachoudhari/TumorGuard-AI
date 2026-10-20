# ==============================================================================
# TumorGuard AI — Phase 9: Cached Multi-Slice Training Pipeline
# ==============================================================================
# Trains a 2D U-Net on 1200 tumor-containing slices from 30 BraTS2020 patients.
# Caches the prepared tensors to disk on first run for fast re-use.
# Saves the best checkpoint based on validation Dice (no overwrite of dev ckpt).
# ==============================================================================

import os
import json
import math
import time
import warnings

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import tumorguard_core

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR    = r"C:\Medha\TumorGuard AI\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
MODELS_DIR  = r"C:\Medha\TumorGuard AI\models"
DATASET_DIR = r"C:\Medha\TumorGuard AI\datasets"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_PATIENTS = [f"{i:03d}" for i in range(1, 31)]   # 001–030
VAL_PATIENTS   = [f"{i:03d}" for i in range(31, 41)]  # 031–040

MAX_TRAIN_SLICES_PER_PATIENT = 40
MAX_VAL_SLICES_PER_PATIENT   = 40

BATCH_SIZE    = 8
EPOCHS        = 20
LR            = 1e-3
PATIENCE      = 5

BEST_CKPT    = os.path.join(MODELS_DIR, "tumorguard_30patient_best.pth")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history_30patient.json")

TRAIN_X_PATH = os.path.join(DATASET_DIR, "train_x.pt")
TRAIN_Y_PATH = os.path.join(DATASET_DIR, "train_y.pt")
VAL_X_PATH   = os.path.join(DATASET_DIR, "val_x.pt")
VAL_Y_PATH   = os.path.join(DATASET_DIR, "val_y.pt")
META_PATH    = os.path.join(DATASET_DIR, "dataset_metadata.json")


# ==============================================================================
# HELPERS
# ==============================================================================
def get_patient_paths(patient_id):
    d = os.path.join(DATA_DIR, f"BraTS20_Training_{patient_id}")
    def p(mod):
        path = os.path.join(d, f"BraTS20_Training_{patient_id}_{mod}.nii")
        if not os.path.exists(path):
            path += ".gz"
        return path
    return p("flair"), p("t1ce"), p("t2"), p("seg")


def build_and_cache_dataset(patient_ids, label, max_slices, x_path, y_path):
    """Load multi-slice tensors from BraTS NIfTI files and save to disk."""
    print(f"\n  Building {label} tensors from {len(patient_ids)} patients...")
    inputs, targets, all_slices = [], [], []
    patient_slice_counts = {}

    for pid in patient_ids:
        flair_p, t1ce_p, t2_p, seg_p = get_patient_paths(pid)

        # Load seg to find tumor slices
        seg = nib.load(seg_p).get_fdata()
        seg = np.nan_to_num(seg)
        binary = (seg > 0).astype(np.float32)
        tumor_per_slice = binary.sum(axis=(0, 1))
        tumor_slices = np.where(tumor_per_slice > 0)[0]

        if len(tumor_slices) == 0:
            print(f"    [{label}] Patient {pid} -> no tumor slices. Skipping.")
            patient_slice_counts[pid] = 0
            continue

        # Deterministic evenly-spaced selection
        if len(tumor_slices) > max_slices:
            idx = np.unique(np.linspace(0, len(tumor_slices) - 1, max_slices).astype(int))
            selected = tumor_slices[idx]
        else:
            selected = tumor_slices

        patient_slice_counts[pid] = len(selected)
        print(f"    [{label}] Patient {pid} -> {len(selected)} slices")

        for s in selected:
            x = tumorguard_core.load_and_preprocess_mri(flair_p, t1ce_p, t2_p, slice_idx=int(s))
            y = tumorguard_core.load_segmentation_mask(seg_p, slice_idx=int(s))
            inputs.append(x)
            targets.append(y)
            all_slices.append(int(s))

    batch_x = torch.stack(inputs)
    batch_y = torch.stack(targets)

    torch.save(batch_x, x_path)
    torch.save(batch_y, y_path)
    print(f"    Saved {x_path} — shape: {tuple(batch_x.shape)}")
    print(f"    Saved {y_path} — shape: {tuple(batch_y.shape)}")

    return batch_x, batch_y, all_slices, patient_slice_counts


def verify_shapes(name, x, y, expected_x, expected_y):
    """Strict shape verification — stops training if mismatch."""
    if tuple(x.shape) != expected_x:
        raise ValueError(
            f"Shape mismatch for {name} inputs: expected {expected_x}, got {tuple(x.shape)}"
        )
    if tuple(y.shape) != expected_y:
        raise ValueError(
            f"Shape mismatch for {name} targets: expected {expected_y}, got {tuple(y.shape)}"
        )
    print(f"  [{name}] Input: {tuple(x.shape)}  Target: {tuple(y.shape)}  ✓")


# ==============================================================================
# MAIN TRAINING ROUTINE
# ==============================================================================
def train():
    print("=" * 60)
    print("TumorGuard AI — Phase 9 Training")
    print("=" * 60)

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {str(device).upper()}")

    # ── 1. Dataset: load from cache or rebuild ────────────────────
    print("\n--- [1/5] Dataset ---")
    metadata = {}

    if all(os.path.exists(p) for p in [TRAIN_X_PATH, TRAIN_Y_PATH, VAL_X_PATH, VAL_Y_PATH, META_PATH]):
        print("  Cache found. Loading from disk...")
        train_x = torch.load(TRAIN_X_PATH, weights_only=True)
        train_y = torch.load(TRAIN_Y_PATH, weights_only=True)
        val_x   = torch.load(VAL_X_PATH,   weights_only=True)
        val_y   = torch.load(VAL_Y_PATH,   weights_only=True)
        with open(META_PATH) as f:
            metadata = json.load(f)
        print(f"  Loaded from cache.")
    else:
        print("  No cache found. Building from BraTS NIfTI files...")
        t0 = time.time()
        train_x, train_y, train_slices, train_counts = build_and_cache_dataset(
            TRAIN_PATIENTS, "train", MAX_TRAIN_SLICES_PER_PATIENT,
            TRAIN_X_PATH, TRAIN_Y_PATH
        )
        val_x, val_y, val_slices, val_counts = build_and_cache_dataset(
            VAL_PATIENTS, "val", MAX_VAL_SLICES_PER_PATIENT,
            VAL_X_PATH, VAL_Y_PATH
        )
        metadata = {
            "training_patients":   TRAIN_PATIENTS,
            "validation_patients": VAL_PATIENTS,
            "max_train_slices_per_patient": MAX_TRAIN_SLICES_PER_PATIENT,
            "max_val_slices_per_patient":   MAX_VAL_SLICES_PER_PATIENT,
            "train_patient_slice_counts":   train_counts,
            "val_patient_slice_counts":     val_counts,
            "train_slice_range": [int(min(train_slices)), int(max(train_slices))],
            "val_slice_range":   [int(min(val_slices)),   int(max(val_slices))],
            "total_train_slices": len(train_slices),
            "total_val_slices":   len(val_slices),
        }
        with open(META_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"  Metadata saved -> {META_PATH}")
        print(f"  Build time: {(time.time()-t0)/60:.1f} min")

    # ── 2. Shape verification ─────────────────────────────────────
    print("\n--- [2/5] Shape Verification ---")
    verify_shapes("train", train_x, train_y,
                  expected_x=(1200, 3, 128, 128), expected_y=(1200, 1, 128, 128))
    verify_shapes("val",   val_x,   val_y,
                  expected_x=(398,  3, 128, 128), expected_y=(398,  1, 128, 128))

    # ── 3. DataLoaders ────────────────────────────────────────────
    print("\n--- [3/5] DataLoaders ---")
    train_ds = TensorDataset(train_x, train_y)
    val_ds   = TensorDataset(val_x,   val_y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── 4. Model, optimizer, loss ─────────────────────────────────
    print("\n--- [4/5] Model ---")
    model = tumorguard_core.TumorGuardUNet(in_channels=3, out_channels=1, dropout_rate=0.3)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    bce_loss  = nn.BCEWithLogitsLoss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  TumorGuardUNet  |  Parameters: {num_params:,}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")

    # ── 5. Training loop ──────────────────────────────────────────
    print("\n--- [5/5] Training ---")
    history = []
    best_val_dice = -1.0
    best_epoch    = -1
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # — Train —
        model.train()
        total_train_loss = 0.0
        total_train_dice = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = bce_loss(logits, by) + tumorguard_core.calculate_dice_loss(logits, by)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_dice += tumorguard_core.calculate_dice_score(logits, by)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_dice = total_train_dice / len(train_loader)

        # — Validate —
        model.eval()
        total_val_loss = 0.0
        total_val_dice = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = bce_loss(logits, by) + tumorguard_core.calculate_dice_loss(logits, by)
                total_val_loss += loss.item()
                total_val_dice += tumorguard_core.calculate_dice_score(logits, by)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_dice = total_val_dice / len(val_loader)

        # — Log —
        print(
            f"  Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f}  Train Dice: {avg_train_dice:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}  Val Dice: {avg_val_dice:.4f}"
        )
        history.append({
            "epoch":          epoch,
            "train_loss":     round(avg_train_loss, 6),
            "train_dice":     round(avg_train_dice, 6),
            "val_loss":       round(avg_val_loss, 6),
            "val_dice":       round(avg_val_dice, 6),
        })

        # — Checkpoint best —
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_epoch    = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_CKPT)
            print(f"    *** New best Val Dice: {best_val_dice:.4f} — checkpoint saved ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  Early stopping triggered (no improvement for {PATIENCE} epochs).")
                break

    # ── Post-training: reload best and evaluate ───────────────────
    print(f"\n  Reloading best checkpoint (epoch {best_epoch})...")
    model_best = tumorguard_core.TumorGuardUNet(in_channels=3, out_channels=1, dropout_rate=0.3)
    tumorguard_core.load_model_checkpoint(model_best, BEST_CKPT, map_location=device)
    model_best = model_best.to(device)
    model_best.eval()

    per_slice_dice = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model_best(bx)
            preds  = torch.sigmoid(logits)
            for i in range(preds.shape[0]):
                d = tumorguard_core.calculate_dice_score(logits[i:i+1], by[i:i+1])
                per_slice_dice.append(d)

    dice_arr = np.array(per_slice_dice)
    final_epoch = history[-1]

    # ── Save training history ──────────────────────────────────────
    training_history = {
        "training_patients":     TRAIN_PATIENTS,
        "validation_patients":   VAL_PATIENTS,
        "epochs_trained":        len(history),
        "batch_size":            BATCH_SIZE,
        "learning_rate":         LR,
        "early_stopping_patience": PATIENCE,
        "best_epoch":            best_epoch,
        "best_validation_dice":  round(float(best_val_dice), 6),
        "final_training_loss":   final_epoch["train_loss"],
        "final_training_dice":   final_epoch["train_dice"],
        "final_validation_loss": final_epoch["val_loss"],
        "final_validation_dice": final_epoch["val_dice"],
        "model_architecture":    "2D_UNet_MCDropout",
        "dropout_rate":          0.3,
        "image_size":            tumorguard_core.IMG_SIZE,
        "epoch_history":         history,
        "disclaimer":            "Research prototype — not clinically validated.",
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(training_history, f, indent=4)

    # ── Final report ──────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("TumorGuard AI — Phase 9 Training Report")
    print("=" * 56)
    print(f"Training patients:      001–030")
    print(f"Training slices:        1200")
    print(f"Validation patients:    031–040")
    print(f"Validation slices:      398")
    print()
    print(f"Best epoch:             {best_epoch}")
    print(f"Best validation Dice:   {best_val_dice:.4f}")
    print()
    print("Final validation statistics (best checkpoint):")
    print(f"  Overall (batch-avg) Dice: {dice_arr.mean():.4f}")
    print(f"  Mean Dice:    {dice_arr.mean():.4f}")
    print(f"  Median Dice:  {float(np.median(dice_arr)):.4f}")
    print(f"  Min Dice:     {dice_arr.min():.4f}")
    print(f"  Max Dice:     {dice_arr.max():.4f}")
    print()
    print(f"Checkpoint:        {BEST_CKPT}")
    print(f"Training history:  {HISTORY_PATH}")
    print("=" * 56)
    print("NOTE: Research prototype — not clinically validated.")
    print("=" * 56)


if __name__ == "__main__":
    train()
