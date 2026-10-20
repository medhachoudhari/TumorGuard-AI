import os
import json
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import tumorguard_core

# ============================================================
# CONFIG
# ============================================================

# Paths default to the original Windows layout; override with environment
# variables to run elsewhere.
DATASET_DIR = os.environ.get("TUMORGUARD_DATASET_DIR", r"C:\Medha\TumorGuard AI\datasets")
MODELS_DIR = os.environ.get("TUMORGUARD_MODELS_DIR", r"C:\Medha\TumorGuard AI\models")

MODEL_PATH = os.path.join(MODELS_DIR, "tumorguard_model.pt")
TRAINING_HISTORY = os.path.join(MODELS_DIR, "training_history.json")

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# METRICS
# ============================================================

def calculate_metrics(logits, targets, threshold=0.5, smooth=1e-6):

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)

    pred_sum = preds_flat.sum(dim=1)
    target_sum = targets_flat.sum(dim=1)

    dice = (
        (2.0 * intersection + smooth) /
        (pred_sum + target_sum + smooth)
    )

    union = pred_sum + target_sum - intersection

    iou = (
        (intersection + smooth) /
        (union + smooth)
    )

    true_positive = intersection
    false_positive = pred_sum - intersection
    false_negative = target_sum - intersection

    precision = (
        (true_positive + smooth) /
        (true_positive + false_positive + smooth)
    )

    recall = (
        (true_positive + smooth) /
        (true_positive + false_negative + smooth)
    )

    return (
        dice.cpu().numpy(),
        iou.cpu().numpy(),
        precision.cpu().numpy(),
        recall.cpu().numpy(),
        probs.cpu().numpy(),
        preds.cpu().numpy()
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TumorGuard AI — Phase 9 Model Evaluation")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_PATH}")
print()

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation dataset...")

val_x = torch.load(
    os.path.join(DATASET_DIR, "val_x.pt"),
    weights_only=True
)

val_y = torch.load(
    os.path.join(DATASET_DIR, "val_y.pt"),
    weights_only=True
)

print(f"Validation X: {tuple(val_x.shape)}")
print(f"Validation Y: {tuple(val_y.shape)}")
print()

val_dataset = TensorDataset(val_x, val_y)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

import hashlib
import datetime

RUN_UTC = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
with open(MODEL_PATH, "rb") as _f:
    CHECKPOINT_SHA256 = hashlib.sha256(_f.read()).hexdigest()

print("Loading trained TumorGuard U-Net...")

model = tumorguard_core.TumorGuardUNet(
    in_channels=3,
    out_channels=1,
    dropout_rate=0.3
).to(DEVICE)

checkpoint = tumorguard_core.load_model_checkpoint(model, MODEL_PATH, map_location=DEVICE)

model.eval()

print("Model loaded successfully.")
print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"Checkpoint validation loss: {checkpoint.get('val_loss', 'unknown')}")
print()

# ============================================================
# EVALUATION
# ============================================================

print("=" * 70)
print("RUNNING VALIDATION")
print("=" * 70)

all_dice = []
all_iou = []
all_precision = []
all_recall = []

predicted_tumor_pixels = []
actual_tumor_pixels = []

with torch.no_grad():

    for batch_idx, (images, masks) in enumerate(val_loader, 1):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)

        dice, iou, precision, recall, probs, preds = calculate_metrics(
            logits,
            masks
        )

        all_dice.extend(dice.tolist())
        all_iou.extend(iou.tolist())
        all_precision.extend(precision.tolist())
        all_recall.extend(recall.tolist())

        predicted_tumor_pixels.extend(
            preds.sum(axis=(1, 2, 3)).tolist()
        )

        actual_tumor_pixels.extend(
            masks.cpu().numpy().sum(axis=(1, 2, 3)).tolist()
        )

        if batch_idx % 10 == 0:
            print(
                f"Processed validation batches: "
                f"{batch_idx}/{len(val_loader)}"
            )

# ============================================================
# RESULTS
# ============================================================

all_dice = np.array(all_dice)
all_iou = np.array(all_iou)
all_precision = np.array(all_precision)
all_recall = np.array(all_recall)

mean_dice = float(all_dice.mean())
mean_iou = float(all_iou.mean())
mean_precision = float(all_precision.mean())
mean_recall = float(all_recall.mean())

median_dice = float(np.median(all_dice))
min_dice = float(all_dice.min())
max_dice = float(all_dice.max())

print()
print("=" * 70)
print("PHASE 9 EVALUATION RESULTS")
print("=" * 70)

print(f"Validation samples: {len(all_dice)}")
print()

print(f"Mean Dice:       {mean_dice:.4f}")
print(f"Median Dice:     {median_dice:.4f}")
print(f"Min Dice:        {min_dice:.4f}")
print(f"Max Dice:        {max_dice:.4f}")
print()

print(f"Mean IoU:        {mean_iou:.4f}")
print(f"Mean Precision:  {mean_precision:.4f}")
print(f"Mean Recall:     {mean_recall:.4f}")

print()

print("Dice percentage:")
print(f"  {mean_dice * 100:.2f}%")

print()

# ============================================================
# BASIC SANITY CHECKS
# ============================================================

print("=" * 70)
print("SANITY CHECKS")
print("=" * 70)

zero_predictions = int(np.sum(np.array(predicted_tumor_pixels) == 0))

print(
    f"Slices with zero predicted tumor: "
    f"{zero_predictions}/{len(predicted_tumor_pixels)}"
)

print(
    f"Average predicted tumor pixels: "
    f"{np.mean(predicted_tumor_pixels):.1f}"
)

print(
    f"Average actual tumor pixels: "
    f"{np.mean(actual_tumor_pixels):.1f}"
)

if zero_predictions == len(predicted_tumor_pixels):
    print()
    print("WARNING: Model predicts zero tumor on every validation slice.")

elif mean_dice < 0.20:
    print()
    print("WARNING: Dice score is currently low.")

elif mean_dice < 0.50:
    print()
    print("NOTE: Model shows partial segmentation performance.")

else:
    print()
    print("Model shows substantial tumor-mask overlap.")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "tumorguard_model.pt",
    "checkpoint_epoch": checkpoint.get("epoch"),
    "checkpoint_validation_loss": checkpoint.get("val_loss"),

    "validation_samples": len(all_dice),

    "mean_dice": mean_dice,
    "median_dice": median_dice,
    "min_dice": min_dice,
    "max_dice": max_dice,

    "mean_iou": mean_iou,
    "mean_precision": mean_precision,
    "mean_recall": mean_recall,

    "zero_prediction_slices": zero_predictions,

    "average_predicted_tumor_pixels":
        float(np.mean(predicted_tumor_pixels)),

    "average_actual_tumor_pixels":
        float(np.mean(actual_tumor_pixels)),

    "device": str(DEVICE)
}

# Never overwrite the historical models/phase9_evaluation.json: write each run
# to its own file (override with TUMORGUARD_EVAL_OUTPUT).
results["run_utc"] = RUN_UTC
results["checkpoint_sha256"] = CHECKPOINT_SHA256
results["torch_version"] = torch.__version__
results["numpy_version"] = np.__version__

results_path = os.environ.get("TUMORGUARD_EVAL_OUTPUT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs", "evaluations", f"phase9_{RUN_UTC}.json"
)
if os.path.exists(results_path):
    raise FileExistsError(f"Refusing to overwrite existing evaluation result: {results_path}")
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(f"Results saved to:")
print(results_path)
print()
print("Next step: MC Dropout + reliability calibration.")
print("=" * 70)
