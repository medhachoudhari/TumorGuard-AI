import os
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import tumorguard_core

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# PATHS
# ==============================================================================

DATASET_DIR = r"C:\Medha\TumorGuard AI\datasets"
MODELS_DIR = r"C:\Medha\TumorGuard AI\models"

os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_X_PATH = os.path.join(DATASET_DIR, "train_x.pt")
TRAIN_Y_PATH = os.path.join(DATASET_DIR, "train_y.pt")
VAL_X_PATH   = os.path.join(DATASET_DIR, "val_x.pt")
VAL_Y_PATH   = os.path.join(DATASET_DIR, "val_y.pt")

CHECKPOINT_PATH = os.path.join(MODELS_DIR, "tumorguard_model.pt")

# ==============================================================================
# TRAINING CONFIG
# ==============================================================================

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("TumorGuard AI — Cached Dataset Training")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Early stopping patience: {PATIENCE}")
print()

# ==============================================================================
# LOAD CACHED DATASET
# ==============================================================================

print("Loading cached tensors...")

train_x = torch.load(TRAIN_X_PATH, weights_only=True)
train_y = torch.load(TRAIN_Y_PATH, weights_only=True)

val_x = torch.load(VAL_X_PATH, weights_only=True)
val_y = torch.load(VAL_Y_PATH, weights_only=True)

print(f"Train X: {tuple(train_x.shape)}")
print(f"Train Y: {tuple(train_y.shape)}")
print(f"Val X:   {tuple(val_x.shape)}")
print(f"Val Y:   {tuple(val_y.shape)}")
print()

# ==============================================================================
# DATA LOADERS
# ==============================================================================

train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(val_x, val_y)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Training batches:   {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print()

# ==============================================================================
# MODEL
# ==============================================================================

print("Creating U-Net model...")

model = tumorguard_core.TumorGuardUNet(
    in_channels=3,
    out_channels=1,
    dropout_rate=0.3
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# ==============================================================================
# LOSS
# ==============================================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        probs.sum(dim=1) + targets.sum(dim=1) + smooth
    )

    return 1.0 - dice.mean()


def combined_loss(logits, targets):
    return bce_loss(logits, targets) + dice_loss(logits, targets)


# ==============================================================================
# OPTIMIZER
# ==============================================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==============================================================================
# TRAINING
# ==============================================================================

best_val_loss = float("inf")
epochs_without_improvement = 0
history = []

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------------------

    model.train()

    train_loss = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader, 1):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = combined_loss(logits, masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --------------------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            loss = combined_loss(logits, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f}"
    )

    # --------------------------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "device": str(DEVICE),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
            },
            CHECKPOINT_PATH
        )

        print(
            f"  ✓ New best model saved "
            f"(validation loss: {val_loss:.4f})"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"  No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

        if epochs_without_improvement >= PATIENCE:

            print()
            print("Early stopping triggered.")
            break

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model checkpoint: {CHECKPOINT_PATH}")

# ==============================================================================
# SAVE TRAINING HISTORY
# ==============================================================================

history_path = os.path.join(
    MODELS_DIR,
    "training_history.json"
)

with open(history_path, "w") as f:
    json.dump(history, f, indent=4)

print(f"Training history: {history_path}")

print()
print("Next step: validate the trained model and recalibrate reliability.")