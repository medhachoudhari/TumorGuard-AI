import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from skimage.transform import resize

# ==============================================================================
# 1. MODALITY CONFIGURATION
# ==============================================================================
REQUIRED_MODALITIES = ["flair", "t1ce", "t2"]
IMG_SIZE = 128


# ==============================================================================
# 2. SLICE SELECTION STRATEGIES
# ==============================================================================

def select_best_tumor_slice(seg_path):
    """
    TRAINING / EVALUATION ONLY.
    Loads the segmentation mask and returns the axial slice index with the
    most tumor pixels (any label > 0).

    NOTE: This function uses ground-truth labels, which is acceptable for
    selecting an informative slice for evaluation in this tiny development
    experiment. In a real-world live upload workflow, no ground-truth label
    is available and this function must NOT be called. Use
    select_inference_slice() instead.
    """
    seg = nib.load(seg_path).get_fdata()
    seg = np.nan_to_num(seg)
    binary = (seg > 0).astype(np.float32)
    # Sum tumor pixels across each axial slice
    tumor_per_slice = binary.sum(axis=(0, 1))  # shape: (num_slices,)
    best_idx = int(np.argmax(tumor_per_slice))
    return best_idx


def select_inference_slice(flair_path, preferred_idx=75):
    """
    LIVE INFERENCE ONLY (no ground-truth label available).
    Selects the axial slice with the most non-zero brain voxels in the FLAIR
    modality, as a proxy for the most informative brain slice.
    Falls back to preferred_idx if the FLAIR-based selection fails.
    """
    try:
        img = nib.load(flair_path).get_fdata()
        img = np.nan_to_num(img)
        brain_per_slice = (img > 0).sum(axis=(0, 1))
        best_idx = int(np.argmax(brain_per_slice))
        return best_idx
    except Exception:
        return preferred_idx


# ==============================================================================
# 3. DATA LOADING & PREPROCESSING
# ==============================================================================

def load_nifti_slice(path, slice_idx):
    """Loads a NIfTI file, extracts an axial slice at slice_idx, and resizes to IMG_SIZE."""
    img = nib.load(path).get_fdata()
    img = np.nan_to_num(img)
    if slice_idx >= img.shape[2]:
        slice_idx = img.shape[2] // 2
    img_slice = img[:, :, slice_idx]
    img_resized = resize(img_slice, (IMG_SIZE, IMG_SIZE), mode='constant', anti_aliasing=True)
    return img_resized.astype(np.float32)


def normalize_intensity(img_slice):
    """
    Z-score normalization per slice using only the brain-masked region.
    Background (0-valued) voxels are reset to 0 after normalization.
    """
    mask = img_slice > 0
    if mask.sum() == 0:
        return img_slice
    mean = img_slice[mask].mean()
    std = img_slice[mask].std()
    if std > 0:
        img_slice = (img_slice - mean) / std
    img_slice[~mask] = 0.0
    return img_slice


def load_and_preprocess_mri(flair_path, t1ce_path, t2_path, slice_idx):
    """
    Loads FLAIR, T1CE, T2 slices at the given slice_idx, normalizes each,
    and returns a stacked float32 tensor of shape (3, IMG_SIZE, IMG_SIZE).

    Modality order: [FLAIR=0, T1CE=1, T2=2]  — always matches REQUIRED_MODALITIES.
    Normalization:  z-score per modality per slice, background zeroed.
    """
    flair = normalize_intensity(load_nifti_slice(flair_path, slice_idx))
    t1ce  = normalize_intensity(load_nifti_slice(t1ce_path,  slice_idx))
    t2    = normalize_intensity(load_nifti_slice(t2_path,    slice_idx))

    stacked = np.stack([flair, t1ce, t2], axis=0)  # (3, H, W)
    return torch.tensor(stacked, dtype=torch.float32)


def load_segmentation_mask(seg_path, slice_idx):
    """
    Loads the BraTS segmentation label at slice_idx and converts it to a
    binary mask: any tumor class (1, 2, 4) → 1.0, background → 0.0.
    Returns tensor of shape (1, IMG_SIZE, IMG_SIZE).
    """
    seg = nib.load(seg_path).get_fdata()
    seg = np.nan_to_num(seg)
    if slice_idx >= seg.shape[2]:
        slice_idx = seg.shape[2] // 2
    seg_slice = seg[:, :, slice_idx]
    binary = (seg_slice > 0).astype(np.float32)
    seg_resized = resize(binary, (IMG_SIZE, IMG_SIZE), order=0, mode='constant', anti_aliasing=False)
    return torch.tensor(seg_resized, dtype=torch.float32).unsqueeze(0)  # (1, H, W)


# ==============================================================================
# 4. MODEL ARCHITECTURE (2D U-Net with MC Dropout)
# ==============================================================================

class ConvBlock(nn.Module):
    """
    Double-conv block: Conv→BN→ReLU→Dropout→Conv→BN→ReLU
    Dropout is placed after the first activation so it is active during
    MC Dropout inference while BatchNorm remains in eval mode.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),          # standard Dropout (not Dropout2d)
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TumorGuardUNet(nn.Module):
    def __init__(self, in_channels=len(REQUIRED_MODALITIES), out_channels=1, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck  — extra dropout here for stronger stochastic signal
        self.bottleneck = ConvBlock(64, 128, dropout_rate)
        self.bottleneck_drop = nn.Dropout(p=dropout_rate)

        # Decoder
        self.up2    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2   = ConvBlock(128, 64, dropout_rate)

        self.up1    = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1   = ConvBlock(64, 32, dropout_rate)

        self.outc   = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        # Bottleneck
        b = self.bottleneck(self.pool2(e2))
        b = self.bottleneck_drop(b)

        # Decoder with skip connections
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.outc(d1)


# ==============================================================================
# 5. LOSS & METRICS
# ==============================================================================

def calculate_dice_loss(pred_logits, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)


def calculate_dice_score(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union


# ==============================================================================
# 6. MC DROPOUT INFERENCE
# ==============================================================================

def enable_dropout(model):
    """
    Switches all nn.Dropout (and nn.Dropout2d) modules in the model to
    train mode so they apply stochastic masking during inference.
    Everything else (BatchNorm etc.) remains in eval mode.
    """
    model.eval()  # start with everything in eval mode
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def run_mc_dropout_inference(model, input_tensor, num_passes=10):
    """
    Runs MC Dropout inference: keeps Dropout in train mode while keeping
    BatchNorm in eval mode. Returns the mean prediction, variance map, and
    a scalar uncertainty score.

    Returns
    -------
    mean_pred   : Tensor  (B, C, H, W)  — mean probability map
    var_map     : Tensor  (B, C, H, W)  — per-pixel variance across passes
    unc_score   : float                 — mean variance (scalar summary)
    max_diff    : float                 — max |p_i - p_j| between any two passes
    entropy     : float                 — mean predictive entropy
    """
    enable_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(input_tensor)
            probs  = torch.sigmoid(logits)
            all_probs.append(probs)

    # Stack: (num_passes, B, C, H, W)
    stacked    = torch.stack(all_probs, dim=0)
    mean_pred  = stacked.mean(dim=0)
    var_map    = stacked.var(dim=0)

    # Scalar uncertainty
    unc_score = var_map.mean().item()

    # Max difference between any two passes (sample the first two)
    max_diff = (stacked[0] - stacked[-1]).abs().max().item()

    # Mean predictive entropy: -p*log(p) - (1-p)*log(1-p)
    # Numerically stable: sanitize probabilities before logarithms.
    eps = 1e-7
    p = torch.nan_to_num(mean_pred, nan=0.5, posinf=1.0, neginf=0.0)
    p = p.clamp(eps, 1.0 - eps)

    entropy_map = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    entropy = torch.nan_to_num(
        entropy_map,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).mean().item()
    return mean_pred, var_map, unc_score, max_diff, entropy


# ==============================================================================
# 7. SIMILARITY / FAMILIARITY — Feature-vector distance from training distribution
# ==============================================================================

# Number of statistics extracted per modality channel.
# Features per channel: [mean, std, min, max, nonzero_fraction, p75]
_N_STATS_PER_CHANNEL = 6


def extract_familiarity_features(input_tensor):
    """
    Extracts a compact, label-free feature vector from a preprocessed MRI tensor.

    Parameters
    ----------
    input_tensor : Tensor, shape (B, C, H, W) or (C, H, W)
        The already-preprocessed MRI tensor produced by load_and_preprocess_mri().
        No ground-truth label is used — this is safe for live inference.

    Returns
    -------
    features : np.ndarray, shape (C * _N_STATS_PER_CHANNEL,)
        Flattened feature vector.  For 3 modalities: 18 features total.

    Features per channel (in order):
        0  mean              — overall brightness of the brain region
        1  std               — spread / contrast
        2  min               — darkest value after z-score normalisation
        3  max               — brightest value after z-score normalisation
        4  nonzero_fraction  — fraction of pixels with non-zero signal
        5  p75               — 75th percentile (robust high-intensity marker)
    """
    # Support both batched (B, C, H, W) and unbatched (C, H, W) inputs.
    t = input_tensor.detach().cpu()
    if t.dim() == 4:
        # Average features across the batch dimension so the result is always
        # a single per-sample feature vector (works for B=1 or B>1).
        t = t.mean(dim=0)   # (C, H, W)

    features = []
    for c in range(t.shape[0]):
        ch = t[c].numpy().ravel().astype(np.float64)
        feat_mean      = float(ch.mean())
        feat_std       = float(ch.std())
        feat_min       = float(ch.min())
        feat_max       = float(ch.max())
        feat_nonzero   = float((ch != 0).mean())
        feat_p75       = float(np.percentile(ch, 75))
        features.extend([feat_mean, feat_std, feat_min, feat_max,
                         feat_nonzero, feat_p75])

    return np.array(features, dtype=np.float64)


def build_familiarity_reference(train_tensors):
    """
    Computes the training-distribution reference statistics from a list of
    preprocessed training tensors (one per patient).

    Parameters
    ----------
    train_tensors : list of Tensor, each shape (3, H, W)
        One tensor per training patient, from load_and_preprocess_mri().
        Ground-truth labels are NOT used.

    Returns
    -------
    ref_mean : list[float]   — per-feature mean across training patients
    ref_std  : list[float]   — per-feature std  across training patients
                               (clamped to a minimum of 1e-6 to avoid /0)
    feature_names : list[str]
    """
    modality_names = REQUIRED_MODALITIES  # ['flair', 't1ce', 't2']
    stat_names = ["mean", "std", "min", "max", "nonzero_frac", "p75"]
    feature_names = [f"{mod}_{stat}"
                     for mod in modality_names
                     for stat in stat_names]

    # Extract feature vector for each training patient
    all_features = np.array([extract_familiarity_features(t) for t in train_tensors])
    # all_features shape: (N_patients, N_features)

    ref_mean = all_features.mean(axis=0).tolist()
    ref_std  = np.maximum(all_features.std(axis=0), 1e-6).tolist()

    return ref_mean, ref_std, feature_names


def calculate_similarity_score(input_tensor, training_stats):
    """
    Computes a real familiarity score [0, 1] by comparing image-level
    statistics of the input against the training distribution.

    No ground-truth label is used — safe for live inference.

    Algorithm
    ---------
    1. Extract an 18-d feature vector from input_tensor (same features that
       were used to build the reference during training).
    2. Normalise each feature by the training standard deviation:
           z_i = (feat_i - ref_mean_i) / ref_std_i
    3. Compute the mean absolute normalised deviation (a transparent L1
       distance in standardised feature space):
           distance = mean(|z_i|)
    4. Map distance -> score via exponential decay:
           score = exp(-distance)
       Interpretation:
           distance = 0   -> score = 1.0  (identical to training distribution)
           distance = 1   -> score = 0.37 (1 std away on average)
           distance = 2   -> score = 0.14 (2 std away on average)
           distance >> 0  -> score -> 0   (far out-of-distribution)

    Parameters
    ----------
    input_tensor   : Tensor (B, C, H, W) or (C, H, W)
    training_stats : dict loaded from training_stats.json

    Returns
    -------
    score : float in [0, 1]
        Higher = more similar to training distribution.
        Returns 0.5 as a neutral fallback if the reference is missing.
    """
    ref = training_stats.get("familiarity_reference")
    if ref is None:
        # Reference was not yet built; return a neutral mid-point so downstream
        # code doesn't break, but flag the issue clearly.
        import warnings as _w
        _w.warn("familiarity_reference missing from training_stats.json. "
                "Re-run train.py to rebuild. Returning 0.5 as fallback.")
        return 0.5

    ref_mean = np.array(ref["mean"],  dtype=np.float64)
    ref_std  = np.array(ref["std"],   dtype=np.float64)

    feat = extract_familiarity_features(input_tensor)

    # Normalised L1 distance in standardised feature space
    z        = (feat - ref_mean) / ref_std
    distance = float(np.abs(z).mean())

    # Exponential decay: score = exp(-distance)
    score = float(np.exp(-distance))
    return score


# ==============================================================================
# 8. RELIABILITY SCORING & VERDICT
# ==============================================================================

def calculate_reliability_and_verdict(similarity_score, entropy, variance=0.0005, max_diff=0.8):
    """
    Prototype reliability score for TumorGuard AI.
    This describes model reliability only; it does not diagnose cancer.
    """
    import math

    # Familiarity
    fam_score = max(0.0, min(float(similarity_score), 1.0))

    # Entropy confidence
    max_entropy = math.log(2.0)
    entropy = max(0.0, min(float(entropy), max_entropy))
    entropy_confidence = (max_entropy - entropy) / max_entropy
    entropy_confidence = max(0.0, min(entropy_confidence, 1.0))

    # Variance confidence
    variance = max(0.0, float(variance))
    VAR_LOW = 0.0003
    VAR_HIGH = 0.0009
    variance_confidence = 1.0 - ((variance - VAR_LOW) / (VAR_HIGH - VAR_LOW))
    variance_confidence = max(0.0, min(variance_confidence, 1.0))

    # Max-difference confidence
    max_diff = max(0.0, float(max_diff))
    DIFF_LOW = 0.65
    DIFF_HIGH = 1.0
    diff_confidence = 1.0 - ((max_diff - DIFF_LOW) / (DIFF_HIGH - DIFF_LOW))
    diff_confidence = max(0.0, min(diff_confidence, 1.0))

    # Combined reliability
    reliability_score = (
        0.40 * fam_score +
        0.20 * entropy_confidence +
        0.20 * variance_confidence +
        0.20 * diff_confidence
    )

    reliability_score = max(0.0, min(float(reliability_score), 1.0))

    # Prototype verdict
    if reliability_score >= 0.70:
        verdict = "HIGH RELIABILITY"
    elif reliability_score >= 0.50:
        verdict = "MODERATE RELIABILITY"
    elif reliability_score >= 0.30:
        verdict = "LOW RELIABILITY"
    else:
        verdict = "MANUAL REVIEW REQUIRED"

    return reliability_score, verdict
