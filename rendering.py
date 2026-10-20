"""PNG rendering of the three result panels.

All panels go through the same display transform so they stay spatially aligned:
  * array [x, y] -> np.rot90 (90 degrees counter-clockwise), the usual display
    convention for NIfTI slices. The data themselves are not otherwise altered.
    Left/right convention assumes standard BraTS orientation and is not verified
    from the affine.
  * images are written at the native model resolution (no resampling, no smoothing).
"""
import base64
import io

import numpy as np
from PIL import Image

ORIENTATION = "rot90_ccw"
OVERLAY_COLOR = (255, 0, 0)
OVERLAY_ALPHA = 0.5
HEATMAP_MAX_ALPHA = 0.85


def to_display(a):
    return np.ascontiguousarray(np.rot90(np.asarray(a)))


def _png_data_uri(rgb):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)), "RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _gray(flair):
    """Window the z-scored FLAIR slice to 1st-99th percentile of non-zero (brain) pixels."""
    flair = np.nan_to_num(np.asarray(flair, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    brain = flair != 0
    if brain.any():
        lo, hi = np.percentile(flair[brain], [1, 99])
        if hi <= lo:
            hi = lo + 1.0
    else:
        lo, hi = 0.0, 1.0
    g = np.clip((flair - lo) / (hi - lo), 0.0, 1.0)
    g[~brain] = 0.0
    return np.repeat((g * 255.0)[..., None], 3, axis=2)


def generate_visualizations(flair_slice, pred_mask, var_map):
    """
    Returns (mri_png, overlay_png, heatmap_png, info). All inputs are 2D arrays of
    identical shape. `info` records dimensions, orientation and the heatmap scale.
    """
    flair = to_display(flair_slice)
    mask = to_display(np.nan_to_num(np.asarray(pred_mask)) > 0.5)
    var = to_display(np.nan_to_num(np.asarray(var_map, dtype=np.float64)))
    if not (flair.shape == mask.shape == var.shape):
        raise ValueError("MRI, mask and variance arrays must have identical shapes.")

    gray = _gray(flair)

    overlay = gray.copy()
    color = np.array(OVERLAY_COLOR, dtype=np.float64)
    overlay[mask] = gray[mask] * (1 - OVERLAY_ALPHA) + color * OVERLAY_ALPHA

    from matplotlib import colormaps  # colour map only; no pyplot state involved
    vmax = float(var.max())
    norm = var / vmax if vmax > 0 else np.zeros_like(var)
    heat_rgb = colormaps["inferno"](norm)[..., :3] * 255.0
    alpha = (HEATMAP_MAX_ALPHA * norm)[..., None]
    heat = gray * (1 - alpha) + heat_rgb * alpha

    info = {
        "width": int(flair.shape[1]),
        "height": int(flair.shape[0]),
        "orientation": ORIENTATION,
        "resampled": False,
        "heatmap_scale": "linear, 0 to slice maximum variance",
        "heatmap_max_variance": vmax,
        "overlay_alpha": OVERLAY_ALPHA,
    }
    return _png_data_uri(gray), _png_data_uri(overlay), _png_data_uri(heat), info
