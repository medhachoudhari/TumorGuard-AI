import base64
import io
import json
import os

import numpy as np
import pytest
from PIL import Image

import input_validation as iv
import rendering

nib = pytest.importorskip("nibabel")


def decode(uri):
    assert uri.startswith("data:image/png;base64,")
    return np.array(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))


def test_panels_are_aligned_and_native_size():
    rng = np.random.default_rng(0)
    flair = rng.standard_normal((128, 128))
    flair[:20] = 0
    mask = np.zeros((128, 128)); mask[40:60, 70:100] = 1
    var = np.zeros((128, 128)); var[40:60, 70:100] = 0.01
    mri, seg, heat, info = rendering.generate_visualizations(flair, mask, var)
    a, b, c = decode(mri), decode(seg), decode(heat)
    assert a.shape == b.shape == c.shape == (128, 128, 3)
    assert (info["width"], info["height"], info["resampled"]) == (128, 128, False)
    # Overlay differs from the MRI exactly on the (rotated) mask pixels and nowhere else.
    diff = np.any(a != b, axis=2)
    expected = rendering.to_display(mask > 0)
    assert np.array_equal(diff, expected & np.any(a != b, axis=2))
    assert diff.sum() > 0 and not diff[~expected].any()
    # Heatmap only changes pixels with non-zero variance.
    hd = np.any(a != c, axis=2)
    assert not hd[~rendering.to_display(var > 0)].any()
    assert info["heatmap_max_variance"] == pytest.approx(0.01)


def test_no_horizontal_banding_for_smooth_input():
    yy, xx = np.mgrid[:128, :128]
    flair = np.exp(-(((yy - 64) / 40.0) ** 2 + ((xx - 64) / 40.0) ** 2)) + 0.1
    mri, _, _, _ = rendering.generate_visualizations(flair, np.zeros((128, 128)), np.zeros((128, 128)))
    img = decode(mri)[..., 0].astype(float)
    rows = np.abs(np.diff(img, axis=0)).mean()
    cols = np.abs(np.diff(img, axis=1)).mean()
    assert rows < 8 and cols < 8  # smooth input stays smooth


def test_non_finite_and_empty_inputs_do_not_crash():
    flair = np.full((128, 128), np.nan)
    mri, seg, heat, info = rendering.generate_visualizations(flair, np.zeros((128, 128)), np.full((128, 128), np.nan))
    assert decode(mri).max() == 0 and info["heatmap_max_variance"] == 0.0


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        rendering.generate_visualizations(np.zeros((8, 8)), np.zeros((8, 9)), np.zeros((8, 8)))


# ---- input validation -----------------------------------------------------------

def save(tmp_path, name, shape=(48, 48, 6), affine=None, fill=1.0, nan=0.0, dtype=np.float32):
    rng = np.random.default_rng(1)
    v = (rng.random(shape) * fill).astype(dtype)
    if nan:
        v.flat[: int(nan * v.size)] = np.nan
    p = tmp_path / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(v, np.eye(4) if affine is None else affine), str(p))
    return str(p)


def trio(tmp_path, **over):
    paths = {m: save(tmp_path, m) for m in ("flair", "t1ce", "t2")}
    paths.update(over)
    return paths


def test_valid_trio(tmp_path):
    s = iv.inspect_volumes(trio(tmp_path))
    assert s["shape"] == [48, 48, 6] and s["num_slices"] == 6 and s["affine_consistent"]


def test_unusual_axis_order_produces_warning_not_error(tmp_path):
    paths = {m: save(tmp_path, m, shape=(40, 64, 64)) for m in ("flair", "t1ce", "t2")}
    s = iv.inspect_volumes(paths)
    assert len(s["warnings"]) >= 1
    assert iv.inspect_volumes(trio(tmp_path))["warnings"] == []


@pytest.mark.parametrize("case,code", [
    ("shape", "SPATIAL_MISMATCH"), ("affine", "SPATIAL_MISMATCH"), ("4d", "INVALID_SHAPE"),
    ("tiny", "INVALID_SHAPE"), ("zeros", "EMPTY_VOLUME"), ("nan", "NON_FINITE_DATA"),
    ("garbage", "INVALID_NIFTI"),
])
def test_invalid_inputs(tmp_path, case, code):
    bad = {
        "shape": lambda: save(tmp_path, "x1", shape=(48, 48, 7)),
        "affine": lambda: save(tmp_path, "x2", affine=np.diag([2, 2, 2, 1.0])),
        "4d": lambda: save(tmp_path, "x3", shape=(48, 48, 6, 2)),
        "tiny": lambda: save(tmp_path, "x4", shape=(8, 8, 6)),
        "zeros": lambda: save(tmp_path, "x5", fill=0.0),
        "nan": lambda: save(tmp_path, "x6", nan=0.05),
        "garbage": lambda: (tmp_path / "g.nii.gz").write_bytes(b"nope") or str(tmp_path / "g.nii.gz"),
    }[case]()
    with pytest.raises(iv.InputValidationError) as e:
        iv.inspect_volumes(trio(tmp_path, t2=bad))
    assert e.value.code == code
    assert str(tmp_path) not in e.value.message


def test_baseline_copy_matches_historical_artifact():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    art = json.load(open(os.path.join(root, "models", "phase9_evaluation.json")))
    copy = json.load(open(os.path.join(root, "frontend", "src", "data", "historical_baseline.json")))
    for k in ("validation_samples", "mean_dice", "median_dice", "mean_iou", "mean_precision", "mean_recall",
              "zero_prediction_slices", "checkpoint_epoch"):
        assert copy["metrics"][k] == art[k], k
