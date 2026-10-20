"""Tests for spatial, dimensional, affine validation and safe error handling (Phase 2B Stage A)."""
import io
import os
import numpy as np
import pytest

pytest.importorskip("flask_cors")
nib = pytest.importorskip("nibabel")

import tumorguard_core as core
import server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_volume(path, shape=(64, 64, 20), affine=None, zooms=None, fill=50.0):
    if affine is None:
        affine = np.eye(4)
    vol = np.full(shape, fill, dtype=np.float32)
    img = nib.Nifti1Image(vol, affine)
    if zooms is not None:
        img.header.set_zooms(zooms)
    nib.save(img, str(path))
    return str(path)


def make_modality_dict(tmp_path, shape=(64, 64, 20), affine=None, zooms=None):
    paths = {}
    for mod in core.REQUIRED_MODALITIES:
        p = tmp_path / f"{mod}.nii.gz"
        make_volume(p, shape=shape, affine=affine, zooms=zooms)
        paths[mod] = str(p)
    return paths


# ==============================================================================
# 1. CORE VALIDATION TESTS
# ==============================================================================

def test_valid_volumes_pass_validation(tmp_path):
    paths = make_modality_dict(tmp_path)
    images = core.validate_modality_volumes(paths)
    assert set(images.keys()) == set(core.REQUIRED_MODALITIES)
    for mod in core.REQUIRED_MODALITIES:
        assert images[mod].shape == (64, 64, 20)


def test_missing_modality_key_raises(tmp_path):
    paths = make_modality_dict(tmp_path)
    del paths["t2"]
    with pytest.raises(core.SpatialValidationError, match="Missing required modality: 't2'"):
        core.validate_modality_volumes(paths)


def test_file_not_found_raises(tmp_path):
    paths = make_modality_dict(tmp_path)
    paths["flair"] = str(tmp_path / "non_existent.nii.gz")
    with pytest.raises(core.SpatialValidationError, match="File not found for modality 'flair'"):
        core.validate_modality_volumes(paths)


def test_corrupted_header_raises(tmp_path):
    paths = make_modality_dict(tmp_path)
    corrupt_path = tmp_path / "corrupt.nii.gz"
    corrupt_path.write_bytes(b"NOT_A_NIFTI_FILE_CORRUPTED_BYTES")
    paths["t1ce"] = str(corrupt_path)
    with pytest.raises(core.SpatialValidationError, match="Invalid or corrupted NIfTI file for 't1ce'"):
        core.validate_modality_volumes(paths)


def test_2d_volume_rejected(tmp_path):
    paths = make_modality_dict(tmp_path)
    p = tmp_path / "t2_2d.nii.gz"
    img = nib.Nifti1Image(np.ones((64, 64), dtype=np.float32), np.eye(4))
    nib.save(img, str(p))
    paths["t2"] = str(p)
    with pytest.raises(core.SpatialValidationError, match="must be a 3D volume, but got 2D"):
        core.validate_modality_volumes(paths)


def test_4d_volume_rejected(tmp_path):
    paths = make_modality_dict(tmp_path)
    p = tmp_path / "t2_4d.nii.gz"
    img = nib.Nifti1Image(np.ones((64, 64, 20, 2), dtype=np.float32), np.eye(4))
    nib.save(img, str(p))
    paths["t2"] = str(p)
    with pytest.raises(core.SpatialValidationError, match="must be a 3D volume, but got 4D"):
        core.validate_modality_volumes(paths)


def test_mismatched_dimensions_rejected(tmp_path):
    paths = make_modality_dict(tmp_path, shape=(64, 64, 20))
    # Alter T2 to have fewer slices
    p_t2 = tmp_path / "t2_short.nii.gz"
    make_volume(p_t2, shape=(64, 64, 15))
    paths["t2"] = str(p_t2)
    with pytest.raises(core.SpatialValidationError, match="Mismatched volume dimensions across modalities"):
        core.validate_modality_volumes(paths)


def test_mismatched_voxel_spacing_rejected(tmp_path):
    paths = make_modality_dict(tmp_path, zooms=(1.0, 1.0, 1.0))
    p_t1ce = tmp_path / "t1ce_zooms.nii.gz"
    make_volume(p_t1ce, zooms=(1.0, 1.0, 2.5))
    paths["t1ce"] = str(p_t1ce)
    with pytest.raises(core.SpatialValidationError, match="Mismatched voxel spacing across modalities"):
        core.validate_modality_volumes(paths)


def test_incompatible_affines_rejected(tmp_path):
    paths = make_modality_dict(tmp_path)
    p_t2 = tmp_path / "t2_rotated.nii.gz"
    shifted_affine = np.eye(4)
    shifted_affine[0, 3] = 45.0  # spatial translation mismatch
    make_volume(p_t2, affine=shifted_affine)
    paths["t2"] = str(p_t2)
    with pytest.raises(core.SpatialValidationError, match="Incompatible spatial affines across modalities"):
        core.validate_modality_volumes(paths)


def test_non_finite_affine_rejected(tmp_path, monkeypatch):
    paths = make_modality_dict(tmp_path)
    orig_load = nib.load

    def mock_load(p, *args, **kwargs):
        img = orig_load(p, *args, **kwargs)
        if "flair" in str(p):
            bad_affine = np.eye(4)
            bad_affine[0, 0] = np.nan
            monkeypatch.setattr(img, "_affine", bad_affine, raising=False)
        return img

    monkeypatch.setattr(nib, "load", mock_load)
    with pytest.raises(core.SpatialValidationError, match="contains non-finite values in its spatial affine matrix"):
        core.validate_modality_volumes(paths)


def test_non_finite_voxel_data_rejected(tmp_path):
    paths = make_modality_dict(tmp_path)
    p_flair = tmp_path / "flair_nan.nii.gz"
    vol = np.full((64, 64, 20), 50.0, dtype=np.float32)
    vol[10, 10, 10] = np.nan
    img = nib.Nifti1Image(vol, np.eye(4))
    nib.save(img, str(p_flair))
    paths["flair"] = str(p_flair)
    with pytest.raises(core.SpatialValidationError, match="contains non-finite \\(NaN or Inf\\) voxel data"):
        core.validate_modality_volumes(paths)


def test_all_zero_voxel_data_rejected(tmp_path):
    paths = make_modality_dict(tmp_path)
    p_t2 = tmp_path / "t2_zero.nii.gz"
    make_volume(p_t2, fill=0.0)
    paths["t2"] = str(p_t2)
    with pytest.raises(core.SpatialValidationError, match="contains only zero or empty voxel values"):
        core.validate_modality_volumes(paths)


def test_real_brats_scans_pass_validation():
    val_dir = os.path.join(ROOT, "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData", "BraTS20_Validation_001")
    if not os.path.exists(val_dir):
        pytest.skip("BraTS20_Validation_001 directory not present")
    paths = {
        "flair": os.path.join(val_dir, "BraTS20_Validation_001_flair.nii"),
        "t1ce": os.path.join(val_dir, "BraTS20_Validation_001_t1ce.nii"),
        "t2": os.path.join(val_dir, "BraTS20_Validation_001_t2.nii"),
    }
    images = core.validate_modality_volumes(paths)
    assert set(images.keys()) == {"flair", "t1ce", "t2"}
    for mod in ["flair", "t1ce", "t2"]:
        assert images[mod].shape == (240, 240, 155)


# ==============================================================================
# 2. API ENDPOINT REGRESSION TESTS
# ==============================================================================

@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    return server.app.test_client()


def nifti_upload_tuple(path):
    with open(path, "rb") as f:
        data = f.read()
    return (io.BytesIO(data), os.path.basename(path))


def test_api_mismatched_dimensions_returns_safe_400(client, tmp_path):
    paths = make_modality_dict(tmp_path, shape=(64, 64, 20))
    # Make T2 mismatched
    make_volume(tmp_path / "t2_bad.nii.gz", shape=(64, 64, 10))
    paths["t2"] = str(tmp_path / "t2_bad.nii.gz")

    form = {mod: nifti_upload_tuple(paths[mod]) for mod in core.REQUIRED_MODALITIES}
    r = client.post("/predict", data=form, content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["status"] == "error"
    assert "Mismatched volume dimensions across modalities" in body["error"]
    assert "Traceback" not in body["error"]
    assert "C:\\" not in body["error"]


def test_api_incompatible_affines_returns_safe_400(client, tmp_path):
    paths = make_modality_dict(tmp_path)
    bad_affine = np.eye(4)
    bad_affine[1, 3] = 99.0
    make_volume(tmp_path / "t2_shift.nii.gz", affine=bad_affine)
    paths["t2"] = str(tmp_path / "t2_shift.nii.gz")

    form = {mod: nifti_upload_tuple(paths[mod]) for mod in core.REQUIRED_MODALITIES}
    r = client.post("/predict", data=form, content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["status"] == "error"
    assert "Incompatible spatial affines across modalities" in body["error"]


def test_api_non_finite_voxel_data_returns_safe_400(client, tmp_path):
    paths = make_modality_dict(tmp_path)
    p_flair = tmp_path / "flair_nan.nii.gz"
    vol = np.full((64, 64, 20), 50.0, dtype=np.float32)
    vol[5, 5, 5] = np.nan
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p_flair))
    paths["flair"] = str(p_flair)

    form = {mod: nifti_upload_tuple(paths[mod]) for mod in core.REQUIRED_MODALITIES}
    r = client.post("/predict", data=form, content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["status"] == "error"
    assert "contains non-finite" in body["error"]


def test_api_oversized_upload_returns_413(client, tmp_path):
    orig_limit = server.app.config.get("MAX_CONTENT_LENGTH")
    try:
        # Set limit to 200 bytes so any upload is oversized
        server.app.config["MAX_CONTENT_LENGTH"] = 200
        paths = make_modality_dict(tmp_path)
        form = {mod: nifti_upload_tuple(paths[mod]) for mod in core.REQUIRED_MODALITIES}
        r = client.post("/predict", data=form, content_type="multipart/form-data")
        assert r.status_code == 413
        body = r.get_json()
        assert body["status"] == "error"
        assert "Upload payload exceeds maximum allowed size" in body["error"]
        assert "Traceback" not in body["error"]
    finally:
        server.app.config["MAX_CONTENT_LENGTH"] = orig_limit


def test_api_internal_error_does_not_leak_path_or_traceback(client, monkeypatch):
    def fake_inference(*args, **kwargs):
        raise RuntimeError("Simulated internal disk error at C:\\Users\\secret\\path\\file.dat")

    monkeypatch.setattr(core, "run_mc_dropout_inference", fake_inference)

    # Use a dummy temporary form
    p = os.path.join(ROOT, "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData", "BraTS20_Validation_001")
    if not os.path.exists(p):
        pytest.skip("BraTS data not present for internal error test")

    form = {
        "flair": nifti_upload_tuple(os.path.join(p, "BraTS20_Validation_001_flair.nii")),
        "t1ce": nifti_upload_tuple(os.path.join(p, "BraTS20_Validation_001_t1ce.nii")),
        "t2": nifti_upload_tuple(os.path.join(p, "BraTS20_Validation_001_t2.nii")),
    }
    r = client.post("/predict", data=form, content_type="multipart/form-data")
    assert r.status_code == 500
    body = r.get_json()
    assert body["status"] == "error"
    assert "Traceback" not in body["error"]
    assert "C:\\Users" not in body["error"]
    assert "An internal error occurred during inference" in body["error"]


def test_api_real_patient_scan_prediction(client):
    p = os.path.join(ROOT, "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData", "BraTS20_Validation_001")
    if not os.path.exists(p):
        pytest.skip("BraTS data not present for real patient prediction test")

    form = {
        "flair": nifti_upload_tuple(os.path.join(p, "BraTS20_Validation_001_flair.nii")),
        "t1ce": nifti_upload_tuple(os.path.join(p, "BraTS20_Validation_001_t1ce.nii")),
        "t2": nifti_upload_tuple(os.path.join(p, "BraTS20_Validation_001_t2.nii")),
    }
    r = client.post("/predict", data=form, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    body = r.get_json()
    assert body["status"] == "success"
    assert body["verdict"] in core.VERDICTS
    assert 0.0 <= body["reliability_score"] <= 1.0
    assert set(body["images"]) == {"mri", "segmentation_overlay", "uncertainty_heatmap"}
    assert all(v.startswith("data:image/png;base64,") for v in body["images"].values())
