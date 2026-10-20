"""Real-data end-to-end integration tests using genuine BraTS 2020 validation data (Phase 2B Stage C)."""
import io
import os
import numpy as np
import pytest

pytest.importorskip("flask_cors")
nib = pytest.importorskip("nibabel")

import tumorguard_core as core
import server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_DIR = os.path.join(ROOT, "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData", "BraTS20_Validation_001")


def has_real_data():
    if not os.path.isdir(VAL_DIR):
        return False
    for m in core.REQUIRED_MODALITIES:
        f = os.path.join(VAL_DIR, f"BraTS20_Validation_001_{m}.nii")
        if not os.path.isfile(f):
            return False
    return True


pytestmark = pytest.mark.skipif(not has_real_data(), reason="Genuine BraTS20_Validation_001 dataset not present")


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    return server.app.test_client()


def real_patient_form(**overrides):
    files = {}
    for mod in core.REQUIRED_MODALITIES:
        p = os.path.join(VAL_DIR, f"BraTS20_Validation_001_{mod}.nii")
        with open(p, "rb") as f:
            data = f.read()
        files[mod] = (io.BytesIO(data), f"BraTS20_Validation_001_{mod}.nii")
    files.update(overrides)
    return files


def test_real_data_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["training_stats_loaded"] is True


def test_real_data_predict_automatic_slice(client):
    data = real_patient_form()
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    body = r.get_json()

    assert body["status"] == "success"
    # Automatic selection picks slice 69 for patient 001
    assert body["slice_index"] == 69
    assert body["slice_selection_mode"] == "automatic_flair_heuristic"
    assert body["prediction_method"] == "mc_dropout_consensus_10_passes"

    # Verdict and scores
    assert body["verdict"] in core.VERDICTS
    assert 0.0 <= body["reliability_score"] <= 1.0
    assert np.isfinite(body["familiarity_score"])
    assert np.isfinite(body["uncertainty_score"])
    assert np.isfinite(body["predictive_entropy"])
    assert np.isfinite(body["max_probability_difference"])

    # Predictions
    assert "mc_prediction" in body
    assert isinstance(body["mc_prediction"]["tumor_pixels"], int)
    assert "deterministic_prediction" in body
    assert isinstance(body["deterministic_prediction"]["tumor_pixels"], int)

    # Images
    for img_key in ("mri", "segmentation_overlay", "uncertainty_heatmap"):
        assert img_key in body["images"]
        assert body["images"][img_key].startswith("data:image/png;base64,")

    # Security check: no filesystem paths
    raw_text = r.get_data(as_text=True)
    assert "C:\\Users" not in raw_text
    assert "Traceback" not in raw_text


def test_real_data_predict_explicit_slice(client):
    data = real_patient_form(slice_index="75")
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["slice_index"] == 75
    assert body["slice_selection_mode"] == "explicit"
    assert body["verdict"] in core.VERDICTS


def test_real_data_reject_out_of_range_slice(client):
    data = real_patient_form(slice_index="300")
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["status"] == "error"
    assert "out of range" in body["error"]
    assert "300" in body["error"]


def test_real_data_reject_negative_slice(client):
    data = real_patient_form(slice_index="-10")
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["status"] == "error"
    assert "out of range" in body["error"]


def test_real_data_reject_missing_modality(client):
    data = real_patient_form()
    del data["t2"]
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["status"] == "error"
    assert "Missing required modalities" in body["error"]


def test_real_data_reject_corrupted_modality(client):
    data = real_patient_form(t2=(io.BytesIO(b"CORRUPTED_NON_NIFTI_BYTES"), "t2.nii.gz"))
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["status"] == "error"
    assert "Invalid or corrupted NIfTI file" in body["error"]
