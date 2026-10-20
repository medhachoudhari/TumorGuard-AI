"""Tests for Stage B: Inference Correctness (Slice Selection & Deterministic vs MC Dropout)."""
import io
import os
import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("flask_cors")
nib = pytest.importorskip("nibabel")

import tumorguard_core as core
import server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    return server.app.test_client()


def make_volume(path, shape=(64, 64, 20), affine=None, zooms=None, fill=50.0):
    if affine is None:
        affine = np.eye(4)
    vol = np.full(shape, fill, dtype=np.float32)
    img = nib.Nifti1Image(vol, affine)
    if zooms is not None:
        img.header.set_zooms(zooms)
    nib.save(img, str(path))
    return str(path)


def make_form_data(tmp_path, shape=(64, 64, 20), **extra_fields):
    data = {}
    for mod in core.REQUIRED_MODALITIES:
        p = tmp_path / f"{mod}.nii.gz"
        make_volume(p, shape=shape)
        data[mod] = (io.BytesIO(p.read_bytes()), f"{mod}.nii.gz")
    data.update(extra_fields)
    return data


# ==============================================================================
# 1. SLICE SELECTION UNIT & API TESTS
# ==============================================================================

def test_explicit_slice_valid(client, tmp_path):
    data = make_form_data(tmp_path, shape=(64, 64, 20), slice_index="5")
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "success"
    assert body["slice_index"] == 5
    assert body["slice_selection_mode"] == "explicit"


def test_explicit_slice_first_slice(client, tmp_path):
    data = make_form_data(tmp_path, shape=(64, 64, 20), slice_index="0")
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["slice_index"] == 0
    assert body["slice_selection_mode"] == "explicit"


def test_explicit_slice_last_slice(client, tmp_path):
    data = make_form_data(tmp_path, shape=(64, 64, 20), slice_index="19")
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["slice_index"] == 19
    assert body["slice_selection_mode"] == "explicit"


def test_explicit_slice_negative_rejected(client, tmp_path):
    data = make_form_data(tmp_path, shape=(64, 64, 20), slice_index="-1")
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["status"] == "error"
    assert "out of range" in body["error"]


def test_explicit_slice_out_of_range_rejected(client, tmp_path):
    data = make_form_data(tmp_path, shape=(64, 64, 20), slice_index="20")
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["status"] == "error"
    assert "out of range" in body["error"]


def test_explicit_slice_non_integer_rejected(client, tmp_path):
    for bad in ["abc", "3.5", "ten"]:
        data = make_form_data(tmp_path, shape=(64, 64, 20), slice_index=bad)
        r = client.post("/predict", data=data, content_type="multipart/form-data")
        assert r.status_code == 400
        body = r.get_json()
        assert body["status"] == "error"
        assert "must be an integer" in body["error"]


def test_automatic_slice_selection_fallback(client, tmp_path):
    data = make_form_data(tmp_path, shape=(64, 64, 20))
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "success"
    assert isinstance(body["slice_index"], int)
    assert 0 <= body["slice_index"] < 20
    assert body["slice_selection_mode"] == "automatic_flair_heuristic"


# ==============================================================================
# 2. DETERMINISTIC VS MC DROPOUT EXECUTION TESTS
# ==============================================================================

def test_deterministic_inference_eval_mode():
    model = core.TumorGuardUNet()
    x = torch.randn(1, 3, 128, 128)
    probs, preds = core.run_deterministic_inference(model, x, threshold=0.5)

    assert probs.shape == (1, 1, 128, 128)
    assert preds.shape == (1, 1, 128, 128)
    assert ((preds == 0.0) | (preds == 1.0)).all()
    assert torch.isfinite(probs).all()

    # Running twice produces identical numbers bit-for-bit
    probs2, preds2 = core.run_deterministic_inference(model, x, threshold=0.5)
    assert torch.equal(probs, probs2)
    assert torch.equal(preds, preds2)

    # Verify model is left in eval mode
    assert not model.training
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            assert not m.training


def test_mc_dropout_stochastic_passes():
    model = core.TumorGuardUNet()
    x = torch.randn(1, 3, 128, 128)
    mean_pred, var_map, unc_score, max_diff, entropy = core.run_mc_dropout_inference(model, x, num_passes=10)

    assert mean_pred.shape == (1, 1, 128, 128)
    assert var_map.shape == (1, 1, 128, 128)
    assert np.isfinite(unc_score)
    assert np.isfinite(max_diff)
    assert np.isfinite(entropy)

    # Verify variance across stochastic passes is non-negative
    assert (var_map >= 0.0).all()

    # Verify model is reset to eval mode
    assert not model.training
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            assert not m.training


def test_api_prediction_response_contract(client, tmp_path):
    data = make_form_data(tmp_path, shape=(64, 64, 20))
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()

    # Existing frontend contract preserved
    assert body["status"] == "success"
    assert "segmentation_overlay" in body["images"]
    assert "uncertainty_heatmap" in body["images"]
    assert "mri" in body["images"]
    assert body["verdict"] in core.VERDICTS
    assert 0.0 <= body["reliability_score"] <= 1.0

    # New metadata fields
    assert body["prediction_method"] == "mc_dropout_consensus_10_passes"
    assert "mc_prediction" in body
    assert isinstance(body["mc_prediction"]["tumor_pixels"], int)
    assert body["mc_prediction"]["threshold"] == 0.5

    assert "deterministic_prediction" in body
    assert isinstance(body["deterministic_prediction"]["tumor_pixels"], int)
    assert body["deterministic_prediction"]["threshold"] == 0.5
    assert body["deterministic_prediction"]["method"] == "single_pass_eval"
