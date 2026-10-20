"""API smoke tests with SYNTHETIC volumes. They check software behaviour only,
not segmentation quality."""
import io
import os

import numpy as np
import pytest

pytest.importorskip("flask_cors")
nib = pytest.importorskip("nibabel")

import tumorguard_core as core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(ROOT, "models", "tumorguard_model.pt")):
    pytest.skip("repository checkpoint not present", allow_module_level=True)

os.environ.setdefault("MPLBACKEND", "Agg")
import server  # noqa: E402  (loads the checkpoint strictly at import)


def nifti_bytes(tmp_path, name, shape=(64, 64, 20), seed=0):
    rng = np.random.default_rng(seed)
    x, y, z = np.mgrid[:shape[0], :shape[1], :shape[2]]
    brain = ((x - shape[0] / 2) ** 2 / 600 + (y - shape[1] / 2) ** 2 / 600 + (z - shape[2] / 2) ** 2 / 60) < 1
    vol = (brain * (100 + 10 * rng.standard_normal(shape))).astype(np.float32)
    path = tmp_path / name
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))
    return path.read_bytes()


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    return server.app.test_client()


def form(tmp_path, **overrides):
    data = {m: (io.BytesIO(nifti_bytes(tmp_path, f"{m}.nii.gz", seed=i)), f"{m}.nii.gz")
            for i, m in enumerate(["flair", "t1ce", "t2"])}
    data.update(overrides)
    return data


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok", "model_loaded": True, "training_stats_loaded": True}


def test_missing_modality_rejected(client, tmp_path):
    data = form(tmp_path)
    del data["t2"]
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400


def test_segmentation_upload_rejected(client, tmp_path):
    data = form(tmp_path, seg=(io.BytesIO(b"x"), "seg.nii.gz"))
    assert client.post("/predict", data=data, content_type="multipart/form-data").status_code == 400


def test_wrong_extension_rejected(client, tmp_path):
    data = form(tmp_path, flair=(io.BytesIO(b"x"), "flair.jpg"))
    assert client.post("/predict", data=data, content_type="multipart/form-data").status_code == 400


def test_corrupt_nifti_rejected(client, tmp_path):
    data = form(tmp_path, t1ce=(io.BytesIO(b"not nifti"), "t1ce.nii.gz"))
    assert client.post("/predict", data=data, content_type="multipart/form-data").status_code == 400


def test_synthetic_prediction_contract(client, tmp_path):
    r = client.post("/predict", data=form(tmp_path), content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    body = r.get_json()
    assert body["status"] == "success"
    assert body["verdict"] in core.VERDICTS
    assert 0.0 <= body["reliability_score"] <= 1.0
    assert core.verdict_from_score(body["reliability_score"]) == body["verdict"]
    assert set(body["images"]) == {"mri", "segmentation_overlay", "uncertainty_heatmap"}
    assert all(v.startswith("data:image/png;base64,") for v in body["images"].values())
    for key in ("familiarity_score", "uncertainty_score", "predictive_entropy", "max_probability_difference"):
        assert np.isfinite(body[key])
    assert "does NOT diagnose cancer" in body["disclaimer"]
