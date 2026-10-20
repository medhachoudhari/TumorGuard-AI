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
    body = r.get_json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["training_stats_loaded"] is True
    assert body["limits"]["max_upload_mb"] > 0


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


# ---- Phase-2B-style API behaviour added in this release (synthetic volumes) ----------

def post(client, tmp_path, extra=None, **overrides):
    data = form(tmp_path, **overrides)
    data.update(extra or {})
    return client.post("/predict", data=data, content_type="multipart/form-data")


def test_original_response_fields_preserved_and_new_fields_added(client, tmp_path):
    b = post(client, tmp_path).get_json()
    for k in ("status", "slice_index", "familiarity_score", "uncertainty_score", "predictive_entropy",
              "max_probability_difference", "reliability_score", "verdict", "images", "disclaimer"):
        assert k in b
    for k in ("slice_selection_mode", "num_slices", "prediction_method", "mc_prediction", "deterministic_prediction",
              "prediction_settings", "mask_pixels", "image_info",
              "input_validation", "reliability_breakdown", "reliability_evidence_layer"):
        assert k in b
    assert b["slice_selection_mode"] == "automatic_flair_heuristic" and b["num_slices"] == 20
    assert b["prediction_settings"]["num_passes"] == 10
    assert b["prediction_method"] == "mc_dropout_consensus_10_passes"  # Phase 2B field preserved
    assert b["mc_prediction"]["tumor_pixels"] == b["mask_pixels"]
    assert b["deterministic_prediction"]["method"] == "single_pass_eval"
    ev = b["reliability_evidence_layer"]
    assert ev["legacy_verdict"] == b["verdict"] and ev["legacy_reliability_score"] == pytest.approx(b["reliability_score"])
    assert ev["provenance"]["checkpoint_sha256"]
    assert b["image_info"]["width"] == 128


def test_explicit_slice_selection(client, tmp_path):
    b = post(client, tmp_path, {"slice_index": "7"}).get_json()
    assert b["slice_index"] == 7 and b["slice_selection_mode"] == "explicit"


@pytest.mark.parametrize("bad", ["-1", "20", "999", "abc", "1.5"])
def test_invalid_slice_index_rejected(client, tmp_path, bad):
    r = post(client, tmp_path, {"slice_index": bad})
    assert r.status_code == 400 and r.get_json()["error_code"] == "INVALID_SLICE_INDEX"
    assert r.get_json()["status"] == "error"


def test_spatial_mismatch_rejected_via_api(client, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    r = post(client, tmp_path, t2=(io.BytesIO(nifti_bytes(other, "t2.nii.gz", shape=(64, 64, 21))), "t2.nii.gz"))
    assert r.status_code == 400 and r.get_json()["error_code"] == "SPATIAL_MISMATCH"


def test_error_bodies_do_not_leak_paths_or_tracebacks(client, tmp_path):
    cases = [
        post(client, tmp_path, t1ce=(io.BytesIO(b"not nifti"), "t1ce.nii.gz")),
        post(client, tmp_path, {"slice_index": "x"}),
        client.post("/predict", data={}, content_type="multipart/form-data"),
        client.get("/predict"),
        client.get("/nope"),
    ]
    for r in cases:
        text = r.get_data(as_text=True)
        assert r.status_code >= 400
        assert "Traceback" not in text and "/tmp" not in text and str(tmp_path) not in text
        assert "error_code" in r.get_json()


def test_internal_error_is_generic(client, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret /home/user/path detail")
    monkeypatch.setattr(server.tumorguard_core, "run_mc_dropout_inference", boom)
    r = post(client, tmp_path)
    assert r.status_code == 500
    assert "secret" not in r.get_data(as_text=True) and r.get_json()["error_code"] == "INTERNAL_ERROR"


def test_oversized_upload_rejected(client, tmp_path, monkeypatch):
    monkeypatch.setitem(server.app.config, "MAX_CONTENT_LENGTH", 1024)
    r = post(client, tmp_path)
    assert r.status_code == 413 and r.get_json()["error_code"] == "UPLOAD_TOO_LARGE"


def test_missing_familiarity_reference_yields_assessment_unavailable(client, tmp_path, monkeypatch):
    stats = {k: v for k, v in server.training_stats.items() if k != "familiarity_reference"}
    monkeypatch.setattr(server, "training_stats", stats)
    b = post(client, tmp_path).get_json()
    assert b["reliability_evidence_layer"]["recommendation"] == "ASSESSMENT_UNAVAILABLE"
    assert b["verdict"] in core.VERDICTS  # legacy behaviour (D-16) is unchanged


def test_repeated_requests_are_deterministic_in_shape_not_values(client, tmp_path):
    a = post(client, tmp_path).get_json(); b = post(client, tmp_path).get_json()
    assert a["slice_index"] == b["slice_index"] and a["image_info"] == b["image_info"] | {
        "heatmap_max_variance": a["image_info"]["heatmap_max_variance"]}
