"""Audit tests: MC Dropout run-to-run variation and the real-inference contract.

SYNTHETIC input only. These tests show how the software behaves; they say nothing about
how the model behaves on real MRI. Variation statistics are written to
$TUMORGUARD_VARIATION_OUT (if set) so they can be quoted without being hard-coded.
"""
import io
import json
import os
import statistics

import numpy as np
import pytest
import torch
import torch.nn as nn

nib = pytest.importorskip("nibabel")
pytest.importorskip("flask_cors")

import tumorguard_core as core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "tumorguard_model.pt")
if not os.path.exists(CKPT):
    pytest.skip("repository checkpoint not present", allow_module_level=True)

import server  # noqa: E402

SHAPE = (64, 64, 20)


def write_trio(tmp_path):
    x, y, z = np.mgrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    brain = (((x - 32) / 20) ** 2 + ((y - 32) / 24) ** 2 + ((z - 10) / 8) ** 2) < 1
    paths = {}
    for i, m in enumerate(("flair", "t1ce", "t2")):
        rng = np.random.default_rng(i)
        v = (brain * (100 + 12 * rng.standard_normal(SHAPE))).astype(np.float32)
        v[((x - 38) ** 2 + (y - 32) ** 2 + (z - 10) ** 2) < 20] += 100
        p = tmp_path / f"{m}.nii.gz"
        nib.save(nib.Nifti1Image(v, np.eye(4)), str(p))
        paths[m] = str(p)
    return paths


@pytest.fixture(scope="module")
def trio(tmp_path_factory):
    return write_trio(tmp_path_factory.mktemp("trio"))


@pytest.fixture()
def model():
    m = core.TumorGuardUNet()
    core.load_model_checkpoint(m, CKPT)
    return m.eval()


def tensor(trio, s=10):
    return core.load_and_preprocess_mri(trio["flair"], trio["t1ce"], trio["t2"], slice_idx=s).unsqueeze(0)


# ---- where randomness comes from --------------------------------------------------

def test_only_dropout_modules_are_stochastic_during_mc(model):
    core.enable_dropout(model)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            assert m.training
        elif len(list(m.children())) == 0:  # leaf modules: BatchNorm, Conv, ReLU, ...
            assert not m.training, type(m).__name__
    assert sum(isinstance(m, nn.Dropout) for m in model.modules()) > 0


def test_model_state_is_restored_after_mc_inference(model, trio):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    core.run_mc_dropout_inference(model, tensor(trio), num_passes=10)
    assert not any(m.training for m in model.modules()), "a module was left in train mode"
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "weights or BN buffers changed"


def test_deterministic_pass_is_bitwise_repeatable_and_unaffected_by_mc(model, trio):
    x = tensor(trio)
    p1, b1 = core.run_deterministic_inference(model, x)
    core.run_mc_dropout_inference(model, x, num_passes=10)
    p2, b2 = core.run_deterministic_inference(model, x)
    assert torch.equal(p1, p2) and torch.equal(b1, b2)


def test_preprocessing_is_deterministic(trio):
    assert torch.equal(tensor(trio), tensor(trio))


def test_mc_is_reproducible_only_with_a_seed(model, trio):
    x = tensor(trio)
    torch.manual_seed(123); a = core.run_mc_dropout_inference(model, x, 10)
    torch.manual_seed(123); b = core.run_mc_dropout_inference(model, x, 10)
    torch.manual_seed(124); c = core.run_mc_dropout_inference(model, x, 10)
    assert torch.equal(a[0], b[0]) and a[2:] == b[2:]
    assert not torch.equal(a[0], c[0]) and a[2] != c[2]


# ---- API response is derived from the real model passes -----------------------------

def test_api_outputs_equal_direct_pipeline_with_same_seed(trio):
    """Same seed + same file + same slice => identical mask size, uncertainty and verdict inputs.
    This can only hold if the API returns what the real model passes produce."""
    s = 10
    x = tensor(trio, s)
    torch.manual_seed(7)
    # server runs the deterministic pass first (no RNG use), then MC: mirror that order.
    core.run_deterministic_inference(server.model, x)
    mean, var, unc, md, ent = core.run_mc_dropout_inference(server.model, x, 10)
    expected_mask = int((mean > 0.5).sum().item())
    fam = core.calculate_similarity_score(x, server.training_stats)
    exp_score, exp_verdict = core.calculate_reliability_and_verdict(fam, ent, unc, md)

    client = server.app.test_client()
    data = {m: (io.BytesIO(open(p, "rb").read()), f"{m}.nii.gz") for m, p in trio.items()}
    data["slice_index"] = str(s)
    torch.manual_seed(7)
    body = client.post("/predict", data=data, content_type="multipart/form-data").get_json()

    assert body["slice_selection_mode"] == "explicit" and body["slice_index"] == s
    assert body["mc_prediction"]["tumor_pixels"] == body["mask_pixels"] == expected_mask
    assert body["uncertainty_score"] == pytest.approx(unc, rel=1e-6)
    assert body["predictive_entropy"] == pytest.approx(ent, rel=1e-6)
    assert body["max_probability_difference"] == pytest.approx(md, rel=1e-6)
    assert body["familiarity_score"] == pytest.approx(fam, rel=1e-9)
    assert body["reliability_score"] == pytest.approx(exp_score, rel=1e-9) and body["verdict"] == exp_verdict
    ev = body["reliability_evidence_layer"]
    assert ev["legacy_verdict"] == exp_verdict  # evidence layer reports, never replaces
    assert body["deterministic_prediction"]["method"] == "single_pass_eval"
    assert body["prediction_method"] == "mc_dropout_consensus_10_passes"


def test_evidence_layer_does_not_change_inputs(trio):
    import copy
    import reliability_evidence as ev
    args = dict(input_summary={"shape": [1, 1, 1]}, familiarity=0.3, familiarity_reference_available=True,
                entropy=0.03, variance=0.0002, max_diff=0.7, mask_pixels=10, reliability_score=0.55,
                verdict=core.VERDICT_MODERATE)
    snap = copy.deepcopy(args)
    ev.build_evidence_report(**args)
    assert args == snap


# ---- recorded run-to-run variation -----------------------------------------------

N_RUNS = int(os.environ.get("TUMORGUARD_VARIATION_RUNS", "40"))


def test_record_verdict_variation_over_repeated_unseeded_runs(model, trio):
    x = tensor(trio)
    fam = core.calculate_similarity_score(x, server.training_stats)
    scores, verdicts, masks, uncs, mds = [], [], [], [], []
    for _ in range(N_RUNS):
        mean, var, unc, md, ent = core.run_mc_dropout_inference(model, x, 10)
        s, v = core.calculate_reliability_and_verdict(fam, ent, unc, md)
        scores.append(s); verdicts.append(v); masks.append(int((mean > 0.5).sum())); uncs.append(unc); mds.append(md)
    counts = {v: verdicts.count(v) for v in sorted(set(verdicts))}
    stats = {
        "runs": N_RUNS, "passes_per_run": 10, "input": "synthetic volume (not MRI)", "familiarity": fam,
        "score_min": min(scores), "score_max": max(scores), "score_mean": statistics.mean(scores),
        "score_std": statistics.pstdev(scores), "verdict_counts": counts,
        "mask_pixels_min": min(masks), "mask_pixels_max": max(masks),
        "max_diff_min": min(mds), "max_diff_max": max(mds), "unc_min": min(uncs), "unc_max": max(uncs),
        "distance_of_mean_score_to_nearest_cutoff": min(abs(statistics.mean(scores) - c) for c in (0.3, 0.5, 0.7)),
    }
    out = os.environ.get("TUMORGUARD_VARIATION_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(stats, f, indent=2)
    assert all(v in core.VERDICTS for v in verdicts)
    assert all(np.isfinite(scores))
    assert statistics.pstdev(scores) > 0  # stochastic by design; seeding would be needed for equality
