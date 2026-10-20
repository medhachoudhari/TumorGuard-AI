import copy

import pytest
import torch

import tumorguard_core as core


@pytest.fixture(scope="module")
def reference_model():
    torch.manual_seed(0)
    return core.TumorGuardUNet()


def fresh_model():
    torch.manual_seed(1)
    return core.TumorGuardUNet()


def same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def save(obj, tmp_path, name="ckpt.pt"):
    path = tmp_path / name
    torch.save(obj, path)
    return str(path)


def test_raw_state_dict_loads(reference_model, tmp_path):
    path = save(reference_model.state_dict(), tmp_path)
    target = fresh_model()
    assert not same_weights(target, reference_model)
    meta = core.load_model_checkpoint(target, path)
    assert same_weights(target, reference_model)
    assert meta == {}


@pytest.mark.parametrize("key", ["model_state_dict", "state_dict"])
def test_wrapped_state_dict_loads_with_metadata(reference_model, tmp_path, key):
    path = save({key: reference_model.state_dict(), "epoch": 20, "val_loss": 0.35}, tmp_path)
    target = fresh_model()
    meta = core.load_model_checkpoint(target, path)
    assert same_weights(target, reference_model)
    assert meta == {"epoch": 20, "val_loss": 0.35}


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.load_model_checkpoint(fresh_model(), str(tmp_path / "nope.pt"))


@pytest.mark.parametrize("obj", [[1, 2, 3], torch.zeros(2), "text", 5])
def test_non_dict_payload_rejected(tmp_path, obj):
    with pytest.raises(core.CheckpointError):
        core.load_model_checkpoint(fresh_model(), save(obj, tmp_path))


def test_unreadable_file_rejected(tmp_path):
    p = tmp_path / "garbage.pt"
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(core.CheckpointError):
        core.load_model_checkpoint(fresh_model(), str(p))


def test_arbitrary_pickle_blocked_by_weights_only(tmp_path):
    class Evil:
        def __reduce__(self):
            return (eval, ("1+1",))

    p = tmp_path / "evil.pt"
    torch.save({"model_state_dict": Evil()}, p)
    with pytest.raises(core.CheckpointError):
        core.load_model_checkpoint(fresh_model(), str(p))


def test_empty_and_unknown_wrapper_rejected(tmp_path):
    for obj in ({}, {"model": {"w": torch.zeros(1)}}, {"model_state_dict": {}}, {"model_state_dict": [1]}):
        with pytest.raises(core.CheckpointError):
            core.load_model_checkpoint(fresh_model(), save(obj, tmp_path))


def test_ambiguous_wrapper_rejected(reference_model, tmp_path):
    sd = reference_model.state_dict()
    with pytest.raises(core.CheckpointError, match="Ambiguous"):
        core.load_model_checkpoint(fresh_model(), save({"model_state_dict": sd, "state_dict": sd}, tmp_path))


def test_missing_key_rejected_and_model_unmodified(reference_model, tmp_path):
    sd = dict(reference_model.state_dict())
    dropped = next(iter(sd))
    del sd[dropped]
    target = fresh_model()
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(core.CheckpointError, match="missing keys"):
        core.load_model_checkpoint(target, save(sd, tmp_path))
    assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())


def test_unexpected_key_rejected_strict_but_allowed_non_strict(reference_model, tmp_path):
    sd = dict(reference_model.state_dict())
    sd["extra.weight"] = torch.zeros(1)
    path = save(sd, tmp_path)
    with pytest.raises(core.CheckpointError, match="unexpected keys"):
        core.load_model_checkpoint(fresh_model(), path)
    target = fresh_model()
    core.load_model_checkpoint(target, path, strict=False)
    assert same_weights(target, reference_model)


def test_shape_mismatch_rejected_even_non_strict(reference_model, tmp_path):
    sd = dict(reference_model.state_dict())
    key = next(k for k, v in sd.items() if v.ndim == 4)
    sd[key] = torch.zeros(1, 1, 1, 1)
    path = save(sd, tmp_path)
    for strict in (True, False):
        with pytest.raises(core.CheckpointError, match="shape mismatches"):
            core.load_model_checkpoint(fresh_model(), path, strict=strict)


def test_wrong_input_channels_rejected(tmp_path):
    other = core.TumorGuardUNet(in_channels=1)
    with pytest.raises(core.CheckpointError, match="shape mismatches"):
        core.load_model_checkpoint(fresh_model(), save(other.state_dict(), tmp_path))


def test_non_finite_weights_rejected(reference_model, tmp_path):
    sd = {k: v.clone() for k, v in reference_model.state_dict().items()}
    key = next(k for k, v in sd.items() if v.is_floating_point())
    sd[key].view(-1)[0] = float("nan")
    with pytest.raises(core.CheckpointError, match="non-finite"):
        core.load_model_checkpoint(fresh_model(), save(sd, tmp_path))


def test_error_message_does_not_leak_directory(tmp_path):
    p = tmp_path / "secret_dir_name"
    p.mkdir()
    with pytest.raises(FileNotFoundError) as e:
        core.load_model_checkpoint(fresh_model(), str(p / "missing.pt"))
    assert "secret_dir_name" not in str(e.value)


def test_map_location_cpu_for_tensors(reference_model, tmp_path):
    payload = core.load_checkpoint_payload(save(reference_model.state_dict(), tmp_path), map_location="cpu")
    assert all(v.device.type == "cpu" for v in payload.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_onto_cuda_model(reference_model, tmp_path):
    target = fresh_model().to("cuda")
    core.load_model_checkpoint(target, save(reference_model.state_dict(), tmp_path), map_location="cuda")
    assert next(target.parameters()).device.type == "cuda"


def test_repository_checkpoint_loads_strictly():
    import os
    path = os.path.join(os.path.dirname(core.__file__), "models", "tumorguard_model.pt")
    if not os.path.exists(path):
        pytest.skip("repository checkpoint not present")
    model = core.TumorGuardUNet()
    meta = core.load_model_checkpoint(model, path)
    assert isinstance(meta, dict)
    out = model.eval()(torch.zeros(1, 3, 128, 128))
    assert out.shape == (1, 1, 128, 128)
