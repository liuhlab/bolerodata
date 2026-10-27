"""Tests for :func:`bolerodata.get_flashzoi_checkpoint`.

The HuggingFace download and torch's file IO are faked, so these tests need no
network, no lab data and no torch (bolerodata's lean env has none).
"""

import pathlib
import pickle
import sys
import types

import pytest

import bolerodata
from bolerodata import flashzoi

# A miniature of the public johahi/flashzoi-replicate-* state dict: one key per
# kind the rename has to handle (plain, transformer sub-layer, output heads).
PUBLIC_STATE = {
    "conv_dna.conv_layer.weight": "w0",
    "transformer.0.0.fn.0.weight": "t00",
    "transformer.7.1.fn.3.bias": "t71",
    "human_head.weight": "hh",
    "mouse_head.bias": "mh",
}
# What bolero's ``Borzoi`` expects, written out by hand from its module names.
BOLERO_STATE = {
    "conv_dna.conv_layer.weight": "w0",
    "transformer.0.layers.0.fn.0.weight": "t00",
    "transformer.7.layers.1.fn.3.bias": "t71",
}


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Fake torch IO and the HuggingFace download; record each download call."""
    fake_torch = types.ModuleType("torch")

    def load(path, map_location=None, weights_only=False):
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    fake_torch.load, fake_torch.save = load, save
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    calls = []

    def hf_hub_download(repo_id, filename, revision=None, cache_dir=None, **kw):
        calls.append((repo_id, filename, revision))
        src = tmp_path / "hub" / repo_id / filename
        src.parent.mkdir(parents=True, exist_ok=True)
        save(PUBLIC_STATE, src)
        return str(src)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setenv("BOLERODATA_HOME", str(tmp_path / "cache"))
    # Pretend the lab filesystem is absent, as for an external user.
    monkeypatch.setattr(flashzoi, "_LAB_PATH", str(tmp_path / "nolab/fold_{fold}.pt"))
    return calls


def test_returns_path_with_bolero_keys(fake_hub):
    path = bolerodata.get_flashzoi_checkpoint(0)

    assert pathlib.Path(path).is_file()
    with open(path, "rb") as f:
        assert pickle.load(f) == BOLERO_STATE
    assert fake_hub == [
        ("johahi/flashzoi-replicate-0", "pytorch_model.bin", flashzoi.REVISIONS[0])
    ]


def test_second_call_uses_cache(fake_hub):
    first = bolerodata.get_flashzoi_checkpoint(1)
    second = bolerodata.get_flashzoi_checkpoint(1)

    assert first == second
    assert len(fake_hub) == 1


def test_lab_checkpoint_used_when_present(fake_hub, tmp_path, monkeypatch):
    lab = tmp_path / "lab" / "fold_2.renamed.pt"
    lab.parent.mkdir()
    lab.write_bytes(b"lab")
    lab_pattern = str(tmp_path / "lab" / "fold_{fold}.renamed.pt")
    monkeypatch.setattr(flashzoi, "_LAB_PATH", lab_pattern)

    assert bolerodata.get_flashzoi_checkpoint(2) == lab
    assert fake_hub == []


@pytest.mark.parametrize("fold", [4, -1, "0", 1.0, True])
def test_unsupported_fold_raises(fake_hub, fold):
    with pytest.raises(ValueError, match="fold"):
        bolerodata.get_flashzoi_checkpoint(fold)
    assert fake_hub == []


def test_localize_maps_config_backbone_path(fake_hub, monkeypatch):
    # Published model configs name the backbone by its lab path's logical form.
    from bolerodata import _sync

    monkeypatch.setattr(_sync, "_lab_roots", lambda: [])
    path = _sync.localize("wmb/model/Borzoi/flashzoi_checkpoints/fold_1.renamed.pt")

    assert path == bolerodata.get_flashzoi_checkpoint(1)
    with open(path, "rb") as f:
        assert pickle.load(f) == BOLERO_STATE
