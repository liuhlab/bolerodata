"""Bolero-AP1-Score and its AP-1 score file resolve from HuggingFace off-lab.

The HuggingFace download is faked, so these run without network or lab data.
"""

import json
import pathlib

import huggingface_hub
import pytest

from bolerodata import MODELS, metadata
from bolerodata._sync import (
    HF_DATA_REPO,
    HF_MODELS_REPO,
    FileNotAvailableError,
    _to_logical,
    manifest,
)
from bolerodata.data import SCORE_DATA_DIR

AP1_SCORE_MODEL = "Bolero10M+AP1+260104"
AP1_SCORE_FILE = "AP1.chromvar_score.scale.joblib.gz"


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Replace ``hf_hub_download`` with a fake that writes a stub file."""
    monkeypatch.setenv("BOLERODATA_HOME", str(tmp_path / "cache"))
    calls = []

    def fake_download(*, repo_id, repo_type, revision, filename, cache_dir):
        calls.append({"repo_id": repo_id, "repo_type": repo_type, "filename": filename})
        out = pathlib.Path(cache_dir) / repo_id / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}" if filename.endswith(".json") else "stub")
        return str(out)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return calls


def _is_lab_file(path):
    return pathlib.Path(path).exists()


def test_ap1_score_path_downloads_from_bolero_data(fake_hub):
    lab_path = pathlib.Path(SCORE_DATA_DIR) / AP1_SCORE_FILE
    if _is_lab_file(lab_path):
        pytest.skip("lab file present: the getter returns it without downloading")

    path = metadata.get_chromvar_score_path("AP1")

    assert path.exists()
    assert fake_hub == [
        {
            "repo_id": HF_DATA_REPO,
            "repo_type": "dataset",
            "filename": f"score_data/{AP1_SCORE_FILE}",
        }
    ]


def test_ap1_score_path_offline_raises_when_not_cached(fake_hub):
    if _is_lab_file(pathlib.Path(SCORE_DATA_DIR) / AP1_SCORE_FILE):
        pytest.skip("lab file present: the getter returns it without downloading")

    with pytest.raises(FileNotAvailableError, match="allow_download=False"):
        metadata.get_chromvar_score_path("AP1", allow_download=False)
    assert fake_hub == []


def test_ap1_score_model_resolves_from_bolero_models(fake_hub):
    row = metadata.MODEL_ZOO.loc[AP1_SCORE_MODEL]
    if _is_lab_file(row["ConfigPath"]):
        pytest.skip("lab files present: the model resolves without downloading")

    model = MODELS[AP1_SCORE_MODEL]
    ckpt = model.ckpt_path

    assert ckpt.exists()
    assert [c["filename"] for c in fake_hub] == [
        f"BorzoiSignalMultiDS/{AP1_SCORE_MODEL}/config.json",
        f"BorzoiSignalMultiDS/{AP1_SCORE_MODEL}/checkpoint.pt",
    ]
    assert {c["repo_id"] for c in fake_hub} == {HF_MODELS_REPO}


def test_only_the_ap1_score_model_is_shared():
    zoo = metadata.MODEL_ZOO
    man = manifest()
    motif_rows = zoo[zoo.index.str.endswith("+260104")]
    assert len(motif_rows) == 26
    for key, row in motif_rows.iterrows():
        cfg = man[_to_logical(row["ConfigPath"])]
        assert cfg["share"] is (key == AP1_SCORE_MODEL), key
    # all 26 rows map one checkpoint file; it is shared under the AP-1 key
    ckpt = man[_to_logical(motif_rows.loc[AP1_SCORE_MODEL, "CkptPath"])]
    assert ckpt["share"] and ckpt["slimmed"]
    assert ckpt["repo_path"] == f"BorzoiSignalMultiDS/{AP1_SCORE_MODEL}/checkpoint.pt"


def test_other_motif_score_files_are_not_shared():
    shared = [k for k, e in manifest().items() if e["share"] and "chromvar_score" in k]
    assert shared == [f"projects/260102-ScoreModel/score_data/{AP1_SCORE_FILE}"]


def test_manifest_has_no_lab_paths():
    text = json.dumps(manifest())
    assert "/large_storage" not in text


def test_unpublished_motif_score_file_raises_naming_the_published_ones(fake_hub):
    with pytest.raises(ValueError, match="AP1"):
        metadata.get_chromvar_score_path("CTCF")
    assert fake_hub == []
