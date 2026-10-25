"""Tests for the reference-GTF registry, with the HuggingFace download faked."""

import pathlib
import subprocess
import sys

import huggingface_hub
import pytest

import bolerodata
from bolerodata._sync import FileNotAvailableError, manifest


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Replace ``hf_hub_download`` with a stub that writes the file locally."""
    calls = []

    def _fake(*, repo_id, repo_type, filename, cache_dir, **kwargs):
        calls.append({"repo_id": repo_id, "repo_type": repo_type, "filename": filename})
        local = tmp_path / "hub" / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("fake")
        return str(local)

    monkeypatch.setenv("BOLERODATA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake)
    return calls


def test_returns_cached_gffutils_db_by_default(fake_hub):
    path = bolerodata.get_gtf_path("hg38")

    assert pathlib.Path(path).is_file()
    assert fake_hub == [
        {
            "repo_id": "arcinstitute/bolero-data",
            "repo_type": "dataset",
            "filename": "reference_gtf/hg38/gencode.v30.annotation.gffutils.db",
        }
    ]


def test_db_false_returns_the_raw_gtf(fake_hub):
    path = bolerodata.get_gtf_path("mCalJac1.pat.X", db=False)

    assert str(path).endswith("Callithrix_jacchus.mCalJac1.pat.X.114.chr.gtf")
    assert fake_hub[0]["filename"] == (
        "reference_gtf/mCalJac1.pat.X/Callithrix_jacchus.mCalJac1.pat.X.114.chr.gtf"
    )


def test_unknown_genome_raises_keyerror_naming_known_genomes(fake_hub):
    with pytest.raises(KeyError) as err:
        bolerodata.get_gtf_path("hg19")

    msg = str(err.value)
    for genome in ("mm10", "hg38", "calJac4", "mCalJac1.pat.X", "monDom5.split",
                   "panPan3", "rheMac10"):
        assert genome in msg
    assert fake_hub == []


def test_allow_download_false_raises_when_not_cached(fake_hub):
    with pytest.raises(FileNotAvailableError):
        bolerodata.get_gtf_path("hg38", allow_download=False)
    assert fake_hub == []


def test_manifest_shares_every_gtf_and_database_under_reference_gtf():
    gtf = [e for e in manifest().values() if e.get("category") == "reference-gtf"]

    assert len(gtf) == 14
    assert all(e["share"] and e["repo_id"] == "arcinstitute/bolero-data" for e in gtf)
    assert all(e["size"] > 0 for e in gtf)
    assert "reference_gtf/rheMac10/rheMac10.ensGene.gffutils.db" in {
        e["repo_path"] for e in gtf
    }


def test_import_pulls_in_no_bolero():
    code = "import sys, bolerodata; print('bolero' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "False"
