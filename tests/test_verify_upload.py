"""``scripts/verify_upload.py`` checks a HuggingFace file listing against the manifest.

The live listing is faked (plain dicts), so these run without network or lab data.
"""

import importlib.util
import pathlib
import sys

_SCRIPT = (
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "verify_upload.py"
)
_spec = importlib.util.spec_from_file_location("verify_upload", _SCRIPT)
verify_upload = importlib.util.module_from_spec(_spec)
sys.modules["verify_upload"] = verify_upload
_spec.loader.exec_module(verify_upload)

DATA = "arcinstitute/bolero-data"
MODELS = "arcinstitute/bolero-models"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _shared(repo_path, size, repo_id=DATA, **extra):
    return {
        "share": True,
        "kind": "file",
        "category": "motif-scan",
        "tier": "core",
        "size": size,
        "repo_id": repo_id,
        "repo_path": repo_path,
        **extra,
    }


def test_matching_listing_is_clean():
    manifest = {"standard/motif_scan/A.feather": _shared("motif_scan/A.feather", 10)}
    live = {DATA: {"motif_scan/A.feather": {"size": 10, "sha256": SHA_A}}}

    report = verify_upload.verify(manifest, live)

    assert report.clean
    assert report.ok == [(DATA, "motif_scan/A.feather")]


def test_missing_and_mismatched_files_are_reported():
    manifest = {
        "a": _shared("motif_scan/A.feather", 10),
        "b": _shared("motif_scan/B.feather", 20),
        "c": _shared("motif_scan/C.feather", 30, sha256=SHA_A),
    }
    live = {
        DATA: {
            "motif_scan/B.feather": {"size": 21, "sha256": SHA_A},
            "motif_scan/C.feather": {"size": 30, "sha256": SHA_B},
        }
    }

    report = verify_upload.verify(manifest, live)

    assert not report.clean
    assert report.missing == [(DATA, "motif_scan/A.feather")]
    assert report.size_mismatch == [(DATA, "motif_scan/B.feather", 20, 21)]
    assert report.sha256_mismatch == [(DATA, "motif_scan/C.feather", SHA_A, SHA_B)]


def test_uploaded_withheld_files_are_flagged_apart_from_other_extras():
    withheld = {"share": False, "kind": "file", "category": "motif-scan", "size": 5}
    manifest = {
        "standard/motif_scan/A.feather": _shared("motif_scan/A.feather", 10),
        # withheld entry: no repo_path, so its upload path is derived from its category
        "standard/motif_scan/Secret.feather": withheld,
        "diff_analysis/BigWig/x.bw": {**withheld, "category": "diff-bigwig"},
    }
    live = {
        DATA: {
            "motif_scan/A.feather": {"size": 10, "sha256": SHA_A},
            "motif_scan/Secret.feather": {"size": 5, "sha256": SHA_A},
            "diff_analysis/BigWig/x.bw": {"size": 5, "sha256": SHA_A},
            "metadata/cell/Torpor.cell_metadata.feather": {"size": 1, "sha256": SHA_A},
            "stray/notes.txt": {"size": 1, "sha256": None},
            ".gitattributes": {"size": 1, "sha256": None},
            "README.md": {"size": 1, "sha256": None},
        }
    }

    report = verify_upload.verify(manifest, live)

    assert not report.clean
    assert report.withheld == [
        (DATA, "diff_analysis/BigWig/x.bw", "diff_analysis/BigWig/x.bw"),
        (DATA, "metadata/cell/Torpor.cell_metadata.feather", "excluded dataset Torpor"),
        (DATA, "motif_scan/Secret.feather", "standard/motif_scan/Secret.feather"),
    ]
    assert report.extra == [(DATA, "stray/notes.txt")]


def test_tar_packed_directory_is_found_and_size_checked_loosely():
    # A directory artifact is uploaded as <repo_path>.tar; the manifest size is the
    # directory's content size, so the tar may differ by its headers and padding.
    def tar_entry(name, size):
        return _shared(f"metacell/parquet/{name}", size, kind="dir", pack="tar")

    gib = 1 << 30
    manifest = {"x": tar_entry("X", 10 * gib), "y": tar_entry("Y", 10 * gib)}
    live = {
        DATA: {
            "metacell/parquet/X.tar": {"size": 10 * gib + 3_000_000, "sha256": SHA_A},
            "metacell/parquet/Y.tar": {"size": 5 * gib, "sha256": SHA_A},
        }
    }

    report = verify_upload.verify(manifest, live)

    assert report.ok == [(DATA, "metacell/parquet/X.tar")]
    assert report.size_mismatch == [(DATA, "metacell/parquet/Y.tar", 10 * gib, 5 * gib)]
    assert report.missing == report.extra == []


def test_files_expected_outside_the_manifest_are_checked_too():
    # reference_gtf/ files are not in the manifest; get_gtf_path names them and a
    # sha256sum file of the staged copies gives their checksums.
    gtf = "reference_gtf/hg38/gencode.v30.annotation.gtf"
    db = "reference_gtf/hg38/gencode.v30.annotation.gffutils.db"
    extra = verify_upload.with_checksums(
        {(DATA, gtf): {}, (DATA, db): {}},
        f"{SHA_A}  {gtf}\n{SHA_A}  {db}\n",
    )
    live = {DATA: {gtf: {"size": 7, "sha256": SHA_A}, db: {"size": 9, "sha256": SHA_B}}}

    report = verify_upload.verify({}, live, extra_expected=extra)

    assert report.ok == [(DATA, gtf)]
    assert report.sha256_mismatch == [(DATA, db, SHA_A, SHA_B)]
    assert report.extra == []
