"""
Check the live HuggingFace repos against the manifest's shared entries.

Lists every file in bolero-data and bolero-models (path, size, LFS sha256) and
compares the listing with what should be there:

- every ``share = True`` manifest entry (a tar-packed directory as
  ``<repo_path>.tar``);
- files the package fetches outside the manifest: the ``reference_gtf/`` files
  named by :mod:`bolerodata.gtf` (when that module is present).

It reports files that are missing, mismatched (size, or sha256 where an
expected checksum is known), withheld (an uploaded file that is a
``share = False`` manifest entry or belongs to an excluded dataset) and extra
(any other unexpected file). The manifest carries no sha256 unless built with
``build_manifest.py --hash``; pass ``--checksums`` (``sha256sum`` format,
repo paths) to check more files by content.

Usage::

    python scripts/verify_upload.py [--checksums FILE ...] [--out REPORT.md]

Exits 0 when the report is clean, 1 otherwise. Read-only: it never uploads.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # for build_manifest

from build_manifest import MANIFEST_PATH, _data_repo_path

from bolerodata._sync import EXCLUDED_DATASETS, HF_DATA_REPO, HF_MODELS_REPO

#: Repo housekeeping files that are expected but not in the manifest.
REPO_META_FILES = {".gitattributes", "README.md"}

#: A tar's size differs from its directory's ``du -sb`` size by tar headers and
#: padding (up) and directory inode sizes (down); allow this fraction plus 1 MiB.
#: The 77 uploaded tars measured 0.00%-0.08% larger than their directories.
TAR_SIZE_REL_TOL = 0.005

#: repo_id -> repo_type of the repos to check.
REPOS = {HF_DATA_REPO: "dataset", HF_MODELS_REPO: "model"}


@dataclasses.dataclass
class Report:
    """Result of comparing a live listing with the expected files."""

    ok: list = dataclasses.field(default_factory=list)
    missing: list = dataclasses.field(default_factory=list)
    size_mismatch: list = dataclasses.field(default_factory=list)
    sha256_mismatch: list = dataclasses.field(default_factory=list)
    withheld: list = dataclasses.field(default_factory=list)
    extra: list = dataclasses.field(default_factory=list)
    #: (repo_id, path) of present files whose content could not be checked.
    sha256_unchecked: list = dataclasses.field(default_factory=list)

    @property
    def clean(self):
        """True when nothing is missing, extra, withheld or mismatched."""
        return not (
            self.missing
            or self.size_mismatch
            or self.sha256_mismatch
            or self.withheld
            or self.extra
        )


# ---------------------------------------------------------------------------
# Expected files
# ---------------------------------------------------------------------------
def expected_from_manifest(manifest):
    """
    Return the files the manifest's shared entries are uploaded as.

    Parameters
    ----------
    manifest : dict
        ``logical_path -> entry`` as in ``hf_manifest.json``.

    Returns
    -------
    dict
        ``(repo_id, repo_path) -> {"size", "sha256", "tar"}``. ``repo_path`` of
        a tar-packed directory ends in ``.tar``.
    """
    out = {}
    for entry in manifest.values():
        if not entry.get("share"):
            continue
        tar = entry.get("kind") == "dir" and entry.get("pack") == "tar"
        path = entry["repo_path"] + (".tar" if tar else "")
        out[(entry["repo_id"], path)] = {
            "size": entry.get("size"),
            "sha256": entry.get("sha256"),
            "tar": tar,
        }
    return out


def gtf_expected():
    """
    Return the ``reference_gtf/`` files :mod:`bolerodata.gtf` downloads.

    Returns
    -------
    dict
        ``(repo_id, repo_path) -> {}`` for each genome's GTF and gffutils
        database; empty if this checkout has no ``bolerodata.gtf``.
    """
    try:
        from bolerodata import gtf
    except ImportError:
        return {}
    return {
        (HF_DATA_REPO, gtf._repo_path(genome, db)): {}
        for genome in gtf.GTF_FILES
        for db in (False, True)
    }


def with_checksums(expected, sha256sum_text):
    """
    Add sha256 values from ``sha256sum`` output to the expected files.

    Parameters
    ----------
    expected : dict
        ``(repo_id, repo_path) -> info``.
    sha256sum_text : str
        Lines ``<sha256>  <repo_path>``; paths with no expected file are ignored.

    Returns
    -------
    dict
        A copy of ``expected`` with ``sha256`` set where a line names the path.
    """
    sums = {}
    for line in sha256sum_text.splitlines():
        if line.strip():
            sha, path = line.split(maxsplit=1)
            sums[path.lstrip("*").removeprefix("./")] = sha
    out = {}
    for (repo_id, path), info in expected.items():
        info = dict(info)
        if path in sums:
            info["sha256"] = sums[path]
        out[(repo_id, path)] = info
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _size_ok(info, live_size):
    """Exact size match for files; a tar may differ by headers and padding."""
    if live_size is None:
        return False
    if info.get("tar"):
        slack = TAR_SIZE_REL_TOL * info["size"] + (1 << 20)
        return abs(live_size - info["size"]) <= slack
    return live_size == info["size"]


def _withheld_paths(manifest):
    """Map each path a withheld entry could be uploaded under to its logical key."""
    out = {}
    for logical, entry in manifest.items():
        if entry.get("share"):
            continue
        for path in (logical, _data_repo_path(entry.get("category", ""), logical)):
            out[path] = logical
            out[path + ".tar"] = logical
    return out


def _withheld_reason(path, withheld_paths):
    """Return why an unexpected upload is withheld material, or ``None``."""
    if path in withheld_paths:
        return withheld_paths[path]
    for name in EXCLUDED_DATASETS:
        if name in path:
            return f"excluded dataset {name}"
    return None


def verify(manifest, live, extra_expected=None):
    """
    Compare a live repo listing with the expected files.

    Parameters
    ----------
    manifest : dict
        ``logical_path -> entry`` as in ``hf_manifest.json``.
    live : dict
        ``repo_id -> {repo_path: {"size": int, "sha256": str or None}}``.
    extra_expected : dict, optional
        More expected files, ``(repo_id, repo_path) -> {"size"?, "sha256"?}``,
        e.g. :func:`gtf_expected`. They override manifest-derived values.

    Returns
    -------
    Report
    """
    expected = expected_from_manifest(manifest)
    for key, info in (extra_expected or {}).items():
        expected[key] = {**expected.get(key, {}), **info}

    report = Report()
    for (repo_id, path), info in sorted(expected.items()):
        got = live.get(repo_id, {}).get(path)
        if got is None:
            report.missing.append((repo_id, path))
            continue
        bad = False
        if info.get("size") is not None and not _size_ok(info, got.get("size")):
            report.size_mismatch.append((repo_id, path, info["size"], got.get("size")))
            bad = True
        if info.get("sha256"):
            if got.get("sha256") != info["sha256"]:
                report.sha256_mismatch.append(
                    (repo_id, path, info["sha256"], got.get("sha256"))
                )
                bad = True
        else:
            report.sha256_unchecked.append((repo_id, path))
        if not bad:
            report.ok.append((repo_id, path))

    withheld_paths = _withheld_paths(manifest)
    for repo_id, files in sorted(live.items()):
        for path in sorted(files):
            if (repo_id, path) in expected or path in REPO_META_FILES:
                continue
            reason = None
            if repo_id == HF_DATA_REPO:
                reason = _withheld_reason(path, withheld_paths)
            if reason:
                report.withheld.append((repo_id, path, reason))
            else:
                report.extra.append((repo_id, path))
    return report


# ---------------------------------------------------------------------------
# Live listing (the only network step)
# ---------------------------------------------------------------------------
def list_live(repos=None, revision="main"):
    """
    List every file of the HuggingFace repos with its size and LFS sha256.

    Parameters
    ----------
    repos : dict, optional
        ``repo_id -> repo_type``; default :data:`REPOS`.
    revision : str
        Branch, tag or commit to list.

    Returns
    -------
    dict
        ``repo_id -> {repo_path: {"size", "sha256"}}``; ``sha256`` is ``None``
        for files stored in git rather than LFS/Xet.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    out = {}
    for repo_id, repo_type in (repos or REPOS).items():
        files = {}
        for item in api.list_repo_tree(
            repo_id, repo_type=repo_type, revision=revision, recursive=True
        ):
            if not isinstance(item, RepoFile):
                continue
            lfs = item.lfs
            sha = (
                lfs.get("sha256")
                if isinstance(lfs, dict)
                else getattr(lfs, "sha256", None)
            )
            files[item.path] = {"size": item.size, "sha256": sha}
        out[repo_id] = files
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render(report, live):
    """Render a report as Markdown."""
    n_live = {r: len(f) for r, f in live.items()}
    lines = [
        "# HuggingFace upload check",
        "",
        "Live files: " + ", ".join(f"{r} {n}" for r, n in n_live.items()),
        "",
        f"- clean: {report.clean}",
        f"- ok: {len(report.ok)}",
        f"- missing: {len(report.missing)}",
        f"- size mismatch: {len(report.size_mismatch)}",
        f"- sha256 mismatch: {len(report.sha256_mismatch)}",
        f"- withheld but uploaded: {len(report.withheld)}",
        f"- extra: {len(report.extra)}",
        f"- present, sha256 not checked (no expected checksum): "
        f"{len(report.sha256_unchecked)}",
    ]
    sections = [
        ("Missing", report.missing),
        ("Size mismatch (repo, path, expected, live)", report.size_mismatch),
        ("sha256 mismatch (repo, path, expected, live)", report.sha256_mismatch),
        ("Withheld but uploaded (repo, path, why)", report.withheld),
        ("Extra", report.extra),
    ]
    for title, rows in sections:
        if not rows:
            continue
        lines += ["", f"## {title} ({len(rows)})", ""]
        lines += ["- " + " | ".join(str(x) for x in row) for row in rows]
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Run the check against the live repos and print or save the report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument(
        "--checksums",
        nargs="*",
        default=[],
        help="sha256sum-format files keyed by repo path",
    )
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", help="write the Markdown report here")
    ap.add_argument("--json", help="write the full report as JSON here")
    args = ap.parse_args(argv)

    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    extra = gtf_expected()
    live = list_live(revision=args.revision)
    if args.checksums:
        text = "".join(pathlib.Path(p).read_text() for p in args.checksums)
        extra = with_checksums({**expected_from_manifest(manifest), **extra}, text)
    report = verify(manifest, live, extra_expected=extra)

    md = render(report, live)
    if args.out:
        pathlib.Path(args.out).write_text(md)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(dataclasses.asdict(report)))
    print(md)
    return 0 if report.clean else 1


if __name__ == "__main__":
    sys.exit(main())
