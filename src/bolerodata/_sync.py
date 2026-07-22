"""
Locate bolerodata artifacts locally or fetch them from HuggingFace.

`bolerodata` is a registry: its tables and joblib dicts map short keys to file
paths. On the lab filesystem those paths are absolute (``/large_storage/...``)
and the files already exist, so nothing needs to be downloaded. Off-lab (i.e.
for external users installing from PyPI) the files do not exist locally; this
module translates each lab path to a repo-relative *logical* path, looks it up
in a bundled manifest, and downloads the shared ones from HuggingFace into a
local cache. Artifacts that are deliberately not redistributed raise a friendly
:class:`FileNotAvailableError` instead of a bare download error.

The single entry point is :func:`localize`. It serves two personas from one
code path:

- **Author (lab machine):** the absolute path exists → it is returned unchanged,
  so behaviour is identical to before this layer existed (no network, no cache).
- **External user:** the path does not exist → it is mapped to a logical key,
  looked up in the manifest, and either downloaded (``share = True``) or refused
  with a friendly message (``share = False`` / unknown).
"""

import json
import os
import pathlib
import shutil
import tarfile
import tempfile
from functools import lru_cache

__all__ = [
    "FileNotAvailableError",
    "cache_root",
    "localize",
    "manifest",
    "prefetch",
    "sync",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#: HuggingFace repositories the public artifacts live in. Overridable via env
#: so a fork / mirror can be pointed at without editing the package.
HF_MODELS_REPO = os.environ.get("BOLERO_HF_MODELS_REPO", "arcinstitute/bolero-models")
HF_DATA_REPO = os.environ.get("BOLERO_HF_DATA_REPO", "arcinstitute/bolero-data")

#: Where users are pointed to obtain non-shared artifacts / report problems.
CONTACT_URL = "https://github.com/liuhlab/bolerodata"

#: Kept in sync with ``bolerodata.data.DEFAULT_STANDARD_DIR``.
_DEFAULT_STANDARD_DIR = "/large_storage/zhoulab/hanliu/wmb/standard"

#: Datasets withheld from the public release (unpublished). The committed source
#: tables keep them (the lab package is unchanged); they are stripped only at
#: PUBLISH time — from the HuggingFace upload (scripts/build_manifest.py) and from
#: the shipped wheel tables (hatch_build.py). Single source of truth for both.
EXCLUDED_DATASETS = (
    "HYSocialRebound",
    "HumanKidney_CSC2025",
    "CZIBodyAtlas",
    "Torpor",
)


class FileNotAvailableError(RuntimeError):
    """Raised when an artifact is not available locally and cannot be fetched."""


# ---------------------------------------------------------------------------
# lab-root -> repo-subdir mapping
# ---------------------------------------------------------------------------
# Ordered longest-prefix-first at use time. Each lab filesystem root maps to a
# clean subdirectory used as the *logical* namespace (the manifest is keyed on
# these logical paths). The actual on-HuggingFace location is the manifest
# entry's ``repo_path`` (which may differ, e.g. clean per-model-key checkpoint
# names), so this table only needs to be stable, not pretty.
_STATIC_ROOTS = [
    ("/large_storage/zhoulab/hanliu/wmb/GTEx", "gtex/"),
    ("/large_storage/zhoulab/hanliu/wmb/Zeng2024Science", "zeng2024/"),
    ("/large_storage/zhoulab/hanliu/wmb/standard", "standard/"),
    ("/large_storage/zhoulab/hanliu/wmb", "wmb/"),
    ("/large_storage/zhoulab/hanliu/250924-ScoreModel", "projects/250924-ScoreModel/"),
    ("/large_storage/zhoulab/hanliu/260102-ScoreModel", "projects/260102-ScoreModel/"),
    (
        "/large_storage/zhoulab/hanliu/250907-MultiDatasetModel",
        "projects/250907-MultiDatasetModel/",
    ),
    (
        "/large_storage/zhoulab/hanliu/250728-EmbeddingBenchmark",
        "projects/250728-EmbeddingBenchmark/",
    ),
    (
        "/large_storage/zhoulab/hanliu/250618-MouseBrainGeneralization",
        "projects/250618-MouseBrainGeneralization/",
    ),
    ("/large_storage/zhoulab/hanliu/260121-KidneyModel", "projects/260121-KidneyModel/"),
    ("/large_storage/zhoulab/hanliu/251105-DiffAnalysis", "diff_analysis/"),
    ("/large_storage/zhoulab/shengmao/sc-eQTL", "sc-eqtl/"),
    (
        "/large_storage/zhoulab/shengmao/250901-GeneCountPred",
        "projects/250901-GeneCountPred/",
    ),
    (
        "/large_storage/zhoulab/shengmao/250907-MultiDatasetModel",
        "projects/shengmao-250907-MultiDatasetModel/",
    ),
    ("/large_storage/zhoulab/zhoujt/project", "zhoujt/"),
    ("/scratch/zhoulab/hanliu", "scratch/"),
    ("//hms-hanqing-archive", "raw-archive/"),
    ("//hms-hanqing-wmb", "raw-wmb/"),
]


@lru_cache(maxsize=1)
def _lab_roots():
    """Return the ordered (lab_prefix, repo_subdir) map, longest-prefix first."""
    roots = list(_STATIC_ROOTS)
    # Include the live $STANDARD_DIR so an overridden data root still maps to
    # ``standard/`` (both it and the hardcoded default resolve to the same tree).
    std = os.environ.get("STANDARD_DIR", _DEFAULT_STANDARD_DIR)
    roots.append((std, "standard/"))
    roots.append((_DEFAULT_STANDARD_DIR, "standard/"))
    seen = {}
    for prefix, sub in roots:
        key = _norm(prefix).rstrip("/")
        if key and key not in seen:
            seen[key] = sub
    return sorted(seen.items(), key=lambda kv: len(kv[0]), reverse=True)


def _norm(s):
    """Normalize a path-ish string to forward-slash, collapsed-slash form."""
    s = str(s).replace("\\", "/")
    leading = "//" if s.startswith("//") else ("/" if s.startswith("/") else "")
    body = s[len(leading):]
    while "//" in body:
        body = body.replace("//", "/")
    return leading + body


def _to_logical(abs_path):
    """
    Map a lab absolute path to a repo-relative logical key.

    Returns
    -------
    str or None
        The logical key, or ``None`` if the path is under no known lab root.
    """
    s = _norm(abs_path)
    for prefix, sub in _lab_roots():
        if s == prefix or s.startswith(prefix + "/"):
            tail = s[len(prefix):].lstrip("/")
            return _norm(sub + tail).lstrip("/")
    return None


def _to_lab_abs(logical):
    """
    Inverse of :func:`_to_logical`: map a logical key back to a lab path.

    Used only to let an author who happens to hold a *logical* path hit the
    local fast-path when the lab filesystem is mounted. Returns the first
    candidate that exists, else the first candidate, else ``None``.
    """
    logical = _norm(logical).lstrip("/")
    candidates = []
    for prefix, sub in _lab_roots():
        sub_n = _norm(sub).lstrip("/")
        if logical == sub_n.rstrip("/") or logical.startswith(sub_n):
            tail = logical[len(sub_n):].lstrip("/")
            candidates.append(pathlib.Path(prefix) / tail)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Manifest + cache
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def manifest():
    """
    Load the bundled HuggingFace manifest.

    Returns
    -------
    dict
        Mapping ``logical_path -> {kind, share, category, tier, size, sha256,
        repo_id, repo_type, repo_path, pack}``. Empty dict if the manifest has
        not been built/shipped yet (authors do not need it).
    """
    try:
        from importlib.resources import files

        text = files("bolerodata.data").joinpath("hf_manifest.json").read_text()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def cache_root():
    """
    Return the local cache directory for downloaded bolerodata artifacts.

    Resolution order: ``$BOLERODATA_HOME`` → ``$BOLERO_HOME`` →
    :func:`platformdirs.user_cache_dir` → ``~/.cache/bolero``.
    """
    env = os.environ.get("BOLERODATA_HOME") or os.environ.get("BOLERO_HOME")
    if env:
        root = pathlib.Path(env).expanduser()
    else:
        try:
            import platformdirs

            root = pathlib.Path(platformdirs.user_cache_dir("bolero"))
        except ModuleNotFoundError:
            root = pathlib.Path.home() / ".cache" / "bolero"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _hf_cache():
    d = cache_root() / "hf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store_root():
    d = cache_root() / "store"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# The public seam
# ---------------------------------------------------------------------------
def localize(path, *, allow_download=True, desc=None):
    """
    Resolve a bolerodata artifact to a concrete local path.

    Parameters
    ----------
    path : str or os.PathLike
        An absolute lab path (as stored in the registry tables / joblib dicts)
        or a repo-relative logical path.
    allow_download : bool
        If ``False``, never hit the network; raise if the artifact is not
        already present locally (useful for strict offline runs).
    desc : str, optional
        Human-readable label used in error messages.

    Returns
    -------
    pathlib.Path
        A path that exists on the local filesystem.

    Raises
    ------
    FileNotAvailableError
        If the artifact is unknown, marked not-shared, or (with
        ``allow_download=False``) simply not cached.
    """
    s = os.fspath(path)
    p = pathlib.Path(s)

    # Persona A: the real file/dir is present locally (lab machine) -> use as-is.
    if p.is_absolute() and p.exists():
        return p

    # Compute the repo-relative logical key.
    logical = _to_logical(s) if p.is_absolute() else _norm(s).lstrip("/")

    # Persona A': author holding a logical path while the lab FS is mounted.
    if logical is not None:
        lab_abs = _to_lab_abs(logical)
        if lab_abs is not None and lab_abs.exists():
            return lab_abs

    if logical is None:
        raise FileNotAvailableError(_msg_unknown(s, None, desc))

    entry = manifest().get(logical)
    if entry is None:
        raise FileNotAvailableError(_msg_unknown(s, logical, desc))
    if not entry.get("share", False):
        raise FileNotAvailableError(_msg_noshare(logical, entry, desc))
    if not allow_download:
        raise FileNotAvailableError(
            f"{desc or logical!r} is not in the local cache and downloads are "
            f"disabled (allow_download=False)."
        )
    return _download(logical, entry)


def _download(logical, entry):
    """Download a manifest entry into the cache and return its local path."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ModuleNotFoundError as e:
        raise FileNotAvailableError(
            "huggingface_hub is required to download bolerodata artifacts. "
            "Install it with: pip install huggingface_hub"
        ) from e

    repo_id = entry.get("repo_id", HF_DATA_REPO)
    repo_type = entry.get("repo_type", "dataset")
    revision = entry.get("revision", "main")
    repo_path = entry.get("repo_path", logical)
    kind = entry.get("kind", "file")

    if kind == "file":
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            filename=repo_path,
            cache_dir=str(_hf_cache()),
        )
        return pathlib.Path(local)

    # Directory artifact.
    if entry.get("pack") == "tar":
        tar_local = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            filename=repo_path + ".tar",
            cache_dir=str(_hf_cache()),
        )
        return _extract_once(tar_local, _store_root() / logical)

    snap = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        allow_patterns=[repo_path + "/**"],
        cache_dir=str(_hf_cache()),
    )
    return pathlib.Path(snap) / repo_path


def _extract_once(tar_path, dest_dir):
    """Extract a tar artifact to ``dest_dir`` exactly once (atomic, lock-guarded)."""
    dest_dir = pathlib.Path(dest_dir)
    done = dest_dir.parent / (dest_dir.name + ".done")
    if done.exists() and dest_dir.exists():
        return dest_dir

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with _filelock(dest_dir.parent / (dest_dir.name + ".lock")):
        if done.exists() and dest_dir.exists():
            return dest_dir
        tmp = pathlib.Path(
            tempfile.mkdtemp(dir=str(dest_dir.parent), prefix=".extract-")
        )
        try:
            with tarfile.open(tar_path) as tf:
                _safe_extract(tf, tmp)
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            os.replace(tmp, dest_dir)
            done.write_text("ok")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
    return dest_dir


def _safe_extract(tf, dest):
    """Extract a tarfile, refusing members that would escape ``dest``."""
    dest = pathlib.Path(dest).resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if dest not in target.parents and target != dest:
            raise FileNotAvailableError(
                f"Refusing to extract unsafe path from archive: {member.name!r}"
            )
    tf.extractall(dest)


def _filelock(path):
    from filelock import FileLock

    return FileLock(str(path))


# ---------------------------------------------------------------------------
# Bulk pre-fetch
# ---------------------------------------------------------------------------
def prefetch(tier=None, category=None, datasets=None, allow_download=True):
    """
    Pre-download shared artifacts matching the given filters into the cache.

    Parameters
    ----------
    tier : str, optional
        Only fetch entries whose ``tier`` equals this (e.g. ``"core"``, ``"eval"``).
    category : str, optional
        Only fetch entries whose ``category`` equals this.
    datasets : iterable of str, optional
        Keep an entry only if one of these substrings appears in its logical path.
    allow_download : bool
        Passed through to the downloader; if ``False`` this is a no-op sanity pass.

    Returns
    -------
    dict
        ``{"downloaded": int, "errors": [(logical, message), ...]}``.
    """
    man = manifest()
    datasets = list(datasets) if datasets else None
    downloaded, errors = 0, []
    for logical, entry in man.items():
        if not entry.get("share", False):
            continue
        if tier is not None and entry.get("tier") != tier:
            continue
        if category is not None and entry.get("category") != category:
            continue
        if datasets is not None and not any(d in logical for d in datasets):
            continue
        try:
            if allow_download:
                _download(logical, entry)
            downloaded += 1
        except Exception as e:  # noqa: BLE001 - collect and report, don't abort the batch
            errors.append((logical, str(e)))
    return {"downloaded": downloaded, "errors": errors}


def sync(**filters):
    """Download all shared artifacts (optionally filtered). Alias of :func:`prefetch`."""
    return prefetch(**filters)


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------
def _msg_noshare(logical, entry, desc=None):
    category = entry.get("category", "raw/intermediate")
    name = desc or logical
    return (
        f"'{name}' is a {category} artifact that is not redistributed publicly on "
        f"HuggingFace. It is part of the Bolero project's internal data and is "
        f"available from the authors on request — see {CONTACT_URL}."
    )


def _msg_unknown(orig, logical, desc=None):
    name = desc or orig
    hint = f" (no manifest entry for '{logical}')" if logical else ""
    return (
        f"'{name}' is not a recognized bolerodata artifact{hint}. If you are a "
        f"Bolero author, mount the lab filesystem or set $STANDARD_DIR so the file "
        f"resolves locally; otherwise this artifact is not part of the public "
        f"release. See {CONTACT_URL}."
    )
