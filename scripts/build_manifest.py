#!/usr/bin/env python
"""
Build the bolerodata HuggingFace manifest + upload list from the registry.

Run this **on the lab filesystem** (where the referenced files exist). It:

1. Enumerates every file the registry can reach, tagging each with a *category*
   by its provenance (which TSV column / joblib dict it came from).
2. Maps each absolute lab path to a repo-relative *logical* key
   (``bolerodata._sync._to_logical``).
3. Joins ``huggingface_digest.md`` to decide share / noshare + tier + repo.
4. Stats the shared files (size, file / dir, tar-pack for many-file dirs).
5. Emits:
   - ``src/bolerodata/data/hf_manifest.json`` — consumed by ``localize`` (keys and
     repo paths are logical / clean; it contains **no** lab absolute paths).
   - ``dist/hf_upload_list.tsv`` + ``dist/hf_upload.sh`` — the share-only upload
     commands (these DO contain local source paths; not shipped in the wheel).
   - ``dist/hf_review.md`` — a per-category summary for the GitHub review issue.
   - Appends any un-categorized categories to ``huggingface_digest.md``.

Usage
-----
    python scripts/build_manifest.py [--hash] [--out DIR]

``--hash`` additionally computes sha256 for shared files (slow on TB-scale
checkpoints; off by default).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import sys

import joblib
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # for slim_checkpoint
sys.path.insert(0, str(SRC))

import bolerodata.data as dm  # noqa: E402
from bolerodata._sync import (  # noqa: E402
    EXCLUDED_DATASETS,
    HF_DATA_REPO,
    HF_MODELS_REPO,
    _to_logical,
)
from bolerodata.data._path_patch import PATH_PATCH_DICT  # noqa: E402

DIGEST_PATH = REPO_ROOT / "huggingface_digest.md"
MANIFEST_PATH = SRC / "bolerodata" / "data" / "hf_manifest.json"

# Directory artifacts with more than this many files are packed as a single tar.
TAR_FILE_COUNT_THRESHOLD = 200

# joblib path-dict attribute -> category
JOBLIB_CATEGORY = {
    "META_CELL_ADATA_PATH_DICT": "metacell-adata",
    "META_CELL_PARQUET_PATH_DICT": "metacell-parquet",
    "META_CELL_PSEUDOBULK_RECORDS_PATH_DICT": "pseudobulk-embedding",
    "META_CELL_COND_PSEUDOBULK_RECORDS_PATH_DICT": "pseudobulk-condition",
    "META_CELL_METADATA_PSEUDOBULK_RECORDS_PATH_DICT": "pseudobulk-metadata",
    "META_CELL_REFERENCE_BIGWIG_PATH_DICT": "reference-bigwig",
    "PSEUDOBULK_GENE_AND_HVG_PATH_DICT": "gene-hvg",
}

# DATASET_METADATA column -> category (joined dataset_metadata + embedding tsvs)
DS_COL_CATEGORY = {
    "cell_metadata": "original-cell-metadata",
    "peaks": "peaks",
    "peak_adata": "peak-adata",
    "gene_adata": "gene-adata",
    "fragment_files": "raw-fragments",
    "snap_files": "raw-snap",
    "per_group_peaks": "per-group-peaks",
    "within_dataset_cell_embedding": "cell-embedding",
    "chromvar_within_dataset": "chromvar",
    "chromvar_cross_dataset": "chromvar-cross",
    "chromvar_motif": "chromvar-motif",
    "parquet_dataset": "metacell-parquet",
}

def _excluded_logicals(refs, md):
    """Logicals belonging to an excluded dataset: name-match ∪ dataset_metadata rows.

    The name-match catches the vast majority (dataset name in a path segment or
    filename); the dataset_metadata row-paths additionally catch datasets whose
    files carry no dataset token (e.g. HumanKidney_CSC2025 -> ``zhoujt/DISC5_kidney/``).
    """
    ex = set()
    for lg in refs:
        segs = lg.split("/")
        fn = segs[-1]
        for ds in EXCLUDED_DATASETS:
            if (ds in segs or fn.startswith((ds + ".", ds + "+", ds + "-"))
                    or ("+" + ds) in lg or (ds + "+") in lg):
                ex.add(lg)
                break
    dsmeta = md.DATASET_METADATA
    for ds in EXCLUDED_DATASETS:
        if ds in dsmeta.index:
            for val in dsmeta.loc[ds].values:
                for p in iter_paths(val):
                    lg = _to_logical(p)
                    if lg in refs:
                        ex.add(lg)
    return ex


def _scrub_ensemble(src, category, stage_dir):
    """Stage an uploaded ensemble file with excluded-dataset records removed.

    ``path-table-index`` joblib dicts: drop keys mentioning an excluded dataset.
    ``snap-index`` csv: drop rows whose ``dataset`` is excluded. Returns
    ``(path, size)`` — the staged scrubbed copy if anything changed, else the
    original ``src``.
    """
    if not src.exists():
        return src, 0

    def _mentions_excluded(text):
        return any(ds in str(text) for ds in EXCLUDED_DATASETS)

    def _sanitize_value(v):
        """Rewrite lab-abs path strings in a dict value to portable logical keys.

        Values are single paths or comma-joined lists; ``localize()`` accepts both
        lab-abs and logical, so this only removes the cosmetic ``/large_storage/…``
        leak from the uploaded index. Non-path / unmappable parts are left as-is.
        """
        if not isinstance(v, str):
            return v
        out, changed = [], False
        for part in v.split(","):
            p = part.strip()
            if p.startswith("/"):
                lg = _to_logical(p)
                if lg:
                    out.append(lg)
                    changed = True
                    continue
            out.append(part)
        return ",".join(out) if changed else v

    if category == "path-table-index" and src.suffix == ".joblib":
        d = joblib.load(src)
        if hasattr(d, "items"):
            # Drop excluded-dataset keys AND sanitize lab paths out of the values.
            kept = {k: _sanitize_value(v)
                    for k, v in d.items() if not _mentions_excluded(k)}
            stage_dir.mkdir(parents=True, exist_ok=True)
            out = stage_dir / src.name
            joblib.dump(kept, out)
            return out, out.stat().st_size
    elif category == "snap-index" and src.suffix == ".csv":
        df = pd.read_csv(src)
        if "dataset" in df.columns:
            keep = df[~df["dataset"].astype(str).isin(EXCLUDED_DATASETS)]
            if len(keep) != len(df):
                stage_dir.mkdir(parents=True, exist_ok=True)
                out = stage_dir / src.name
                keep.to_csv(out, index=False)
                return out, out.stat().st_size
    return src, src.stat().st_size


#: Home-mirror prefix seen inside model configs; aliased to the lab root so the
#: same _to_logical mapping applies (e.g. base_checkpoint / data_path entries).
_HOME_ALIAS = ("/home/hanliu/data/", "/large_storage/zhoulab/hanliu/")
_CONFIG_NULL_FIELDS = ("wandb_project", "wandb_job_type", "wandb_group",
                       "wandb_name", "savename", "output_dir")
_LEAK_MARKERS = ("/large_storage", "/scratch/zhoulab", "/home/", "hms-hanqing",
                 "/gale/")


def _scrub_config(src, stage_dir):
    """Stage a model-config JSON with lab paths + internal names sanitized.

    - Every absolute-path string is rewritten to a portable logical key
      (``localize`` accepts logical); home-mirror paths are aliased to the lab
      root first, and anything still absolute afterwards is redacted.
    - Experiment-tracking cruft (``wandb_*`` / ``savename`` / ``output_dir``) is
      nulled.
    - ``dataset_records`` entries for excluded (unpublished) datasets are dropped.

    The committed source is untouched; the returned staged copy is uploaded.
    Raises if any lab-path marker survives (turns a leak into a build failure).
    """
    if not src.exists() or src.suffix != ".json":
        return src, (src.stat().st_size if src.exists() else 0)

    # The -gene models reference gene-count inputs (deg_list / gene_data_path) that
    # live outside STANDARD_DIR (250901-GeneCountPred/prepare/gene_data/), but the
    # byte-identical file is already shared under standard/gene_data/gene_data/.
    # Map those by basename so the configs point at the distributed copy.
    gene_dir = dm.STANDARD_DIR / "gene_data" / "gene_data"
    gene_index = ({p.name: f"standard/gene_data/gene_data/{p.name}"
                   for p in gene_dir.iterdir() if p.is_file()}
                  if gene_dir.exists() else {})

    def _san_str(s):
        if not isinstance(s, str) or not s.startswith("/"):
            return s
        p = s
        if p.startswith(_HOME_ALIAS[0]):
            p = _HOME_ALIAS[1] + p[len(_HOME_ALIAS[0]):]
        lg = _to_logical(p)
        if lg:
            return lg
        if "gene_data" in p:
            name = pathlib.Path(p).name
            if name in gene_index:
                return gene_index[name]
        return "<not-distributed>"

    def _walk(o):
        if isinstance(o, dict):
            return {k: _walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_walk(v) for v in o]
        return _san_str(o)

    d = json.load(open(src))
    dr = d.get("dataset_records")
    if isinstance(dr, dict):
        d["dataset_records"] = {
            k: v for k, v in dr.items()
            if not any(ds in k for ds in EXCLUDED_DATASETS)
        }
    for f in _CONFIG_NULL_FIELDS:
        if f in d:
            d[f] = None
    d = _walk(d)

    text = json.dumps(d, indent=1)
    leaks = [m for m in _LEAK_MARKERS if m in text]
    if leaks:
        raise SystemExit(f"config sanitize leak in {src.name}: {leaks}")

    stage_dir.mkdir(parents=True, exist_ok=True)
    out = stage_dir / src.name
    out.write_text(text + "\n")
    return out, out.stat().st_size


def qtl_category(source: str) -> str:
    """Map a ``qtl_collection.tsv`` Source to a category."""
    s = str(source)
    if "Zeng2024" in s:
        return "qtl-zeng2024"
    if "Onek1k" in s or "eqtl_catalog" in s:
        return "qtl-sc-eqtl"
    return "qtl-GTEx"


def iter_paths(value):
    """Yield absolute path strings from a table cell (handles NaN / comma-lists)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return
    for part in str(value).split(","):
        part = part.strip()
        if part.startswith("/"):  # absolute lab path (incl. //hms-* shares)
            yield part


# ---------------------------------------------------------------------------
# Enumerate referenced paths -> {logical: {"category", "src"}}
# ---------------------------------------------------------------------------
def enumerate_refs():
    """Return ``{logical: {"category", "src"}}`` and a list of unmapped abs paths."""
    refs: dict[str, dict] = {}
    unmapped: list[tuple[str, str]] = []

    def add(category, abs_path):
        logical = _to_logical(str(abs_path))
        if logical is None:
            unmapped.append((category, str(abs_path)))
            return
        # First provenance wins; a shared category overrides a later noshare alias.
        refs.setdefault(logical, {"category": category, "src": str(abs_path)})

    md = dm.metadata

    # model zoo: checkpoints + configs
    zoo = md.MODEL_ZOO
    for _key, row in zoo.iterrows():
        for p in iter_paths(row.get("CkptPath")):
            add("model-checkpoints", p)
        for p in iter_paths(row.get("ConfigPath")):
            add("model-configs", p)

    # qtl collection feathers
    qtl = md.QTL_COLLECTION
    for _idx, row in qtl.iterrows():
        for p in iter_paths(row.get("FilePath")):
            add(qtl_category(row.get("Source")), p)

    # qtl training-region feathers
    try:
        from bolerodata.qtl import QTL_TRAINING_PATH_DIR

        tdir = pathlib.Path(QTL_TRAINING_PATH_DIR)
        if tdir.exists():
            for p in sorted(tdir.glob("*.qtl_model_regions.feather")):
                add("qtl-training-regions", p)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] qtl training regions: {e}", file=sys.stderr)

    # dataset-level path columns (joined dataset_metadata + embedding_and_meta_cell)
    ds = md.DATASET_METADATA
    for col, category in DS_COL_CATEGORY.items():
        if col not in ds.columns:
            continue
        for value in ds[col].tolist():
            for p in iter_paths(value):
                add(category, p)

    # differential-analysis records (in-package DA.TotalRecords.csv.gz)
    da = md.DA_COLLECTION
    for col in ("Group1BigWig", "Group2BigWig"):
        if col in da.columns:
            for value in da[col].tolist():
                for p in iter_paths(value):
                    add("diff-bigwig", p)
    if "DiffPath" in da.columns:
        for value in da["DiffPath"].tolist():
            for p in iter_paths(value):
                add("diff-tables", p)

    # curated metadata dirs under STANDARD_DIR
    for d, category in [
        (dm.STANDARD_CELL_METADATA_DIR, "curated-cell-metadata"),
        (dm.STANDARD_SAMPLE_METADATA_DIR, "curated-sample-metadata"),
        (dm.STANDARD_CLUSTER_METADATA_DIR, "curated-cluster-metadata"),
    ]:
        d = pathlib.Path(d)
        if d.exists():
            for p in sorted(d.glob("*.feather")):
                add(category, p)

    # motif scan
    motif_dir = dm.STANDARD_DIR / "motif_scan"
    if motif_dir.exists():
        for p in sorted(motif_dir.glob("*.feather")):
            add("motif-scan", p)

    # file_path_table joblib dicts: the index files themselves + their values
    for attr, category in JOBLIB_CATEGORY.items():
        dict_path = getattr(dm, attr, None)
        if dict_path is None:
            continue
        add("path-table-index", dict_path)
        if not pathlib.Path(dict_path).exists():
            continue
        d = joblib.load(dict_path)
        d.update(PATH_PATCH_DICT.get(attr, {}))
        for value in d.values():
            for p in iter_paths(value):
                cat = category
                # pseudobulk-condition: only the canonical wmb/standard copies are
                # the released models' inputs (Bolero10M config dataset_records point
                # here). The 250728-EmbeddingBenchmark patches are a separate
                # method-comparison experiment -> withhold, do not publish.
                if (category == "pseudobulk-condition"
                        and "coverage_pseudobulk_with_condition" not in p):
                    cat = "pseudobulk-benchmark"
                # pseudobulk-metadata: only the canonical wmb/standard copies are
                # released; the _path_patch 250728-EmbeddingBenchmark method-specific
                # copies (+PANVI/+PVI/+SCVI/+SCANVI/+STATE) are a benchmark experiment.
                if (category == "pseudobulk-metadata"
                        and "/standard/embedding/metadata_pseudobulk/" not in p):
                    cat = "pseudobulk-benchmark"
                add(cat, p)

    # whole-directory pseudobulk shares: the path tables above reference only the
    # cov5000000 version, but the canonical dirs hold other data versions too
    # (e.g. reference_pseudobulk.joblib). Add every data file, skipping the dev
    # artifacts (notebooks / checkpoints / logs / sbatch). First-provenance-wins
    # keeps already-added records/bigwigs in their categories.
    pb_junk_ext = {".ipynb", ".sh", ".log", ".txt", ".finish", ".output", ".error"}

    def _pb_is_data(path):
        if ".ipynb_checkpoints" in path.parts:
            return False
        if any(s in path.name for s in
               ("papermill", "-checkpoint", ".job_id.", ".output.", ".error.")):
            return False
        return path.suffix not in pb_junk_ext

    for sub, base_cat in [
        ("embedding/coverage_pseudobulk_with_condition", "pseudobulk-condition"),
        ("embedding/metadata_pseudobulk", "pseudobulk-metadata"),
    ]:
        root = dm.STANDARD_DIR / sub
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or not _pb_is_data(p):
                continue
            if p.suffix == ".bw":
                cat = "reference-bigwig"
            elif p.name == "reference_pseudobulk.joblib":
                cat = "pseudobulk-reference"
            else:
                cat = base_cat
            add(cat, p)

    # snap file table (index of raw snaps)
    snap_csv = getattr(dm, "SAMPLE_SNAP_TABLE_PATH", None)
    if snap_csv is not None:
        add("snap-index", snap_csv)

    return refs, unmapped


# ---------------------------------------------------------------------------
# Digest parsing
# ---------------------------------------------------------------------------
def parse_digest():
    """Parse the category table -> ``{category: {"status", "tier", "repo"}}``."""
    rules = {}
    if not DIGEST_PATH.exists():
        return rules
    header_seen = False
    for line in DIGEST_PATH.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            header_seen = False
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        if cells[0].lower() == "category":
            header_seen = True
            continue
        if not header_seen or set(cells[0]) <= {"-", ":"}:
            continue
        category, _desc, tier, status, repo = cells[:5]
        status = status.replace("⚠️", "").strip().lower()
        if status in ("share", "noshare"):
            rules[category] = {"status": status, "tier": tier, "repo": repo}
    return rules


# ---------------------------------------------------------------------------
# Stat + repo placement
# ---------------------------------------------------------------------------
def _du(path, timeout=90):
    """Directory size in bytes via ``du -sb``; None on timeout/error (huge trees)."""
    try:
        out = subprocess.run(
            ["du", "-sb", str(path)], capture_output=True, text=True, timeout=timeout
        )
        if out.returncode == 0:
            return int(out.stdout.split()[0])
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return None


def _size_dirs(dir_paths, workers=4, timeout=90):
    """Parallel du over a set of directory paths -> {path: bytes-or-None}."""
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_du, p, timeout): p for p in dir_paths}
        for fut in concurrent.futures.as_completed(futs):
            result[futs[fut]] = fut.result()
    return result


def _safe_key(s):
    return "".join(c if (c.isalnum() or c in "-_.+") else "_" for c in str(s))


# ---------------------------------------------------------------------------
# HuggingFace on-repo layout (repo_path), decoupled from the logical key.
# ---------------------------------------------------------------------------
# For the data repo we present a clean *by-type* tree: each category lands under
# a human-readable top-level directory, and the lab source prefix is stripped so
# internal roots (``scratch/``, ``projects/250728-…``, ``poissonmultivi/``, the
# doubled ``gene_data/gene_data/``) do not leak. Enough of the tail is kept to
# stay unique — basenames alone collide (every ``peak_adata`` dir is named
# ``peak_adata``; DA feathers repeat across cell types). ``build()`` asserts the
# resulting repo_paths are collision-free, so this table is safe to tune.
#
# Each entry: category -> (clean_top_dir, lab_prefix_to_strip). The repo_path is
# ``clean_top_dir/<logical with lab_prefix removed>``; if a logical does not start
# with its prefix, the full logical is kept under clean_top_dir (still unique).
DATA_LAYOUT = {
    "curated-cell-metadata":    ("metadata/cell",            "standard/cell_metadata/"),
    "curated-sample-metadata":  ("metadata/sample",          "standard/sample_metadata/"),
    "curated-cluster-metadata": ("metadata/cluster",         "standard/cluster_metadata/"),
    "motif-scan":               ("motif_scan",               "standard/motif_scan/"),
    "cell-embedding":           ("cell_embedding",           "standard/embedding/poissonmultivi/"),
    "chromvar":                 ("chromvar/within_dataset",  "standard/embedding/chromvar/"),
    "chromvar-cross":           ("chromvar/cross_dataset",   "standard/embedding/cross_dataset_chromvar/"),
    "reference-bigwig":         ("reference_bigwig",         "standard/embedding/coverage_pseudobulk_with_condition/"),
    "pseudobulk-condition":     ("pseudobulk/condition",     "standard/embedding/coverage_pseudobulk_with_condition/"),
    "pseudobulk-reference":     ("pseudobulk/reference",     "standard/embedding/coverage_pseudobulk_with_condition/"),
    "pseudobulk-embedding":     ("pseudobulk/embedding",     "standard/embedding/coverage_pseudobulk/"),
    "pseudobulk-metadata":      ("pseudobulk/metadata",      "standard/embedding/metadata_pseudobulk/"),
    "gene-hvg":                 ("gene_data",                "standard/gene_data/gene_data/"),
    "diff-tables":              ("diff_analysis",            "diff_analysis/aggregate/dar/"),
    "peaks":                    ("peaks/peak_bed",           "scratch/"),
    "per-group-peaks":          ("peaks/peak_by_group",      "scratch/"),
    "peak-adata":               ("peak_adata",               "scratch/"),
    "gene-adata":               ("gene_adata",               "scratch/"),
    "metacell-adata-standard":  ("metacell/adata",           "standard/metacell_dataset/adata/"),
    "metacell-parquet-32bp":    ("metacell/parquet",         "standard/metacell_dataset/parquet/"),
    "qtl-training-regions":     ("qtl/gtex_training_regions", "gtex/QTLModelTrainingFile/borzoi_regions/"),
    "qtl-GTEx":                 ("qtl/gtex",                 "gtex/"),
    "qtl-sc-eqtl":              ("qtl/sc_eqtl",              "sc-eqtl/"),
    "qtl-zeng2024":             ("qtl/zeng2024",             "zeng2024/"),
    "path-table-index":         ("index",                    "standard/file_path_table/"),
    "snap-index":               ("index",                    "standard/file_path_table/"),
}

# Clean, uniform on-repo filenames for the model repo (one dir per model_key).
MODEL_FILENAME = {"model-checkpoints": "checkpoint.pt", "model-configs": "config.json"}


def _data_repo_path(category, logical):
    """Clean by-type repo_path for a data-repo entry (see :data:`DATA_LAYOUT`)."""
    top, prefix = DATA_LAYOUT.get(category, (category, None))
    tail = logical[len(prefix):] if prefix and logical.startswith(prefix) else logical
    return f"{top}/{tail}".replace("//", "/")


def repo_placement(category, logical, rules, model_group=None, model_key=None):
    """Return (repo_id, repo_type, repo_path) for a shared entry."""
    repo = rules.get(category, {}).get("repo", "bolero-data")
    if repo == "bolero-models":
        repo_id, repo_type = HF_MODELS_REPO, "model"
        # <ModelGroup>/<model_key>/checkpoint.pt | config.json — uniform + clean.
        fname = MODEL_FILENAME.get(category, pathlib.Path(logical).name)
        if model_group and model_key:
            repo_path = f"{_safe_key(model_group)}/{_safe_key(model_key)}/{fname}"
        else:
            repo_path = fname
    else:
        repo_id, repo_type = HF_DATA_REPO, "dataset"
        repo_path = _data_repo_path(category, logical)
    return repo_id, repo_type, repo_path


def build(do_hash=False, out_dir=None):
    out_dir = pathlib.Path(out_dir) if out_dir else REPO_ROOT / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Enumerating referenced files ...", file=sys.stderr)
    refs, unmapped = enumerate_refs()

    # publish-time removal of unpublished datasets (committed source untouched)
    excluded = _excluded_logicals(refs, dm.metadata)
    for lg in excluded:
        refs.pop(lg, None)
    print(f"  excluded {len(excluded)} files from unpublished datasets "
          f"{EXCLUDED_DATASETS}", file=sys.stderr)

    # opt-in: share the standard 32bp metacell parquet — the model-training
    # resolution (Bolero10M config data_path points to standard/.../-MetaCell-32bp).
    # 1bp and project/benchmark copies stay withheld under metacell-parquet.
    for logical, info in refs.items():
        if (info["category"] == "metacell-parquet"
                and logical.startswith("standard/")
                and "-MetaCell-32bp" in logical):
            info["category"] = "metacell-parquet-32bp"
        # metacell-adata: share only the standard/ copies; benchmark/project
        # copies (250728/250618) stay withheld under metacell-adata.
        if info["category"] == "metacell-adata" and logical.startswith("standard/"):
            info["category"] = "metacell-adata-standard"

    rules = parse_digest()
    print(f"  {len(refs)} logical paths, {len(unmapped)} unmapped", file=sys.stderr)

    # model naming: first model_key wins per checkpoint (matches slim task list)
    zoo = dm.metadata.MODEL_ZOO
    ckpt_owner = {}  # logical -> (ModelGroup, model_key)
    for key, row in zoo.iterrows():
        grp = row.get("ModelGroup")
        for col in ("CkptPath", "ConfigPath"):
            for p in iter_paths(row.get(col)):
                lg = _to_logical(p)
                if lg:
                    ckpt_owner.setdefault(lg, (grp, key))

    # per-model selection (model_selection.tsv) + checkpoint slim locations
    from slim_checkpoint import SLIM_DIR, model_records, read_selection

    selection = read_selection()
    ckpt_models, config_model = {}, {}
    for r in model_records():
        if r["ckpt_logical"]:
            ckpt_models.setdefault(r["ckpt_logical"], set()).add(r["model_key"])
        if r["config_logical"]:
            config_model[r["config_logical"]] = r["model_key"]

    def _selected(category, logical):
        if selection is None:
            return True  # no selection file -> fall back to category status only
        if category == "model-checkpoints":
            return any(selection.get(k, False) for k in ckpt_models.get(logical, ()))
        if category == "model-configs":
            return selection.get(config_model.get(logical), False)
        return True

    # ---- pass 1: classify each ref, resolve slim source, collect dirs to size
    unknown_categories = set()
    recs, dir_paths = [], set()
    for logical, info in sorted(refs.items()):
        category = info["category"]
        rule = rules.get(category)
        if rule is None:
            unknown_categories.add(category)
            share, tier = False, "?"
        else:
            share = rule["status"] == "share" and _selected(category, logical)
            tier = rule["tier"]

        src = pathlib.Path(info["src"])
        exists = src.exists()
        kind = "dir" if (exists and src.is_dir()) else "file"

        # checkpoints upload the slimmed copy when it exists
        local_src, slimmed, size = src, False, None
        if category == "model-checkpoints" and logical in ckpt_owner:
            grp, mkey = ckpt_owner[logical]
            slim_p = SLIM_DIR / f"{_safe_key(grp)}/{_safe_key(mkey)}/{src.name}"
            if slim_p.exists():
                local_src, slimmed, size = slim_p, True, slim_p.stat().st_size
        # uploaded ensemble indices are scrubbed of excluded-dataset records
        elif share and category in ("path-table-index", "snap-index"):
            local_src, size = _scrub_ensemble(src, category, out_dir / "staging")
        # model configs: sanitize lab paths + drop excluded-dataset records
        elif share and category == "model-configs":
            local_src, size = _scrub_config(src, out_dir / "staging")

        if size is None:
            if exists and kind == "file":
                size = src.stat().st_size
            elif exists and kind == "dir":
                dir_paths.add(str(src))  # sized below via du (share AND noshare)
            else:
                size = 0

        recs.append({"logical": logical, "category": category, "tier": tier,
                     "share": share, "kind": kind, "exists": exists, "src": src,
                     "local_src": local_src, "size": size, "slimmed": slimmed})

    # ---- pass 2: size directories in parallel so withheld items are estimated too
    print(f"  sizing {len(dir_paths)} directories (du) ...", file=sys.stderr)
    dir_sizes = _size_dirs(dir_paths) if dir_paths else {}

    # ---- pass 3: assemble manifest / upload list / per-category counts
    manifest, upload_rows, cat_counts = {}, [], {}
    for rec in recs:
        size, size_known = rec["size"], True
        if size is None:
            size = dir_sizes.get(str(rec["src"]))
            size_known = size is not None
            size = size or 0

        cc = cat_counts.setdefault(rec["category"], {
            "count": 0, "size": 0, "share": rec["share"], "missing": 0,
            "sample": [], "partial": False})
        cc["count"] += 1
        cc["size"] += size
        cc["share"] = cc["share"] or rec["share"]
        if not rec["exists"]:
            cc["missing"] += 1
        if not size_known:
            cc["partial"] = True
        if len(cc["sample"]) < 3:
            cc["sample"].append(rec["logical"])

        entry = {"kind": rec["kind"], "share": rec["share"],
                 "category": rec["category"], "tier": rec["tier"], "size": size}
        if not size_known:
            entry["size_unknown"] = True
        if rec["share"]:
            grp, mkey = ckpt_owner.get(rec["logical"], (None, None))
            repo_id, repo_type, repo_path = repo_placement(
                rec["category"], rec["logical"], rules, grp, mkey)
            entry.update(repo_id=repo_id, repo_type=repo_type, repo_path=repo_path)
            if rec["slimmed"]:
                entry["slimmed"] = True
            pack = "tar" if rec["kind"] == "dir" else None
            if pack:
                entry["pack"] = pack
            if do_hash and rec["exists"] and rec["kind"] == "file":
                entry["sha256"] = _sha256(rec["local_src"])
            upload_rows.append({
                "repo_id": repo_id, "repo_type": repo_type,
                "local_src": str(rec["local_src"]), "repo_path": repo_path,
                "category": rec["category"], "kind": rec["kind"], "pack": pack or "",
                "exists": rec["exists"], "slimmed": rec["slimmed"], "size": size})
        manifest[rec["logical"]] = entry

    # Safety gate: every uploaded artifact must have a unique (repo_id, repo_path).
    # A clash would silently overwrite one file with another on HuggingFace, so a
    # collision is a hard build failure — fix DATA_LAYOUT / model naming and rerun.
    seen, collisions = {}, []
    for lg, e in manifest.items():
        if not e.get("share"):
            continue
        rk = (e["repo_id"], e["repo_path"])
        if rk in seen:
            collisions.append((rk, seen[rk], lg))
        else:
            seen[rk] = lg
    if collisions:
        for (rk, a, b) in collisions[:20]:
            print(f"  [COLLISION] {rk[0]}:{rk[1]}\n      <- {a}\n      <- {b}",
                  file=sys.stderr)
        raise SystemExit(
            f"repo_path collision: {len(collisions)} clash(es) — the by-type remap "
            f"maps two logicals to one HF path. Fix DATA_LAYOUT and rebuild.")

    _write_manifest(manifest)
    _write_upload_list(out_dir, upload_rows)
    _write_review(out_dir, cat_counts, rules, unmapped)
    if unknown_categories:
        _append_uncategorized(unknown_categories)

    n_share = sum(1 for e in manifest.values() if e["share"])
    print(
        f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}: {len(manifest)} entries "
        f"({n_share} share).",
        file=sys.stderr,
    )
    if unknown_categories:
        print(
            f"  [action] {len(unknown_categories)} uncategorized categories appended "
            f"to {DIGEST_PATH.name}: {sorted(unknown_categories)}",
            file=sys.stderr,
        )
    missing = [c for c, r in cat_counts.items() if r["missing"]]
    if missing:
        print(f"  [warn] categories with missing files on disk: {missing}", file=sys.stderr)


def _sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _write_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def _write_upload_list(out_dir, rows):
    tsv = out_dir / "hf_upload_list.tsv"
    cols = ["repo_id", "repo_type", "local_src", "repo_path", "category", "kind",
            "pack", "exists", "slimmed", "size"]
    with open(tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    sh = out_dir / "hf_upload.sh"
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by scripts/build_manifest.py — uploads ONLY shared files.",
        "# Review dist/hf_review.md and the GitHub sign-off issue before running.",
        "set -euo pipefail",
        "export HF_HUB_ENABLE_HF_TRANSFER=1",
        "",
    ]
    for r in rows:
        if not r["exists"]:
            lines.append(f'echo "SKIP (missing on disk): {r["local_src"]}"')
            continue
        if r["pack"] == "tar":
            lines.append(
                f'# DIR (tar): tar -cf "{r["repo_path"]}.tar" -C "{r["local_src"]}" . '
                f'&& huggingface-cli upload {r["repo_id"]} "{r["repo_path"]}.tar" '
                f'"{r["repo_path"]}.tar" --repo-type {r["repo_type"]}'
            )
        elif r["kind"] == "dir":
            lines.append(
                f'huggingface-cli upload-large-folder {r["repo_id"]} '
                f'"{r["local_src"]}" --path-in-repo "{r["repo_path"]}" '
                f'--repo-type {r["repo_type"]}'
            )
        else:
            if r["category"] == "model-checkpoints" and not r["slimmed"]:
                lines.append(
                    f'echo "WARNING: uploading NON-slimmed checkpoint (run slim first): '
                    f'{r["repo_path"]}"'
                )
            lines.append(
                f'huggingface-cli upload {r["repo_id"]} "{r["local_src"]}" '
                f'"{r["repo_path"]}" --repo-type {r["repo_type"]}'
            )
    sh.write_text("\n".join(lines) + "\n")
    sh.chmod(0o755)


def _human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def _write_review(out_dir, cat_counts, rules, unmapped):
    md = ["# bolerodata HuggingFace upload — review summary", ""]
    md.append("| Category | Status | Tier | Repo | Files | Total size | Missing |")
    md.append("|---|---|---|---|---|---|---|")
    for category in sorted(cat_counts):
        cc = cat_counts[category]
        rule = rules.get(category, {})
        status = "share" if cc["share"] else "noshare"
        size_str = _human(cc["size"]) + ("+" if cc.get("partial") else "")
        md.append(
            f"| {category} | {status} | {rule.get('tier','?')} | {rule.get('repo','—')} "
            f"| {cc['count']} | {size_str} | {cc['missing']} |"
        )
    md.append("")
    md.append("## Sample logical paths per category")
    for category in sorted(cat_counts):
        md.append(f"- **{category}**: " + ", ".join(cat_counts[category]["sample"]))
    if unmapped:
        md.append("")
        md.append(f"## Unmapped paths ({len(unmapped)}) — no LAB_ROOTS prefix matched")
        for category, p in unmapped[:50]:
            md.append(f"- [{category}] {p}")
    (out_dir / "hf_review.md").write_text("\n".join(md) + "\n")


def _append_uncategorized(categories):
    text = DIGEST_PATH.read_text()
    block = "\n".join(f"| {c} | (auto-detected) | ? | noshare | — |" for c in sorted(categories))
    marker = "_(none yet — run `python scripts/build_manifest.py` to populate)_"
    stamp = (
        "The following categories were found in the data but are missing from the "
        "table above. Classify each (move it into the table) and re-run:\n\n"
        "| Category | Description | Tier | Status | Repo |\n|---|---|---|---|---|\n"
        + block
    )
    text = text.replace(marker, stamp) if marker in text else text + "\n\n" + stamp
    DIGEST_PATH.write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hash", action="store_true", help="compute sha256 for shared files")
    ap.add_argument("--out", default=None, help="output dir for upload list / review")
    args = ap.parse_args()
    build(do_hash=args.hash, out_dir=args.out)


if __name__ == "__main__":
    main()
