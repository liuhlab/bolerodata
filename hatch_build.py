"""
Hatchling build hook: sanitize lab absolute paths out of the shipped data tables.

The git-committed TSVs (and ``DA.TotalRecords.csv.gz``) keep absolute
``/large_storage/...`` paths so lab authors' fast path is unchanged. This hook
rewrites those path columns to portable *logical* paths **in the wheel only**, so
published artifacts never expose internal filesystem layout. Resolution still
works because ``bolerodata._sync.localize()`` accepts both absolute-lab and
logical paths.

The core (:func:`sanitize_tables`) is importable/testable on its own; the
hatchling hook merely calls it and force-includes the rewritten copies.
"""

import csv
import gzip
import importlib.util
import pathlib
import tempfile

HERE = pathlib.Path(__file__).parent
SRC = HERE / "src"
DATA_DIR = SRC / "bolerodata" / "data"

# Shipped tables that may carry lab paths, with their on-disk format. We sanitize
# EVERY cell (not a hardcoded column list) — only cells whose parts start with "/"
# and match a known lab root are rewritten, so non-path cells are untouched. This
# is robust to new/renamed path columns.
TABLES = [
    ("dataset_metadata.tsv", "tsv"),
    ("embedding_and_meta_cell.tsv", "tsv"),
    ("model_zoo.tsv", "tsv"),
    ("qtl_collection.tsv", "tsv"),
    ("DA.TotalRecords.csv.gz", "gzcsv"),
]

# Markers that must NOT survive into the wheel. Any cell still containing one of
# these after logical-mapping (unmapped local roots, or cloud URLs to internal
# buckets like gs://hms-hanqing-*) is redacted — these only occur in raw/noshare
# columns (fragment/snap files) that `localize` never reads.
LAB_MARKERS = ("/large_storage", "/scratch/zhoulab", "hms-hanqing", "/gale/", "/ref/")
REDACTED = "<not-distributed>"


def _load_sync():
    """Load ``_sync.py`` standalone (no pandas-heavy package import) for its helpers."""
    spec = importlib.util.spec_from_file_location(
        "_bolerodata_sync_standalone", SRC / "bolerodata" / "_sync.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_cell_sanitizer(to_logical):
    def sanitize_cell(value):
        if not isinstance(value, str) or "/" not in value:
            return value
        out = []
        for part in value.split(","):
            p = part.strip()
            mapped = part
            if p.startswith("/"):
                logical = to_logical(p)
                if logical:
                    mapped = logical
            # Redact anything that still exposes an internal location (cloud URLs
            # to internal buckets, or a local root missing from LAB_ROOTS).
            if any(m in mapped for m in LAB_MARKERS):
                mapped = REDACTED
            out.append(mapped)
        return ",".join(out)

    return sanitize_cell


def _rewrite_all(rows, sanitize_cell):
    """Sanitize every cell of every data row (header row left untouched)."""
    for row in rows[1:]:
        for i in range(len(row)):
            row[i] = sanitize_cell(row[i])
    return rows


def _filter_excluded_rows(name, rows, excluded):
    """Drop rows for unpublished datasets from a shipped table (header kept).

    - ``dataset_metadata.tsv`` / ``embedding_and_meta_cell.tsv``: first column is
      the dataset -> drop matching rows.
    - ``model_zoo.tsv``: drop rows whose ``DefaultDataset`` is excluded (the
      dataset-specific model rows), and blank the free-text ``Notes`` column
      (it references training-history datasets, incl. excluded ones).
    - ``DA.TotalRecords.csv.gz``: drop any row mentioning an excluded dataset.
    - ``qtl_collection.tsv``: untouched ("Kidney" there is a GTEx tissue).
    """
    if not rows or name == "qtl_collection.tsv":
        return rows
    header = rows[0]
    ex = set(excluded)
    if name in ("dataset_metadata.tsv", "embedding_and_meta_cell.tsv"):
        return [header] + [r for r in rows[1:] if r and r[0] not in ex]
    if name == "model_zoo.tsv":
        di = header.index("DefaultDataset") if "DefaultDataset" in header else None
        ni = header.index("Notes") if "Notes" in header else None
        out = [header]
        for r in rows[1:]:
            if di is not None and len(r) > di and r[di] in ex:
                continue  # drop dataset-specific model rows for excluded datasets
            if ni is not None and len(r) > ni:
                r = list(r)
                r[ni] = ""  # strip Notes free-text at publish time
            out.append(r)
        return out
    if name == "DA.TotalRecords.csv.gz":
        return [header] + [r for r in rows[1:]
                           if not any(ds in cell for cell in r for ds in ex)]
    return rows


def _read_rows(src, fmt):
    if fmt == "gzcsv":
        with gzip.open(src, "rt", newline="") as f:
            return list(csv.reader(f))
    with open(src, newline="") as f:
        return list(csv.reader(f, delimiter="\t"))


def _write_rows(out, rows, fmt):
    if fmt == "gzcsv":
        with gzip.open(out, "wt", newline="") as f:
            csv.writer(f).writerows(rows)
    else:
        with open(out, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(rows)


def sanitize_tables(out_dir, strict=True):
    """
    Write sanitized copies of every shipped path-bearing table into ``out_dir``.

    Parameters
    ----------
    out_dir
        Directory to write the rewritten tables into.
    strict
        If True (default), raise if any lab-path marker survives sanitization —
        i.e. a path under a root not in ``bolerodata._sync.LAB_ROOTS``. This turns
        an accidental leak into a hard build failure.

    Returns
    -------
    dict
        ``{absolute_output_path: "bolerodata/data/<name>"}`` for hatchling
        ``force_include``.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sync = _load_sync()
    sanitize_cell = _make_cell_sanitizer(sync._to_logical)
    excluded = getattr(sync, "EXCLUDED_DATASETS", ())
    force_include = {}
    leaks = {}

    for name, fmt in TABLES:
        src = DATA_DIR / name
        if not src.exists():
            continue
        rows = _filter_excluded_rows(name, _read_rows(src, fmt), excluded)
        rows = _rewrite_all(rows, sanitize_cell)
        out = out_dir / name
        _write_rows(out, rows, fmt)
        force_include[str(out)] = f"bolerodata/data/{name}"

        blob = gzip.open(out, "rt").read() if fmt == "gzcsv" else out.read_text()
        n = sum(blob.count(m) for m in LAB_MARKERS)
        if n:
            leaks[name] = n

    if strict and leaks:
        raise RuntimeError(
            "Lab paths survived sanitization (roots missing from "
            f"bolerodata._sync.LAB_ROOTS): {leaks}. Add the missing roots and rebuild."
        )
    return force_include


try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface

    class SanitizePathsHook(BuildHookInterface):
        """Rewrite shipped-table path columns to logical form in the wheel."""

        PLUGIN_NAME = "sanitize-paths"

        def initialize(self, version, build_data):
            tmp = pathlib.Path(tempfile.mkdtemp(prefix="bolerodata-sanitize-"))
            mapping = sanitize_tables(tmp)
            fi = build_data.setdefault("force_include", {})
            fi.update(mapping)
            # The generated manifest is (re)built by scripts/build_manifest.py and
            # may be untracked; force-include it so localize() always has it.
            manifest = DATA_DIR / "hf_manifest.json"
            if manifest.exists():
                fi[str(manifest)] = "bolerodata/data/hf_manifest.json"

except ModuleNotFoundError:  # allow importing sanitize_tables without hatchling
    BuildHookInterface = None
