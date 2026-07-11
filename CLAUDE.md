# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## What bolerodata is

`bolerodata` is the **dataset, model-zoo and metadata registry** for the paper *"Bolero:
predicting cell-state-specific gene regulation from DNA sequence."* It is the companion
package to [`bolero`](https://github.com/liuhlab/bolero) (the model/training/inference code
base): where `bolero` holds the *model*, `bolerodata` holds the uniform Python API used to
**find and load the data behind it** — the single-cell ATAC/RNA datasets (Bolero-10M),
trained model checkpoints, QTL collections and differential-accessibility records that
train Bolero and generate the paper's results.

It is a **thin registry / index layer**, not a data-processing engine: it maps short keys
(dataset names, model keys, tissue/cell-type names) to on-disk artifacts and returns them
as pandas/AnnData objects. The heavy lifting (models, Ray pipelines, attribution, plotting)
lives in `bolero`.

## Relationship to bolero (read this first)

- **Two repos, one project.** `bolero` is the **main** package and the environment owner
  (Python + PyTorch/CUDA + Ray + flash-attn, all via pixi). `bolerodata` is a lightweight
  satellite that is **consumed by `bolero`**: `bolero`'s `pyproject.toml` lists `bolerodata`
  as a **git-based pixi dependency**, so installing `bolero` brings `bolerodata` along.
- **`bolerodata` runs inside the `bolero` environment.** Its own pixi env is intentionally
  lean (no GPU stack), so several features only work when `bolero` is importable.
- **`bolerodata` imports `bolero`** — mostly *lazily*, inside methods (e.g.
  `dataset.py` `Dataset.Genome` → `bolero.Genome`; `model.py` → `bolero.tl.predict`
  predictor classes). The exception is `diff_analysis.py`, which imports `bolero` at module
  top level (`bolero.pl.igv.Browser`, `bolero.utils.understand_regions`), so that module
  requires `bolero` to even import. To avoid an import cycle, keep `bolerodata`'s
  package-level `__init__` free of eager `bolero` imports.
- **Scope split, in one line:** ask *"is this about the model / sequence / training /
  inference machinery?"* → it belongs in `bolero`. *"Is this about which datasets / models /
  QTLs / DA records exist and where their files live?"* → it belongs in `bolerodata`.

## Important: research code, and it is local-configuration bound

This is a research code base finalized to ship with the paper, so — like `bolero` — it
contains **vestigial and experimental code** (datasets, model-zoo rows, helpers that were
tried but are not part of the final published results). Do **not** proactively audit, flag,
or prune "vestigial" code; work on the specific piece under discussion, and when it is
unclear whether something is final vs. vestigial, **ask rather than guess**. The `bolero`
manuscript (`bolero/paper/Draft.md`, git-ignored) is the authoritative spec for what is real.

Crucially, `bolerodata` is **bound to the lab's local filesystem**:

- `data/__init__.py` resolves a **`STANDARD_DIR`** (env var `$STANDARD_DIR`, else the
  hardcoded default `/large_storage/zhoulab/hanliu/wmb/standard`) and **`assert`s it exists
  at import time** — so `import bolerodata` fails on any machine without that data lake
  (e.g. CI). Many methods then read `.feather`/`.h5ad`/`.joblib`/`.bw` files from absolute
  paths under `STANDARD_DIR`.
- `data/_path_patch.py` and `qtl.py` (`QTL_TRAINING_PATH_DIR`) hold **additional hardcoded
  absolute paths** to benchmark/QTL datasets. These are lab-local overrides.
- Consequence: **tests and imports are not runnable on GitHub CI as-is** (see the Test
  workflow) — they need the local data. Treat portability cleanup as future, author-driven work.

## Environment & common commands

Like `bolero`, this repo uses **[pixi](https://pixi.sh)** (config in `pyproject.toml` under
`[tool.pixi.*]`). But it is **lean**: only `linux-64` (no CUDA platform), and no
torch/CUDA/Ray/flash-attn — those come from `bolero` when the two are used together.

Environments: `default` (minimal runtime: `anndata`, `joblib`, `pandas`, `tqdm`), `dev`
(adds `pre-commit` + `pytest`/`coverage`), `docs` (CUDA-free, mkdocs only).

```bash
pixi install                 # lean runtime env
pixi install -e dev          # dev env (tests + lint)

pixi run -e dev test         # = coverage run -m pytest -v --color=yes  (needs $STANDARD_DIR)
pixi run -e dev lint         # = pre-commit run --all-files

pixi run -e docs docs-serve  # mkdocs live preview
pixi run -e docs docs-build  # mkdocs build --strict
pixi run -e docs docs-deploy # build + push to gh-pages
```

Note: to exercise the full API (`Genome`, predictors, IGV browser) you must run inside a
`bolero` environment, since those pull in `bolero`.

CI/CD mirrors `bolero`: `.github/workflows/` has `test`, `build`, `docs` (mkdocs → gh-pages),
and `release` (PyPI on a `v*` tag). Versioning is `hatch-vcs` from git tags using **CalVer**
(`vYYYY.MM.DD`), matching `bolero`.

## Package layout & import conventions

Single top-level package `src/bolerodata/`. Unlike `bolero` (scverse `pp`/`tl`/`pl` layout),
`bolerodata` is flat. The package `__init__` curates the public API:

```python
from bolerodata import metadata, DATASETS, MODELS, GTEx, Zeng2024, GTEx_eqtl_catalog, Onek1k
```

`DA` (differential analysis) is **not** re-exported — import it explicitly with
`from bolerodata.diff_analysis import DA` (that module eagerly imports `bolero`).

The public objects are **module-level singletons** created at import time (`DATASETS`,
`MODELS`, `metadata`, `GTEx`, …). The registries use `__getattr__`/`__getitem__` so datasets
and models are addressed by name (`DATASETS.HumanBrainDev`, `MODELS["<model_key>"]`), with
lazy per-instance `_cache` dicts.

## Core architecture (the registries)

### `data/` — the metadata backbone (`metadata` singleton, `Metadata` class)
- Bundled small tables live **in-package** under `src/bolerodata/data/` and are the only
  committed data assets: `dataset_metadata.tsv`, `embedding_and_meta_cell.tsv`,
  `model_zoo.tsv`, `qtl_collection.tsv`, `JASPAR_motif_cluster{,_info}.tsv`,
  `DA.TotalRecords.csv.gz`. `Metadata` exposes them as cached DataFrames
  (`DATASET_METADATA`, `MODEL_ZOO`, `QTL_COLLECTION`, `JASPAR_MOTIF_CLUSTER*`, `DA_COLLECTION`).
- **Large artifacts are not bundled**: their paths come from `.joblib` dictionaries under
  `STANDARD_DIR/file_path_table/` (metacell adata/parquet, pseudobulk records, reference
  bigwigs, gene/HVG data). `get_misc_data_path()` loads a joblib dict, applies the
  `_path_patch.py` overrides, and looks up a comma-joined key; a missing key raises a
  `KeyError` that suggests near-miss keys.
- `_path_patch.py` — hardcoded absolute-path overrides for benchmark/non-standard datasets
  (embedding benchmark, perturbation control groups, etc.).

### `dataset.py` — `Dataset` + `Datasets` (`DATASETS` singleton)
Per-dataset accessor. `Dataset(name)` pulls its row from `DATASET_METADATA` and exposes:
cell/sample/cluster metadata (`.feather`), original + within-dataset-embedding tables, snap
file tables, peak/gene AnnData paths, chromVAR AnnData (within/cross dataset), peak motif
scans, and a family of path getters for metacell AnnData (`get_meta_cell_adata[_path]`),
32bp/1bp parquet dbs, pseudobulk records (`condition`/`metadata`/`embedding` types, by target
coverage), reference bigwigs, and pseudobulk gene/HVG data. `.Genome` lazily builds a
`bolero.Genome`. `Datasets` is the registry: attribute/index access, iteration, and
`get_datasets[/_and_subset_names](species=, genome=)` filters.

### `model.py` — `BoleroModel` + `Models` (`MODELS` singleton)
Wraps rows of the **model zoo** (`model_zoo.tsv`): resolves `ConfigPath`/`CkptPath`, binds a
default `Dataset`, and — the main entry point — `create_predictor(...)` assembles the right
`bolero.tl.predict` predictor for the model group (`BorzoiSignalMultiDS`/`SingleDS`,
`Scooby`, `BorzoiMultihead`), wiring in pseudobulk records, peak beds, reference bigwigs and
fixed p0 meta cells. `determin_batch_size()` sizes batches from live GPU memory via `torch`.

### `qtl.py` — QTL collections (`GTEx`, `Zeng2024`, `GTEx_eqtl_catalog`, `Onek1k`)
`QTLCollection` filters `qtl_collection.tsv` by `Source` and provides `get_qtl_table()`
(reads a `.feather` via `get_qtl_path`); the subclasses only override
`get_qtl_path(tissue|cell_type, ...)` with source-specific key builders.
`GTExQTLCollection.get_qtl_training_path()` points at a hardcoded `QTL_TRAINING_PATH_DIR`.

### `diff_analysis.py` — `DiffRecords` + `DA` (not re-exported; eagerly imports `bolero`)
`DiffRecords` is one pairwise differential-accessibility/expression record (keyed by `DAKey`
in `DA.TotalRecords.csv.gz`, combining that key's `Gene` and `Peak` rows): gene/peak diff
tables (LFC-filtered), Group1/Group2 bigwig value/stat extraction and peak scans,
significant-peak BED dumping, and an IGV browser via `bolero.pl.igv.Browser`.
`DiffAnalysisCollection` (`DA`) indexes records by `DAKey`.

## Paper ↔ code map (quick reference)

- **Bolero-10M datasets / metacells / pseudobulks** = `DATASETS[...]` accessors +
  `metadata` path getters (files under `$STANDARD_DIR`).
- **Model zoo / building a predictor** = `MODELS[...]` → `BoleroModel.create_predictor()`
  (delegates to `bolero.tl.predict`).
- **Variant-effect eval sets (caQTL/eQTL)** = `GTEx`, `Zeng2024`, `GTEx_eqtl_catalog`,
  `Onek1k` in `qtl.py`.
- **Differential accessibility (DAR / velocity) records + IGV** = `diff_analysis.DA`.

## Conventions

- Formatting/linting: **ruff** (line length 88, numpy-docstring convention) via
  `.pre-commit-config.yaml` — identical config to `bolero`. New Python should carry
  numpy-style docstrings on public functions/classes.
- Registry pattern: module-level singletons with lazy `_cache`, `__getattr__`/`__getitem__`
  name-based access, feather/joblib/AnnData IO.
- Keep runtime dependencies **minimal** and keep the package `__init__` free of eager
  `bolero` imports (import `bolero` lazily inside functions/methods) to avoid an import cycle
  with `bolero`, which depends on `bolerodata`.
