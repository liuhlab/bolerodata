# Installation

## Quick start (registry + data access)

```bash
pip install bolerodata
```

This is enough to browse the registry and **auto-download shared data from
HuggingFace**:

```python
import bolerodata
from bolerodata import DATASETS, MODELS, metadata
metadata.DATASET_METADATA          # works offline; ships in the package
DATASETS["HumanBrainDev"].cell_metadata   # downloads from HuggingFace on first use
```

Core dependencies are intentionally light: `anndata`, `pandas`, `joblib`, `numpy`,
plus `huggingface_hub` and `platformdirs` for downloading/caching.

## What each capability needs

| You want to… | Install |
|---|---|
| Browse the registry, download curated data | `pip install bolerodata` |
| Build & run a model predictor, resolve a `Genome` | `bolerodata` **+ [`bolero`](https://github.com/liuhlab/bolero)** (brings `torch` + GPU stack) |
| Differential-accessibility analysis (`bolerodata.diff_analysis`) | `pip install 'bolerodata[diff]'` **+ `bolero`** |

`torch` and `bolero` are **not** declared as dependencies:

- `torch` is imported lazily (only for GPU batch-sizing / building predictors), so
  `import bolerodata` works without it.
- `bolero` is the companion package (it depends on `bolerodata`, so declaring it
  here would be circular). Install it separately when you need modelling features.

The `diff` extra adds the bioinformatics stack used only by `diff_analysis`
(`pyBigWig`, `pyranges`, `pysam`). If you import `diff_analysis` without it, you
get a clear message telling you to `pip install 'bolerodata[diff]'`.

## Developing bolerodata

The repo uses [pixi](https://pixi.sh) (lean — no GPU stack):

```bash
git clone https://github.com/liuhlab/bolerodata.git
cd bolerodata
pixi install            # runtime env
pixi install -e dev     # + tests + lint (pytest, pre-commit)
pixi install -e diff    # + differential-analysis deps
pixi install -e docs    # + mkdocs
```

Common tasks:

```bash
pixi run -e dev test          # pytest
pixi run -e dev lint          # pre-commit (ruff)
pixi run -e docs docs-serve   # live docs preview
```

## Lab / author mode

If you are on the lab filesystem (the data lake exists at `$STANDARD_DIR`), files
resolve directly from disk and **nothing is downloaded** — behaviour is unchanged
from before the HuggingFace layer existed. See
[Data setup & caching](data-setup.md#author-mode-lab-filesystem).
