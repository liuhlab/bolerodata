# bolerodata

`bolerodata` is the **dataset, model-zoo and metadata registry** for
[Bolero](https://github.com/liuhlab/bolero) — a model that predicts
cell-state-specific gene regulation from DNA sequence. It is a thin, uniform
Python API that maps short keys (dataset names, model keys, tissue / cell-type
names) to the artifacts behind the paper — single-cell ATAC/RNA datasets, trained
model checkpoints, QTL collections and differential-accessibility records — and
returns them as pandas / AnnData objects and ready-to-use file paths.

## What it provides

| Object | What it gives you |
|---|---|
| `metadata` | Curated tables: dataset metadata, the model zoo, QTL collection, JASPAR motif clusters, DA records |
| `DATASETS` | Per-dataset accessor (`DATASETS.HumanBrainDev`): cell/sample/cluster metadata, meta-cell AnnData, pseudobulks, reference bigwigs, motif scans |
| `MODELS` | The model zoo (`MODELS["<key>"]`): resolves checkpoints + configs and builds a `bolero` predictor |
| `GTEx`, `Zeng2024`, `GTEx_eqtl_catalog`, `Onek1k` | QTL evaluation collections (variant-effect eval sets) |
| `diff_analysis.DA` | Differential-accessibility records + IGV browsing (optional `diff` extra) |

```python
import bolerodata
from bolerodata import DATASETS, MODELS, metadata, GTEx

metadata.MODEL_ZOO.head()                 # browse the model zoo (no downloads)
ds = DATASETS["HumanBrainDev"]
ds.cell_metadata                          # auto-downloads the feather on first use
model = MODELS["<model_key>"]             # cheap; the checkpoint downloads lazily
```

## How the data reaches you

`bolerodata` ships only small registry tables. The large artifacts (checkpoints,
metadata feathers, QTL tables, …) are **downloaded on demand from HuggingFace** the
first time you touch them and cached locally — see [Data setup & caching](data-setup.md).
The curated subset that is published is decided by `huggingface_digest.md`;
anything withheld raises a friendly error rather than a broken path.

- HuggingFace: **[`arcinstitute/bolero-models`](https://huggingface.co/arcinstitute/bolero-models)**
  (checkpoints) · **[`arcinstitute/bolero-data`](https://huggingface.co/datasets/arcinstitute/bolero-data)** (everything else)

## Relationship to `bolero`

`bolerodata` is the data/registry half of the project; the modelling code lives in
[`bolero`](https://github.com/liuhlab/bolero). Browsing the registry and downloading
data needs only `bolerodata`. **Building and running a predictor** (`MODELS[...].create_predictor()`),
resolving a `Genome`, or using `diff_analysis` additionally needs `bolero` (and a GPU
stack) installed — `bolerodata` imports it lazily. See [Installation](install.md).

## Next steps

- [Installation](install.md)
- [Data setup & caching](data-setup.md)
- [API reference](api.md)
