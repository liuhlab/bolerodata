# bolerodata

**Dataset, model-zoo and metadata registry for [Bolero](https://github.com/liuhlab/bolero).**

A uniform Python API over the single-cell ATAC/RNA datasets, model checkpoints, QTL
collections and differential-accessibility records behind Bolero. It maps short keys
to artifacts and returns them as pandas / AnnData objects and ready-to-use file paths.
Large artifacts are **downloaded on demand from HuggingFace** and cached locally.

Documentation: <https://liuhlab.github.io/bolerodata/>

## Installation

```bash
pip install bolerodata            # registry + auto-download of shared data
pip install 'bolerodata[diff]'    # + differential-analysis stack
```

```python
import bolerodata
from bolerodata import DATASETS, MODELS, metadata, GTEx

metadata.MODEL_ZOO.head()                    # browse the model zoo (offline)
DATASETS["HumanBrainDev"].cell_metadata      # downloads from HuggingFace on first use
model = MODELS["<model_key>"]                # checkpoint downloads lazily
```

Building / running a model predictor, resolving a `Genome`, or using
`diff_analysis` additionally needs the companion
[`bolero`](https://github.com/liuhlab/bolero) package (GPU stack); `bolerodata`
imports it lazily. See the [installation guide](https://liuhlab.github.io/bolerodata/install/).

Data is fetched from **[`arcinstitute/bolero-models`](https://huggingface.co/arcinstitute/bolero-models)**
(checkpoints) and **[`arcinstitute/bolero-data`](https://huggingface.co/datasets/arcinstitute/bolero-data)**
into `$BOLERODATA_HOME` (default: the platform cache dir). See
[Data setup & caching](https://liuhlab.github.io/bolerodata/data-setup/).

## Developing

The repo uses [pixi](https://pixi.sh) (lean — no GPU stack):

```bash
git clone https://github.com/liuhlab/bolerodata.git
cd bolerodata
pixi install            # runtime env; or: pixi install -e dev / -e diff / -e docs
```

Maintainers publishing data to HuggingFace: see
[`huggingface_digest.md`](huggingface_digest.md) and the
[sharing guide](https://liuhlab.github.io/bolerodata/sharing/).

## License

MIT — see [LICENSE](LICENSE).
