---
license: mit
library_name: pytorch
tags:
  - genomics
  - single-cell
  - ATAC-seq
  - gene-regulation
---

# Bolero models

Trained checkpoints for **Bolero**, a sequence-to-function model that predicts
cell-state-specific chromatin accessibility and gene expression from DNA sequence.
Each checkpoint holds LoRA weights trained on a Borzoi backbone.

## Layout

One folder per model: `<ModelGroup>/<model_key>/{checkpoint.pt, config.json}`.
`-Gene` models add a gene-expression head; `+AP1` models also take an AP-1 motif
score as input.

## Usage

Load the models through [`bolerodata`](https://github.com/liuhlab/bolerodata), which
downloads the files on first use. Building a predictor also needs
[`bolero`](https://github.com/liuhlab/bolero) and a GPU.

```bash
pip install bolerodata
```

```python
from bolerodata import MODELS

model = MODELS["WholeHuman"]          # registry entry, nothing downloaded yet
model.dataset = "HumanBrainDev"       # dataset whose cell states to predict
predictor = model.create_predictor()  # downloads checkpoint, config and inputs
```

Inputs (pseudobulk records, peaks, reference bigwigs) come from
[`arcinstitute/bolero-data`](https://huggingface.co/datasets/arcinstitute/bolero-data).

## Citation

Liu H. *Bolero: predicting cell-state-specific gene regulation from DNA sequence.* 2026.
