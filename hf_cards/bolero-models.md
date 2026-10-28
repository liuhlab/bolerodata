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

Liu H, Mao S, Gallent T, Chen Y, Yan Z, Vogel I, Traunmuller L, Zhang R,
Buenrostro JD, Zhou J, Greenberg ME. *Predicting cell-state-specific gene regulation
from DNA across species.* bioRxiv (2026).
[doi:10.64898/2026.10.19.760026](https://doi.org/10.64898/2026.10.19.760026)

```bibtex
@article{liu2026bolero,
  title   = {Predicting cell-state-specific gene regulation from DNA across species},
  author  = {Liu, H. and Mao, S. and Gallent, T. and Chen, Y. and Yan, Z. and
             Vogel, I. and Traunmuller, L. and Zhang, R. and Buenrostro, J. D. and
             Zhou, J. and Greenberg, M. E.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.19.760026}
}
```
