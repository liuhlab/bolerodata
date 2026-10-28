---
license: mit
tags:
  - genomics
  - single-cell
  - ATAC-seq
  - gene-regulation
---

# Bolero data

Data behind **Bolero**, a sequence-to-function model that predicts
cell-state-specific chromatin accessibility and gene expression from DNA sequence:
the Bolero-10M single-cell ATAC/RNA collection, the inputs the released models
need, and the evaluation sets used in the paper. Raw fragment files are not
included.

## Layout

| Folder | Contents |
|---|---|
| `metadata/` | Cell, sample and cluster metadata per dataset |
| `metacell/` | Metacell AnnData and 32 bp parquet stores (training corpus) |
| `pseudobulk/` | Pseudobulk records used as model inputs |
| `reference_bigwig/` | Reference coverage bigwigs |
| `peaks/`, `peak_adata/`, `gene_adata/` | Peak sets and peak / gene count matrices |
| `gene_data/` | Pseudobulk gene expression and highly variable genes |
| `cell_embedding/`, `chromvar/`, `motif_scan/` | Cell embeddings, chromVAR scores, JASPAR motif hits |
| `score_data/` | Motif score inputs for the `+AP1` models |
| `qtl/` | caQTL / eQTL evaluation sets |
| `diff_analysis/` | Differential accessibility tables |
| `reference_gtf/` | Gene annotations |
| `index/` | Key-to-path lookup tables used by `bolerodata` |

## Usage

Address files by name through [`bolerodata`](https://github.com/liuhlab/bolerodata),
which downloads each file on first use:

```bash
pip install bolerodata
```

```python
from bolerodata import DATASETS, metadata

metadata.DATASET_METADATA.head()            # list datasets (offline)
DATASETS["HumanBrainDev"].cell_metadata     # downloads and returns a DataFrame
```

Model checkpoints live in
[`arcinstitute/bolero-models`](https://huggingface.co/arcinstitute/bolero-models).

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
