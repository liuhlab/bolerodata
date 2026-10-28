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

Liu H. *Bolero: predicting cell-state-specific gene regulation from DNA sequence.* 2026.
