<!--
  huggingface_digest.md — the single source of truth for WHAT bolerodata publishes.

  How it works
  ------------
  bolerodata references >10 TB of files across the lab filesystem. Only a curated
  subset is redistributed publicly on HuggingFace. This file decides that subset.

  Every file the registry can reach is assigned a CATEGORY by its *provenance*
  (which table column or joblib dict it came from — not by guessing from its path).
  `scripts/build_manifest.py` enumerates all referenced files, tags each with its
  category, then looks the category up in the table below to decide:

    - Status = share   -> the file is uploaded and auto-downloaded by end users
    - Status = noshare -> the file is withheld; asking for it raises a friendly
                          FileNotAvailableError (see bolerodata/_sync.py)

  To include or exclude a whole class of files, edit its **Status** cell below and
  re-run `python scripts/build_manifest.py`. Categories that appear in the data but
  are missing from this table are treated as `noshare` and listed under
  UNCATEGORIZED for you to classify.

  Repo column: bolero-models (HF model repo) or bolero-data (HF dataset repo).
  Tier: core = needed to load a model and run inference; eval = needed to reproduce
  the paper's evaluation figures; full = full training corpus (excluded by default).

  DO NOT change the category NAMES (the build script matches on them); only edit
  Status / Tier / Repo, or add new rows for UNCATEGORIZED categories.
-->

# HuggingFace publishing digest for `bolerodata`

Release scope (chosen): **inference + evaluation**. The full meta-cell training
corpus (`metacell-adata`, `metacell-parquet`) is **excluded**.

HuggingFace org **`arcinstitute`** · model repo **`arcinstitute/bolero-models`** ·
dataset repo **`arcinstitute/bolero-data`** · visibility **public**.

## Category table

| Category | Description | Tier | Status | Repo |
|---|---|---|---|---|
| model-checkpoints | Trained LoRA checkpoints (`*.pt`) from `model_zoo.tsv` `CkptPath` | core | share | bolero-models |
| model-configs | Model config JSON from `model_zoo.tsv` `ConfigPath` | core | share | bolero-models |
| curated-cell-metadata | `standard/cell_metadata/*.feather` (Dataset.cell_metadata) | core | share | bolero-data |
| curated-sample-metadata | `standard/sample_metadata/*.feather` | core | share | bolero-data |
| curated-cluster-metadata | `standard/cluster_metadata/*.feather` | core | share | bolero-data |
| motif-scan | `standard/motif_scan/*.JASPAR_motif_hit.feather` | core | share | bolero-data |
| reference-bigwig | Reference coverage bigwigs (`META_CELL_REFERENCE_BIGWIG_PATH_DICT`) | core | share | bolero-data |
| pseudobulk-condition | Condition pseudobulk records — canonical `standard/embedding/coverage_pseudobulk_with_condition/*.cond*.joblib` (released-model inputs) | core | share | bolero-data |
| pseudobulk-benchmark | 250728-EmbeddingBenchmark condition **and metadata** pseudobulk (method-specific `+PANVI/+PVI/+SCVI/+SCANVI/+STATE`; not used by released models) | full | noshare | — |
| pseudobulk-reference | Reference pseudobulk (`reference_pseudobulk.joblib`) from `coverage_pseudobulk_with_condition/` | core | share | bolero-data |
| pseudobulk-metadata | Metadata pseudobulk records | core | share | bolero-data |
| pseudobulk-embedding | Embedding pseudobulk records | core | share | bolero-data |
| gene-hvg | Pseudobulk gene / HVG / ref-log1pcpm joblib | core | share | bolero-data |
| peaks | Per-dataset peak BED (`dataset_metadata.tsv` `peaks`) | core | share | bolero-data |
| path-table-index | The `file_path_table/*.joblib` dicts (key → path lookups) | core | share | bolero-data |
| original-cell-metadata | Original-study cell metadata feather (`cell_metadata` col) | eval | noshare | bolero-data |
| cell-embedding | Within-dataset cell embedding feather | eval | share | bolero-data |
| chromvar | Within-dataset chromVAR `.h5ad` | eval | share | bolero-data |
| chromvar-cross | Cross-dataset chromVAR `.h5ad` | eval | share | bolero-data |
| chromvar-motif | chromVAR motif reference | eval | share | bolero-data |
| diff-bigwig | Differential-analysis Group1/Group2 bigwigs (`DA.TotalRecords`) | eval | noshare | bolero-data |
| diff-tables | Differential-analysis feather tables (`DiffPath`) | eval | share | bolero-data |
| qtl-GTEx | GTEx caQTL/eQTL feathers ⚠️ redistribution — confirm before publish | eval | share | bolero-data |
| qtl-sc-eqtl | sc-eQTL (eQTL-catalog / OneK1K) feathers ⚠️ redistribution — confirm | eval | share | bolero-data |
| qtl-zeng2024 | Zeng2024 caQTL feathers | eval | share | bolero-data |
| qtl-training-regions | GTEx QTL model training-region feathers | eval | share | bolero-data |
| peak-adata | Per-dataset peak-count AnnData (`peak_adata`) | full | share | bolero-data |
| gene-adata | Per-dataset gene-count AnnData (`gene_adata`) | full | share | bolero-data |
| per-group-peaks | Per-group peak BEDs (`per_group_peaks`) | full | share | bolero-data |
| metacell-adata-standard | Standard metacell AnnData (166 files, model-training corpus) | full | share | bolero-data |
| metacell-adata | Meta-cell AnnData — non-standard (250728/250618 benchmark/project) copies | full | noshare | — |
| metacell-parquet-32bp | Standard 32bp metacell parquet datasets — the model-training resolution (Bolero10M config `data_path`); LARGE ~1.56 TB | full | share | bolero-data |
| metacell-parquet | Meta-cell parquet datasets — non-32bp (1bp) + project/benchmark copies — training corpus | full | noshare | — |
| diff-peak-scan | Diff-analysis `peaks/peaks.bed` + `*.npz` scan arrays | full | noshare | — |
| raw-fragments | Raw ATAC fragment stores (`fragment_files`, `//hms-*`) | raw | noshare | — |
| raw-snap | SnapATAC `.snap` files (`snap_files`) | raw | noshare | — |
| snap-index | `file_path_table/sample_snap_file_path.csv` (points at raw snaps) | raw | share | — |

## Notes

- **⚠️ GTEx / OneK1K redistribution.** `qtl-GTEx` and `qtl-sc-eqtl` derive from
  studies with data-use terms (dbGaP / consent). Summary-level QTL tables are
  usually redistributable, but **confirm licensing before `bolero-data` goes
  public** — this is called out again in the review-gate issue.
- **`path-table-index` leaks lab paths cosmetically.** The joblib dicts store
  absolute lab paths as *values*; `localize()` resolves them fine for end users,
  but the strings themselves reveal internal directory names. Optional future
  polish: rewrite the dict values to logical form at upload time.
- **Checkpoints are ~13 GB each (~1.3 TB total).** Upload with hf-xet / hf_transfer.

## UNCATEGORIZED

<!-- build_manifest.py appends any category found in the data but missing above.
     Classify each (add a row to the table) and re-run the build. -->

_(none yet — run `python scripts/build_manifest.py` to populate)_
