# dataset files patch
CTXGlutBenchmarkDir = "/large_storage/zhoulab/hanliu/250618-MouseBrainGeneralization"
EmbeddingBenchmarkDatasetDir = (
    "/large_storage/zhoulab/hanliu/250728-EmbeddingBenchmark/metacell_dataset"
)
EmbeddingBenchmarkDir = "/large_storage/zhoulab/hanliu/250728-EmbeddingBenchmark"

embedding_benchmark_paths = {}
embedding_benchmark_parquet_paths = {}
embedding_benchmark_cond_pseudobulk_paths = {}
embedding_benchmark_metadata_pseudobulk_paths = {}
for dataset in ["HumanBrainDev", "AIBSDev"]:
    for emb_type in ["PVI", "PANVI", "SCVI", "SCANVI", "STATE"]:
        _d = {
            (
                "cell",
                dataset,
                emb_type,
            ): f"{EmbeddingBenchmarkDatasetDir}/adata/metadata_and_coords/{dataset}+{emb_type}.cell.h5ad",
            (
                "metadata",
                dataset,
                emb_type,
            ): f"{EmbeddingBenchmarkDatasetDir}/adata/metadata_and_coords/{dataset}+{emb_type}.metacell.h5ad",
            (
                "peak",
                dataset,
                emb_type,
            ): f"{EmbeddingBenchmarkDatasetDir}/adata/peak/{dataset}+{emb_type}.meta_cell.peak_count.h5ad",
        }
        embedding_benchmark_paths.update(_d)
        _d = {
            (
                "32bp",
                dataset,
                emb_type,
            ): f"{EmbeddingBenchmarkDatasetDir}/parquet/{dataset}+{emb_type}/{dataset}-MetaCell-32bp",
        }
        embedding_benchmark_parquet_paths.update(_d)
        _d = {
            (
                5000000,
                dataset,
                emb_type,
            ): f"{EmbeddingBenchmarkDir}/pseudobulk/{dataset}+{emb_type}/pseudobulk_records_and_cond.cov5000000.joblib",
        }
        embedding_benchmark_cond_pseudobulk_paths.update(_d)
        _d = {
            (
                5000000,
                dataset,
                emb_type,
            ): f"{EmbeddingBenchmarkDir}/metadata_pseudobulk/{dataset}+{emb_type}/pseudobulk_records_and_cond.cov5000000.joblib",
        }
        embedding_benchmark_metadata_pseudobulk_paths.update(_d)

# perturb dataset control group
_ctrl_dir = "/large_storage/zhoulab/hanliu/wmb/standard/embedding/coverage_pseudobulk_with_condition/control_only"
control_benchmark_cond_pseudobulk_paths = {
    (
        5000000,
        dataset,
        "Control",
    ): f"{_ctrl_dir}/{dataset}.pseudobulk_records_and_cond.cov5000000.joblib"
    for dataset in ["EE", "KA", "Torpor", "Fear", "Stress", "MultipleSclerosis"]
}


PATH_PATCH_DICT = {
    "META_CELL_ADATA_PATH_DICT": {
        (
            "peak",
            "Zu2023Nature",
            "CTXGlutBenchmark",
        ): f"{CTXGlutBenchmarkDir}/metacell_dataset/adata/Zu2023Nature+CTXGlut.meta_cell.peak_count.h5ad",
        (
            "cell",
            "Zu2023Nature",
            "CTXGlutBenchmark",
        ): f"{CTXGlutBenchmarkDir}/metacell_dataset/adata/Zu2023Nature+CTXGlut.cell.h5ad",
        (
            "metadata",
            "Zu2023Nature",
            "CTXGlutBenchmark",
        ): f"{CTXGlutBenchmarkDir}/metacell_dataset/adata/Zu2023Nature+CTXGlut.metacell.h5ad",
        (
            "metadata",
            "Zu2023Nature",
            "CTXGlutPVI",
        ): f"{EmbeddingBenchmarkDatasetDir}/adata/metadata_and_coords/Zu2023Nature+CTXGlutPVI.metacell.h5ad",
        (
            "cell",
            "Zu2023Nature",
            "CTXGlutPVI",
        ): f"{EmbeddingBenchmarkDatasetDir}/adata/metadata_and_coords/Zu2023Nature+CTXGlutPVI.cell.h5ad",
        (
            "peak",
            "Zu2023Nature",
            "CTXGlutPVI",
        ): f"{EmbeddingBenchmarkDatasetDir}/adata/peak/Zu2023Nature+CTXGlutPVI.meta_cell.peak_count.h5ad",
        **embedding_benchmark_paths,
    },
    "META_CELL_PARQUET_PATH_DICT": {
        (
            "1bp",
            "Zu2023Nature",
            "CTXGlutBenchmark",
        ): f"{CTXGlutBenchmarkDir}/metacell_dataset/parquet/Zu2023Nature-MetaCell-1bp",
        (
            "32bp",
            "Zu2023Nature",
            "CTXGlutBenchmark",
        ): f"{CTXGlutBenchmarkDir}/metacell_dataset/parquet/Zu2023Nature-MetaCell-32bp",
        (
            "32bp",
            "Zu2023Nature",
            "CTXGlutPVI",
        ): f"{EmbeddingBenchmarkDatasetDir}/parquet/Zu2023Nature+CTXGlutPVI/Zu2023Nature-MetaCell-32bp",
        **embedding_benchmark_parquet_paths,
    },
    "META_CELL_COND_PSEUDOBULK_RECORDS_PATH_DICT": {
        (
            5000000,
            "Zu2023Nature",
            "CTXGlutPVI",
        ): f"{EmbeddingBenchmarkDir}/pseudobulk/Zu2023Nature+CTXGlutPVI/pseudobulk_records_and_cond.cov5000000.joblib",
        **embedding_benchmark_cond_pseudobulk_paths,
        **control_benchmark_cond_pseudobulk_paths,
    },
    "META_CELL_METADATA_PSEUDOBULK_RECORDS_PATH_DICT": {
        (
            5000000,
            "Zu2023Nature",
            "CTXGlutPVI",
        ): f"{EmbeddingBenchmarkDir}/metadata_pseudobulk/Zu2023Nature+CTXGlutPVI/pseudobulk_records_and_cond.cov5000000.joblib",
        **embedding_benchmark_metadata_pseudobulk_paths,
    },
}
