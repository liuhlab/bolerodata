# dataset files patch
CTXGlutBenchmarkDir = "/large_storage/zhoulab/hanliu/250618-MouseBrainGeneralization"
EmbeddingBenchmarkDir = (
    "/large_storage/zhoulab/hanliu/250728-EmbeddingBenchmark/metacell_dataset"
)

embedding_benchmark_paths = {}
embedding_benchmark_parquet_paths = {}
for dataset in ["HumanBrainDev", "AIBSDev"]:
    for emb_type in ["PVI", "PANVI", "SCVI", "SCANVI", "STATE"]:
        _d = {
            (
                "cell",
                dataset,
                emb_type,
            ): f"{EmbeddingBenchmarkDir}/adata/metadata_and_coords/{dataset}+{emb_type}.cell.h5ad",
            (
                "metadata",
                dataset,
                emb_type,
            ): f"{EmbeddingBenchmarkDir}/adata/metadata_and_coords/{dataset}+{emb_type}.metacell.h5ad",
            (
                "peak",
                dataset,
                emb_type,
            ): f"{EmbeddingBenchmarkDir}/adata/peak/{dataset}+{emb_type}.meta_cell.peak_count.h5ad",
        }
        embedding_benchmark_paths.update(_d)
        _d = {
            (
                "32bp",
                dataset,
                emb_type,
            ): f"{EmbeddingBenchmarkDir}/parquet/{dataset}+{emb_type}/{dataset}-MetaCell-32bp",
        }
        embedding_benchmark_parquet_paths.update(_d)

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
        ): f"{EmbeddingBenchmarkDir}/adata/metadata_and_coords/Zu2023Nature+CTXGlutPVI.metacell.h5ad",
        (
            "cell",
            "Zu2023Nature",
            "CTXGlutPVI",
        ): f"{EmbeddingBenchmarkDir}/adata/metadata_and_coords/Zu2023Nature+CTXGlutPVI.cell.h5ad",
        (
            "peak",
            "Zu2023Nature",
            "CTXGlutPVI",
        ): f"{EmbeddingBenchmarkDir}/adata/peak/Zu2023Nature+CTXGlutPVI.meta_cell.peak_count.h5ad",
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
        ): f"{EmbeddingBenchmarkDir}/parquet/Zu2023Nature+CTXGlutPVI/Zu2023Nature-MetaCell-32bp",
        **embedding_benchmark_parquet_paths,
    },
}
