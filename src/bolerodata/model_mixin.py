from typing import TYPE_CHECKING

import joblib
import pandas as pd
import torch

if TYPE_CHECKING:
    from bolero.tl.predict.predictor_borzoi import BorzoiSignalPredictor

    from .dataset import Dataset


def determin_batch_size(gb_per_sample: int) -> int:
    """
    Determine the batch size based on the available GPU memory.
    """
    gpu_mem = 0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem += torch.cuda.get_device_properties(i).total_memory / (
                1024**3
            )  # bytes → GB
    batch_size = int((gpu_mem + 1) // gb_per_sample)
    return batch_size


class BorzoiModelMixin:
    def get_default_p0_meta_cells(
        self: "Dataset", subset_name: str = None, random_state: int = 0
    ) -> list[str]:
        """
        Get up to 1000 meta cells for creating fixed p0 meta cells.
        """
        meta_adata = self.get_meta_cell_adata(subset_name=subset_name)
        fixed_p0_meta_cells = meta_adata.obs.sample(
            min(meta_adata.obs_names.size, 1000), random_state=random_state
        ).index.tolist()
        return fixed_p0_meta_cells

    @property
    def pred_batch_size(self: "Dataset") -> int:
        """Batch size for prediction tasks."""
        return determin_batch_size(5)

    @property
    def attr_batch_size(self: "Dataset") -> int:
        """Batch size for attribution tasks."""
        return determin_batch_size(10)

    def _create_predictor(
        self: "Dataset",
        subset_name: str = None,
        pseudobulk_records_path: str = None,
        peak_bed_path: str = None,
        pseudobulk_type: str = "condition",
        sample_n_pseudobulks: int = 100,
        target_cov: int = 5000000,
        nosignal: bool = False,
        use_ref_bw: bool = True,
    ) -> "BorzoiSignalPredictor":
        from bolero.tl.predict.predictor_borzoi import BorzoiSignalPredictor

        # pseudobulk info
        if pseudobulk_records_path is None:
            pseudobulk_records_path = self.get_meta_cell_pseudobulk_records_path(
                target_cov,
                subset_name,
                pseudobulk_type=pseudobulk_type,
            )
        pseudobulk_records = joblib.load(pseudobulk_records_path)
        pseudobulk_ids = list(pseudobulk_records["pseudobulk_records"].keys())
        if len(pseudobulk_ids) > sample_n_pseudobulks:
            pseudobulk_ids = sorted(
                pd.Series(pseudobulk_ids)
                .sample(sample_n_pseudobulks, random_state=0)
                .tolist()
            )

        db_path = self.get_meta_cell_parquet_path("32bp", subset_name=subset_name)
        fixed_p0_meta_cells = self.get_default_p0_meta_cells(subset_name=subset_name)
        model_path_dict = self.get_borzoi_signal_model_path(subset_name=subset_name)
        if peak_bed_path is None:
            peak_bed_path = self.peak_bed_path
        if use_ref_bw:
            ref_bw_path = self.get_reference_bigwig_path(subset_name=subset_name)
            bigwig_paths = {"bigwig_paths": {"reference": ref_bw_path}}
        else:
            bigwig_paths = {}
        config = {
            "checkpoint_path": model_path_dict["ckpt_path"],
            "genome": self.genome,
            "train_config": model_path_dict["config_path"],
            "pseudobulk_records_path": pseudobulk_records_path,
            "dataset_path": db_path,
            "peak_path": peak_bed_path,
            "designs": pseudobulk_ids,
            "fixed_p0_meta_cells": fixed_p0_meta_cells,
            **bigwig_paths,
            # For no signal model inference, uncomment this:
            "nosignal": nosignal,
        }

        predictor = BorzoiSignalPredictor(config)
        return predictor
