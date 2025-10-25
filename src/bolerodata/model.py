import json
import pathlib
from typing import TYPE_CHECKING

import joblib
import pandas as pd
import torch

from .data import metadata
from .dataset import DATASETS, Dataset

if TYPE_CHECKING:
    from bolero.tl.predict.predictor import GenericPredictor

MODEL_ZOO: pd.DataFrame = metadata.MODEL_ZOO


def _get_predictor_class(model_type: str) -> type:
    if model_type == "BorzoiSignalMultiDS":
        from bolero.tl.predict.predictor_borzoi import BorzoiSignalPredictorMultiModel

        return BorzoiSignalPredictorMultiModel
    elif model_type == "BorzoiSignalSingleDS":
        from bolero.tl.predict.predictor_borzoi import BorzoiSignalPredictor

        return BorzoiSignalPredictor
    elif model_type == "BorzoiMultihead":
        from bolero.tl.predict.predictor_borzoi import BorzoiMultiHeadPredictor

        return BorzoiMultiHeadPredictor
    elif model_type == "Scooby":
        from bolero.tl.predict.predictor_borzoi import BorzoiPredictor

        return BorzoiPredictor
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_dict(model_key: str) -> dict:
    """Get the model record dictionary."""
    if model_key not in MODEL_ZOO.index:
        return {}
    data = MODEL_ZOO.loc[model_key].to_dict()
    return data


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


class BoleroModel:
    model_key: str
    model_metadata: dict
    config_path: pathlib.Path
    model_config: dict
    ckpt_path: pathlib.Path
    _has_gene_count_head: bool
    _tf_score_input: bool
    model_group: str
    default_dataset: str
    _current_dataset: "Dataset | None"

    def __init__(self, model_key: str, **additional_model_metadata: dict):
        self.model_key = model_key
        _model_dict = get_model_dict(model_key)
        _model_dict.update(additional_model_metadata)

        self.model_metadata = _model_dict
        self.config_path = (
            pathlib.Path(self.model_metadata["ConfigPath"]).absolute().resolve()
        )
        with open(self.config_path) as f:
            self.model_config = json.load(f)

        self.ckpt_path = (
            pathlib.Path(self.model_metadata["CkptPath"]).absolute().resolve()
        )
        self._has_gene_count_head = self.model_metadata["Gene Count Head"]
        self._tf_score_input = self.model_metadata["TF Score Input"]

        self.model_group = self.model_metadata["ModelGroup"]

        self.default_dataset = self.model_metadata["DefaultDataset"]
        self._current_dataset = None

    @property
    def dataset(self) -> "Dataset":
        """Get the dataset for the model."""
        if self._current_dataset is None and isinstance(self.default_dataset, str):
            self._current_dataset = DATASETS[self.default_dataset]
        assert self._current_dataset is not None, "Dataset is not set"
        return self._current_dataset

    @dataset.setter
    def dataset(self, dataset: "Dataset"):
        """Set the dataset for the model."""
        self._current_dataset = DATASETS[dataset]

    def get_default_p0_meta_cells(
        self, subset_name: str = None, random_state: int = 0
    ) -> list[str]:
        """
        Get up to 1000 meta cells for creating fixed p0 meta cells.
        """
        if subset_name == "Control":
            return None

        meta_adata = self.dataset.get_meta_cell_adata(subset_name=subset_name)
        fixed_p0_meta_cells = meta_adata.obs.sample(
            min(meta_adata.obs_names.size, 1000), random_state=random_state
        ).index.tolist()
        return fixed_p0_meta_cells

    @property
    def pred_batch_size(self) -> int:
        """Batch size for prediction tasks."""
        return determin_batch_size(5)

    @property
    def attr_batch_size(self) -> int:
        """Batch size for attribution tasks."""
        return determin_batch_size(10)

    def create_predictor(
        self,
        subset_name: str = None,
        pseudobulk_records_path: str = None,
        pseudobulk_ids: list[str] = None,
        target_cov: int = 5000000,
        pseudobulk_type: str = "condition",
        sample_n_pseudobulks: int = 100,
        peak_bed_path: str = None,
        nosignal: bool = False,
        use_ref_bw: bool = True,
        embedding_only_mode: bool = False,
        **kwargs,
    ) -> "GenericPredictor":
        """
        Create a predictor for the model.

        Parameters
        ----------
        subset_name: str
            The name of the subset to create the predictor for.
        pseudobulk_records_path: str or dict
            The path to the pseudobulk records file, if None, will use the dataset's default pseudobulk records file.
            If a dict, will use the pseudobulk records from the dict.
        pseudobulk_ids: list[str]
            The list of pseudobulk ids to use. If None, will use all pseudobulk ids in the pseudobulk records file.
        peak_bed_path: str
            The path to the peak bed file. If None, will use the dataset's default peak bed file.
        pseudobulk_type: str
            The type of dataset's default pseudobulk file to use. Must be one of ['condition', 'metadata'].
        sample_n_pseudobulks: int
            The number of pseudobulks to sample.
        target_cov: int
            The target coverage to use for the default pseudobulk records file.
        nosignal: bool
            Whether to use the DNA only (no signal) model.
        use_ref_bw: bool
            Whether to use the reference bigwig file to represent the reference signal.
        embedding_only_mode: bool
            Whether to use the embedding only mode to create the predictor.
            If True, will skip ytrue data preparation and skip all the metrics calculation.

        Returns
        -------
        GenericPredictor
            The predictor for the model.
        """
        _predictor_class = _get_predictor_class(self.model_group)
        kwargs["_predictor_class"] = _predictor_class

        # pseudobulk info
        if pseudobulk_records_path is None:
            pseudobulk_records_path = (
                self.dataset.get_meta_cell_pseudobulk_records_path(
                    target_cov,
                    subset_name,
                    pseudobulk_type=pseudobulk_type,
                )
            )

        kwargs["pseudobulk_records_path"] = pseudobulk_records_path
        if not isinstance(pseudobulk_records_path, dict):
            pseudobulk_records = joblib.load(pseudobulk_records_path)
        else:
            pseudobulk_records = pseudobulk_records_path
        if pseudobulk_ids is None:
            pseudobulk_ids = list(pseudobulk_records["pseudobulk_records"].keys())
        if (
            sample_n_pseudobulks is not None
            and len(pseudobulk_ids) > sample_n_pseudobulks
        ):
            pseudobulk_ids = sorted(
                pd.Series(pseudobulk_ids)
                .sample(sample_n_pseudobulks, random_state=0)
                .tolist()
            )
        kwargs["pseudobulk_ids"] = pseudobulk_ids

        fixed_p0_meta_cells = self.get_default_p0_meta_cells(subset_name=subset_name)
        kwargs["fixed_p0_meta_cells"] = fixed_p0_meta_cells

        if peak_bed_path is None:
            peak_bed_path = self.dataset.peak_bed_path
        kwargs["peak_bed_path"] = peak_bed_path

        if use_ref_bw:
            ref_bw_path = self.dataset.get_reference_bigwig_path(
                subset_name=subset_name
            )
            bigwig_paths = {"bigwig_paths": {"reference": ref_bw_path}}
        else:
            bigwig_paths = {}
        kwargs["bigwig_paths"] = bigwig_paths

        kwargs["nosignal"] = nosignal
        kwargs["subset_name"] = subset_name

        if self.model_group == "BorzoiSignalMultiDS":
            kwargs["embedding_only_mode"] = embedding_only_mode
            return self._create_multi_dataset_model_predictor(**kwargs)
        elif self.model_group == "BorzoiSignalSingleDS":
            return self._create_single_dataset_model_predictor(**kwargs)
        elif self.model_group == "Scooby":
            return self._create_single_dataset_model_predictor(**kwargs)
        elif self.model_group == "BorzoiMultihead":
            return self._create_single_dataset_model_predictor(**kwargs)
        else:
            raise ValueError(f"Unknown model group: {self.model_group}")

    def _create_single_dataset_model_predictor(
        self,
        _predictor_class: type,
        subset_name: str,
        pseudobulk_records_path: str,
        pseudobulk_ids: list[str],
        peak_bed_path: str,
        fixed_p0_meta_cells: list[str],
        nosignal: bool,
        bigwig_paths: dict,
    ) -> "GenericPredictor":
        db_path = self.dataset.get_meta_cell_parquet_path(
            "32bp", subset_name=subset_name
        )

        config = {
            "checkpoint_path": str(self.ckpt_path),
            "genome": self.dataset.genome,
            "train_config": self.model_config,
            "pseudobulk_records_path": pseudobulk_records_path,
            "dataset_path": db_path,
            "peak_path": peak_bed_path,
            "designs": pseudobulk_ids,
            "fixed_p0_meta_cells": fixed_p0_meta_cells,
            **bigwig_paths,
            # For no signal model inference, uncomment this:
            "nosignal": nosignal,
        }

        predictor = _predictor_class(config)
        return predictor

    def _create_multi_dataset_model_predictor(
        self,
        _predictor_class: type,
        subset_name: str | None,
        pseudobulk_records_path: str,
        pseudobulk_ids: list[str],
        peak_bed_path: str,
        nosignal: bool,
        bigwig_paths: dict,
        fixed_p0_meta_cells: list[str],
        embedding_only_mode: bool,
    ) -> "GenericPredictor":
        if subset_name is None:
            dataset_key = self.dataset.name
        else:
            dataset_key = f"{self.dataset.name}+{subset_name}"

        config = {
            "checkpoint_path": str(self.ckpt_path),
            "train_config": self.model_config,
            "peak_path": peak_bed_path,
            "designs": pseudobulk_ids,
            "pseudobulk_records_path": pseudobulk_records_path,
            "fixed_p0_meta_cells": fixed_p0_meta_cells,
            "nosignal": nosignal,
            "embedding_only_mode": embedding_only_mode,
            **bigwig_paths,
        }

        predictor = _predictor_class(dataset_key, config)
        return predictor

    def __repr__(self):
        _ds = self.dataset.name if self._current_dataset is not None else "Not set"
        doc = (
            f"BoleroModel(model_key={self.model_key})\n"
            f"  - Model group: {self.model_group}\n"
            f"  - Has gene count head: {self._has_gene_count_head}\n"
            f"  - TF score input: {self._tf_score_input}\n"
            f"  - Current Dataset: {_ds}\n"
        )
        return doc


class Models:
    def __init__(self):
        """
        Models is a singleton class that contains all the models.
        """
        self._cache = {}
        self.available_models = metadata.MODEL_ZOO.index.tolist()

    def __getattr__(self, name):
        if name not in self._cache:
            self._cache[name] = BoleroModel(name)
        return self._cache[name]

    def __getitem__(self, name):
        return getattr(self, name)


MODELS = Models()
