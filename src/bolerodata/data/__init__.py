import os
import pathlib

import joblib
import pandas as pd

from ._path_patch import PATH_PATCH_DICT

DEFAULT_STANDARD_DIR = "/large_storage/zhoulab/hanliu/wmb/standard"


try:
    STANDARD_DIR = pathlib.Path(os.environ["STANDARD_DIR"])
    assert (
        STANDARD_DIR.exists()
    ), f"Standard directory from $STANDARD_DIR does not exist at {STANDARD_DIR}."
except KeyError:
    # If the environment variable is not set, use the default directory
    # and check if it exists.
    STANDARD_DIR = pathlib.Path(DEFAULT_STANDARD_DIR)
    assert STANDARD_DIR.exists(), (
        "Standard directory does not exist. ",
        "Please set environment variable STANDARD_DIR to a valid path.",
    )


STANDARD_CELL_METADATA_DIR = STANDARD_DIR / "cell_metadata"
STANDARD_SAMPLE_METADATA_DIR = STANDARD_DIR / "sample_metadata"
STANDARD_CLUSTER_METADATA_DIR = STANDARD_DIR / "cluster_metadata"

# Curated dataset files
# columns: [sample, dataset, snap_path]
SAMPLE_SNAP_TABLE_PATH = STANDARD_DIR / "file_path_table/sample_snap_file_path.csv"
META_CELL_ADATA_PATH_DICT = STANDARD_DIR / "file_path_table/meta_cell.adata_path.joblib"
META_CELL_PARQUET_PATH_DICT = (
    STANDARD_DIR / "file_path_table/meta_cell.parquet_dataset_path.joblib"
)
META_CELL_PSEUDOBULK_RECORDS_PATH_DICT = (
    STANDARD_DIR / "file_path_table/meta_cell.pseudobulk_records_path.joblib"
)
META_CELL_COND_PSEUDOBULK_RECORDS_PATH_DICT = (
    STANDARD_DIR / "file_path_table/meta_cell.condition_pseudobulk_records_path.joblib"
)
META_CELL_REFERENCE_BIGWIG_PATH_DICT = (
    STANDARD_DIR / "file_path_table/meta_cell.reference_bigwig_path.joblib"
)
META_CELL_METADATA_PSEUDOBULK_RECORDS_PATH_DICT = (
    STANDARD_DIR / "file_path_table/meta_cell.metadata_pseudobulk_records_path.joblib"
)

# Trained Model files
BORZOI_SIGNAL_MODEL_PATH = (
    STANDARD_DIR / "file_path_table/250823_borzoi_signal_model_paths.joblib"
)


class Metadata:
    def __init__(self):
        self._cwd = pathlib.Path(__file__).parent
        self._cache = {}

        self.STANDARD_CELL_METADATA_DIR = STANDARD_CELL_METADATA_DIR
        self.STANDARD_SAMPLE_METADATA_DIR = STANDARD_SAMPLE_METADATA_DIR
        self.STANDARD_CLUSTER_METADATA_DIR = STANDARD_CLUSTER_METADATA_DIR
        self.SAMPLE_SNAP_TABLE_PATH = SAMPLE_SNAP_TABLE_PATH
        self.META_CELL_ADATA_PATH_DICT = META_CELL_ADATA_PATH_DICT
        self.META_CELL_PARQUET_PATH_DICT = META_CELL_PARQUET_PATH_DICT
        self.META_CELL_PSEUDOBULK_RECORDS_PATH_DICT = (
            META_CELL_PSEUDOBULK_RECORDS_PATH_DICT
        )
        self.META_CELL_COND_PSEUDOBULK_RECORDS_PATH_DICT = (
            META_CELL_COND_PSEUDOBULK_RECORDS_PATH_DICT
        )
        self.META_CELL_REFERENCE_BIGWIG_PATH_DICT = META_CELL_REFERENCE_BIGWIG_PATH_DICT
        self.META_CELL_METADATA_PSEUDOBULK_RECORDS_PATH_DICT = (
            META_CELL_METADATA_PSEUDOBULK_RECORDS_PATH_DICT
        )
        self.BORZOI_SIGNAL_MODEL_PATH = BORZOI_SIGNAL_MODEL_PATH

    @property
    def DATASET_METADATA(self):
        """Misc dataset metadata."""
        if "DATASET_METADATA" not in self._cache:
            _metadata_df = pd.read_table(
                self._cwd / "dataset_metadata.tsv", index_col=0
            )
            _emb_and_meta_cell_df = pd.read_table(
                self._cwd / "embedding_and_meta_cell.tsv", index_col=0
            )
            _metadata_df = _metadata_df.join(
                _emb_and_meta_cell_df.loc[
                    :, ~_emb_and_meta_cell_df.columns.isin(_metadata_df.columns)
                ],
                how="left",
                on="dataset",
            )
            self._cache["DATASET_METADATA"] = _metadata_df
        return self._cache["DATASET_METADATA"]

    def get_sample_snap_files(self, dataset_name):
        """Get the sample snap files table."""
        if "sample_snap_table" not in self._cache:
            self._cache["sample_snap_table"] = pd.read_csv(self.SAMPLE_SNAP_TABLE_PATH)
        snap_table = self._cache["sample_snap_table"]
        use_snap_table = snap_table[snap_table["dataset"] == dataset_name].copy()
        return use_snap_table

    def get_misc_data_path(self, key, path_attr):
        """
        Get the misc data path from the dictionary.
        """
        if isinstance(key, tuple):
            key = ",".join(map(str, key))

        try:
            path_dict = self._cache[path_attr]
        except KeyError:
            d = joblib.load(getattr(self, path_attr))

            # any spatial patch for non-standard datasets
            patch = PATH_PATCH_DICT.get(path_attr, {})
            d.update(patch)

            self._cache[path_attr] = {",".join(map(str, k)): v for k, v in d.items()}
            path_dict = self._cache[path_attr]

        try:
            return path_dict[key]
        except KeyError:
            startswith = [k for k in path_dict.keys() if k.startswith(key)]
            raise KeyError(
                f"Key {key} not found in the dictionary."
                + (f" Possible keys are {startswith}" if len(startswith) > 0 else "")
            ) from None

    def get_metacell_adata_path(self, key):
        """
        Get the metacell adata path from the dictionary.
        """
        return self.get_misc_data_path(key, "META_CELL_ADATA_PATH_DICT")

    def get_metacell_parquet_path(self, key):
        """
        Get the metacell parquet path from the dictionary.
        """
        return self.get_misc_data_path(key, "META_CELL_PARQUET_PATH_DICT")

    def get_metacell_pseudobulk_records_path(self, key):
        """
        Get the metacell pseudobulk records path from the dictionary.
        """
        return self.get_misc_data_path(key, "META_CELL_PSEUDOBULK_RECORDS_PATH_DICT")

    def get_metacell_cond_pseudobulk_records_path(self, key):
        """
        Get the metacell condition pseudobulk records path from the dictionary.
        """
        return self.get_misc_data_path(
            key, "META_CELL_COND_PSEUDOBULK_RECORDS_PATH_DICT"
        )

    def get_reference_bigwig_path(self, key):
        """
        Get the reference bigwig path from the dictionary.
        """
        return self.get_misc_data_path(key, "META_CELL_REFERENCE_BIGWIG_PATH_DICT")

    def get_metacell_metadata_pseudobulk_records_path(self, key):
        """
        Get the metacell metadata pseudobulk records path from the dictionary.
        """
        return self.get_misc_data_path(
            key, "META_CELL_METADATA_PSEUDOBULK_RECORDS_PATH_DICT"
        )

    def get_borzoi_signal_model_path(self, key):
        """
        Get the borzoi signal model path from the dictionary.
        """
        return self.get_misc_data_path(key, "BORZOI_SIGNAL_MODEL_PATH")


metadata = Metadata()

if __name__ == "__main__":
    print(metadata.DATASET_METADATA)
