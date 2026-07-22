import os
import pathlib

import joblib
import pandas as pd

from bolerodata._sync import localize

from ._path_patch import PATH_PATCH_DICT

DEFAULT_STANDARD_DIR = "/large_storage/zhoulab/hanliu/wmb/standard"


# STANDARD_DIR is the lab data-lake root. On lab machines it exists and files are
# read straight from it. Off-lab it does not exist; the paths built from it below
# become *logical keys* that ``bolerodata._sync.localize()`` maps to a HuggingFace
# download into the local cache. We therefore do NOT assert it exists at import
# time — that assert used to make ``import bolerodata`` fail on any machine
# without the data lake (e.g. CI, or any external user).
STANDARD_DIR = pathlib.Path(os.environ.get("STANDARD_DIR", DEFAULT_STANDARD_DIR))


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

PSEUDOBULK_GENE_AND_HVG_PATH_DICT = (
    STANDARD_DIR / "file_path_table/pseudobulk.gene_and_hvg_data.joblib"
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
        self.PSEUDOBULK_GENE_AND_HVG_PATH_DICT = PSEUDOBULK_GENE_AND_HVG_PATH_DICT

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

    @property
    def MODEL_ZOO(self):
        """Model zoo."""
        if "MODEL_ZOO" not in self._cache:
            self._cache["MODEL_ZOO"] = pd.read_table(
                self._cwd / "model_zoo.tsv", index_col=0
            )
            assert (
                self._cache["MODEL_ZOO"].index.duplicated().sum() == 0
            ), "Model zoo table has duplicated index."
        return self._cache["MODEL_ZOO"]

    @property
    def QTL_COLLECTION(self):
        """QTL collection."""
        if "QTL_COLLECTION" not in self._cache:
            self._cache["QTL_COLLECTION"] = pd.read_table(
                self._cwd / "qtl_collection.tsv", index_col=0
            )
        return self._cache["QTL_COLLECTION"]

    @property
    def JASPAR_MOTIF_CLUSTER_INFO(self):
        """JASPAR motif cluster info."""
        if "JASPAR_MOTIF_CLUSTER_INFO" not in self._cache:
            self._cache["JASPAR_MOTIF_CLUSTER_INFO"] = pd.read_table(
                self._cwd / "JASPAR_motif_cluster_info.tsv", index_col=0
            )
        return self._cache["JASPAR_MOTIF_CLUSTER_INFO"]

    @property
    def JASPAR_MOTIF_CLUSTER(self):
        """JASPAR motif cluster."""
        if "JASPAR_MOTIF_CLUSTER" not in self._cache:
            self._cache["JASPAR_MOTIF_CLUSTER"] = pd.read_table(
                self._cwd / "JASPAR_motif_cluster.tsv", index_col=0
            )
        return self._cache["JASPAR_MOTIF_CLUSTER"]

    @property
    def DA_COLLECTION(self):
        """Differential analysis collection."""
        if "DA_COLLECTION" not in self._cache:
            self._cache["DA_COLLECTION"] = pd.read_csv(
                self._cwd / "DA.TotalRecords.csv.gz", index_col=("DAKey", "DAType")
            )
        return self._cache["DA_COLLECTION"]

    def get_sample_snap_files(self, dataset_name):
        """Get the sample snap files table."""
        if "sample_snap_table" not in self._cache:
            self._cache["sample_snap_table"] = pd.read_csv(
                localize(self.SAMPLE_SNAP_TABLE_PATH)
            )
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
            # The joblib path-dict itself lives under STANDARD_DIR; localize it so
            # external users fetch it from HuggingFace before looking keys up.
            d = joblib.load(localize(getattr(self, path_attr)))

            # any spatial patch for non-standard datasets
            patch = PATH_PATCH_DICT.get(path_attr, {})
            d.update(patch)

            self._cache[path_attr] = {",".join(map(str, k)): v for k, v in d.items()}
            path_dict = self._cache[path_attr]

        try:
            # The dict value is an absolute lab path (or logical key); localize it
            # to a concrete local file, downloading from HuggingFace if needed.
            return localize(path_dict[key])
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

    def get_pseudobulk_gene_and_hvg_path(self, key):
        """
        Get the pseudobulk gene and hvg path from the dictionary.
        """
        return self.get_misc_data_path(key, "PSEUDOBULK_GENE_AND_HVG_PATH_DICT")

    def get_peak_motif_scan_path(self, dataset_name):
        """
        Get the peak motif scan path.
        """
        return localize(
            STANDARD_DIR / f"motif_scan/{dataset_name}.JASPAR_motif_hit.feather"
        )


metadata = Metadata()

if __name__ == "__main__":
    print(metadata.DATASET_METADATA)
