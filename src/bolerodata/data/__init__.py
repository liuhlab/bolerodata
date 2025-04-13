import pandas as pd
import pathlib
import os

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
        f"Standard directory does not exist. ",
        f"Please set environment variable STANDARD_DIR to a valid path.",
    )


STANDARD_CELL_METADATA_DIR = STANDARD_DIR / "cell_metadata"
STANDARD_SAMPLE_METADATA_DIR = STANDARD_DIR / "sample_metadata"
STANDARD_CLUSTER_METADATA_DIR = STANDARD_DIR / "cluster_metadata"

# columns: [sample, dataset, snap_path]
SAMPLE_SNAP_TABLE_PATH = STANDARD_DIR / "file_path_table/sample_snap_file_path.csv"


class Metadata:
    def __init__(self):
        self._cwd = pathlib.Path(__file__).parent
        self._cache = {}

        self.STANDARD_CELL_METADATA_DIR = STANDARD_CELL_METADATA_DIR
        self.STANDARD_SAMPLE_METADATA_DIR = STANDARD_SAMPLE_METADATA_DIR
        self.STANDARD_CLUSTER_METADATA_DIR = STANDARD_CLUSTER_METADATA_DIR
        self.SAMPLE_SNAP_TABLE_PATH = SAMPLE_SNAP_TABLE_PATH

    @property
    def DATASET_METADATA(self):
        if "DATASET_METADATA" not in self._cache:
            _metadata_df = pd.read_table(
                self._cwd / "dataset_metadata.tsv", index_col=0
            )
            _emb_and_meta_cell_df = pd.read_table(
                self._cwd / "embedding_and_meta_cell.tsv", index_col=0
            )
            _metadata_df = _metadata_df.join(
                _emb_and_meta_cell_df.loc[:, ~_emb_and_meta_cell_df.columns.isin(_metadata_df.columns)],
                how="left",
                on="dataset",
            )
            self._cache["DATASET_METADATA"] = _metadata_df
        return self._cache["DATASET_METADATA"]

    def get_sample_snap_files(self, dataset_name):
        if "sample_snap_table" not in self._cache:
            self._cache["sample_snap_table"] = pd.read_csv(self.SAMPLE_SNAP_TABLE_PATH)
        snap_table = self._cache["sample_snap_table"]
        use_snap_table = snap_table[snap_table["dataset"] == dataset_name].copy()
        return use_snap_table


metadata = Metadata()

if __name__ == "__main__":
    print(metadata.DATASET_METADATA)
