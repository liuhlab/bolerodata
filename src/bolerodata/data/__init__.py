import pandas as pd
import pathlib

STANDARD_CELL_METADATA_DIR = pathlib.Path("/large_storage/zhoulab/hanliu/wmb/standard/cell_metadata")
STANDARD_SAMPLE_METADATA_DIR = pathlib.Path("/large_storage/zhoulab/hanliu/wmb/standard/sample_metadata")
STANDARD_CLUSTER_METADATA_DIR = pathlib.Path("/large_storage/zhoulab/hanliu/wmb/standard/cluster_metadata")

class Metadata:
    def __init__(self):
        self._cwd = pathlib.Path(__file__).parent
        self._cache = {}

    @property
    def DATASET_METADATA(self):
        if "DATASET_METADATA" not in self._cache:
            self._cache["DATASET_METADATA"] = pd.read_table(
                self._cwd / "dataset_metadata.tsv", index_col=0
            )
        return self._cache["DATASET_METADATA"]


metadata = Metadata()

if __name__ == "__main__":
    print(metadata.DATASET_METADATA)
