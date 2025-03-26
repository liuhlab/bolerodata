import pandas as pd
from .data import metadata


class Dataset:
    def __init__(self, name):
        self.name = name
        self._cache = {}

        self.metadata = metadata.DATASET_METADATA.loc[self.name].to_dict()
        self.cell_metadata_path = self.metadata["cell_metadata"]
        self.sample_metadata_path = self.metadata["sample_metadata"]
        self.peak_bed_path = self.metadata["peaks"]
        self.peak_adata_path = self.metadata["peak_adata"]
        self.gene_adata_path = self.metadata["gene_adata"]

    @property
    def cell_metadata(self):
        if "cell_metadata" not in self._cache:
            self._cache["cell_metadata"] = pd.read_feather(self.cell_metadata_path)
        return self._cache["cell_metadata"]


class Datasets:
    def __init__(self):
        self._cache = {}
        self.available_datasets = metadata.DATASET_METADATA.index.tolist()
        self._dataset_alias = {
            # add dataset alias here
        }

    def __getattr__(self, name):
        name = self._dataset_alias.get(name, name)
        if name not in self.available_datasets:
            raise AttributeError(
                f"Dataset {name} is not available. Possible datasets are {self.available_datasets}"
            )
        if name not in self._cache:
            self._cache[name] = Dataset(name)
        return self._cache[name]

    def __getitem__(self, name):
        return getattr(self, name)


DATASETS = Datasets()


"""


Snap file table
Dataset
Experiment ID
Snap path
- Snap file may contain additional unfiltered cells not occuring in cell metadata
- Snap file obs_names is only cell barcode (barcode seq or "BARCODE_SEQ-1" for 10X). Prepend "{Experiment ID}+{cell barcode}" to match with cell metadata

RNA adata table (per dataset)
- obs_names: "{Experiment ID}+{cell barcode}"
- var_names: Ensemble Gene ID base
- X: raw counts

Peak adata table (within dataset peak set, per dataset)
- obs_names: "{Experiment ID}+{cell barcode}"
- var_names: "{chrom}:{start}-{end}"
- X: raw counts

Peak table
- Per dataset per peak
- Standard to 500 bp

Pseudobulk joblib
- For each dataset
- Pseudobulk ID - by - whole genome chunked csc

"""
