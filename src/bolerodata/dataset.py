import pandas as pd
from .data import metadata


class Dataset:
    def __init__(self, name: str):
        self.name = name
        self._cache = {}

        self.metadata = metadata.DATASET_METADATA.loc[self.name].to_dict()

        # dataset level metadata
        self.study_group = self.metadata["study_group"]
        self.species = self.metadata["species"]
        self.technology = self.metadata["technology"]
        self.genome = self.metadata["genome"]

        self.peak_bed_path = self.metadata["peaks"]
        self.peak_adata_path = self.metadata["peak_adata"]
        self.gene_adata_path = self.metadata["gene_adata"]

    @property
    def cell_metadata(self) -> pd.DataFrame:
        """
        Standard minimum cell metadata table.

        Schema:
        - Index: cell barcode in "{sample}+{cell barcode}" format.
            For 10X experiments, barcode is in "GEX_BARCODE-1" format
        - Columns:
            1. sample  # link to sample metadata
            2. cluster  # link to cluster metadata, deepest level of clustering
            3. n_fragments  # From ATAC
            4. tsse  # From ATAC
            5. n_umi  # From RNA, if any
            6. n_genes  # From RNA, if any
        """
        if "standard_cell_metadata" not in self._cache:
            self._cache["standard_cell_metadata"] = pd.read_feather(
                metadata.STANDARD_CELL_METADATA_DIR
                / f"{self.name}.cell_metadata.feather"
            )
        return self._cache["standard_cell_metadata"]
    
    def make_cell_metadata(self, sample_meta=True, cluster_meta=True):
        """
        Make cell metadata table with sample and cluster metadata.
        """
        cell_meta = self.cell_metadata.copy()
        if sample_meta:
            cell_meta = cell_meta.join(self.sample_metadata, on="sample", how="left")
        if cluster_meta:
            cell_meta = cell_meta.join(self.cluster_metadata, on="cluster", how="left")
        return cell_meta

    @property
    def n_cells(self) -> int:
        """
        Number of cells in the dataset.
        """
        return self.cell_metadata.shape[0]

    @property
    def n_samples(self) -> int:
        """
        Number of samples in the dataset.
        """
        return self.sample_metadata.shape[0]

    @property
    def sample_metadata(self) -> pd.DataFrame:
        """
        Standard sample metadata table.

        Schema:
        - Index: sample_id
            - This id match the sample column in all cell_metadata.
            - This correspond to technical sample (e.g., one 10X experiments)
              the same barcode within sample should come from a single cell.
            - For biological sample, record in donor column
        - Columns:
            1. age  # time, standard age group
            2. age_int  # Estimated postconceptional age in days, standard age group int
            3. tissue  # space, experiment tissue source information
            4. DissectionRegion  # space, anatomical descriptive vocabulary of the tissue
            5. sex  # exp design
            6. donor  # exp design
        - Additional Columns:
            - **age_details
            - **tissue_details
            - **exp_details
        """
        if "sample_metadata" not in self._cache:
            self._cache["sample_metadata"] = pd.read_feather(
                metadata.STANDARD_SAMPLE_METADATA_DIR
                / f"{self.name}.sample_metadata.feather"
            )
        return self._cache["sample_metadata"]

    @property
    def cluster_metadata(self) -> pd.DataFrame:
        """
        Standard cluster metadata table.

        Schema:
        - Index: cluster_id
            - This id match the cluster column in all cell_metadata.
            - This correspond to the deepest level of clustering
        - Columns:
            - Columns:
            1. subclass: Most meaningful detail cell annotation level
            2. class: Most meaningful major cell annotation level
        - Additional Columns:
            - **Other cluster levels
        """
        if "cluster_metadata" not in self._cache:
            self._cache["cluster_metadata"] = pd.read_feather(
                metadata.STANDARD_CLUSTER_METADATA_DIR
                / f"{self.name}.cluster_metadata.feather"
            )
        return self._cache["cluster_metadata"]

    @property
    def snap_file_table(self) -> pd.DataFrame:
        """
        Standard snap file table.

        Columns:
        - sample: Sample ID
            - EXCEPT Li2023Science, which is using cell type to group snap
        - dataset: Dataset Name
        - snap_path: Path to the snap file
            - Contains nan, some snap files might be missing.
        """
        return metadata.get_sample_snap_files(self.name)

    @property
    def original_cell_metadata(self):
        """
        Cell metadata table organized from original study, contain misc information.
        """
        if "cell_metadata" not in self._cache:
            self._cache["cell_metadata"] = pd.read_feather(
                self.metadata["cell_metadata"]
            )
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

    def __contains__(self, name):
        name = self._dataset_alias.get(name, name)
        return name in self.available_datasets

    def __iter__(self):
        for name in self.available_datasets:
            dataset = Dataset(name)
            yield dataset


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
