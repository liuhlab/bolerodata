import anndata
import pandas as pd

from ._sync import localize
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

        # meta cell
        _gs = self.metadata["meta_cell_groups"]
        self.meta_cell_groups = _gs.split(",") if isinstance(_gs, str) else []

    @property
    def Genome(self):
        """
        Return a bolero.Genome object for the dataset genome.
        """
        genome_obj = self._cache.get("_genome_obj", None)
        if genome_obj is None:
            from bolero import Genome

            self._cache["_genome_obj"] = Genome(self.genome)
        return self._cache["_genome_obj"]

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
                localize(
                    metadata.STANDARD_CELL_METADATA_DIR
                    / f"{self.name}.cell_metadata.feather"
                )
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
                localize(
                    metadata.STANDARD_SAMPLE_METADATA_DIR
                    / f"{self.name}.sample_metadata.feather"
                )
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
                localize(
                    metadata.STANDARD_CLUSTER_METADATA_DIR
                    / f"{self.name}.cluster_metadata.feather"
                )
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
                localize(self.metadata["cell_metadata"])
            )
        return self._cache["cell_metadata"]

    @property
    def cell_embedding(self):
        """
        Cell embedding table organized from original study, contain misc information.
        """
        if "within_dataset_cell_embedding" not in self._cache:
            self._cache["within_dataset_cell_embedding"] = pd.read_feather(
                localize(self.metadata["within_dataset_cell_embedding"])
            )
        return self._cache["within_dataset_cell_embedding"]

    @property
    def peak_motif_scan_path(self):
        """
        Get the path to the peak motif scan table.
        """
        return metadata.get_peak_motif_scan_path(self.name)

    @property
    def peak_motif_scan_table(self):
        """
        Peak motif scan binary table.
        """
        return pd.read_feather(self.peak_motif_scan_path)

    @property
    def chromvar_adata(self):
        """Chromvar adata object."""
        adata = self._cache.get("chromvar_adata", None)
        if adata is None:
            chromvar_path = self.metadata.get("chromvar_within_dataset", None)
            if chromvar_path is None:
                raise ValueError(
                    f"Dataset {self.name} does not have chromvar data. "
                    "Please check the dataset metadata."
                )

            genome = self.Genome
            motif_db = genome.get_motif_db()
            motif_names = pd.Series(
                {mid: motif.name for mid, motif in motif_db.motif_id_dict.items()}
            )

            adata = anndata.read_h5ad(localize(chromvar_path))
            adata.var["motif_name"] = motif_names

            self._cache["chromvar_adata"] = adata
        return adata

    @property
    def cross_dataset_chromvar_adata(self):
        """Cross dataset chromvar adata object."""
        adata = self._cache.get("cross_dataset_chromvar_adata", None)
        if adata is None:
            chromvar_path = self.metadata.get("chromvar_cross_dataset", None)
            if chromvar_path is None:
                raise ValueError(
                    f"Dataset {self.name} does not have cross dataset chromvar data. "
                    "Please check the dataset metadata."
                )
            adata = anndata.read_h5ad(localize(chromvar_path))
            self._cache["cross_dataset_chromvar_adata"] = adata
        return adata

    def get_meta_cell_adata_path(self, kind: str = "metadata", subset_name: str = None):
        """
        Get the path to the meta cell adata file.
        """
        assert kind in [
            "metadata",
            "gene",
            "peak",
            "cell",
        ], f"kind must be one of ['metadata', 'gene', 'peak', 'cell'], got {kind}"
        key = [kind, self.name]
        if subset_name:
            key.append(subset_name)
        key = ",".join(key)
        return metadata.get_metacell_adata_path(key)

    def get_meta_cell_adata(
        self, kind: str = "metadata", subset_name: str = None, backed=None
    ):
        """
        Get the meta cell adata object.

        Parameters
        ----------
        kind: str
            The kind of meta cell adata file to get:
            - "metadata": metadata and embedding only
            - "gene": gene expression data
            - "peak": peak count data
            - "cell": cell level metadata and embedding only
        subset_name: str
            The name of the subset to get. If None, return the full dataset.
        backed: str or None
            If not None, load the adata in backed mode. This is useful for large datasets.

        Returns
        -------
        anndata.AnnData
            The meta cell adata object.
        """
        path = self.get_meta_cell_adata_path(kind, subset_name)
        adata = anndata.read_h5ad(path, backed=backed)
        return adata

    def get_meta_cell_parquet_path(self, kind: str, subset_name: str = None):
        """
        Get the path to the meta cell parquet dataset dir.
        """
        assert kind in [
            "1bp",
            "32bp",
        ], f"kind must be one of ['1bp', '32bp'], got {kind}"
        key = [kind, self.name]
        if subset_name:
            key.append(subset_name)
        key = ",".join(key)
        return metadata.get_metacell_parquet_path(key)

    def get_meta_cell_pseudobulk_records_path(
        self, target_coverage, subset_name: str = None, pseudobulk_type="condition"
    ):
        """
        Get the path to the meta cell pseudobulk records file.

        Parameters
        ----------
        target_coverage: int
            The target_coverage of meta cell pseudobulk records file to get:
            - 5000000: pseudobulk records with 5000000 target coverage
            - 25000000: pseudobulk records with 25000000 target coverage
        subset_name: str
            The name of the subset to get. If None, return the full dataset.
        pseudobulk_type: str or None
            The type of pseudobulk to get. Must be one of ['embedding', 'condition', 'metadata'].
            Embedding: pseudobulk grouped based on embedding, containing embedding information only.
            Condition: pseudobulk grouped based on embedding, containing embedding and condition information.
            Metadata: pseudobulk grouped based on metadata, containing embedding and condition information.

        Returns
        -------
        str
            The path to the meta cell pseudobulk records file.
        """
        target_coverage = str(int(target_coverage))
        assert target_coverage in [
            "5000000",
            "25000000",
        ], f"target_coverage must be one of [5000000, 25000000], got {target_coverage}"
        key = [target_coverage, self.name]
        if subset_name:
            key.append(subset_name)
        key = ",".join(key)
        if pseudobulk_type == "condition":
            return metadata.get_metacell_cond_pseudobulk_records_path(key)
        elif pseudobulk_type == "metadata":
            return metadata.get_metacell_metadata_pseudobulk_records_path(key)
        elif pseudobulk_type == "embedding":
            return metadata.get_metacell_pseudobulk_records_path(key)
        else:
            raise ValueError(
                f"pseudobulk_type must be one of ['embedding', 'condition', 'metadata'], got {pseudobulk_type}"
            )

    def get_reference_bigwig_path(self, subset_name: str = None):
        """
        Get the path to the reference bigwig file.
        """
        key = [self.name]
        if subset_name:
            key.append(subset_name)
        key = ",".join(key)
        return metadata.get_reference_bigwig_path(key)

    def get_meta_cell_cond_pseudobulk_records_path(
        self, target_coverage, subset_name: str = None
    ):
        """
        Get the path to the meta cell condition pseudobulk records file.
        """
        return self.get_meta_cell_pseudobulk_records_path(
            target_coverage, subset_name, pseudobulk_type="condition"
        )

    def get_meta_cell_metadata_pseudobulk_records_path(
        self, target_coverage, subset_name: str = None
    ):
        """
        Get the path to the meta cell metadata pseudobulk records file.
        """
        return self.get_meta_cell_pseudobulk_records_path(
            target_coverage, subset_name, pseudobulk_type="metadata"
        )

    def get_pseudobulk_gene_data_path(self, subset_name: str = None):
        """
        Get the path to the pseudobulk gene data file.
        """
        key = ["gene_count", self.name]
        if subset_name:
            key.append(subset_name)
        key = ",".join(key)
        return metadata.get_pseudobulk_gene_and_hvg_path(key)

    def get_pseudobulk_hvg_list_path(self, subset_name: str = None):
        """
        Get the path to the pseudobulk hvg list file.
        """
        key = ["hvg_list", self.name]
        if subset_name:
            key.append(subset_name)
        key = ",".join(key)
        return metadata.get_pseudobulk_gene_and_hvg_path(key)

    def get_dataset_reference_gene_log1pcpm_path(self, subset_name: str = None):
        """
        Get the path to the dataset reference gene log1pcpm file.
        """
        key = ["ref_gene_log1pcpm", self.name]
        if subset_name:
            key.append(subset_name)
        key = ",".join(key)
        return metadata.get_pseudobulk_gene_and_hvg_path(key)


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

    def get_datasets(self, species=None, genome=None) -> list[Dataset]:
        """Get the datasets with the given species and genome."""
        datasets = []
        for dataset in self:
            if species is not None and dataset.species.lower() != species.lower():
                continue
            if genome is not None and dataset.genome.lower() != genome.lower():
                continue
            datasets.append(dataset)
        return datasets

    def get_datasets_and_subset_names(
        self, species=None, genome=None
    ) -> list[tuple[str, str]]:
        """Get the datasets and subset names with the given species and genome."""
        datasets = self.get_datasets(species=species, genome=genome)
        recs = []
        for dataset in datasets:
            subset_list = dataset.meta_cell_groups
            if len(subset_list) == 0:
                subset_list = [None]
            for subset in subset_list:
                recs.append((dataset.name, subset))
        return recs


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
