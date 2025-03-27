# %%
import pandas as pd
from bolerodata import DATASETS

# %%
def test_datasets():
    for dataset in DATASETS:
        name = dataset.name
        cell_metadata = dataset.cell_metadata
        assert not cell_metadata.index.duplicated().all()

        sample_metadata = dataset.sample_metadata
        assert (
            not sample_metadata.index.duplicated().all()
        ), f"Sample metadata for {name} has duplicate indices"
        assert (
            pd.Index(cell_metadata["sample"].dropna().unique()).isin(sample_metadata.index).all()
        ), f"Sample metadata for {name} does not match cell metadata"

        cluster_metadata = dataset.cluster_metadata
        assert (
            not cluster_metadata.index.duplicated().all()
        ), f"Cluster metadata for {name} has duplicate indices"
        assert (
            pd.Index(cell_metadata["cluster"].dropna().unique()).isin(cluster_metadata.index).all()
        ), f"Cluster metadata for {name} does not match cell metadata"

        DATASETS._cache.clear()

