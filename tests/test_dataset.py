# %%
import pathlib

import pandas as pd
import pytest

from bolerodata import DATASETS
from bolerodata.data import STANDARD_DIR

# These tests read curated metadata from the lab filesystem. Off-lab / on CI the
# files are not present (and withheld datasets are not downloadable by design), so
# skip rather than fail when the lab data lake is not mounted.
pytestmark = pytest.mark.skipif(
    not (pathlib.Path(STANDARD_DIR) / "cell_metadata").is_dir(),
    reason="lab data (STANDARD_DIR/cell_metadata) not available; skipped off-lab/CI",
)


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
            pd.Index(cell_metadata["sample"].dropna().unique())
            .isin(sample_metadata.index)
            .all()
        ), f"Sample metadata for {name} does not match cell metadata"

        cluster_metadata = dataset.cluster_metadata
        assert (
            not cluster_metadata.index.duplicated().all()
        ), f"Cluster metadata for {name} has duplicate indices"
        assert (
            pd.Index(cell_metadata["cluster"].dropna().unique())
            .isin(cluster_metadata.index)
            .all()
        ), f"Cluster metadata for {name} does not match cell metadata"

        DATASETS._cache.clear()
