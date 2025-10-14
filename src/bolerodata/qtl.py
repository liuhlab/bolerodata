import pandas as pd

from .data import metadata


class QTLCollection:
    def __init__(self, source: str):
        self.source: str = source
        self._path_table = metadata.QTL_COLLECTION.query(f"Source == '{source}'")

    @property
    def qtl_path_table(self) -> pd.DataFrame:
        """QTL path table."""
        return self._path_table

    @property
    def tissue_list(self) -> list[str]:
        """Tissue list."""
        return self._path_table["TissueORCellType"].unique().tolist()

    def get_qtl_path(self, key: str) -> str:
        """Get the QTL path."""
        return self._path_table.loc[key, "FilePath"]

    def get_qtl_table(self, *args, **kwargs) -> pd.DataFrame:
        """Get the QTL data."""
        return pd.read_feather(self.get_qtl_path(*args, **kwargs))


class GTExQTLCollection(QTLCollection):
    def get_qtl_path(
        self, tissue: str, gene_sel: str = "protein_coding"
    ) -> pd.DataFrame:
        """Get the QTL path."""
        key = f"GTEx.{tissue}.{gene_sel}"
        return super().get_qtl_path(key)


class Zeng2024QTLCollection(QTLCollection):
    def get_qtl_path(self, cell_type: str, all_pair: bool = False) -> pd.DataFrame:
        """Get the QTL path."""
        key = f"Zeng2024Science.{cell_type}"
        if all_pair:
            key += ".all_pair"
        return super().get_qtl_path(key)


GTEx = GTExQTLCollection("GTEx")
Zeng2024 = Zeng2024QTLCollection("Zeng2024Science")
