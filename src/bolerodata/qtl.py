import pandas as pd

from .data import metadata


class GTExCollection:
    def __init__(self):
        self._path_table = metadata.QTL_COLLECTION.query("Source == 'GTEx'")

    @property
    def qtl_path_table(self) -> pd.DataFrame:
        """QTL path table."""
        return self._path_table

    @property
    def tissue_list(self) -> list[str]:
        """Tissue list."""
        return self._path_table["BioType"].unique().tolist()

    def get_qtl_path(self, tissue: str, gene_sel: str = "protein_coding") -> str:
        """Get the QTL path."""
        key = f"GTEx.{tissue}.{gene_sel}"
        return self._path_table.loc[key, "FilePath"]

    def get_qtl_table(
        self, tissue: str, gene_sel: str = "protein_coding"
    ) -> pd.DataFrame:
        """Get the QTL data."""
        return pd.read_feather(self.get_qtl_path(tissue, gene_sel))


GTEx = GTExCollection()
