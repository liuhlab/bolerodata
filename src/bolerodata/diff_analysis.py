import os
import pathlib

import pandas as pd

from bolerodata import DATASETS
from bolerodata._sync import localize
from bolerodata.data import metadata


def _require_diff():
    """
    Import the optional heavy dependencies for differential analysis.

    Differential analysis needs pyBigWig / pyranges / pysam (the ``diff`` extra)
    and the companion ``bolero`` package. They are imported lazily so that merely
    importing :mod:`bolerodata` (or this module) does not pull them in; the check
    runs when a :class:`DiffRecords` is constructed.
    """
    try:
        import numpy  # noqa: F401
        import pyBigWig  # noqa: F401
        import pyranges  # noqa: F401
        import pysam  # noqa: F401
        from bolero.pl.igv import Browser  # noqa: F401
        from bolero.utils import understand_regions  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "bolerodata.diff_analysis requires the optional 'diff' dependencies "
            "(pyBigWig, pyranges, pysam) and the companion `bolero` package. "
            "Install with: pip install 'bolerodata[diff]' (and install bolero)."
        ) from e


def _parse_region(region: str | tuple[str, int, int]):
    if isinstance(region, str):
        # convert to chromosome, start, end
        chrom, coords = region.replace(",", "").split(":")
        start, end = map(int, coords.split("-"))
    else:
        chrom, start, end = region
    return chrom, start, end


class DiffRecords:
    """
    A single pair-wise differential analysis record.
    """

    def __init__(
        self, gene_rec=None, peak_rec=None, lfc_cutoff=0.2, igv_base_dir="igv"
    ):
        """
        Initialize the DiffRecords object with gene and/or peak records.
        """
        _require_diff()
        assert (peak_rec is not None) or (gene_rec is not None)
        _rec = gene_rec if gene_rec is not None else peak_rec
        self.da_key = _rec.name[0]
        self.name = self.da_key
        self.dataset_name = _rec["DatasetName"]
        self.dataset = DATASETS[self.dataset_name]
        self.genome = self.dataset.genome
        self.da_group = _rec["DAGroup"]
        self.variable = _rec["Variable"]
        self.group1 = _rec["Group1"]
        self.group2 = _rec["Group2"]
        self.lfc_cutoff = lfc_cutoff

        self.group1_bw_path = pathlib.Path(_rec["Group1BigWig"])
        self.group2_bw_path = pathlib.Path(_rec["Group2BigWig"])

        self.gene_diff_path = None if gene_rec is None else gene_rec["DiffPath"]
        self.peak_diff_path = None if peak_rec is None else peak_rec["DiffPath"]
        self.igv_base_dir = pathlib.Path(igv_base_dir)
        self.igv_base_dir.mkdir(parents=True, exist_ok=True)

        self._caches = {}

    def _filter_by_lfc_cutoff(self, diff_table):
        """
        Filter differential analysis table by abs log2 fold change cutoff.
        """
        return diff_table[diff_table["log2FoldChange"].abs() > self.lfc_cutoff].copy()

    @property
    def gene_diff_table(self):
        """
        Gene differential analysis table.
        """
        if self.gene_diff_path is None:
            return None
        if "gene_diff_table" not in self._caches:
            table = pd.read_feather(localize(self.gene_diff_path))
            table.index = table.pop("gene").astype(str)
            self._caches["gene_diff_table"] = self._filter_by_lfc_cutoff(table)
        return self._caches["gene_diff_table"]

    @property
    def peak_diff_table(self):
        """
        Peak differential analysis table.
        """
        if self.peak_diff_path is None:
            return None
        if "peak_diff_table" not in self._caches:
            table = pd.read_feather(localize(self.peak_diff_path))
            table.index = table.pop("gene").astype(str)
            self._caches["peak_diff_table"] = self._filter_by_lfc_cutoff(table)
        return self._caches["peak_diff_table"]

    @property
    def group1_bw_handle(self):
        """
        BigWig file handle for group 1.
        """
        if "bw1_handle" not in self._caches:
            import pyBigWig

            self._caches["bw1_handle"] = pyBigWig.open(str(localize(self.group1_bw_path)))
        return self._caches["bw1_handle"]

    @property
    def group2_bw_handle(self):
        """
        BigWig file handle for group 2.
        """
        if "bw2_handle" not in self._caches:
            import pyBigWig

            self._caches["bw2_handle"] = pyBigWig.open(str(localize(self.group2_bw_path)))
        return self._caches["bw2_handle"]

    def get_bw_values(self, region: str | tuple[str, int, int]):
        """
        Get Group1 and Group2 bigwig values for a given region.
        """
        chrom, start, end = _parse_region(region)
        values = (
            pd.DataFrame(
                {
                    self.group1: self.group1_bw_handle.values(
                        chrom, start, end, numpy=True
                    ),
                    self.group2: self.group2_bw_handle.values(
                        chrom, start, end, numpy=True
                    ),
                }
            )
            .fillna(0)
            .astype("float32")
        )
        return values

    def get_bw_stats(self, region: str | tuple[str, int, int], stat_type: str = "mean"):
        """
        Get Group1 and Group2 bigwig stats for a given region.
        """
        chrom, start, end = _parse_region(region)
        values = pd.Series(
            {
                self.group1: self.group1_bw_handle.stats(
                    chrom, start, end, type=stat_type
                )[0],
                self.group2: self.group2_bw_handle.stats(
                    chrom, start, end, type=stat_type
                )[0],
            }
        )
        values.index.name = "group"
        values.name = f"{chrom}:{start}-{end}"
        return values

    def scan_peaks(self, regions):
        """
        Scan peaks on Group1 and Group2 bigwig files.
        """
        from bolero.utils import understand_regions

        regions = understand_regions(regions, as_df=True)
        if "Name" not in regions.columns:
            regions["Name"] = (
                regions.iloc[:, 0].astype(str)
                + ":"
                + regions.iloc[:, 1].astype(str)
                + "-"
                + regions.iloc[:, 2].astype(str)
            )
        value_col = []
        for _, (chrom, start, end, *_) in regions.iterrows():
            stats = self.get_bw_stats((chrom, start, end))
            value_col.append(stats)
        values = (
            pd.DataFrame(value_col, index=regions["Name"]).fillna(0).astype("float32")
        )
        return values

    @property
    def dataset_peak_scan(self):
        """
        Get dataset peak scan values for Group1 and Group2.
        """
        import numpy as np
        import pyranges as pr

        g1_bw_path = self.group1_bw_path
        g2_bw_path = self.group2_bw_path
        peaks_path = str(localize(g1_bw_path.parent / "peaks/peaks.bed"))
        if peaks_path not in self._caches:
            peaks_bed = pr.read_bed(peaks_path, as_df=True)
            self._caches[peaks_path] = peaks_bed
        else:
            peaks_bed = self._caches[peaks_path]

        peak_scan_path = localize(
            g1_bw_path.parent / f"peaks/{g1_bw_path.name[:-3]}.npz"
        )
        g1_values = np.load(peak_scan_path)["data"]
        g1_values = pd.Series(g1_values, index=peaks_bed["Name"].values)
        g2_values = np.load(
            localize(g2_bw_path.parent / f"peaks/{g2_bw_path.name[:-3]}.npz")
        )["data"]
        g2_values = pd.Series(g2_values, index=peaks_bed["Name"].values)
        values = pd.DataFrame(
            {
                self.group1: g1_values,
                self.group2: g2_values,
            }
        )
        values.index.name = "peak"
        values.columns.name = "group"
        return values

    def _dump_regions(self, regions, output_bed_path):
        import pyranges as pr
        import pysam
        from bolero.utils import understand_regions

        bed = pr.PyRanges(understand_regions(regions)).sort()
        bed.to_bed(output_bed_path)

        comp_out_path = f"{output_bed_path}.gz"
        pysam.tabix_compress(str(output_bed_path), comp_out_path, force=True)
        pysam.tabix_index(comp_out_path, preset="bed", force=True)
        output_bed_path.unlink()
        return comp_out_path

    def dump_sig_peak_bed(self, output_dir="./"):
        """
        Dump significant peak bed files for IGV browser.
        """
        if self.peak_diff_path is None:
            return None, None

        output_dir = pathlib.Path(output_dir)
        diff_table = self.peak_diff_table
        g1_high_bed_path = (
            output_dir / f"{self.da_key}.{self.group1}_high.{self.lfc_cutoff}.bed"
        )
        g2_high_bed_path = (
            output_dir / f"{self.da_key}.{self.group2}_high.{self.lfc_cutoff}.bed"
        )

        sign = diff_table["log2FoldChange"] > self.lfc_cutoff
        regions = diff_table[sign]["gene"]
        if len(regions) > 0:
            g1_high_bed_path = self._dump_regions(regions, g1_high_bed_path)
        else:
            g1_high_bed_path = None

        sign = diff_table["log2FoldChange"] < -self.lfc_cutoff
        regions = diff_table[sign]["gene"]
        if len(regions) > 0:
            g2_high_bed_path = self._dump_regions(regions, g2_high_bed_path)
        else:
            g2_high_bed_path = None
        return g1_high_bed_path, g2_high_bed_path

    def setup_igv_tracks(self):
        """
        Create IGV tracks table for IGV browser.
        """
        self.igv_dir = self.igv_base_dir / self.name.replace(":", "_")
        if not self.igv_dir.exists():
            self.igv_dir.mkdir(parents=True)

        # soft link bigwig to igv dir (localize first so external users link the
        # cached download rather than a non-existent lab path)
        g1_bw = self.igv_dir / self.group1_bw_path.name
        if not g1_bw.exists():
            os.symlink(localize(self.group1_bw_path), g1_bw)
        g2_bw = self.igv_dir / self.group2_bw_path.name
        if not g2_bw.exists():
            os.symlink(localize(self.group2_bw_path), g2_bw)
        tracks = [
            {
                "name": self.group1,
                "url": g1_bw,
                "color": "#fc8d62",
                "autoscaleGroup": "defaultgroup",
            },
            {
                "name": self.group2,
                "url": g2_bw,
                "color": "#8da0cb",
                "autoscaleGroup": "defaultgroup",
            },
        ]
        if self.peak_diff_path is not None:
            g1_high_bed_path, g2_high_bed_path = self.dump_sig_peak_bed(self.igv_dir)
            if g1_high_bed_path is not None:
                tracks.append(
                    {
                        "name": f"{self.group1}.high",
                        "url": g1_high_bed_path,
                        "indexURL": f"{g1_high_bed_path}.tbi",
                    }
                )
            if g2_high_bed_path is not None:
                tracks.append(
                    {
                        "name": f"{self.group2}.high",
                        "url": g2_high_bed_path,
                        "indexURL": f"{g2_high_bed_path}.tbi",
                    }
                )
        tracks = pd.DataFrame(tracks)
        tracks.to_csv(self.igv_dir / "tracks.csv", index=None)
        return tracks

    def igv_browser(self, locus=None):
        """
        Open IGV browser for the differential analysis record.
        """
        from bolero.pl.igv import Browser

        tracks = self.setup_igv_tracks()
        browser = Browser(genome=self.genome, locus=locus)
        browser.load_track_table(tracks, windowFunction="mean")
        return browser


class DiffAnalysisCollection:
    def __init__(self):
        self.da_table = metadata.DA_COLLECTION

    def __getitem__(self, key):
        try:
            gene_rec = self.da_table.loc[(key, "Gene")]
        except KeyError:
            gene_rec = None
        try:
            peak_rec = self.da_table.loc[(key, "Peak")]
        except KeyError:
            peak_rec = None
        return DiffRecords(gene_rec=gene_rec, peak_rec=peak_rec)


DA = DiffAnalysisCollection()
