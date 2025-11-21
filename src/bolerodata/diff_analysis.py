import os
import pathlib

import pandas as pd
import pyranges as pr
import pysam
from bolero.pl.igv import Browser
from bolero.utils import understand_regions

from bolerodata import DATASETS
from bolerodata.data import metadata


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
        self._gene_diff_table = None
        self.peak_diff_path = None if peak_rec is None else peak_rec["DiffPath"]
        self._peak_diff_table = None
        self.igv_base_dir = pathlib.Path(igv_base_dir)
        self.igv_base_dir.mkdir(parents=True, exist_ok=True)

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
        if self._gene_diff_table is None:
            table = pd.read_feather(self.gene_diff_path)
            self._gene_diff_table = self._filter_by_lfc_cutoff(table)
        return self._gene_diff_table

    @property
    def peak_diff_table(self):
        """
        Peak differential analysis table.
        """
        if self.peak_diff_path is None:
            return None
        if self._peak_diff_table is None:
            table = pd.read_feather(self.peak_diff_path)
            self._peak_diff_table = self._filter_by_lfc_cutoff(table)
        return self._peak_diff_table

    def _dump_regions(self, regions, output_bed_path):
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

        # soft link bigwig to igv dir
        g1_bw = self.igv_dir / self.group1_bw_path.name
        if not g1_bw.exists():
            os.symlink(self.group1_bw_path, g1_bw)
        g2_bw = self.igv_dir / self.group2_bw_path.name
        if not g2_bw.exists():
            os.symlink(self.group2_bw_path, g2_bw)
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
        tracks = self.setup_igv_tracks()
        browser = Browser(genome=self.genome, locus=locus)
        browser.load_track_table(tracks, windowFunction="mean")
        return browser


class DiffAnalysisCollection:
    def __init__(self):
        self.da_table = metadata.DA_COLLECTION

    def __getitem__(self, key):
        gene_rec = self.da_table.loc[(key, "Gene")]
        peak_rec = self.da_table.loc[(key, "Peak")]
        return DiffRecords(gene_rec=gene_rec, peak_rec=peak_rec)
