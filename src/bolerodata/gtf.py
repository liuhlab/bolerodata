"""
Reference gene annotations (GTF) for the genomes bolero works with.

Each genome's GTF and its prebuilt gffutils database are hosted on the
bolero-data HuggingFace repo under ``reference_gtf/<genome>/``.
:func:`get_gtf_path` downloads one on first use and returns the cached copy.
"""

from . import _sync

__all__ = ["GTF_FILES", "get_gtf_path"]

#: genome -> GTF filename. The prebuilt database next to it is named
#: ``<stem>.gffutils.db``, the name ``bolero.pp.gtf.load_gtf`` derives from a GTF.
GTF_FILES = {
    "mm10": "modified_gencode.vM23.primary_assembly.annotation.gtf",
    "hg38": "gencode.v30.annotation.gtf",
    "calJac4": "ncbiRefSeq.gtf",
    "mCalJac1.pat.X": "Callithrix_jacchus.mCalJac1.pat.X.114.chr.gtf",
    "monDom5.split": "monDom5_evodevoCerebellum_extended_codingOverlapsRM.gtf",
    "panPan3": "ncbiRefSeq.gtf",
    "rheMac10": "rheMac10.ensGene.gtf",
}

_REPO_DIR = "reference_gtf"


def _repo_path(genome, db):
    """Return the bolero-data repo path of a genome's GTF or gffutils database."""
    name = GTF_FILES[genome]
    if db:
        name = name.removesuffix(".gtf") + ".gffutils.db"
    return f"{_REPO_DIR}/{genome}/{name}"


def get_gtf_path(genome, db=True):
    """
    Return a local path to a genome's reference annotation.

    The file is downloaded from the bolero-data HuggingFace repo on first use
    and served from the local cache afterwards.

    Parameters
    ----------
    genome : str
        Genome name, one of :data:`GTF_FILES` (e.g. ``"hg38"``, ``"mm10"``).
    db : bool
        If ``True`` (default), return the prebuilt gffutils database, which
        ``bolero.pp.gtf.load_gtf`` opens directly. If ``False``, return the
        raw ``.gtf``.

    Returns
    -------
    pathlib.Path
        Path to the cached file.

    Raises
    ------
    KeyError
        If ``genome`` has no registered annotation.
    """
    if genome not in GTF_FILES:
        raise KeyError(
            f"No reference GTF for genome {genome!r}. "
            f"Known genomes: {', '.join(GTF_FILES)}."
        )
    repo_path = _repo_path(genome, db)
    entry = {"repo_id": _sync.HF_DATA_REPO, "repo_type": "dataset",
             "repo_path": repo_path, "kind": "file"}
    return _sync._download(repo_path, entry)
