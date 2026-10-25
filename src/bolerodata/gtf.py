"""
Reference gene annotations (GTF) for the genomes bolero works with.

Each genome's GTF and its prebuilt gffutils database are hosted on the
bolero-data HuggingFace repo under ``reference_gtf/<genome>/``.
:func:`get_gtf_path` resolves one through :func:`bolerodata._sync.localize`:
the lab copy when the lab filesystem is mounted, else a download on first use
and the cached copy afterwards.
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

#: genome -> logical directory of the lab copy (the manifest key's directory).
_LOGICAL_DIRS = {
    "mm10": "wmb/ref/mm10/gtf/biccn",
    "hg38": "wmb/ref/hg38/gtf",
    "calJac4": "wmb/ref/calJac4",
    "mCalJac1.pat.X": "wmb/ref/mCalJac1.pat.X/gtf",
    "monDom5.split": "wmb/ref/monDom5/gtf",
    "panPan3": "wmb/ref/panPan3/gtf",
    "rheMac10": "wmb/ref/rheMac10",
}

_REPO_DIR = "reference_gtf"


def _file_name(genome, db):
    """Return the file name of a genome's GTF or gffutils database."""
    name = GTF_FILES[genome]
    if db:
        name = name.removesuffix(".gtf") + ".gffutils.db"
    return name


def logical_path(genome, db):
    """Return the manifest key of a genome's GTF or gffutils database."""
    return f"{_LOGICAL_DIRS[genome]}/{_file_name(genome, db)}"


def repo_path(genome, db):
    """Return the bolero-data repo path of a genome's GTF or gffutils database."""
    return f"{_REPO_DIR}/{genome}/{_file_name(genome, db)}"


def get_gtf_path(genome, db=True, *, allow_download=True):
    """
    Return a local path to a genome's reference annotation.

    On the lab filesystem the lab copy is returned. Elsewhere the file is
    downloaded from the bolero-data HuggingFace repo on first use and served
    from the local cache afterwards.

    Parameters
    ----------
    genome : str
        Genome name, one of :data:`GTF_FILES` (e.g. ``"hg38"``, ``"mm10"``).
    db : bool
        If ``True`` (default), return the prebuilt gffutils database, which
        ``bolero.pp.gtf.load_gtf`` opens directly. If ``False``, return the
        raw ``.gtf``.
    allow_download : bool
        If ``False``, never hit the network; raise if the file is not local.

    Returns
    -------
    pathlib.Path
        Path to the local file.

    Raises
    ------
    KeyError
        If ``genome`` has no registered annotation.
    FileNotAvailableError
        If the file is not local and ``allow_download`` is ``False``.
    """
    if genome not in GTF_FILES:
        raise KeyError(
            f"No reference GTF for genome {genome!r}. "
            f"Known genomes: {', '.join(GTF_FILES)}."
        )
    kind = "gffutils database" if db else "GTF"
    return _sync.localize(
        logical_path(genome, db),
        allow_download=allow_download,
        desc=f"{genome} {kind}",
    )
