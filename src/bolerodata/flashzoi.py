"""
Flashzoi backbone weights in the layout bolero's ``Borzoi`` class loads.

Bolero adapts the public Flashzoi replicates (``johahi/flashzoi-replicate-{fold}``
on HuggingFace). Their state-dict keys differ from bolero's ``Borzoi`` module in
two ways: each transformer block nests its sub-layers under ``.layers``, and the
human/mouse output heads are absent from bolero's backbone. This module downloads
the public weights once, applies that rename, and caches the result.

The renamed public weights were checked against the lab's own renamed
checkpoints for all four folds: same keys, bit-identical tensors.
"""

import os
import pathlib
import re

from ._sync import FileNotAvailableError, _filelock, _hf_cache, cache_root

__all__ = ["FLASHZOI_FOLDS", "get_flashzoi_checkpoint"]

#: Folds published as ``johahi/flashzoi-replicate-{fold}``.
FLASHZOI_FOLDS = (0, 1, 2, 3)

#: Pinned HuggingFace commit per fold, so the weights cannot change under a user.
REVISIONS = {
    0: "72040fb525d4b9abab21ead1c53b76b6bd8b6d67",
    1: "3e54f34c90e007e6da7ebe7bc83282c2d70bfc18",
    2: "524b05b3d3ed7f9acf28e51fc283f37affc26c28",
    3: "336e45f85cbf0eafe8c8880984f1b561c79a5fe9",
}

_REPO = "johahi/flashzoi-replicate-{fold}"
_FILENAME = "pytorch_model.bin"
_HEADS = ("human_head", "mouse_head")

#: The lab's renamed checkpoints; used as-is when the lab filesystem is mounted.
_LAB_PATH = (
    "/large_storage/zhoulab/hanliu/wmb/model/Borzoi/flashzoi_checkpoints/"
    "fold_{fold}.renamed.pt"
)


def _to_bolero_key(key):
    """Rename a key: ``transformer.3.1.x`` -> ``transformer.3.layers.1.x``."""
    return re.sub(r"(transformer\.\d+)\.(\d+)", r"\1.layers.\2", key)


def get_flashzoi_checkpoint(fold=0, *, allow_download=True):
    """
    Return a local path to Flashzoi weights that bolero's ``Borzoi`` loads.

    On first use the public ``johahi/flashzoi-replicate-{fold}`` weights are
    downloaded from HuggingFace (pinned commit), their keys renamed to bolero's
    ``Borzoi`` module names, the output heads dropped, and the result saved to
    the bolerodata cache. Later calls return the cached file. On the lab
    filesystem the lab's own renamed checkpoint is returned instead.

    Parameters
    ----------
    fold : int
        Flashzoi replicate, one of :data:`FLASHZOI_FOLDS`.
    allow_download : bool
        If ``False``, never hit the network; raise if the file is not cached.

    Returns
    -------
    pathlib.Path
        A ``torch.save`` state dict for ``Borzoi.load_state_dict(strict=True)``.

    Raises
    ------
    ValueError
        If ``fold`` is not a published fold.
    FileNotAvailableError
        If the file is not cached and ``allow_download`` is ``False``.
    """
    if type(fold) is not int or fold not in FLASHZOI_FOLDS:
        raise ValueError(
            f"Unsupported Flashzoi fold {fold!r}; choose one of {FLASHZOI_FOLDS}."
        )

    lab = pathlib.Path(_LAB_PATH.format(fold=fold))
    if lab.exists():
        return lab

    out = cache_root() / "flashzoi" / f"fold_{fold}.renamed.pt"
    if out.exists():
        return out
    if not allow_download:
        raise FileNotAvailableError(
            f"Flashzoi fold {fold} is not in the local cache and downloads are "
            f"disabled (allow_download=False)."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with _filelock(out.parent / (out.name + ".lock")):
        if not out.exists():
            _download_and_rename(fold, out)
    return out


def _download_and_rename(fold, out):
    """Download one public fold, rename its keys, and write it to ``out``."""
    import torch
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(
        repo_id=_REPO.format(fold=fold),
        filename=_FILENAME,
        revision=REVISIONS[fold],
        cache_dir=str(_hf_cache()),
    )
    state = torch.load(src, map_location="cpu", weights_only=True)
    renamed = {
        _to_bolero_key(k): v
        for k, v in state.items()
        if k.split(".")[0] not in _HEADS
    }
    tmp = out.with_name(out.name + ".tmp")
    torch.save(renamed, tmp)
    os.replace(tmp, out)
