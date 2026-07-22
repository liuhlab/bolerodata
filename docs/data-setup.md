# Data setup & caching

`bolerodata` ships only small registry tables. Everything large — model
checkpoints, metadata feathers, reference bigwigs, QTL tables — is fetched **on
demand from HuggingFace** and cached on your machine. You rarely need to configure
anything; this page explains what happens and how to control it.

## How resolution works

Every file the registry hands you goes through a single resolver,
[`bolerodata._sync.localize`][bolerodata._sync.localize]:

1. **If the file already exists locally** (you are on the lab filesystem, or it is
   already cached) → it is returned as-is, no network.
2. **Otherwise** the path is mapped to a repo-relative key and looked up in the
   bundled manifest:
   - **shared** → downloaded from HuggingFace into your cache, then returned;
   - **not shared** → a friendly [`FileNotAvailableError`][bolerodata._sync.FileNotAvailableError]
     explains it is not part of the public release;
   - **unknown** → a friendly error (you are likely off-lab and asking for something
     that was never published).

Downloads are lazy and per-file: `MODELS["k"]` is cheap; the ~13 GB checkpoint is
only fetched when you actually build a predictor.

## Where files are cached

The cache directory is chosen in this order:

1. `$BOLERODATA_HOME`
2. `$BOLERO_HOME`
3. `platformdirs.user_cache_dir("bolero")` (e.g. `~/.cache/bolero` on Linux)

```bash
export BOLERODATA_HOME=/scratch/$USER/bolero_cache   # put the cache on a big disk
```

Layout under the cache root:

- `hf/` — the HuggingFace download cache (content-addressed; safe to share between
  projects, resumable, deduplicated).
- `store/` — unpacked directory artifacts (extracted once, atomically).

Deleting the cache is always safe; files re-download on next use.

## Pre-fetching (warm cache / offline use)

To download a set of artifacts ahead of time (e.g. before going offline or onto a
compute node without internet):

```python
from bolerodata import _sync

_sync.prefetch(tier="core")                 # everything needed to run inference
_sync.prefetch(category="curated-cell-metadata")
_sync.prefetch(datasets=["HumanBrainDev"])  # only this dataset's shared files
_sync.sync()                                # all shared artifacts (large!)
```

Each returns `{"downloaded": N, "errors": [...]}`. To force a strict offline run
that never hits the network, resolve with `allow_download=False`:

```python
from bolerodata._sync import localize
localize(path, allow_download=False)   # raises if not already cached
```

## Private / gated repositories

If the HuggingFace repos are private or gated, log in once so `huggingface_hub`
picks up your token:

```bash
huggingface-cli login
```

You can point at a fork / mirror with `$BOLERO_HF_MODELS_REPO` and
`$BOLERO_HF_DATA_REPO`.

## Author mode (lab filesystem)

On the lab, set `$STANDARD_DIR` to the data lake (or rely on the built-in default).
Because the referenced files exist on disk, `localize` returns them directly and
**nothing downloads** — identical to the pre-HuggingFace behaviour. Off-lab, leave
`$STANDARD_DIR` unset; resolution falls through to the HuggingFace cache.

!!! note
    Model *building*, `Genome`, and `diff_analysis` also need the companion
    [`bolero`](https://github.com/liuhlab/bolero) package installed — see
    [Installation](install.md). Genome references are managed by `bolero`, not
    `bolerodata`.
