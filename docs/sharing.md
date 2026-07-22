# What is shared (for maintainers)

This page is for **maintainers** publishing `bolerodata` data to HuggingFace. Users
do not need it.

## The digest is the source of truth

[`huggingface_digest.md`](https://github.com/liuhlab/bolerodata/blob/main/huggingface_digest.md)
decides what is published. Every referenced file is assigned a **category** by its
provenance (which table column / joblib dict it came from); each category has a
**Status** of `share` or `noshare`. To include or exclude a class of files, edit its
Status cell and rebuild the manifest.

Files that are not shared raise a friendly
[`FileNotAvailableError`][bolerodata._sync.FileNotAvailableError] when requested,
naming the category and pointing users to the authors — never a broken path or a
raw 404.

## Build the manifest

```bash
python scripts/build_manifest.py
```

This enumerates every referenced file, maps each lab path to a portable *logical*
path, joins the digest, and writes:

- `src/bolerodata/data/hf_manifest.json` — consumed by `localize` (ships in the
  wheel; contains **no** lab absolute paths);
- `dist/hf_upload_list.tsv` + `dist/hf_upload.sh` — the share-only upload commands;
- `dist/hf_review.md` — a per-category summary.

Any category found in the data but missing from the digest is treated as `noshare`
and appended to the digest's **UNCATEGORIZED** section for you to classify.

## Review gate (required before uploading)

Nothing is uploaded until the file list is signed off on GitHub:

```bash
python scripts/post_review_issues.py            # dry run → dist/review_issue.md
python scripts/post_review_issues.py --post     # open the sign-off issue
# reviewer checks boxes / comments on the issue …
python scripts/reconcile_review.py --issue N --apply   # fold decisions into the digest
python scripts/build_manifest.py                # regenerate with the approved set
```

Checked box = approved to share; unchecked = withheld. The issue also carries a
**licensing checkbox** for the GTEx / OneK1K-derived QTL categories — confirm
redistribution terms before approving those.

## Upload

```bash
huggingface-cli login          # once, from a machine with the data
bash dist/hf_upload.sh         # uploads only approved files
```

Checkpoints go to `arcinstitute/bolero-models` (clean `<ModelGroup>/<model_key>/…` names);
everything else to `arcinstitute/bolero-data`. Enable `HF_HUB_ENABLE_HF_TRANSFER=1` (the
script does) for the ~1.3 TB of checkpoints.

## Publishing the package

At build time a hatch hook ([`hatch_build.py`](https://github.com/liuhlab/bolerodata/blob/main/hatch_build.py))
rewrites the shipped tables' path columns to logical form **in the wheel only** and
fails the build if any lab path survives — so the published package never exposes
internal filesystem layout, while the git-committed tables keep absolute paths for
the author fast path.
