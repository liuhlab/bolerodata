# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][], and this project uses
[Calendar Versioning][] (`vYYYY.MM.DD`) to stay in step with the `bolero` package
it ships with.

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[calendar versioning]: https://calver.org/

## [Unreleased]

## [2026.07.22]

First public release — `pip install bolerodata`, with the curated data/model
subset published to `arcinstitute/bolero-data` and `arcinstitute/bolero-models`
on HuggingFace.

### Added

-   Dataset, model-zoo, QTL and differential-analysis registries for Bolero
-   **HuggingFace auto-download + cache layer** (`bolerodata._sync`): artifacts are
    fetched on demand from `arcinstitute/bolero-models` and `arcinstitute/bolero-data` and cached
    under `$BOLERODATA_HOME` / `$BOLERO_HOME` / the platform cache dir. `localize()`
    serves lab files directly and downloads shared ones otherwise;
    `prefetch()` / `sync()` warm the cache.
-   `FileNotAvailableError` — a friendly error for artifacts that are deliberately
    not redistributed publicly.
-   `huggingface_digest.md` (share/noshare categories) with `scripts/build_manifest.py`,
    `scripts/post_review_issues.py`, `scripts/reconcile_review.py` for the publish +
    review-gate workflow.
-   Build-time sanitization of shipped tables so the wheel never exposes lab paths.
-   `diff` optional extra (`pyBigWig`, `pyranges`, `pysam`) for `diff_analysis`.
-   Documentation: installation, data setup & caching, sharing, API reference.

### Changed

-   `import bolerodata` no longer requires the lab filesystem: the import-time
    `$STANDARD_DIR` existence assertion was removed and `torch` is now imported
    lazily, so the package imports and browses the registry off-lab.
-   Declared dependencies now include `numpy`, `huggingface_hub`, `platformdirs`
    (dropped the unused `tqdm`).
