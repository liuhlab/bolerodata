# bolerodata

`bolerodata` is the **dataset, model-zoo and metadata registry** for
[Bolero](https://github.com/liuhlab/bolero) — the cell-state-conditioned
sequence-to-function model. It provides a thin, uniform Python API over the
single-cell datasets, trained model checkpoints, QTL collections and
differential-accessibility records used to train Bolero and produce the paper's
results.

`bolerodata` is designed to be used **together with `bolero`**: it imports `bolero`
lazily and is normally installed as a dependency of `bolero`. Most functionality
also relies on the lab's local data lake (`$STANDARD_DIR`), so it is primarily a
reproducibility/registry layer rather than a standalone, portable package.

Documentation is under construction — more to come.
