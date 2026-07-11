# bolerodata

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/liuhlab/bolerodata/test.yaml?branch=main
[link-tests]: https://github.com/liuhlab/bolerodata/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/github/deployments/liuhlab/bolerodata/github-pages?label=docs

`bolerodata` is the **dataset, model-zoo and metadata registry** for
[Bolero](https://github.com/liuhlab/bolero) — the cell-state-conditioned
sequence-to-function model. It provides a thin, uniform Python API over the
single-cell datasets, trained model checkpoints, QTL collections and
differential-accessibility records used to train Bolero and to produce the
paper's results.

It is meant to be used **together with `bolero`** (it imports `bolero` lazily and
is normally installed as a git dependency of `bolero`), and most functionality
relies on the lab's local data lake (`$STANDARD_DIR`) — so it is primarily a
reproducibility/registry layer rather than a portable, standalone package.

## Getting started

Please refer to the [documentation][link-docs].

## Installation

You normally get `bolerodata` for free by installing `bolero`, which lists it as a
git dependency. To work on `bolerodata` itself, use [pixi](https://pixi.sh):

```bash
git clone https://github.com/liuhlab/bolerodata.git
cd bolerodata
pixi install            # lean runtime env; or `pixi install -e dev` for tests/lint
```

Unlike `bolero`, this environment is intentionally lightweight — it carries no
PyTorch/CUDA/`ray` stack. To exercise the full API, run `bolerodata` inside a
`bolero` environment.

## Release notes

See the [changelog](CHANGELOG.md).

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[issue-tracker]: https://github.com/liuhlab/bolerodata/issues
[link-docs]: https://liuhlab.github.io/bolerodata/
