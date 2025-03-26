# bolero data

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/lhqing/bolerodata/test.yaml?branch=main
[link-tests]: https://github.com/lhqing/bolerodata/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/bolerodata

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniforge](https://github.com/conda-forge/miniforge).

```bash
# 1. Create a environment named bolerodata
mamba env create -f environment.yaml
# OR if you use conda
# conda env create -f environment.yaml
# Note that conda can be very slow in solving complex dependencies

# 2. Install this package
pip install bolerodata

# or install the package with dev mode
git clone https://github.com/lhqing/bolerodata.git
cd bolerodata
pip install -e ".[dev,test]"
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/lhqing/bolerodata/issues
[changelog]: https://bolerodata.readthedocs.io/latest/changelog.html
[link-docs]: https://bolerodata.readthedocs.io
[link-api]: https://bolerodata.readthedocs.io/latest/api.html
