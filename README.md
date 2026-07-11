# bolerodata

**Dataset, model-zoo and metadata registry for [Bolero](https://github.com/liuhlab/bolero).**

A uniform Python API over the single-cell datasets, model checkpoints, QTL collections and
differential-accessibility records behind Bolero. It is meant to be used **together with
`bolero`** (imported lazily; normally installed as a git dependency of `bolero`) and relies
on the lab's local data lake (`$STANDARD_DIR`).

Documentation: <https://liuhlab.github.io/bolerodata/>

## Installation

You normally get `bolerodata` automatically when you install `bolero`. To work on it
directly, use [pixi](https://pixi.sh) (lean — no GPU stack):

```bash
git clone https://github.com/liuhlab/bolerodata.git
cd bolerodata
pixi install            # runtime env; or: pixi install -e dev
```

---

> README and documentation are a work in progress.
