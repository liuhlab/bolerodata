#!/usr/bin/env python
"""
Slim Bolero model checkpoints for public sharing, and manage per-model selection.

A ``*.lora.best_checkpoint.pt`` is ~13 GB: a ~4.7 GB ``state_dict`` (the model
parameters) plus a ~7.6 GB ``optimizer`` state dump (Adam moments) that is only
needed to *resume training*. The inference loader (bolero
``load_checkpoint_from_path``) reads only ``state_dict``. This tool writes a slim
checkpoint containing just ``{"state_dict": ...}`` — ~63% smaller, bit-identical
weights — into ``/scratch/zhoulab/hanliu/huggingfase_model``.

It also owns the **per-model selection** (`model_selection.tsv`): only checkpoints
whose model is marked share are published, so in-development checkpoints do not
confuse users.

Subcommands
-----------
    plan     enumerate checkpoints -> dist/slim_tasks.tsv + refresh model_selection.tsv
    run      slim one task: --index N (from tasks file) or --src X --dst Y

The heavy `torch` import is lazy, so the helper functions (used by
build_manifest.py) import without it.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from bolerodata._sync import _to_logical  # noqa: E402

SLIM_DIR = pathlib.Path(
    os.environ.get("BOLERO_SLIM_DIR", "/scratch/zhoulab/hanliu/huggingfase_model")
)
MODEL_ZOO_TSV = SRC / "bolerodata" / "data" / "model_zoo.tsv"
MODEL_SELECTION = REPO_ROOT / "model_selection.tsv"
TASKS_TSV = REPO_ROOT / "dist" / "slim_tasks.tsv"


def safe_key(s):
    """Filesystem-safe token for a model key / group."""
    return "".join(c if (c.isalnum() or c in "-_.+") else "_" for c in str(s))


def model_repo_path(group, model_key, basename):
    """Clean per-model path inside the bolero-models repo."""
    return f"{safe_key(group)}/{safe_key(model_key)}/{basename}"


def _first_path(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    for part in str(value).split(","):
        part = part.strip()
        if part.startswith("/"):
            return part
    return None


def _ckpt_date(basename):
    m = re.match(r"(\d{6})", basename)
    return m.group(1) if m else ""


def model_records():
    """One record per model_zoo row (model_key). Includes checkpoint stat + slim path."""
    zoo = pd.read_table(MODEL_ZOO_TSV, index_col=0)
    recs = []
    for key, row in zoo.iterrows():
        src = _first_path(row.get("CkptPath"))
        cfg = _first_path(row.get("ConfigPath"))
        group = str(row.get("ModelGroup"))
        rec = {
            "model_key": str(key),
            "group": group,
            "dataset": str(row.get("DefaultDataset")),
            "ckpt_src": src,
            "ckpt_logical": _to_logical(src) if src else None,
            "config_src": cfg,
            "config_logical": _to_logical(cfg) if cfg else None,
            "basename": pathlib.Path(src).name if src else "",
            "ckpt_date": _ckpt_date(pathlib.Path(src).name) if src else "",
            "exists": bool(src) and pathlib.Path(src).exists(),
        }
        if rec["exists"]:
            st = pathlib.Path(src).stat()
            rec["size"] = st.st_size
            rec["mtime"] = st.st_mtime
            rec["repo_path"] = model_repo_path(group, key, rec["basename"])
            rec["slim"] = str(SLIM_DIR / rec["repo_path"])
        else:
            rec.update(size=0, mtime=0, repo_path=None, slim=None)
        recs.append(rec)
    return recs


def read_selection():
    """Return {model_key: bool} from model_selection.tsv, or None if the file is absent."""
    if not MODEL_SELECTION.exists():
        return None
    df = pd.read_table(MODEL_SELECTION, dtype=str).fillna("")
    truthy = {"1", "true", "yes", "y", "x", "share", "keep"}
    return {r["model_key"]: str(r.get("Share", "")).strip().lower() in truthy
            for _, r in df.iterrows()}


def slim_tasks(selected_only=False):
    """Unique existing checkpoints -> [(src, dst)], deduped by source path."""
    sel = read_selection() if selected_only else None
    seen, tasks = set(), []
    for r in model_records():
        if not r["exists"] or r["ckpt_src"] in seen:
            continue
        if sel is not None and not sel.get(r["model_key"], False):
            continue
        seen.add(r["ckpt_src"])
        tasks.append((r["ckpt_src"], r["slim"]))
    return tasks


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def cmd_plan(args):
    recs = model_records()
    # slim tasks (all existing unique checkpoints, so sizes are known for review)
    tasks = slim_tasks(selected_only=args.selected_only)
    TASKS_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(TASKS_TSV, "w") as f:
        f.write("src\tdst\n")
        for src, dst in tasks:
            f.write(f"{src}\t{dst}\n")
    print(f"Wrote {TASKS_TSV.relative_to(REPO_ROOT)}: {len(tasks)} checkpoints to slim.")

    # refresh model_selection.tsv WITHOUT clobbering existing Share choices
    prev = read_selection() or {}
    cols = ["model_key", "ModelGroup", "DefaultDataset", "CkptDate", "CkptSizeGB",
            "CkptExists", "Share"]
    rows = []
    for r in sorted(recs, key=lambda x: (x["group"], x["dataset"], x["ckpt_date"])):
        rows.append({
            "model_key": r["model_key"],
            "ModelGroup": r["group"],
            "DefaultDataset": r["dataset"],
            "CkptDate": r["ckpt_date"],
            "CkptSizeGB": f"{r['size']/2**30:.1f}" if r["exists"] else "",
            "CkptExists": "yes" if r["exists"] else "no",
            "Share": "yes" if prev.get(r["model_key"]) else "",
        })
    pd.DataFrame(rows, columns=cols).to_csv(MODEL_SELECTION, sep="\t", index=False)
    n_sel = sum(1 for v in prev.values() if v)
    print(f"Wrote {MODEL_SELECTION.name}: {len(rows)} models "
          f"({n_sel} currently marked Share).")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def slim_one(src, dst, force=False):
    """Write {'state_dict': ...} of `src` to `dst`. Returns (status, dst_size)."""
    import torch

    src, dst = pathlib.Path(src), pathlib.Path(dst)
    if (dst.exists() and not force and dst.stat().st_size > 0
            and dst.stat().st_mtime >= src.stat().st_mtime):
        return "skip", dst.stat().st_size

    ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        slim = {"state_dict": ck["state_dict"]}
    elif isinstance(ck, dict) and "model_state_dict" in ck:
        slim = {"state_dict": ck["model_state_dict"]}
    elif isinstance(ck, dict) and ck and all(torch.is_tensor(v) for v in ck.values()):
        slim = {"state_dict": ck}
    else:
        raise SystemExit(f"Unrecognized checkpoint structure in {src}: keys={list(ck)[:8]}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(slim, tmp)
    os.replace(tmp, dst)
    return "slimmed", dst.stat().st_size


def cmd_run(args):
    if args.src and args.dst:
        status, size = slim_one(args.src, args.dst, force=args.force)
        print(f"[{status}] {size/2**30:.2f}GB  {args.dst}")
        return

    rows = TASKS_TSV.read_text().splitlines()[1:]
    total = len(rows)
    if args.stride:
        # round-robin partition: task `index` handles lines index, index+stride, ...
        indices = list(range(args.index, total + 1, args.stride))
    else:
        indices = [args.index]
    done = failed = 0
    for n in indices:
        if not (1 <= n <= total):
            continue
        src, dst = rows[n - 1].split("\t")
        try:
            status, size = slim_one(src, dst, force=args.force)
            print(f"[{status}] line {n} {size/2**30:.2f}GB {dst}", flush=True)
            done += 1
        except Exception as e:  # noqa: BLE001 - keep going; the array is idempotent
            print(f"[ERROR] line {n} {src}: {e}", flush=True)
            failed += 1
    print(f"chunk complete: {done} ok, {failed} failed of {len(indices)}", flush=True)
    if failed:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write slim task list + model_selection.tsv")
    p.add_argument("--selected-only", action="store_true",
                   help="only slim checkpoints whose model is marked Share")
    p.set_defaults(func=cmd_plan)

    r = sub.add_parser("run", help="slim checkpoint(s)")
    r.add_argument("--index", type=int, help="1-based line in dist/slim_tasks.tsv")
    r.add_argument("--stride", type=int,
                   help="round-robin: also do lines index+stride, index+2*stride, ...")
    r.add_argument("--src")
    r.add_argument("--dst")
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
