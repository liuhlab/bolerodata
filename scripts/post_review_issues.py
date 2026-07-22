#!/usr/bin/env python
"""
Post the bolerodata upload list to GitHub for review (the pre-upload sign-off gate).

Builds a checklist issue from ``hf_manifest.json`` (one checkbox per category:
checked = approved to upload/share) and creates it on the GitHub repo via the
``gh`` CLI. Nothing is uploaded here — this only opens the review.

Workflow
--------
    python scripts/build_manifest.py            # (re)build the candidate manifest
    python scripts/post_review_issues.py        # DRY RUN: writes dist/review_issue.md
    python scripts/post_review_issues.py --post  # actually create the GitHub issue
    # ... reviewer checks boxes / comments on the issue ...
    python scripts/reconcile_review.py --issue N --apply   # fold decisions into digest
    python scripts/build_manifest.py            # regenerate; then upload approved files

Requires ``gh`` authenticated (`gh auth status`) for ``--post``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "src" / "bolerodata" / "data" / "hf_manifest.json"
DIGEST = REPO_ROOT / "huggingface_digest.md"
DEFAULT_REPO = "liuhlab/bolerodata"

# Categories that need explicit licensing confirmation before going public.
LICENSE_FLAG = {"qtl-GTEx", "qtl-sc-eqtl"}


def _human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def summarize(manifest):
    cats = {}
    for logical, e in manifest.items():
        c = cats.setdefault(
            e["category"],
            {"share": e["share"], "tier": e.get("tier", "?"), "repo": [],
             "count": 0, "size": 0, "sample": []},
        )
        c["count"] += 1
        c["size"] += e.get("size", 0) or 0
        if e.get("repo_id") and e["repo_id"] not in c["repo"]:
            c["repo"].append(e["repo_id"])
        if len(c["sample"]) < 3:
            c["sample"].append(logical)
    return cats


# Checkpoints/configs are selected per-model (section C), not by category.
MODEL_CATEGORIES = {"model-checkpoints", "model-configs"}


def _share_rows(share, tier):
    """Checklist rows for one tier, biggest category first."""
    rows = []
    items = sorted(
        ((c, v) for c, v in share.items() if v["tier"] == tier),
        key=lambda kv: kv[1]["size"], reverse=True,
    )
    n = sum(v["count"] for _, v in items)
    sub = sum(v["size"] for _, v in items)
    for cat, v in items:
        flag = " ⚠️ **licensing**" if cat in LICENSE_FLAG else ""
        repo = ", ".join(v["repo"]) or "—"
        rows.append(f"- [ ] **{cat}** — {v['count']} files, **{_human(v['size'])}** "
                    f"→ `{repo}`{flag}")
        rows.append(f"  <sub>e.g. `{'`, `'.join(v['sample'])}`</sub>")
    return rows, n, sub


def _model_rows():
    """Section C checkboxes — one per unique checkpoint FILE.

    Several model_zoo rows can map to the SAME checkpoint (e.g. in-development
    motif variants aliasing one stand-in checkpoint). Collapse to the physical
    file so sizes/totals are honest; key the checkbox on the first ("owner")
    model_key — matching build_manifest's repo_path and per-model selection —
    and flag heavily-aliased files so placeholders are obvious.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from slim_checkpoint import model_records

    uniq = {}  # ckpt_src -> [rows]; first row seen is the owner
    for r in model_records():
        if r["exists"] and r["ckpt_src"]:
            uniq.setdefault(r["ckpt_src"], []).append(r)

    groups, total = {}, 0
    for rows in uniq.values():
        owner, aliases = rows[0], rows[1:]
        slim = pathlib.Path(owner["slim"]) if owner["slim"] else None
        if slim and slim.exists():
            size = slim.stat().st_size
            size_str = _human(size)
        else:  # unslimmed (should not happen post-slim) — estimate
            size = int(owner["size"] * 0.37)  # ~63% smaller once slimmed
            size_str = f"~{_human(size)} (not slimmed)"
        total += size
        groups.setdefault(owner["group"], []).append((owner, aliases, size_str))

    lines = []
    for group in sorted(groups):
        items = sorted(groups[group], key=lambda x: (x[0]["dataset"], x[0]["ckpt_date"]))
        lines.append(f"#### {group}  ·  {len(items)} checkpoints")
        for owner, aliases, size_str in items:
            lines.append(f"- [ ] **{owner['model_key']}** — {owner['dataset']}, "
                         f"{owner['ckpt_date']}, {size_str}")
            if aliases:
                shown = ", ".join(a["model_key"] for a in aliases[:12])
                more = f" +{len(aliases) - 12} more" if len(aliases) > 12 else ""
                base = pathlib.Path(owner["ckpt_src"]).name
                lines.append(
                    f"  <sub>⚠️ one file (`{base}`) also mapped by {len(aliases)} "
                    f"other config(s): {shown}{more} — likely in-development "
                    f"placeholders; checking shares this single file once.</sub>")
    return lines, len(uniq), total


def build_body(manifest):
    cats = summarize(manifest)
    data = {k: v for k, v in cats.items() if k not in MODEL_CATEGORIES}
    share = {k: v for k, v in data.items() if v["share"]}
    noshare = {k: v for k, v in data.items() if not v["share"]}
    tiers = sorted({v["tier"] for v in share.values()})

    model_lines, n_models, model_total = _model_rows()
    n_data = sum(v["count"] for v in share.values())
    data_total = sum(v["size"] for v in share.values())

    lines = [
        "## bolerodata → HuggingFace: upload sign-off",
        "",
        f"Pre-upload review gate. Proposed: **{n_data} data files (~{_human(data_total)})** "
        f"plus **checkpoints selected per-model** below (~{_human(model_total)} across "
        f"{n_models} unique checkpoints). **Check a box to approve**; leave unchecked to withhold. "
        "Nothing uploads until this is signed off.",
        "",
        "Checkpoints are **slimmed to parameters only** — optimizer state dropped, "
        "~63% smaller, weights bit-identical. Repos: `bolero-models` (checkpoints) · "
        "`bolero-data` (data).",
        "",
        "### ⚠️ Licensing — confirm before approving the QTL categories",
        "- [ ] I confirm **qtl-GTEx** / **qtl-sc-eqtl** (GTEx / OneK1K-derived) may be "
        "publicly redistributed (dbGaP / consent terms reviewed).",
        "",
        "## A. Data to share — check to approve",
    ]
    tier_titles = {"core": "Core — inference", "eval": "Eval — reproducibility",
                   "full": "Training data — opt-in (large)"}
    for tier in tiers:
        rows, n, sub = _share_rows(share, tier)
        if rows:
            lines += ["", f"### {tier_titles.get(tier, tier)}  ·  {n} files, ~{_human(sub)}",
                      *rows]

    lines += ["", "## B. Withheld data — check to INCLUDE (else stays withheld)"]
    for cat in sorted(noshare, key=lambda c: -noshare[c]["size"]):
        v = noshare[cat]
        lines.append(f"- [ ] **{cat}** — {v['count']} files, **{_human(v['size'])}**")
        lines.append(f"  <sub>e.g. `{'`, `'.join(v['sample'])}`</sub>")

    lines += [
        "", "## C. Checkpoints — select per model (check to share)",
        "<sub>Grouped by model type; newest date last. Unchecked keeps "
        "in-development checkpoints from confusing users.</sub>", "",
        *model_lines,
        "", "---",
        "<sub>Detail: `hf_manifest.json`, `model_selection.tsv`, `huggingface_digest.md`. "
        "After review: `reconcile_review.py --issue <#> --apply` then `build_manifest.py`.</sub>",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/repo")
    ap.add_argument("--post", action="store_true", help="actually create the issue (else dry run)")
    ap.add_argument("--edit", type=int, metavar="ISSUE",
                    help="update an existing issue's body in place instead of creating")
    ap.add_argument("--title", default="bolerodata → HuggingFace upload sign-off")
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit("hf_manifest.json not found — run scripts/build_manifest.py first.")
    manifest = json.loads(MANIFEST.read_text())
    body = build_body(manifest)

    out = REPO_ROOT / "dist" / "review_issue.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(body)
    print(f"Wrote {out.relative_to(REPO_ROOT)}", file=sys.stderr)

    if args.edit:
        res = subprocess.run(
            ["gh", "issue", "edit", str(args.edit), "--repo", args.repo,
             "--body-file", str(out)],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            sys.exit(f"gh issue edit failed:\n{res.stderr}")
        print(res.stdout.strip() or f"updated issue #{args.edit}")
        return

    if not args.post:
        print("DRY RUN — inspect dist/review_issue.md, then re-run with --post.", file=sys.stderr)
        print(body)
        return

    res = subprocess.run(
        ["gh", "issue", "create", "--repo", args.repo, "--title", args.title,
         "--body-file", str(out), "--label", "upload-review"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        # retry without the label (repo may not have it)
        res = subprocess.run(
            ["gh", "issue", "create", "--repo", args.repo, "--title", args.title,
             "--body-file", str(out)],
            capture_output=True, text=True,
        )
    if res.returncode != 0:
        sys.exit(f"gh issue create failed:\n{res.stderr}")
    print(res.stdout.strip())


if __name__ == "__main__":
    main()
