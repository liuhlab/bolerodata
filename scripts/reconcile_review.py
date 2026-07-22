#!/usr/bin/env python
"""
Fold a reviewed GitHub sign-off issue back into ``huggingface_digest.md``.

Reads the issue posted by ``post_review_issues.py`` (via ``gh``), parses the
approve-to-SHARE checklist, and updates the digest's ``Status`` cells:

    checked  box  -> category Status = share
    unchecked box -> category Status = noshare   (reviewer withheld it)

Free-form reviewer comments and the licensing checkbox are printed for you to act
on manually (they cannot be applied automatically). After applying, re-run
``scripts/build_manifest.py`` to regenerate the manifest + upload list.

Usage
-----
    python scripts/reconcile_review.py --issue 42            # dry run: show changes
    python scripts/reconcile_review.py --issue 42 --apply    # write huggingface_digest.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DIGEST = REPO_ROOT / "huggingface_digest.md"
DEFAULT_REPO = "liuhlab/bolerodata"

# "- [x] **name** ..." (name may contain +, for model keys like Li2023Science+Exc)
CHECK_RE = re.compile(r"^\s*-\s*\[( |x|X)\]\s*\*\*([A-Za-z0-9._+-]+)\*\*")


def fetch_issue(repo, number):
    res = subprocess.run(
        ["gh", "issue", "view", str(number), "--repo", repo, "--json", "body,comments"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        sys.exit(f"gh issue view failed:\n{res.stderr}")
    return json.loads(res.stdout)


def parse_decisions(body):
    """
    Parse the issue checklists.

    Returns
    -------
    (cat, model) : tuple[dict, dict]
        ``cat``   maps data category -> 'share'/'noshare' (sections A & B).
        ``model`` maps model_key -> bool (section C, per-model checkpoints).
    """
    cat, model = {}, {}
    mode = None
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("## A.") or s.startswith("## B."):
            mode = "data"
            continue
        if s.startswith("## C."):
            mode = "model"
            continue
        if s.startswith("## ") or s.startswith("---"):
            mode = None
            continue
        if mode is None:
            continue
        m = CHECK_RE.match(line)
        if m:
            checked = m.group(1).lower() == "x"
            if mode == "data":
                cat[m.group(2)] = "share" if checked else "noshare"
            else:
                model[m.group(2)] = checked
    return cat, model


def apply_to_selection(model_decisions, apply):
    """Write per-model checkbox states into model_selection.tsv Share column."""
    import csv

    sel = REPO_ROOT / "model_selection.tsv"
    if not sel.exists() or not model_decisions:
        return []
    rows = list(csv.reader(sel.open(), delimiter="\t"))
    header = rows[0]
    mi, si = header.index("model_key"), header.index("Share")
    changes = []
    for row in rows[1:]:
        if len(row) <= si:
            continue
        key = row[mi]
        if key in model_decisions:
            want = "yes" if model_decisions[key] else ""
            if row[si] != want:
                changes.append((key, row[si] or "-", want or "-"))
                row[si] = want
    if apply and changes:
        with sel.open("w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(rows)
    return changes


def apply_to_digest(decisions, apply):
    lines = DIGEST.read_text().splitlines()
    changes = []
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or cells[0] == "Category" or set(cells[0]) <= {"-", ":"}:
            continue
        category, desc, tier, status, repo = cells[:5]
        want = decisions.get(category)
        if want and want != status.replace("⚠️", "").strip().lower():
            changes.append((category, status, want))
            cells[3] = want
            lines[i] = "| " + " | ".join(cells[:5]) + " |"
    if apply and changes:
        DIGEST.write_text("\n".join(lines) + "\n")
    return changes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--issue", type=int, required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--apply", action="store_true", help="write huggingface_digest.md")
    args = ap.parse_args()

    issue = fetch_issue(args.repo, args.issue)
    cat_decisions, model_decisions = parse_decisions(issue["body"])
    changes = apply_to_digest(cat_decisions, args.apply)
    model_changes = apply_to_selection(model_decisions, args.apply)

    n_sel = sum(1 for v in model_decisions.values() if v)
    print(f"Parsed {len(cat_decisions)} data-category + {len(model_decisions)} "
          f"per-model decisions from issue #{args.issue} ({n_sel} models checked).")
    if changes:
        print("\nData category Status changes:")
        for cat, old, new in changes:
            print(f"  {cat}: {old} -> {new}")
    if model_changes:
        print(f"\nmodel_selection.tsv changes: {len(model_changes)}")
        for key, old, new in model_changes[:20]:
            print(f"  {key}: {old} -> {new}")
        if len(model_changes) > 20:
            print(f"  ... and {len(model_changes) - 20} more")
    if not changes and not model_changes:
        print("No changes (digest + selection already match the checkboxes).")

    comments = issue.get("comments") or []
    if comments:
        print(f"\n--- {len(comments)} reviewer comment(s) (apply manually) ---")
        for c in comments:
            author = c.get("author", {}).get("login", "?")
            print(f"[{author}] {c.get('body', '').strip()[:500]}")

    if "I confirm **qtl-GTEx**" in issue["body"]:
        lic = re.search(r"-\s*\[( |x|X)\]\s*I confirm \*\*qtl-GTEx\*\*", issue["body"])
        state = "CHECKED" if (lic and lic.group(1).lower() == "x") else "NOT checked"
        print(f"\nLicensing confirmation (GTEx/OneK1K): {state}")

    any_changes = changes or model_changes
    if args.apply and any_changes:
        print("\nApplied. Now run: python scripts/build_manifest.py")
    elif any_changes:
        print("\nDry run — re-run with --apply to write digest + model_selection.tsv.")


if __name__ == "__main__":
    main()
