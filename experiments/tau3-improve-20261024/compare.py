#!/usr/bin/env python3
"""Compare two run labels on the same tasks and summarize failure modes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUN_DIR = Path(__file__).parent


def load(label: str) -> dict[str, dict]:
    out = {}
    for f in sorted((RUN_DIR / "transcripts" / label).glob("*.json")):
        d = json.loads(f.read_text())
        out[d["task"]] = d
    return out


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    ra, rb = load(a), load(b)
    ids = sorted(set(ra) & set(rb))
    print(f"{'task':<12} {a:<8} {b:<8} delta")
    deltas = []
    for t in ids:
        va = ra[t]["reward"]
        vb = rb[t]["reward"]
        deltas.append(vb - va)
        mark = "  <-- flipped" if va != vb else ""
        print(f"{t:<12} {va:<8.2f} {vb:<8.2f} {vb-va:+.2f}{mark}")
    only_a = set(ra) - set(rb)
    only_b = set(rb) - set(ra)
    if only_a:
        print(f"only in {a}: {sorted(only_a)}")
    if only_b:
        print(f"only in {b}: {sorted(only_b)}")
    if ids:
        sa = sum(ra[t]["reward"] for t in ids) / len(ids)
        sb = sum(rb[t]["reward"] for t in ids) / len(ids)
        print(f"\navg {a}={sa:.3f}  {b}={sb:.3f}  delta={sb-sa:+.3f}  (n={len(ids)})")


if __name__ == "__main__":
    main()
