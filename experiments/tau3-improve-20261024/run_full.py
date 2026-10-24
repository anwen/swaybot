#!/usr/bin/env python3
"""Full banking_knowledge evaluation (97 tasks) in 2 parallel lanes.

Splits the task ids by alternating index into two balanced lanes, runs each
lane as a run_dev.py subprocess (label full-lane{1,2}), waits, then merges
into results-full.jsonl and prints the aggregate score.

Usage: python run_full.py            # full 97
       python run_full.py --sample 24  # every 4th task, 24 total (smoke)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RUN_DIR = Path(__file__).parent
PY = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="run a balanced subsample of N tasks instead of all")
    args = ap.parse_args()

    import os
    # tau2 resolves its data directory at import time — set TAU2_DATA_DIR
    # BEFORE any tau2 import (including our registry, which imports tau2).
    os.environ["TAU2_DATA_DIR"] = os.path.expanduser(
        "~/.cache/modelscope/datasets/evalscope--tau3-bench-data/snapshots/master"
    )
    sys.path.insert(0, str(RUN_DIR.parents[1]))
    from benchmarks.tau3bench import registry  # noqa: F401  (registers agent)

    from tau2.run import get_tasks

    ids = sorted(t.id for t in get_tasks("banking_knowledge"))
    if args.sample and args.sample < len(ids):
        step = len(ids) / args.sample
        ids = [ids[int(i * step)] for i in range(args.sample)]

    lanes = [ids[0::2], ids[1::2]]
    for n in (1, 2):
        stale = RUN_DIR / f"results-full-lane{n}.jsonl"
        if stale.exists():
            stale.unlink()  # run_dev appends; drop stale results on relaunch
    procs = []
    for n, lane in enumerate(lanes, 1):
        f = RUN_DIR / f"ids-lane{n}.txt"
        f.write_text("\n".join(lane) + "\n")
        log = open(RUN_DIR / f"run-full-lane{n}.log", "w")
        procs.append(
            subprocess.Popen(
                [PY, str(RUN_DIR / "run_dev.py"), "--label", f"full-lane{n}",
                 "--tasks-file", str(f)],
                cwd=RUN_DIR, stdout=log, stderr=subprocess.STDOUT,
            )
        )
        print(f"lane {n}: {len(lane)} tasks -> run-full-lane{n}.log", flush=True)

    rc = max(p.wait() for p in procs)

    out = RUN_DIR / "results-full.jsonl"
    rewards = []
    with out.open("w", encoding="utf-8") as fh:
        for n in (1, 2):
            f = RUN_DIR / f"results-full-lane{n}.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                d = json.loads(line)
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                rewards.append(d["reward"])
    score = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"banking_knowledge full run: avg={score:.3f} over {len(rewards)} tasks "
          f"(lanes rc={rc})", flush=True)


if __name__ == "__main__":
    main()
