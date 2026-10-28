#!/usr/bin/env python3
"""External-knowledge validation: inject web-retrieved NetHack guidance.

Compares Eat task with vs without external knowledge (external-knowledge.md).
Baseline = no guidance; external = wiki-derived rules injected verbatim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUN_DIR = Path(__file__).parent
sys.path.insert(0, str(RUN_DIR.parents[1]))

from run_experiment import _episode

from _env import load_env


def main() -> None:
    load_env()
    from swaybot.llm_brain import LLMBrain

    external = (RUN_DIR / "external-knowledge.md").read_text(encoding="utf-8")
    results = []
    for arm, guidance in (("baseline", ""), ("external", external)):
        for ep in range(4):
            res = _episode("MiniHack-Eat-v0", LLMBrain(temperature=0.0, max_tokens=64),
                           guidance, 100)
            print(f"[{arm} ep{ep}] success={res['success']} steps={res['steps']}", flush=True)
            results.append({"arm": arm, "ep": ep, "success": res["success"], "steps": res["steps"]})
            with (RUN_DIR / "results-external.jsonl").open("a") as fh:
                fh.write(json.dumps(results[-1]) + "\n")

    for arm in ("baseline", "external"):
        rows = [r for r in results if r["arm"] == arm]
        succ = sum(r["success"] for r in rows)
        avg = sum(r["steps"] for r in rows) / len(rows)
        print(f"{arm}: success {succ}/{len(rows)} avg_steps={avg:.1f}")


if __name__ == "__main__":
    main()
