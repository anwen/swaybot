#!/usr/bin/env python3
"""Knowledge-transfer A/B on tau3 banking_knowledge (architecture validation).

Phases:
  0 baseline  — test-set without any learned guidance            → score B0
  1 learning  — learn-set WITH progressively accumulated lessons:
               before each task, dump memory theories to the guidance
               file; after each task, extract <=2 durable rules from the
               transcript + reward into swaybot's MemoryStore
                                                                → score L
  2 transfer  — test-set again WITH the full learned file         → score B1

Signal: B1 - B0 on identical tasks (temperature 0, fixed order).
Lessons live in swaybot's MemoryStore (merge/corroboration included);
the guidance file is the projection injected via the adapter hook.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RUN_DIR = Path(__file__).parent
GUIDANCE_FILE = RUN_DIR / "learned-guidance.md"
MEMORY_FILE = RUN_DIR / "memory.json"
RESULTS = RUN_DIR / "results.jsonl"

DOMAIN = "banking_knowledge"
LEARN_IDS = ["task_001", "task_002", "task_003", "task_004", "task_005"]
TEST_IDS = ["task_006", "task_007", "task_008", "task_010", "task_012"]


def _env_from_dotenv():
    for line in Path(os.environ.get("SWAYBOT_ENV", ".env")).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _setup():
    _env_from_dotenv()
    model = os.environ["SWAYBOT_MODEL"]
    os.environ["OPENAI_API_KEY"] = os.environ["SWAYBOT_API_KEY"]
    os.environ["OPENAI_API_BASE"] = os.environ["SWAYBOT_API_BASE"]
    os.environ["TAU2_DATA_DIR"] = os.path.expanduser(
        "~/.cache/modelscope/datasets/evalscope--tau3-bench-data/snapshots/master"
    )
    os.environ.setdefault("SWAYBOT_LITELLM_TIMEOUT", "180")

    from benchmarks.tau3bench import registry  # noqa: F401
    from tau2.run import get_tasks
    from tau2.run import run_single_task
    from tau2.data_model.simulation import TextRunConfig

    def make_config():
        return TextRunConfig(
            domain=DOMAIN,
            agent="swaybot",
            user="user_simulator",
            llm_agent=model,
            llm_user=f"openai/{model}",
            max_steps=50,
            num_trials=1,
            seed=0,
            # 新 tau2 的 golden_retrieval 需要 srt 沙箱二进制；bm25 无 shell
            # 依赖。A/B 两臂同用 bm25，实验内部同口径（与八月历史不可比）。
            retrieval_config="bm25",
        )

    tasks = {t.id: t for t in get_tasks(DOMAIN)}
    return make_config, run_single_task, tasks


def _reward(result) -> float:
    info = getattr(result, "reward_info", None)
    return float(getattr(info, "reward", 0.0)) if info else 0.0


def _transcript(result, limit=6000) -> str:
    msgs = getattr(result, "messages", None) or []
    out = []
    for m in msgs:
        role = getattr(m, "role", "?")
        text = json.dumps(getattr(m, "content", ""), ensure_ascii=False)
        out.append(f"{role}: {text[:400]}")
    joined = "\n".join(out)
    return joined[:limit]


def _memory():
    from swaybot.memory import MemoryStore

    return MemoryStore(path=MEMORY_FILE)


def _write_guidance(store):
    theories = store.query(kind="theory", scope="long_term", limit=8)
    lines = [
        f"- {m.content}" for m in theories if not str(m.content).startswith("Run '")
    ]
    GUIDANCE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ask(prompt: str) -> str:
    from swaybot.llm_brain import LLMBrain
    from swaybot.parsing import strip_markdown_fences

    brain = LLMBrain(temperature=0.0)
    perception = {
        "task": prompt[:200],
        "step": 0,
        "max_steps": 2,
        "messages": [{"role": "user", "content": prompt}],
        "tool_descriptions": [],
    }
    action = brain.think(perception, ["final_answer"])
    if action.get("name") == "final_answer":
        return strip_markdown_fences(str(action.get("args", {}).get("answer", "")))
    return ""


def _extract_lessons(task_id: str, transcript: str, reward: float, store):
    verdict_word = "PASS" if reward >= 0.99 else ("PARTIAL" if reward > 0 else "FAIL")
    answer = _ask(
        "You just ran a banking customer-service task. Learn from it.\n"
        f"Task: {task_id}. Final reward: {reward:.2f} ({verdict_word}).\n"
        f"Transcript (truncated):\n{transcript}\n\n"
        "State up to 2 DURABLE domain rules that would help on FUTURE tasks "
        "in this domain (tool usage patterns, policy facts, user-need "
        "interpretation, common failure modes). One rule per line, each "
        "starting with '- '. Only rules you actually saw evidence for; "
        "if the task failed, the rule may be about what to do differently."
    )
    rules = [
        line[2:].strip()
        for line in answer.splitlines()
        if line.strip().startswith("- ") and len(line.strip()) > 12
    ][:2]
    from swaybot.memory import Memory

    for rule in rules:
        store.add(
            Memory(
                content=rule,
                kind="theory",
                scope="long_term",
                credibility=0.6,
                tags=[DOMAIN],
                source="tau3-experiment",
            )
        )
    return rules


def run_phase(name, task_ids, make_config, run_single_task, tasks, store, learn: bool):
    rewards = []
    for tid in task_ids:
        if learn:
            _write_guidance(store)
            os.environ["SWAYBOT_EXTRA_GUIDANCE_FILE"] = str(GUIDANCE_FILE)
        else:
            os.environ.pop("SWAYBOT_EXTRA_GUIDANCE_FILE", None)
        try:
            result = run_single_task(make_config(), tasks[tid])
            reward = _reward(result)
            transcript = _transcript(result) if learn else ""
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] {tid}: ERROR {type(exc).__name__}: {exc}", flush=True)
            reward, transcript = 0.0, ""
        rewards.append(reward)
        line = {"phase": name, "task": tid, "reward": reward}
        if learn:
            rules = _extract_lessons(tid, transcript, reward, store) if transcript else []
            line["learned"] = rules
        with RESULTS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"[{name}] {tid}: reward={reward:.2f}", flush=True)
    score = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"== phase {name}: avg={score:.3f} ==", flush=True)
    return score


def main():
    make_config, run_single_task, tasks = _setup()
    store = _memory()
    b0 = run_phase("baseline", TEST_IDS, make_config, run_single_task, tasks, store, learn=False)
    learn_score = run_phase("learning", LEARN_IDS, make_config, run_single_task, tasks, store, learn=True)
    b1 = run_phase("transfer", TEST_IDS, make_config, run_single_task, tasks, store, learn=True)
    print(
        f"\nRESULT: baseline={b0:.3f} learning={learn_score:.3f} "
        f"transfer={b1:.3f} delta={b1 - b0:+.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
