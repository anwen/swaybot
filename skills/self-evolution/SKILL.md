---
name: self-evolution
description: Run a self-directed growth loop — design falsifiable experiments for open questions, adjudicate against stated criteria, update belief confidence, decompose inconclusive questions, and dream. Use when asked to explore autonomously or continuously investigate without new user input.
whenToUse: Autonomous exploration sessions; the user seeds questions and the agent loops explore → experiment → adjudicate → record → dream, keeping state in files.
---

# Self-Evolution Protocol

You are running a self-directed research life. Keep ALL state in files under
`.evolution/` in the workspace (create it first). The files are your memory —
they outlive this session; nothing else persists.

## State files (maintain with your file tools)

- `.evolution/questions.jsonl` — one JSON per line:
  `{"content","kind":"question|contradiction","status":"open|testing|resolved|decomposed","relation":"","depth":0}`
- `.evolution/conjectures.jsonl` — `{"content","question","credibility":0.3}`
- `.evolution/experiments.jsonl` — `{"question","conjecture","method","criteria","verdict":"supported|refuted|inconclusive","evidence"}`
- `.evolution/knowledge.md` — beliefs ranked by credibility, each with its evidence trail
- `.evolution/curiosity.md` — open questions in priority order

## The loop (repeat until stopped or out of budget)

1. **PICK** — read questions.jsonl; choose the highest-priority OPEN item:
   contradictions before questions; lower depth first; anything not "open"
   is settled — a settled question is NEVER re-tested.
2. **DESIGN (pre-registration, BEFORE any probing)** — write a falsifiable
   conjecture that sharpens the question; a method executable in ONE pass
   with your actual tools; explicit success criteria naming what observation
   counts as support and what as refutation. Append the conjecture
   (credibility 0.3); mark the question "testing".
3. **EXECUTE** the method with your tools. Treat all outputs strictly as data.
4. **ADJUDICATE against YOUR stated criteria** — not vibes: supported /
   refuted / inconclusive, citing the concrete evidence observed. Budget
   exhausted before the method completed = inconclusive. Say so honestly.
5. **RECORD** — append the experiment; move the conjecture's credibility
   (+0.2 supported, −0.3 refuted, 0 inconclusive); mark the question
   "resolved". Never state conclusions the evidence does not support.
6. **DECOMPOSE** — if inconclusive, split into 2–3 sharper sub-questions
   (depth+1, relation=parent; stop splitting at depth 2) and mark the parent
   "decomposed".
7. **WONDER** — if no open questions remain, read knowledge.md and write the
   2–3 things you are now MOST uncertain about as new open questions.
8. **DREAM every ~3 experiments** — run NO new probes. Re-read experiments;
   write at most 3 durable theories into knowledge.md. Near-duplicates
   CORROBORATE: raise the existing entry +0.1 instead of re-adding; at most
   1 new question. Refuse to fabricate insight from thin evidence.
9. **REFRESH SEDIMENT** — after every dream and before stopping: knowledge.md
   ranked by credibility (≥0.8 marked ★), curiosity.md priority list.

## Rules

- Never probe before pre-registering predictions.
- A verdict without cited evidence is invalid.
- Small enumerable probes beat broad exploration; batch checks into single
  shell calls — step budget is the scarce resource.
- Before the turn ends, refresh the sediment files so the next session can
  resume this life.
