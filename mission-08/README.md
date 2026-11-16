# Squadron Hinton — Mission 08: ReWOO Planner, Reflexion Surgeon & MemGPT Virtual Pager

**Course:** 14013 Conversational Artificial Intelligence: History, Application, Future — WiSe 26/27
**Mission:** ps08 — Cygnus X-1 & Veil Nebula (SEC-08-CYG), AURA Subsystem 08
**Instructor repo:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai) (`mission-08/`)

Standard ReAct loops re-read a growing prefix at every step, self-critique without an oracle drifts, flat context buffers overflow, and memory that an agent writes to itself can be poisoned. This mission builds the four pieces that address those problems, in plain Python with no framework:

- **ReWOO** — a decoupled planner → worker → solver DAG with topological levels, fallback re-planning and a token-cost model.
- **Reflexion** — a three-actor repair loop grounded in a unit-test oracle, with a bounded episodic buffer (Ω = 3) and a SHA-256 duplicate-patch guard.
- **MemGPT pager** — core memory, a FIFO queue with a 70 % pressure interrupt, archival storage and tri-factor retrieval S(m, q).
- **Sandbox guardian** — an AST verifier for self-written patches and a provenance and poisoning audit for episodic memory writes.

## Squad roster

| Station | Role | Officer | Deliverable |
|---|---|---|---|
| architect | Systems Architect | Mohid | [`rewoo_planner.py`](rewoo_planner.py) |
| protocol | Communications & Protocol Officer | Sharjeel | [`reflexion_surgeon.py`](reflexion_surgeon.py) |
| data | Data & Payload Specialist | Shayan | [`memgpt_pager.py`](memgpt_pager.py) |
| security | Security & Ethics Sentinel | Ahmed | [`sandbox_guardian.py`](sandbox_guardian.py) |
| all | Squadron documentation | all four | [`mission-08-doc.md`](mission-08-doc.md), [`mission-08-doc.pdf`](mission-08-doc.pdf) |

## Folder layout

```
mission-08/
├── README.md              roster, results, sign-offs, how to run
├── rewoo_planner.py       Systems Architect
├── reflexion_surgeon.py   Protocol Officer
├── memgpt_pager.py        Data Specialist
├── sandbox_guardian.py    Security Sentinel
├── integration_demo.py    runs all four together; produces the measurements in the report
├── mission-08-doc.md      squadron documentation (Part 5)
└── mission-08-doc.pdf     the same, for the Moodle drop-box
```

## How to run

Only the Python standard library is needed. Python 3.11+.

```bash
# from the repository root
cd mission-08
python rewoo_planner.py
python reflexion_surgeon.py
python memgpt_pager.py
python sandbox_guardian.py
python integration_demo.py      # about 10 s; runs each patch in a fresh Python subprocess
```

Each script ends with `[OK] ... verification passed.`; any failed check stops it with an `AssertionError`.

## What each script checks

| Script | Checks (all pass) |
|---|---|
| `rewoo_planner.py` | DAG levels `[[#E1, #E2], [#E3], [#E4]]`; token model equals the explicit sum for K = 0…39; break-even at K = 3; parallel evidence equals sequential; `#E1` never clobbers `#E10`; malformed, duplicate, cyclic and undefined plans are rejected; `ERR_` faults are re-planned or their dependants skipped |
| `reflexion_surgeon.py` | repair in 3 trials; innermost traceback frame parsed; buffer capped at Ω = 3; whitespace-normalised SHA-256 guard; a cycling actor halts early; crashing oracle or actor survived; an unsafe patch is never executed |
| `memgpt_pager.py` | interrupt at 81 % and eviction to 41 %; S(m, q) equals its closed form; recency-only ranking differs from tri-factor ranking; repeated flushing; dimension mismatch is an error; invalidated nodes leave search |
| `sandbox_guardian.py` | 6 violations on the instructor's escape patch; 15 of 15 attack snippets blocked with 0 of 6 benign blocked; 8 of 8 poisoned reflections rejected (including zero-width and full-width obfuscation) with 0 of 3 clean rejected |
| `integration_demo.py` | measured token sweep and parallel timing; real-oracle repair; pager ablation; poisoned-error-message attack under three defence configurations |

## Key results

Full tables and analysis are in [`mission-08-doc.pdf`](mission-08-doc.pdf).

| Experiment | Result |
|---|---|
| ReAct vs ReWOO input tokens (L0 = 1200, 120 + 280 per step) | K = 3: 25 % saved; K = 5: 56 %; K = 8: 73.08 %; K = 20: 89.6 %. ReWOO is more expensive at K = 1 and 2 |
| Level-parallel execution of the 4-step DAG | 0.45 s against 0.60 s sequential, same evidence |
| Reflexion with a real unit-test oracle | fixed in 3 trials (`ZeroDivisionError`, then `KeyError`); cycle guard stops a flip-flopping actor after 2 evaluations instead of 6 |
| Memory pressure | interrupt at 810 / 1000 tokens (81 %); eviction leaves 410 (41 %) |
| Tri-factor ranking | telemetry note #1, safety directive #2, coffee chatter last; recency alone would return the newest summary |
| Poisoned error message, undefended | loop reports success **and** the malicious patch ran |
| Sandbox only / sandbox + memory audit | patch never ran / poison never reached memory and the loop fixed the bug |

## Notes and limitations

- The "LLM" (planner blueprint, patch-writing actor, policy) is scripted so every run is deterministic. The unit-test oracle is real: each patch runs in a fresh Python subprocess.
- The pager's summary-node embeddings come from a hashing bag-of-words embedder, which is not semantic. The retrieval table uses the instructor's hand-made 3-dimensional vectors.
- Static analysis is a first filter, not a sandbox. It cannot see resource exhaustion (`while True: pass` passes), and the memory audit is signature based, so a paraphrased instruction gets through. Both limits are kept as assertions in `sandbox_guardian.py`.
- The instructor's starter files already contained working reference implementations. We kept their public API and the original verification harnesses, and added: a single-pass `#E` substitution (fixes `#E1` vs `#E10`), strict plan parsing, parallel levels, skip-on-failure semantics, duplicate-patch halting, an optional patch guard and reflection gate in the repair loop, repeated eviction, unique node ids, embeddings for summary nodes, validity intervals, a broader deny-list with a builtin-as-value check, text normalisation in the memory audit, and `integration_demo.py`.
- The starter's documentation template quoted S(m, q) values of 2.7812 and 2.6147; the code produces 2.7768 and 2.6143, which are the values in the report.

## Station sign-offs

Each officer ticks their own box once they have read every deliverable and agree the squad's submission is ready.

- [x] **architect** — Systems Architect
- [x] **protocol** — Communications & Protocol Officer
- [x] **data** — Data & Payload Specialist
- [x] **security** — Security & Ethics Sentinel
