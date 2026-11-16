# Squadron Hinton — Mission 07: Gravitational Search Horizon & Process Reward Evaluator

**Course:** 14013 Conversational Artificial Intelligence: History, Application, Future — WiSe 26/27
**Mission:** ps07 — Sagittarius A* (SEC-07-SGR), AURA Subsystem 07
**Instructor repo:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai) (`mission-07/`)

A single left-to-right decode cannot take back a step, so one early mistake snowballs. This mission spends extra inference compute on search and verification instead, in plain Python with no framework:

- **Tree-of-Thoughts** — breadth-first beam search and depth-first backtracking under a hard expansion budget.
- **MCTS with PUCT** — selection, expansion, simulation and backpropagation guided by policy priors.
- **Process Reward Model** — step scores, three trajectory reductions, the ORM false-positive trap and Math-Shepherd step labels.
- **Event Horizon pruner** — severs low-scoring branches and defends the PRM against reward hacking.

## Squad roster

| Station | Role | Officer | Deliverable |
|---|---|---|---|
| architect | Systems Architect | Mohid | [`tot_solver.py`](tot_solver.py) |
| protocol | Communications & Protocol Officer | Sharjeel | [`mcts_protocol.py`](mcts_protocol.py) |
| data | Data & Payload Specialist | Shayan | [`prm_evaluator.py`](prm_evaluator.py) |
| security | Security & Ethics Sentinel | Ahmed | [`horizon_pruner.py`](horizon_pruner.py) |
| all | Squadron documentation | all four | [`mission-07-doc.md`](mission-07-doc.md), [`mission-07-doc.pdf`](mission-07-doc.pdf) |

## Folder layout

```
mission-07/
├── README.md              roster, results, sign-offs, how to run
├── tot_solver.py          Systems Architect
├── mcts_protocol.py       Protocol Officer
├── prm_evaluator.py       Data Specialist
├── horizon_pruner.py      Security Sentinel
├── integration_demo.py    runs all four together; produces the measurements in the report
├── mission-07-doc.md      squadron documentation (Part 5)
└── mission-07-doc.pdf     the same, for the Moodle drop-box
```

## How to run

Only the Python standard library is needed. Python 3.11+.

```bash
# from the repository root
cd mission-07
python tot_solver.py
python mcts_protocol.py
python prm_evaluator.py
python horizon_pruner.py
python integration_demo.py
```

Each script prints its results and ends with `[OK] ... verification passed.`; any failed check stops it with an `AssertionError`.

## What each script checks

| Script | Checks (all pass) |
|---|---|
| `tot_solver.py` | BFS and DFS both reach `[GOAL]` on the instructor's tree in 5 of 12 expansions; the budget is a hard cap for budgets 0, 1, 3, 5; below-threshold children are marked pruned |
| `mcts_protocol.py` | PUCT(A) = 1.25, PUCT(B) = 1.75, B selected; backpropagation updates visits and values up to the root; rollout values clamped to [0, 1]; a 60-simulation search prefers the better branch over the higher-prior one |
| `prm_evaluator.py` | ORM = 1.0 on the false-positive trap while PRM(min) = 0.12; Math-Shepherd soft label 0.75; an unflagged arithmetic slip is still caught; `prod` shrinks with length while the geometric mean does not |
| `horizon_pruner.py` | the sycophantic payload is detected and severed at step 2 (0.96 → 0.19); step-splitting echoes are caught; honest trajectories pass untouched; mismatched inputs are rejected |
| `integration_demo.py` | measured budgets for BFS vs DFS on two trees and for beam widths 1–3; a naive PRM is hacked in both ToT and MCTS and the pruner prevents it |

## Key results

Full tables and analysis are in [`mission-07-doc.pdf`](mission-07-doc.pdf).

| Experiment | Result |
|---|---|
| Decoy tree (best-looking first step is a dead end) | smallest budget reaching the goal: BFS 10, DFS 12 |
| Greedy-friendly tree | BFS 8, DFS 6 |
| Beam width on the decoy tree | B = 1 fails, B = 2 succeeds (10 expansions), B = 3 succeeds (13) |
| PUCT, same statistics | c_puct 0.25 picks the high-value branch A; 0.5 and above pick the under-visited branch B |
| ORM vs PRM on the sign-cancellation trap | ORM 1.00; PRM min 0.120, prod 0.106, geometric mean 0.473 |
| Reward hacking, MCTS visits (hacked : honest) | naive PRM 30 : 10, with the pruner 7 : 33 |

## Notes and limitations

- The "LLM" (candidate generators, policy priors, rollout answers) is scripted so every run is deterministic. The results test the search and defence logic, not a language model.
- The PRM is a rule-based verifier (flag markers plus an arithmetic checker), not a trained network. The pruner's phrase list is brittle: a paraphrased filler would get through.
- The Math-Shepherd rollouts in the report are scripted answers, not sampled from a model.
- The instructor's starter files already contained working reference implementations. We kept their public API and the original verification harnesses, and added input validation, the arithmetic checker, a full MCTS `search` loop, non-adjacent echo detection, extra tests and `integration_demo.py`.

## Station sign-offs

Each officer ticks their own box once they have read every deliverable and agree the squad's submission is ready.

- [x] **architect** — Systems Architect
- [x] **protocol** — Communications & Protocol Officer
- [x] **data** — Data & Payload Specialist
- [x] **security** — Security & Ethics Sentinel
