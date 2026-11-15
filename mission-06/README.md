# Squadron Hinton — Mission 06: Autonomous Warp State Machine & Deliberation Loop

**Course:** 14013 Conversational Artificial Intelligence: History, Application, Future — WiSe 26/27
**Mission:** ps06 — Eta Carinae & Homunculus Nebula (SEC-06-CAR), AURA Subsystem 06
**Instructor repo:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai) (`mission-06/`)

AURA has fallen back to System 1 reflexes: fast, greedy, and stuck repeating the same failed action. This mission upgrades her to a System 2 agent built only from the Python standard library (`enum`, `dataclasses`, `hashlib`, `json`), with no agent framework:

- **State machine** — a strict seven-state graph and a hard step budget (N ≤ 20) that always halts.
- **Tool protocol** — JSON-Schema validation of every tool call and an exception boundary that turns failures into observations.
- **MDP logging** — trajectories τ, empirical discounted return G₀ (γ = 0.95) and a transition-frequency matrix P(s′ | s, a).
- **Sentinel** — SHA-256 state-action hashing, Floyd's tortoise and hare, and an emergency tripwire that forces `HALTED`.

## Squad roster

| Station | Role | Officer | Deliverable |
|---|---|---|---|
| architect | Systems Architect | Mohid | [`01_Agent_State_Machine.py`](01_Agent_State_Machine.py), [`ps06_solution.ipynb`](ps06_solution.ipynb) |
| protocol | Communications & Protocol Officer | Sharjeel | [`02_Tool_Protocol_Dispatcher.py`](02_Tool_Protocol_Dispatcher.py) |
| data | Data & Payload Specialist | Shayan | [`03_MDP_Trajectory_Logger.py`](03_MDP_Trajectory_Logger.py) |
| security | Security & Ethics Sentinel | Ahmed | [`04_Infinite_Loop_Sentinel.py`](04_Infinite_Loop_Sentinel.py), [`04_System2_Warp_Pilot.py`](04_System2_Warp_Pilot.py) |

## Folder layout

```
mission-06/
├── README.md                         roster, results, sign-offs, how to run
├── ps06_common.py                    helper that imports the digit-prefixed station files
├── 01_Agent_State_Machine.py         Systems Architect
├── 02_Tool_Protocol_Dispatcher.py    Protocol Officer
├── 03_MDP_Trajectory_Logger.py       Data Specialist
├── 04_Infinite_Loop_Sentinel.py      Security Sentinel: hashing, Floyd, tripwire
├── 04_System2_Warp_Pilot.py          Security Sentinel: System 2 audit + integrated nebula flight
└── ps06_solution.ipynb               all parts integrated, executed, with the four test suites
```

## How to run

No LLM, API key, GPU or third-party package is needed for the scripts. Python 3.11+.

```bash
# from the repository root
cd mission-06
python 01_Agent_State_Machine.py
python 02_Tool_Protocol_Dispatcher.py
python 03_MDP_Trajectory_Logger.py
python 04_Infinite_Loop_Sentinel.py
python 04_System2_Warp_Pilot.py          # flies the ship three ways and prints the transcripts
```

Or open `ps06_solution.ipynb` and run all cells (a few seconds). It needs `notebook` or VS Code with the Jupyter extension; `pip install -r requirements.txt` from the repository root covers it.

## Design notes

**State machine.** `REFLECTING` is the only state with two successors (`DELIBERATING` while the mission is open, `HALTED` once `mission_completed`). `HALTED` is absorbing. The call that would be transition `max_steps + 1` forces `HALTED` without transitioning, so no more than 20 transitions ever run. `max_steps` outside [1, 20] raises `ValueError`.

**Dispatcher.** The validator is hand-written (types, `required`, `additionalProperties`, bounds, `minLength`). Python's `bool` is rejected as a number or integer, and `NaN`/`inf` are rejected as numbers. Payloads may be dicts or JSON strings; malformed JSON, unknown tools, schema violations and tool exceptions all come back as `Tool Execution Error: <msg>`, so the dispatcher never raises. The shield range check is left in the tool (not the schema) so the `ValueError` path of the error boundary is real.

**MDP.** Rewards are one per action, so G₀ = Σ γᵗ rₜ runs over the actions and the terminal state s_T carries none. A backward recursion G_t = r_t + γ G_{t+1} cross-checks the explicit sum.

**Sentinel.** Two detectors run on every planned (state, action, arguments): a hash counter (3 identical → trip) and Floyd's algorithm on the graph hash_i → hash_{i+1}. Floyd only fires once the tail of the trajectory has gone round the cycle twice, and hashes with conflicting successors add no edge, so one legitimate revisit never halts the ship. The tripwire calls `Agent.halt(...)` and appends a `CRITICAL` entry to `sentinel.violations`.

## What the tests check

| Suite | Checks (all pass) |
|---|---|
| 1 State machine | `IDLE → … → HALTED` in 6 steps with the exact state trace; `REFLECTING` loops back to `DELIBERATING`; 21 forced steps end in `HALTED` with `step_count == 20`; `HALTED` is absorbing; budgets 0 and 21 are rejected |
| 2 Protocol dispatch | invalid JSON payload returns a formatted string and does not crash; 12 malformed calls all return error strings; tool `ValueError` is caught; valid dict and JSON-string calls succeed |
| 3 MDP return | rewards `[10, 0, -5, 20]`, γ = 0.95 → G₀ = 22.635 (explicit sum, backward recursion and logger agree); τ tuple and transitions; matrix rows sum to 1 and match the generating kernel within 0.06 |
| 4 Sentinel cycle | `[s1, s2, s3, s1, s2, s3, …]` gives Floyd (μ, λ) = (0, 3) and halts the agent at trajectory step 6, before 10, with the hash counter disabled; `abcdecde` gives (2, 3); no false trips on a non-repeating or once-revisited trajectory; 3 identical failed actions trip on the third |

## Key results

The same ship and the same scripted policy are flown three ways (`python 04_System2_Warp_Pilot.py`). Its first two reflexes are bad: "lower shields to conserve energy" and "punch it straight ahead".

| Mode | Outcome | Hull | Steps | G₀ | Halted by |
|---|---|---:|---:|---:|---|
| System 1 only | not escaped | 40% | 20 | −19.00 | step budget |
| System 1 + sentinel | not escaped | 40% | 18 | −27.15 | emergency tripwire (3 identical scans) |
| System 2 + sentinel | **escaped** | 100% | 18 | +45.68 | mission completed |

System 2 clamps the 20% shield request to the 60% floor, rejects the blind warp with a reason the policy can read ("no verified escape corridor"), then the reflective policy scans `escape_corridor` and warps along the returned vector.

## Notes and limitations

- The "LLM" in the flight simulation is a deterministic script, not a model. The result shows the control architecture (budget, audit, tripwire), not the quality of any language model.
- The nebula dynamics, rewards and the 60% shield floor are our own modelling choices; the brief does not fix them.
- The brief asks for a "provided historical dataset of past jumps" but none ships with it. The transition matrix is estimated from a **synthetic** log of 3,200 jumps sampled from a known kernel (seed 6), which lets the tests check the estimate against the truth.
- The brief's Part 3 (MDP) and Part 4 (sentinel) differ from the instructor README's file list, which names a loop sentinel as the Data Specialist's file and the System 2 pilot as the Security Sentinel's. We followed the briefing's role split.

## Station sign-offs

Each officer ticks their own box once they have read every deliverable and agree the squad's submission is ready.

- [x] **architect** — Systems Architect
- [x] **protocol** — Communications & Protocol Officer
- [x] **data** — Data & Payload Specialist
- [x] **security** — Security & Ethics Sentinel
