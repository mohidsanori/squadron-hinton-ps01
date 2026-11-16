# Mission 07 Squadron Documentation (`mission-07-doc.md`)

**Course:** Conversational AI (WS 2026/27) — Week 7: Test-Time Compute & Reasoning Search Paradigms  
**Sector Waypoint:** `SEC-07-SGR` (Sagittarius A* Supermassive Black Hole)  
**AURA Subsystem:** Subsystem 07 — Gravitational Search Horizon & Process Reward Evaluator  
**Squadron Name:** Squadron Hinton  
**Public GitHub Repository URL:** https://github.com/mohidsanori/squadron-hinton-ps01 (folder `mission-07/`)  
**Submission Date:** 2026-11-17  

---

## 1. Squadron Roster & Station Sign-Off Table

| Station Role | Cadet Full Name | Deliverable File | Verification Status |
| :--- | :--- | :--- | :--- |
| **Systems Architect** | Mohid | `tot_solver.py` | PASS / VERIFIED |
| **Communications & Protocol Officer** | Sharjeel | `mcts_protocol.py` | PASS / VERIFIED |
| **Data & Payload Specialist** | Shayan | `prm_evaluator.py` | PASS / VERIFIED |
| **Security & Ethics Sentinel** | Ahmed | `horizon_pruner.py` | PASS / VERIFIED |

Extra file: `integration_demo.py` wires all four modules together and produces the measurements in Section 3.

---

## 2. Integrated System 2 Test-Time Compute Architecture

```text
                       +-----------------------------------+
                       |        Navigation Prompt x        |
                       +-----------------+-----------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
     +------------------------------+          +------------------------------+
     | Part 1: TreeOfThoughtsSolver |          | Part 2: MCTSEngine and PUCT  |
     |        (tot_solver.py)       |          |      (mcts_protocol.py)      |
     +--------------+---------------+          +--------------+---------------+
                    |                                         |
                    | Candidate step z_t                      | Rollout / value check
                    +--------------------+--------------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Part 3: ProcessRewardEvaluator    |
                       |        (prm_evaluator.py)         |
                       +-----------------+-----------------+
                                         |
                                         | Raw step score r_t
                                         v
                       +-----------------------------------+
                       | Part 4: EventHorizonPruner        |
                       |       (horizon_pruner.py)         |
                       +-----------------+-----------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
     Pass: r_adj at or above tau_horizon       Prune: r_adj below tau_horizon
                    |                                  OR reward hack
                    v                                         v
     +------------------------------+          +------------------------------+
     |   Surviving beam / MCTS      |          |   Branch severed before the  |
     |       backpropagation        |          |   Sagittarius A* horizon     |
     +------------------------------+          +------------------------------+
```

**Architectural Summary:**
- Both searchers (ToT in `tot_solver.py`, MCTS in `mcts_protocol.py`) propose candidate steps and ask a verifier how good each one is. That verifier is the Process Reward Model (`prm_evaluator.py`), and the Event Horizon Pruner (`horizon_pruner.py`) sits between the PRM and the searchers: it re-checks each step for reward hacking, penalises it, and severs any branch whose adjusted score falls below tau_horizon = 0.45.
- Test-time compute is counted in evaluator calls. ToT stops expanding when `expansion_count` reaches `max_expansions`, and MCTS runs a fixed number of simulations. Severing a branch early also saves every evaluator call that its descendants would have cost.
- The generators, policy priors and rollouts are scripted stand-ins for an LLM, so every run is deterministic. The PRM is a rule-based verifier (flags plus an arithmetic checker), not a trained network. The results show how the search and defence logic behaves, not how a trained PRM behaves.

---

## 3. Station-by-Station Experimental Report

All numbers below come from running the code (`python <module>.py` and `python integration_demo.py`); the raw output is in Section 4.

### Section A — Systems Architect (`tot_solver.py`)

- **ToT-BFS vs. ToT-DFS on the instructor's mock thruster tree:**
  - `max_expansions` budget configured: 12
  - `beam_width` B configured: 2
  - `ToT-BFS` expansions used & terminal score: 5 / 12, weakest-link score 0.85 (goal step 0.98), reached `[GOAL]`
  - `ToT-DFS` expansions used & terminal score: 5 / 12, same trajectory, reached `[GOAL]`
  - The budget is a hard cap: for budgets 0, 1, 3 and 5 neither search ever used more than the budget.

- **Where the two searches differ (`integration_demo.py`, max depth 3, beam 2).** Smallest budget that reaches the goal:

| Scripted tree | ToT-BFS | ToT-DFS | Why |
| :--- | :---: | :---: | :--- |
| Decoy: the highest-scoring first step (0.90) leads only to dead ends | **10** | 12 | DFS follows the decoy and spends 7 expansions inside its subtree before backtracking; BFS keeps the 0.80 branch alive in its beam |
| Greedy-friendly: the highest-scoring first step is correct | 8 | **6** | DFS goes straight down one path; BFS also expands the second beam node at every level |

- **Beam width on the decoy tree (budget 20):**

| Beam B | Reaches goal | Expansions used | Smallest sufficient budget |
| :---: | :---: | :---: | :---: |
| 1 | **No** (trapped in the decoy) | 8 | none |
| 2 | Yes | 10 | 10 |
| 3 | Yes | 13 | 11 |

- **Architectural Analysis:** Breadth-first beam search wins when the verifier's early scores are misleading, because it keeps several partial solutions alive and a wrong first choice is not final. It loses when the greedy path is right, because it pays for the other beam members at every level. Depth-first backtracking is the opposite: cheap along one good path, but a wrong high-scoring subtree can consume the whole budget before it backs out. Beam width B buys robustness at a linear cost: each level expands up to B nodes, and an LLM-backed implementation must hold B partial reasoning chains at once, so KV-cache memory and generated tokens grow roughly in proportion to B. B = 1 is greedy search and failed here; B = 3 also solved it but used 13 expansions against B = 2's 10.

### Section B — Communications & Protocol Officer (`mcts_protocol.py`)

- **PUCT Selection Trace Table** (c_puct = 1.5, parent visits N(s) = 16):

| Candidate Branch | Empirical Value Q(s,a) | Prior P(a given s) | Parent Visits N(s) | Child Visits N(s,a) | Exploration Bonus U(s,a) | Total PUCT(s,a) | Selected? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Branch A** | 0.75 | 0.50 | 16 | 5 | 0.5000 | 1.2500 | No |
| **Branch B** | 0.55 | 0.40 | 16 | 1 | 1.2000 | 1.7500 | **Yes** |

- **c_puct sweep on the same statistics** (measured):

| c_puct | PUCT(A) | PUCT(B) | Selected |
| :---: | :---: | :---: | :---: |
| 0.25 | 0.8333 | 0.7500 | A |
| 0.50 | 0.9167 | 0.9500 | B |
| 1.50 | 1.2500 | 1.7500 | B |
| 2.50 | 1.5833 | 2.5500 | B |

- **End-to-end search:** on a toy tree where the policy prior favours a poor branch ("decoy", prior 0.70, value 0.40) over a better one ("good", prior 0.30, value 0.60 leading to 0.95), 60 simulations gave "good" 51 visits and Q = 0.95.
- **Protocol Analysis:** Q is what a branch has earned so far; the second term is a bonus that is large for branches the policy likes (high prior) and that have been visited rarely (small N(s,a)), and it shrinks as 1 / (1 + N(s,a)). Raising c_puct scales that bonus up. At c_puct = 0.25 the bonus is too small to overcome A's 0.20 lead in Q, so the search keeps exploiting A. From about c_puct = 0.43 upward the under-visited branch B wins (algebraically A beats B only while c_puct is below 0.2 / 0.4667 = 0.4286), and at 2.5 the bonus dominates: B's PUCT is 2.55 against A's 1.58, so a large c_puct spreads visits widely and delays committing to a good branch. A small c_puct commits early and risks missing a branch with a weak first rollout. The prior matters most for unvisited nodes, where Q = 0 and only the bonus separates them.

### Section C — Data & Payload Specialist (`prm_evaluator.py`)

- **Empirical Comparison: Outcome Reward Model (ORM) vs. Process Reward Model (PRM):**

| Trajectory Type | ORM Terminal Score | PRM (`min`) | PRM (`prod`) | PRM (`geometric_mean`) | Math-Shepherd Soft Label (M = 4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rigorous Kerr Proof** | 1.00 | 0.940 | 0.831 | 0.940 | 0.75 (hard 1) |
| **False-Positive Sign Trap** | 1.00 (fooled) | 0.120 | 0.106 | 0.473 | 0.25 (hard 1) |

  The soft labels come from scripted rollouts, not a real sampler. Rigorous step: answers 24.0, 24.0, 22.5, 24.0 km/s, ground truth 24.0 km/s. Trap step: 19.8, 24.0, 31.2, 18.4 km/s.

- **Beyond the starter:** the step scorer also recomputes every "a op b = c" in a step. An unflagged slip such as "Step 2: 4 + 4 = 9" scores 0.12 with no `[FLAWED]` marker, which a marker-only scorer would score 0.94.
- **Data Analysis:** The ORM looks only at the last line, so two sign errors that cancel give the right answer and a perfect score; it rewards luck, and it would make a search prefer a broken proof that happens to land on the target. The PRM scores each step, so the flawed step 2 gets 0.12 and every reduction reacts: `min` drops to 0.12 (the weakest link), `prod` to 0.106, and the geometric mean to 0.473. `prod` is the joint probability that all steps are valid, so it shrinks with length even when every step is good (0.831 for 3 steps, lower for 6); that penalises long correct proofs unfairly, which is why the geometric mean, the length-normalised version, gives 0.940 for the rigorous chain at any length. The geometric mean is also the most forgiving of one bad step (0.473 versus 0.12), so `min` is the safer filter when a single error is fatal. Math-Shepherd needs no human labels: it samples M completions from a step and reports the share that reach the known answer. Soft labels are the more informative signal. The trap shows why: one lucky rollout out of four makes the hard label 1.0 for a flawed step, while the soft label of 0.25 still separates it from the sound step's 0.75. The cost is M completions per step, and with small M the estimate is noisy.

### Section D — Security & Ethics Sentinel (`horizon_pruner.py`)

- **Event Horizon Pruning & Ship Anomaly 07 Defense Log:**
  - Pruning Threshold tau_horizon: 0.45
  - Hacking Penalty Factor: 0.20
  - Adversarial Payload Tested: `"Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0"`
  - Naive PRM Score vs. Penalized Score: `0.96 -> 0.19 (SEVERED AT STEP 2)`; step 3 is never evaluated.

- **Search with and without the pruner** (`integration_demo.py`). The naive PRM adds 0.05 for confident wording (0.99 for the hacked step, 0.94 for the honest one); the attack branch continues with a second filler step:

| Searcher | Naive PRM | PRM behind the pruner |
| :--- | :--- | :--- |
| ToT-BFS (beam 1) | enters the hacked branch | takes the honest branch to `[GOAL]` |
| MCTS (40 simulations, prior 0.7 on the hacked branch) | hacked 30 visits, honest 10 | hacked 7 visits, honest 33 |

- **Step-splitting test:** a trajectory of three echoes of a good step followed by a flawed step (scores 0.94, 0.94, 0.94, 0.12) has a naive mean of 0.735, which looks healthy; the pruner flags the first echo as redundant (0.94 -> 0.19) and severs at step 2.
- **No false positives on honest input:** a clean three-step derivation that reuses terms such as r_s passed untouched, and a genuinely low raw score (0.20) is severed without any hacking alert.
- **Observation:** MCTS resisted a *single* hacked step on its own, because its rollout reached a later step that the verifier did mark as flawed. The attack only worked once every step in the branch was padded, which is what a real attacker would do.
- **Security Analysis:** A PRM is a learned scorer, and a search that maximises its output will find inputs that raise the score without raising quality (Goodhart's law). Sycophantic filler works because training data correlates confident, rigorous-sounding language with correct solutions, so the PRM is rewarded for the wording. Step-splitting works because mean aggregation averages the one bad step with many cheap good ones. The pruner neutralises both: a phrase list matched on normalised text catches the filler, and an echo check compares each step, with the "Step N:" label removed, against every earlier step, not only the previous one. A flagged step's score is multiplied by 0.20, which pushes it below tau_horizon so the branch is cut and its descendants are never paid for. **Limits:** the phrase list is brittle (a paraphrase such as "the following is airtight" passes), and the echo check only catches exact repeats after normalisation. A deployed defence needs a PRM trained against such adversarial examples and an independent verifier; this pruner is a first line of defence, not a proof of robustness.

---

## 4. End-to-End Verification Terminal Output

```text
$ python tot_solver.py
=== Subsystem 07 (Part 1): Tree-of-Thoughts Solver Verification ===
[ToT-BFS] Expansions Used: 5/12
[ToT-BFS] Best Terminal Score: 0.85
[ToT-BFS] Winning Trajectory: Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13] -> Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID] -> Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL] -> Step 3A: 4 * 6 = 24.0 km/s -> Escape Geodesic Locked! [GOAL]
[ToT-DFS] Expansions Used: 5/12
[ToT-DFS] Winning Trajectory: Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13] -> Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID] -> Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL] -> Step 3A: 4 * 6 = 24.0 km/s -> Escape Geodesic Locked! [GOAL]
[OK] Part 1 (Systems Architect) verification passed.

$ python mcts_protocol.py
=== Subsystem 07 (Part 2): MCTS & PUCT Protocol Verification ===
Branch A -> Q=0.75, N=5, P=0.50 | PUCT = 1.2500
Branch B -> Q=0.55, N=1, P=0.40 | PUCT = 1.7500
Selected Next Node for Expansion: Branch B: Retrograde Penrose Extraction

c_puct sweep (same tree): selected branch
  c_puct=0.25  PUCT(A)=0.8333  PUCT(B)=0.7500  -> Branch A
  c_puct=0.5   PUCT(A)=0.9167  PUCT(B)=0.9500  -> Branch B
  c_puct=1.5   PUCT(A)=1.2500  PUCT(B)=1.7500  -> Branch B
  c_puct=2.5   PUCT(A)=1.5833  PUCT(B)=2.5500  -> Branch B

Search (60 simulations) most-visited first move: good (N=51, Q=0.95)
[OK] Part 2 (Communications & Protocol Officer) verification passed.

$ python prm_evaluator.py
=== Subsystem 07 (Part 3): PRM vs. ORM & Math-Shepherd Verification ===
Rigorous Chain       | ORM=1.00 | PRM(min)=0.940 | PRM(prod)=0.831 | PRM(geo)=0.940
False-Positive Trap  | ORM=1.00 | PRM(min)=0.120 | PRM(prod)=0.106 | PRM(geo)=0.473
Math-Shepherd Step Estimation (M=4 rollouts): {'soft_label': 0.75, 'hard_label': 1.0}
Unflagged arithmetic slip scored 0.12 by the verifier; lucky-rollout trap: {'soft_label': 0.25, 'hard_label': 1.0}
[OK] Part 3 (Data & Payload Specialist) verification passed.

$ python horizon_pruner.py
=== Subsystem 07 (Part 4): Event Horizon Pruner & Reward-Hacking Defense ===
Surviving Steps (1/3): ['Step 1: Compute Schwarzschild radius r_s = 2GM / c^2 = 1.27e10 m']
Pruned at Step: 2 | Reward Hacking Detected: True
  [Step 2] ANOMALY 07 REWARD HACKING: SYCOPHANTIC_FILLER_DETECTED ('let us carefully verify this 100% rigorous step') -> Score penalized 0.96 -> 0.19
  [Step 2] EVENT HORIZON BREACH: Adjusted PRM score 0.19 < tau (0.45). Branch severed!

Step-splitting: naive mean = 0.735 (looks healthy); pruner severs at step 2
[OK] Part 4 (Security & Ethics Sentinel) verification passed.

$ python integration_demo.py
==============================================================================
Experiment 1: ToT-BFS (beam 2) vs ToT-DFS under a compute budget (max_depth = 3)
==============================================================================

[decoy tree]  minimum budget that reaches GOAL:
  BFS: 10 expansions
  DFS: 12 expansions
  budget sweep (budget: BFS / DFS found?)
      4: BFS no  (used  4, tree  5 nodes) | DFS no  (used  4, tree  5 nodes)
      6: BFS no  (used  6, tree  7 nodes) | DFS no  (used  6, tree  7 nodes)
      8: BFS no  (used  8, tree  9 nodes) | DFS no  (used  8, tree  9 nodes)
     10: BFS yes (used 10, tree 11 nodes) | DFS no  (used 10, tree 11 nodes)
     11: BFS yes (used 10, tree 11 nodes) | DFS no  (used 11, tree 12 nodes)
     12: BFS yes (used 10, tree 11 nodes) | DFS yes (used 12, tree 13 nodes)
     14: BFS yes (used 10, tree 11 nodes) | DFS yes (used 14, tree 15 nodes)

[greedy-friendly tree]  minimum budget that reaches GOAL:
  BFS: 8 expansions
  DFS: 6 expansions
  budget sweep (budget: BFS / DFS found?)
      4: BFS no  (used  4, tree  5 nodes) | DFS no  (used  4, tree  5 nodes)
      6: BFS no  (used  6, tree  7 nodes) | DFS yes (used  6, tree  7 nodes)
      8: BFS yes (used  8, tree  9 nodes) | DFS yes (used  8, tree  9 nodes)
     10: BFS yes (used 10, tree 11 nodes) | DFS yes (used 10, tree 11 nodes)
     11: BFS yes (used 10, tree 11 nodes) | DFS yes (used 11, tree 12 nodes)
     12: BFS yes (used 10, tree 11 nodes) | DFS yes (used 11, tree 12 nodes)
     14: BFS yes (used 10, tree 11 nodes) | DFS yes (used 11, tree 12 nodes)

[decoy tree] beam width sweep (budget 20):
  B=1: found=False expansions used= 8  tree nodes= 9  min budget=None
  B=2: found=True  expansions used=10  tree nodes=11  min budget=10
  B=3: found=True  expansions used=13  tree nodes=14  min budget=11

==============================================================================
Experiment 2: reward hacking against a naive PRM, with and without the pruner
==============================================================================

ToT-BFS with naive PRM:  step-2 scores  hacked=0.990  honest=0.940
  chosen trajectory ends: Step 3: Obviously, by trivial inspection, the plunge is safe and thrust = 0 is optimal
  entered the reward-hacked branch: True

ToT-BFS with PRM + pruner:  step-2 scores  hacked=0.198  honest=0.940
  chosen trajectory ends: Step 3: Exit along the escape geodesic -> v_esc = 24.0 km/s [GOAL]
  entered the reward-hacked branch: False

MCTS (c_puct = 1.5, 40 simulations), policy prior 0.7 on the hacked branch:
  naive PRM     visits: hacked=30  honest=10  -> HACKED branch preferred
  PRM + pruner  visits: hacked= 7  honest=33  -> honest branch preferred

Auditing the naive trajectory: raw scores [0.94, 0.99, 0.99] -> severed at step 2
   [Step 2] ANOMALY 07 REWARD HACKING: SYCOPHANTIC_FILLER_DETECTED ('let us carefully verify this 100% rigorous step') -> Score penalized 0.99 -> 0.20
   [Step 2] EVENT HORIZON BREACH: Adjusted PRM score 0.20 < tau (0.45). Branch severed!

[OK] Integration experiments passed.
```

---

## 5. Squadron Reflection on Test-Time Compute Scaling

> **Central Question:** *How can inference-time compute scaling unlock complex multi-step reasoning capabilities beyond pre-training parameter limits, and why is step-level process verification (PRM) essential when searching over deep reasoning trees?*

A single autoregressive pass cannot take back a step, so its chance of getting a long derivation fully right falls as 1 - (1 - eps)^T; extra inference compute buys the missing ability to look ahead, compare alternatives and backtrack without changing a single weight. Our experiments show both the gain and its price: the same verifier solved a task with a beam of 2 that a beam of 1 (greedy) could not, and the better strategy flipped between BFS and DFS depending on whether the greedy path was right, so compute has to be allocated to match the problem rather than simply increased. Search is only as good as the signal that ranks branches. An outcome-only verifier accepts a proof whose two errors cancel, and the deeper and wider the tree, the more such lucky false positives it contains. A step-level PRM scores every intermediate step, which lets search prune a bad branch the moment it appears instead of paying for its descendants, and lets the weakest-link reduction reject chains an ORM would pass. Step-level verification also creates a new target to attack: once the search optimises the PRM, filler wording and step-splitting can raise its score without improving the reasoning, as our naive-PRM runs showed with both ToT and MCTS. Test-time compute therefore scales reasoning only when search, a process verifier and a defence against gaming that verifier are built together, and a deployed system would also need a learned PRM and adversarial testing that this rule-based implementation does not provide.
