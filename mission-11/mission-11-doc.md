# Mission 11 Squadron Documentation (`mission-11-doc.md`)

**Course:** Conversational AI (WS 2026/27) — Week 11: Multi-Agent Systems (MAS) & Framework Topologies  
**Sector Waypoint:** `SEC-11-M31` (Andromeda Galaxy Core, `Messier 31` / `NGC 224`, 2,500,000 ly from Sol)  
**AURA Subsystem:** Subsystem 11 — Andromeda Swarm Fleet Coordination Nexus  
**Squadron Name:** Squadron Hinton  
**Public GitHub Repository URL:** https://github.com/mohidsanori/squadron-hinton-ps01 (folder `mission-11/`)  
**Submission Date:** 2026-11-17  

---

## 1. Squadron Roster & Station Sign-Off Table

| Station Role | Cadet Full Name | Deliverable File | Verification Status |
| :--- | :--- | :--- | :--- |
| **Systems Architect** | Mohid | `swarm_state_graph_architect.py` | PASS / VERIFIED |
| **Communications & Protocol Officer** | Sharjeel | `handoff_contract_protocol.py` | PASS / VERIFIED |
| **Data & Payload Specialist** | Shayan | `blackboard_debate_consensus.py` | PASS / VERIFIED |
| **Security & Ethics Sentinel** | Ahmed | `deadlock_byzantine_sentinel.py` | PASS / VERIFIED |
| **Part 5 showcase** | all four | `showcase_adk_offline.py`, `showcase_antigravity_local.py`, `andromeda_crew/`, `andromeda_branch_swarm.py` | RAN OFFLINE (no live-UI screenshot: see Section E) |

Parts 1–4 use only the Python standard library. Extra file: `integration_demo.py` wires the four modules together and produces the measurements in Section 3. The agents in every experiment are scripted stand-ins that do arithmetic, not language, so the experiments test the coordination machinery and not any model.

---

## 2. Integrated Multi-Agent Swarm Architecture

```text
              +-----------------------------------------------------+
              |      Mission query: transit the M31 double nucleus   |
              +--------------------------+--------------------------+
                                         |
                                         v
              +-----------------------------------------------------+
              | Part 4: WaitForGraphMonitor (3-colour DFS)          |
              |   check the delegation plan BEFORE dispatch;        |
              |   a cycle -> Arbiter pre-empts the lowest-priority  |
              |   edge, repeat until acyclic                        |
              +--------------------------+--------------------------+
                                         |
                                         v
   +-----------------------------------------------------------------------+
   | Part 1: SwarmStateGraph                                               |
   |   Coordinator --fan-out--> [Astrogation_P1, Astrogation_P2, Tidal]    |
   |        every worker reads one snapshot, writes a delta                |
   |   barrier merge through channel reducers:                             |
   |   OVERWRITE | APPEND_LIST | SUM_NUMERIC | MAX_CONFIDENCE             |
   |   --gather--> TrajectorySynthesizer    (max_steps circuit breaker)    |
   +------------+-------------------------------------------+-------------+
                |                                           |
   typed handoff envelopes                       shared blackboard + debate
                |                                           |
   +------------v-------------+              +--------------v-------------+
   | Part 2: ContextScopingGate|              | Part 3: VersionedBlackboard|
   |  least-context projection |              |  OCC + namespace ACLs      |
   |  SUBROUTINE_TOOL: push    |              | DebateConsensusAggregator  |
   |  PEER_TRANSFER: no push   |              |  log-odds weighted quorum  |
   |  hop budget, contracts    |              |  echo-chamber alarm        |
   +------------+-------------+              +--------------+-------------+
                |                                           |
                +---------------------+---------------------+
                                      v
              +-----------------------------------------------------+
              | Part 4: ByzantineSwarmFirewall                       |
              |   MAD z-score quarantine of rogue telemetry;         |
              |   swarm token breaker                                |
              +-----------------------------------------------------+
```

**Architectural Summary:**
- The swarm is a typed state graph, not a chat room. Every worker in a fan-out reads one snapshot of the state and returns a delta, and the deltas are merged at a barrier by a reducer chosen per channel, so concurrent writers cannot overwrite each other. A step ceiling stops runaway loops, and the critical-path latency (the slowest worker plus coordination) is what the speedup is measured against.
- A specialist is called through a handoff envelope that names exactly which context keys it may see, which output fields it must return and how many hops the chain may take. The caller is pushed on a return stack for a subroutine call and not for a peer transfer, and the gate counts hops itself so a caller cannot understate them.
- Shared findings go through a versioned blackboard where a stale writer is refused instead of overwriting, and disagreements are settled by a quorum weighted by each agent's log-odds reliability, with an alarm when the voters' errors are correlated.
- Before any delegation is dispatched, the wait-for graph is checked with a 3-colour DFS; a cycle (Ship Anomaly 11) is broken by pre-empting its lowest-priority edge. Telemetry from the workers is screened with a robust MAD z-score, and a swarm-wide token budget trips a latched breaker.

---

## 3. Station-by-Station Experimental Report & Live Studio Showcase

All numbers come from running the code (`python <module>.py`, `python integration_demo.py`, and the two showcase scripts); the raw output is in Section 4.

### Section A — Systems Architect (`swarm_state_graph_architect.py`)

- **State-Graph Topology & Parallel Fan-Out Summary (the instructor's graph):**
  - Nodes Executed: `Coordinator -> [Astrogation_P1, Astrogation_P2, TidalShear_Scanner] -> TrajectorySynthesizer` (5 node executions, 1,960 tokens)
  - Sequential baseline T_seq = 120 + 260 + 300 + 240 + 140 = **1060.0 ms** | Critical path T_crit = 120 + max(260, 300, 240) + 140 = **560.0 ms** | **Agentic Amdahl Speedup S = 1060 / 560 = 1.89x**
- **Reducer Channel Verification Table:**

| State Channel Key | Configured `ReducerMode` | Concurrent Worker Updates | Final Reduced Value | Collision Status |
| :--- | :---: | :--- | :--- | :---: |
| `orbital_findings` | `APPEND_LIST` | P1 apoapsis clearance verified, P2 periapsis disk crossing stable, M31* tidal shear within hull tolerance | 3-item list | `NO_LOST_UPDATES` |
| `compute_units` | `SUM_NUMERIC` | 420 + 390 + 450 | **1260** | `EXACT_SUM` |
| `best_vector` | `MAX_CONFIDENCE` | (0.84, P1-OUTER), (0.93, P2-INCLINED-38DEG), (0.88, M31-POLAR) | (0.93, P2-INCLINED-38DEG) | `MAX_SELECTED` |

- **The lost-update race, measured (5 parallel workers, each reporting one finding and 100 units):**

| Channel type | Findings kept | Units (expected 500) | Reported |
| :--- | :---: | :---: | :--- |
| Plain overwrite (a shared dict) | **1 of 5** | **100** | 1 overwrite conflict reported (parallel branches wrote different values to the same key) |
| Reducer channels (`APPEND_LIST`, `SUM_NUMERIC`) | **5 of 5** | **500** | none |

- **Order independence.** The final state is identical for all 6 orderings of the three parallel workers, and a `MAX_CONFIDENCE` tie is broken by the payload, not by arrival order. `OVERWRITE` is the one reducer that is order-dependent, so two parallel branches writing different values to an `OVERWRITE` key are reported as a `ReducerConflict` (or raised in strict mode) instead of being resolved by thread timing.
- **Modelled against measured speedup.** With real threads and workers of 52, 60 and 48 ms around 25 + 28 ms of coordination, the sequential run took 214.9 ms and the concurrent run 114.9 ms: a measured speedup of **1.87x** against the model's T_seq / T_crit = **1.88x**. The merged state is identical either way.
- **Where Amdahl bites.** The model above has a fixed coordination cost, so the speedup grows without limit as workers are added. In a swarm the coordinator must also read every result, so its cost grows with K. With l_coord = 120 + 15K ms and workers of 250 ms:

| Workers K | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Speedup S | 1.00 | 1.62 | 2.74 | 4.57 | 7.15 | 10.12 | 12.84 |
| Efficiency S / K | 1.00 | 0.81 | 0.69 | 0.57 | 0.45 | 0.32 | 0.20 |

  The speedup saturates toward 1 + 250 / 15 = 17.7x while efficiency falls, so the marginal worker is worth less each time.
- **Runaway loops.** A conditional edge that always returns to its own node is stopped with `MAX_STEPS_EXCEEDED` after 20 steps; a retry loop with a real exit condition runs to completion.
- **Architectural Analysis:** Fan-out helps only when the work is independent and the join is cheap. The reducers make the join safe, because a reducer that is associative (and, where possible, commutative) gives the same result whatever order threads finish in; the engine detects the one case that is not, instead of hiding it. The step ceiling protects the other direction: a graph with cycles is legal (retry loops need them), so termination has to be enforced, not assumed.

---

### Section B — Communications & Protocol Officer (`handoff_contract_protocol.py`)

- **Least-Context Compression & Delegation Stack Matrix:**

| Envelope ID | Mode | Whitelisted Keys | Serialized Payload Reduction (eta_comp) | Caller Stack After Dispatch | Verdict |
| :--- | :---: | :--- | :---: | :---: | :--- |
| `ENV-SUB-01` (instructor harness) | `SUBROUTINE_TOOL` | `sector_id`, `p1_ephemeris` | **94.39 %** | `["AndromedaCoordinator"]` | `SCOPED_OK`; returns to the coordinator and the stack empties |
| `ENV-PEER-02` (a state that also holds `draft_vector`, `safety_bounds`) | `PEER_TRANSFER` | `draft_vector`, `safety_bounds` | **97.38 %** | `["AndromedaCoordinator"]` unchanged by the transfer, active agent `SafetyCritic` | `TRANSFERRED_OK` |
| `ENV-LOOP-99` | `PEER_TRANSFER` | `sector_id` | n/a | n/a | `HOP_LIMIT_EXCEEDED` (hop 6 > max_hops 5) |

  eta_comp = 1 - |serialize(S_scoped)| / |serialize(S_global)|, in serialized characters.
- **The saving grows with the history that never has to be passed on** (a fixed 2-key whitelist; the scoped context stays 154 characters):

| Chat turns in the global state | 4 | 12 | 50 | 200 |
| :--- | :---: | :---: | :---: | :---: |
| Global state (characters) | 1,945 | 2,995 | 8,011 | 27,911 |
| eta_comp | 0.921 | 0.949 | 0.981 | 0.995 |

- **Delegation semantics, traced:** `Coordinator -> Astrogation` (subroutine) pushes the coordinator; `Astrogation -> TidalScanner` (subroutine) pushes Astrogation, giving stack `[Coordinator, Astrogation]`; `TidalScanner -> SafetyCritic` (peer transfer) moves the active pointer and leaves the stack alone; returning unwinds it in order, `Astrogation` then `Coordinator`.
- **Ping-pong.** Two agents that keep handing the request back and forth, with every envelope claiming `hop_count = 0`, are stopped after **5** transfers with the gate (max_hops 5); without it the same pair ran **50** transfers and stopped only because we capped the test. The gate counts hops itself per chain, so an understated `hop_count` does not extend the budget.
- **Additional checks the gate makes:** a handoff whose sender is not the active agent is refused (`SENDER_NOT_ACTIVE`), a return that does not match the top of the stack is refused (`STACK_MISMATCH`), a missing or `None` required output field is a `SCHEMA_CONTRACT_VIOLATION`, and the specialist receives a deep copy, so mutating its view does not change the shared state.
- **Protocol Analysis:** A handoff that forwards everything is a group chat in disguise: the recipient pays to read, and can be misled by, context that has nothing to do with its task. Projecting onto a whitelist makes the cost of a handoff independent of how long the conversation has run and limits what a compromised recipient can see. The mode matters because the two ADK mechanisms have different control flow: an `AgentTool` call is a function call that returns, so the caller stays on the stack, whereas a `sub_agents` transfer hands over the conversation and does not return. Confusing them either loses the return address or grows the stack without bound. Note that the brief defines eta_comp as the share removed, so a higher value is better; the starter's `compression_ratio` was the opposite ratio.

---

### Section C — Data & Payload Specialist (`blackboard_debate_consensus.py`)

- **Optimistic Concurrency & Condorcet / Log-Odds Quorum Benchmark:**

| Evaluation Scenario | Jury Composition | Condorcet P_maj(N, p) | Unweighted Majority | Log-Odds Weighted Verdict | Echo-Chamber Flag |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **3 independent specialists** | N = 3, p = 0.70, rho = 0.10 | **0.7840** | +1 | +1 | False |
| **Specialist vs. 2 weak clones** | 1 x (p 0.92, c 0.95) against 2 x (p 0.55, c 0.60) | n/a (heterogeneous) | **-1** (wrong) | **+1** (rescued, margin 2.0794) | **True** (rho 0.82, 1 flip without evidence) |
| **OCC stale write** | `expected_version = 0` against stored version 1 | n/a | n/a | `CONCURRENCY_CONFLICT` | n/a |

- **Optimistic concurrency under real threads** (8 threads x 100 read-modify-write increments of one key, expected 800; each increment holds the read-modify-write window open for 0.1 ms):

| Writer | Increments kept | Notes |
| :--- | :---: | :--- |
| No version check | **101** | 699 lost updates, silently |
| OCC with retry and jittered backoff | **800** | 788 writes were refused with `CONCURRENCY_CONFLICT` and retried; 0 increments given up on |

  The first version of the retry loop gave up after 50 conflicts under this contention, so a worker stopped early with `OCC_RETRIES_EXHAUSTED`: a loud failure, never a silent loss, but starvation. Randomised exponential backoff fixed it, and the demo counts any worker that gives up.
- **Condorcet majority accuracy P_maj(N, p) for independent voters (exact):**

| N | p = 0.55 | p = 0.70 | p = 0.90 |
| :---: | :---: | :---: | :---: |
| 1 | 0.5500 | 0.7000 | 0.9000 |
| 3 | 0.5747 | 0.7840 | 0.9720 |
| 5 | 0.5931 | 0.8369 | 0.9914 |
| 7 | 0.6083 | 0.8740 | 0.9973 |
| 9 | 0.6214 | 0.9012 | 0.9991 |
| 15 | 0.6535 | 0.9500 | 1.0000 |
| 25 | 0.6937 | 0.9825 | 1.0000 |

- **Correlation erodes the gain (N = 7, p = 0.70; common-shock model, with a 30,000-trial Monte Carlo check):**

| rho | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Effective jury size N_eff | 7.00 | 2.80 | 1.75 | 1.27 | 1.00 |
| P_maj (formula) | 0.8740 | 0.8305 | 0.7870 | 0.7435 | 0.7000 |
| P_maj (Monte Carlo) | 0.8728 | 0.8313 | 0.7871 | 0.7432 | 0.7018 |

- **A calibrated specialist against two correlated clones** (20,000 simulated questions; the specialist is right with probability 0.92, clone A with 0.55, and clone B copies clone A 85 % of the time):

| Aggregation rule | Accuracy |
| :--- | :---: |
| Specialist alone | 0.9225 |
| Unweighted majority of the three | **0.5889** |
| Log-odds weighted quorum | **0.9225** |

  The error correlation estimated from the first 400 questions' error history is **0.87** for the clone pair (0.26 averaged over all three agents), above the 0.50 alarm threshold, so `WARNING_CORRELATED_ECHO_CHAMBER` is raised.
- **Data Analysis:** The Condorcet theorem is a statement about independent voters. Its table is a promise of what adding voters can buy: with p = 0.70, seven independent voters are right 87.4 % of the time and twenty-five 98 %. The correlation table shows how little of that survives in a swarm of clones: at rho = 0.5 seven voters are worth 1.75 independent ones, and at rho = 1 the majority is no better than a single voter. Below p = 0.5 the same effect works the other way, and the majority gets worse as N grows. Unequal competence is the second failure: with the specialist outvoted two to one the unweighted majority scores 0.5889, worse than the specialist alone, while weights of ln(p / (1 - p)) · c give the specialist about 2.3 against 0.12 for each clone, and the weighted quorum recovers 0.9225. The weights are only as good as the calibration of p; an overconfident clone with an inflated p would be given a say it has not earned. An exact tie is a verdict of 0, not an approval. The blackboard's version number turns a silent lost update into an explicit conflict that the writer must handle.

---

### Section D — Security & Ethics Sentinel (`deadlock_byzantine_sentinel.py`)

- **Ship Anomaly 11 — 3-Color DFS Cycle Detection & Byzantine Quarantine Audit:**

| Graph / Telemetry Test Case | Topology / Input | 3-Color DFS / Firewall Output | Action Taken |
| :--- | :--- | :--- | :--- |
| **Diamond convergent DAG** | `Root -> {BranchA, BranchB} -> MergeSink` (a BLACK cross edge) | `cycle = []` (0 false positives) | `ALLOW_DISPATCH` |
| **Ship Anomaly 11 circular wait** | `Astrogation -> ProtocolValidator -> PayloadBuilder -> Astrogation` | `[Astrogation, ProtocolValidator, PayloadBuilder, Astrogation]` | `PREEMPT_EDGE (PayloadBuilder -> Astrogation)`, priority 2, lowest on the cycle; graph acyclic afterwards |
| **Byzantine rogue telemetry** | `[-301.2, -300.8, -301.5, -300.9, +999.0]` km/s | median -300.9, MAD 0.3, z_MAD = **2922.6** > 3.5 | `QUARANTINE_BYZANTINE_NODE (Rogue_Spoofer)` |

- **What the swarm does with and without the monitor** (agents finish when everything they wait for has finished):

| Run | Agents finished | Outcome |
| :--- | :---: | :--- |
| No monitor | **0 of 3** after 30 rounds | frozen (cut off by our 30-round cap) |
| Monitor + arbiter | **3 of 3** | pre-empts `PayloadBuilder -> Astrogation`; finish order `PayloadBuilder`, `ProtocolValidator`, `Astrogation` |

- **Detector accuracy and cost:** on 300 random wait-for DAGs full of diamonds the DFS raised **0** false alarms, and with one back edge added to each it found the cycle in **300 of 300**. Three independent circular waits (3, 2 and 4 agents) were cleared in 3 pre-emptions. The DFS is iterative, so a 200,000-vertex chain is handled with no recursion limit; the time was 33, 179 and 490 ms for chains of 25k, 100k and 200k vertices: 8 times the vertices took 14.8 times as long, a little worse than linear (probably the per-vertex neighbour sort and memory effects on a large graph) and far from the 64 times a quadratic algorithm would need.
- **Byzantine firewall, 11 probes about -301 km/s with r colluding on 999 km/s:**

| r colluding | Rogues quarantined | Honest probes wrongly quarantined | Consensus (median of trusted readings) | Classical mean / std z-score flags |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1 / 1 | 0 / 10 | -300.8 | 0 / 1 |
| 3 | 3 / 3 | 0 / 8 | -300.9 | 0 / 3 |
| 5 | 5 / 5 | 0 / 6 | -300.9 | 0 / 5 |
| **6** | **0 / 6** | **5 / 5** | **+999.1** | 0 / 6 |

  The median and MAD ignore a minority of outliers, so the firewall holds up to 5 of 11, while a classical z-score never flags anything (with n readings the largest possible z is (n - 1) / sqrt(n), under 3 for n = 5). The **breakdown point is 50 %**: with 6 of 11 colluding, the median becomes the lie, the rogues are the trusted majority, and the honest minority is quarantined. If the six rogues report exactly the same value, MAD is 0, nobody is flagged, and the consensus is still the lie.
- **Token breaker:** with five workers spending 900, 700, 800, 650 and 2,600 tokens per round against a 16,000 budget, the breaker tripped in round 3 at 16,950 tokens and named the rogue, the heaviest consumer; once tripped it stays tripped.
- **Security Analysis:** A circular wait is a cycle in the wait-for graph, and a 3-colour DFS finds one in time proportional to the graph: an edge to a GRAY vertex is a back edge and closes a cycle, while an edge to a BLACK vertex only means two branches meet again, which is why the diamond is accepted. Detection before dispatch is what makes the difference, because after the three agents are waiting there is nothing left to run. The arbiter's choice of victim is deterministic (lowest priority, then names), so two runs resolve a deadlock the same way. **Limits:** the monitor sees only the waits that the agents declare; priorities come from the operator, and pre-empting an edge means that delegation fails, which the caller must handle. The firewall assumes honest readings are a majority, treats fewer than 3 readings as unjudgeable, and when most readings are identical there is no scale to measure against, so a single dissenting reading among five identical ones is flagged whatever its size (z = n / 1.2533 = 3.99).

---

### Section E — Part 5 Live Multi-Agent Studio Showcase (`Google ADK 2.0` OR `Google Antigravity`)

**Status, stated plainly: the live-UI screenshot the briefing asks for is NOT included.** Track A needs a Gemini API key (`GOOGLE_API_KEY`) for `adk web`, and Track B needs Google Antigravity (`run_workflow`); neither was available to us. What we did instead is run both tracks' real code as far as it can run without them. The model is the only thing replaced, and every run below is labelled with what is real and what is scripted.

- **Selected Studio Track:** Track A (Google ADK 2.x) run offline with ADK's real runtime, plus Track B's script run against a local runtime.
- **Executed Prompt / Workflow Goal:** `Plan a safe transit of the Andromeda double nucleus.` (ADK) and the instructor's four-station verification workflow (Track B).

**Track A — the instructor's `andromeda_crew/agent.py` executed by `google-adk` 2.14.1** (`showcase_adk_offline.py`). ADK's own `Runner`, session service, `AgentTool` calls and `@node` workflow engine run unchanged; each agent's `model="gemini-flash-latest"` is replaced by a scripted `BaseLlm`, so there is no network and no key. The event stream:

| Run | Event (author: what it did) |
| :--- | :--- |
| **A. Coordinator** | `AndromedaNexusCoordinator`: CALL `AstrogationSpecialist` (a request to draft a P1/P2 trajectory) |
| | `AndromedaNexusCoordinator`: RESULT `AstrogationSpecialist` (velocity 204.5 km/s, inclination 38.2 deg, clearance 0.42 ly) |
| | `AndromedaNexusCoordinator`: CALL `SafetyCritic`, then RESULT `SafetyCritic` (tidal shear within tolerance, P1 and P2 margins verified) |
| | `AndromedaNexusCoordinator`: TEXT final bridge clearance. 3 model calls, 2 `AgentTool` calls, 5 events |
| **B. `@node` workflow** | `AstrogationSpecialist`: draft that mentions only P1 |
| | `deterministic_telemetry_gate` (no LLM): OUTPUT `Missing double-nucleus reference(s): ['P2']` |
| | `andromeda_dynamic_workflow`: STATE `draft_plan`, `lint_findings`, `round` = 1 |
| | `SafetyCritic`: hardened plan that clears the P1 apoapsis and the P2 disk crossing |
| | `deterministic_telemetry_gate`: OUTPUT `findings = ""`, then the workflow OUTPUT is the hardened plan. 1 repair round, 6 events |

  The deterministic `@node` gate caught the missing nucleus without a model call, which is the point of interleaving a non-LLM check with LLM specialists. **We had to fix a bug in the starter to get this to run:** on `google-adk` 2.14.1 the gate's parameters are bound from session state by default, so the first call (made before anything has written `draft_plan` to state) raises `WorkflowDataError: Missing value for parameter "draft_plan"`. The fix is `@node(name="deterministic_telemetry_gate", parameter_binding="node_input")` and calling it with `{"draft_plan": ...}`; it is the only change in our copy of `andromeda_crew/agent.py`.

**Track B — the instructor's `andromeda_branch_swarm.py` executed against a local runtime** (`showcase_antigravity_local.py`). Antigravity supplies `phase`, `agent` and `parallel`; a small local runtime supplies them instead, with deterministic Python in place of the language-model roles. The four verifiers really run our four modules, each in its own copy of the mission folder (`workspace="branch"`), and every result is validated against the JSON-Schema `finish` contract it was given:

| Verifier (isolated branch) | Result | Time |
| :--- | :---: | :---: |
| `handoff_contract_protocol.py` | PASS | 0.12 s |
| `swarm_state_graph_architect.py` | PASS | 0.38 s |
| `blackboard_debate_consensus.py` | PASS | 1.44 s |
| `deadlock_byzantine_sentinel.py` | PASS | 2.06 s |

  The four verifiers ran concurrently in 2.08 s against 4.00 s one after another; each got a distinct directory and the original folder's hash was unchanged afterwards. The Security Sentinel then approved the synthesis (`approved: true`, 4 of 4 modules passed).

**To produce the live screenshot** (for the team): Track A: `pip install google-adk`, set `GOOGLE_API_KEY`, run `adk web` from `mission-11/`, open `http://localhost:8000`, select `andromeda_crew`, send a `SEC-11-M31` query, and capture the Events / Graph Trace view. Track B: run `andromeda_branch_swarm.py` with Antigravity's `run_workflow` and capture the Subagents Auxiliary Pane. Embed the image here as `trace_screenshot.png`.

---

## 4. End-to-End Verification Terminal Output

```text
$ python swarm_state_graph_architect.py
[StateGraph Execution] steps=5 | tokens=1960 | T_seq=1060.0ms | T_crit=560.0ms | Amdahl Speedup=1.8929x
[Reducer Channels] findings=3 | compute_units=1260 | best_vector=(0.93, 'VEC-P2-INCLINED-38DEG')
Real threads: 3 x 60 ms workers took 180.75 ms sequentially and 61.85 ms concurrently.
[PASS] Part 1 Architect: State-Graph, Parallel Reducers & Amdahl Speedup Confirmed.

$ python handoff_contract_protocol.py
[Subroutine Handoff] active=AstrogationSpecialist | stack=['AndromedaCoordinator'] | compression_ratio=0.9439 | reduction=94.39%
[Hop Guard Verified] Caught expected error: HOP_LIMIT_EXCEEDED: Envelope 'ENV-LOOP-99' at hop 6 exceeds max_hops=5.
[PASS] Part 2 Protocol: Least-Context Scoping & Handoff Envelopes Confirmed.

$ python blackboard_debate_consensus.py
[OCC Race Caught] CONCURRENCY_CONFLICT on 'nav/p1_vector': expected v0, found v1.
[Condorcet Jury] P_maj(N=3, p=0.70)=0.7840 | P_maj(N=7, p=0.70)=0.8740
[Quorum Decision] unweighted=-1 | log_odds_weighted=1 (margin=2.0794) | echo_warning=True | flips=1
OCC with retry: 800/800 increments kept; blind read-modify-write without OCC kept 100/800.
[PASS] Part 3 Data: Versioned Blackboard & Condorcet Debate Consensus Confirmed.

$ python deadlock_byzantine_sentinel.py
[Diamond DAG Check] BLACK cross-edge accepted with 0 false positives.
[Ship Anomaly 11 Resolved] cycle=['Astrogation', 'ProtocolValidator', 'PayloadBuilder', 'Astrogation'] | preempted=(PayloadBuilder -> Astrogation) | post_cycle=[]
[Byzantine Firewall] clean=['Probe_Alpha', 'Probe_Beta', 'Probe_Gamma', 'Probe_Delta'] | quarantined=['Rogue_Spoofer']
3-colour DFS on chains of 25k / 100k / 200k vertices: 31 / 170 / 444 ms (no recursion limit hit).
[PASS] Part 4 Security: 3-Color DFS Deadlock Arbiter & Byzantine Firewall Confirmed.

$ python integration_demo.py
==============================================================================
Experiment 1: state graph, reducers and the Amdahl model
==============================================================================
  5 parallel workers, each reports 1 finding and 100 units:
    plain overwrite channels : 1 of 5 findings kept, units = 100 (expected 500), 1 overwrite conflicts reported
    reducer channels         : 5 of 5 findings kept, units = 500

  Real threads, workers of 52 / 60 / 48 ms around 25 + 28 ms of coordination:
    sequential 214.9 ms | concurrent 114.9 ms | measured speedup 1.87x | model T_seq/T_crit = 1.88x

  Agentic Amdahl with coordination cost l_coord = 120 + 15K ms and workers of 250 ms:
       K    T_seq   T_crit  speedup  efficiency S/K
       1      385      385     1.00            1.00
       2      650      400     1.62            0.81
       4     1180      430     2.74            0.69
       8     2240      490     4.57            0.57
      16     4360      610     7.15            0.45
      32     8600      850    10.12            0.32
      64    17080     1330    12.84            0.20
    limit as K grows: 1 + 250/15 = 17.7x

  A conditional edge that always loops back: stopped with MAX_STEPS_EXCEEDED after 20 steps (max_steps = 20).

==============================================================================
Experiment 2: handoff contracts and the least-context principle
==============================================================================
    chat turns in global state  global chars  scoped chars  eta_comp
                             4         1,945           154     0.921
                            12         2,995           154     0.949
                            50         8,011           154     0.981
                           200        27,911           154     0.995
  The scoped context stays the same size, so the saving grows with the history that never has to be passed on.

  handoff                                     mode             active agent   return stack
  Coordinator -> Astrogation                  SUBROUTINE_TOOL  Astrogation    ['Coordinator']
  Astrogation -> TidalScanner                 SUBROUTINE_TOOL  TidalScanner   ['Coordinator', 'Astrogation']
  TidalScanner -> SafetyCritic                PEER_TRANSFER    SafetyCritic   ['Coordinator', 'Astrogation']
  returning unwinds the stack in order: ['Astrogation', 'Coordinator']

  Two agents that keep handing the request back and forth (every envelope claims hop 0): 5 transfers with the gate (max_hops = 5), 50 without it (stopped only by our 50-step cap).

==============================================================================
Experiment 3: blackboard concurrency and debate consensus
==============================================================================
  8 threads x 100 read-modify-write increments (expected 800):
    without OCC : 101 kept (699 lost updates)
    with OCC    : 800 kept; 788 writes were refused with CONCURRENCY_CONFLICT and retried; 0 increments given up on

  Condorcet majority accuracy P_maj(N, p) for independent voters:
       N    p=0.55     p=0.7     p=0.9
       1    0.5500    0.7000    0.9000
       3    0.5747    0.7840    0.9720
       5    0.5931    0.8369    0.9914
       7    0.6083    0.8740    0.9973
       9    0.6214    0.9012    0.9991
      15    0.6535    0.9500    1.0000
      25    0.6937    0.9825    1.0000

  The same jury of N = 7 voters with p = 0.70 when they are correlated (common-shock model, Monte Carlo check):
      rho   N_eff  P_maj formula  Monte Carlo
     0.00    7.00         0.8740       0.8728
     0.25    2.80         0.8305       0.8313
     0.50    1.75         0.7870       0.7871
     0.75    1.27         0.7435       0.7432
     1.00    1.00         0.7000       0.7018

  One calibrated specialist (p = 0.92) and two correlated clones (p = 0.55), 20,000 simulated questions:
    specialist alone       0.9225
    unweighted majority    0.5889
    log-odds weighted      0.9225
    error correlation estimated from the first 400 questions: clone pair 0.87, all three agents 0.26; echo alarm for the clone pair: True

==============================================================================
Experiment 4: deadlock detection and the Byzantine firewall
==============================================================================
  Ship Anomaly 11 (Astrogation -> ProtocolValidator -> PayloadBuilder -> Astrogation):
    no monitor : 0 of 3 agents finished after 30 rounds (frozen; cut off by our 30-round cap)
    with monitor: pre-empted ['PayloadBuilder->Astrogation']; all 3 finished in order ['PayloadBuilder', 'ProtocolValidator', 'Astrogation']

  300 random wait-for DAGs full of diamonds: 0 false alarms; the same graphs with one back edge added: 300 of 300 detected
  Detection time on acyclic chains of 25k / 100k / 200k vertices: 33 / 179 / 490 ms (vertex ratio 8x, time ratio 14.8x)
  Three independent circular waits (3, 2 and 4 agents): cleared in 3 pre-emptions, graph acyclic: True

  Byzantine telemetry: 11 probes read about -301 km/s; r of them collude on 999 km/s
      r  rogues caught  honest wrongly quarantined   consensus (median of trusted)  classical z flags rogue?
      1          1/1                         0/10                         -300.8                     0/1
      3          3/3                         0/8                          -300.9                     0/3
      5          5/5                         0/6                          -300.9                     0/5
      6          0/6                         5/5                           999.1                     0/6

  Swarm token budget 16,000: tripped in round 3 at 16,950 tokens; heaviest consumer: Rogue

[OK] Integration experiments passed.

$ python showcase_adk_offline.py        (google-adk 2.14.1, scripted model)
Google ADK 2.14.1 | python 3.13.9 | model: ScriptedLlm (no network, no API key)

[A] Root coordinator: AgentTool calls into the two specialists (ADK Runner, scripted model)
  AndromedaNexusCoordinator    CALL AstrogationSpecialist({"request": "Draft a P1/P2 trajectory for SEC-11-M31."})
  AndromedaNexusCoordinator    RESULT AstrogationSpecialist -> {'result': 'Orbital plan around M31*: velocity_kms=204.5, inclination_deg=38.2, clearance_
  AndromedaNexusCoordinator    CALL SafetyCritic({"request": "Audit the AstrogationSpecialist plan: velocity 204.5 km/s, inclination 38.2 d)
  AndromedaNexusCoordinator    RESULT SafetyCritic -> {'result': 'Audit: tidal shear at M31* within hull tolerance; P1 and P2 clearance margins 
  AndromedaNexusCoordinator    TEXT Bridge clearance issued: velocity 204.5 km/s, inclination 38.2 deg, clearance 0.42 ly. SafetyCritic verdict: C

[B] @node dynamic workflow: deterministic gate + bounded repair loop
  AstrogationSpecialist        TEXT Orbital plan: velocity_kms=204.5, inclination_deg=38.2, clearance_ly=0.42 around P1 only.
  deterministic_telemetry_gate OUTPUT {'findings': "Missing double-nucleus reference(s): ['P2']"}
  andromeda_dynamic_workflow   STATE {"draft_plan": "Orbital plan: velocity_kms=204.5, inclination_deg=38.2, clearance_ly=0.42 around P1 only.", "l
  SafetyCritic                 TEXT Hardened plan: velocity_kms=204.5, inclination_deg=38.2, clearance_ly=0.42; clears P1 apoapsis and the P2 disk
  deterministic_telemetry_gate OUTPUT {'findings': ''}
  andromeda_dynamic_workflow   OUTPUT Hardened plan: velocity_kms=204.5, inclination_deg=38.2, clearance_ly=0.42; clears P1 apoapsis and the P2 disk

--- checks ---
[A] root coordinator made 3 model calls and 2 AgentTool calls ['AstrogationSpecialist', 'SafetyCritic']; each specialist ran once; 5 events
[B] the draft missing P2 was routed through the critic once; final output mentions P1 and P2; 6 events
[PASS] Part 5 Track A (offline): the instructor's ADK crew executed by the real ADK runtime.

$ python showcase_antigravity_local.py
Local runtime for the instructor's Track B script | python 3.13.9 | roles are deterministic Python, modules run for real

=== phase: 1. Lead Architect Decomposition ===

=== phase: 2. Parallel Branch-Isolated Verification Swarm ===

=== phase: 3. Stateful Security Sentinel Sign-Off ===
{'approved': True, 'summary': '4/4 station modules passed their self-checks in isolated branches'}

--- branch execution log ---
  handoff_contract_protocol.py       branch=branch_lpj4i6kf   0.12s  PASS
  swarm_state_graph_architect.py     branch=branch_llbqn_es   0.38s  PASS
  blackboard_debate_consensus.py     branch=branch_zgwwk2xm   1.44s  PASS
  deadlock_byzantine_sentinel.py     branch=branch_ph__wr2j   2.06s  PASS
  4 verifiers ran concurrently: 2.08s wall clock against 4.00s if run one after another

--- checks ---
4 branches, 4 distinct isolated directories, original folder hash unchanged, finish contracts validated.
[PASS] Part 5 Track B (local runtime): the instructor's Antigravity script executed against the real modules.
```

---

## 5. Squadron Seminar Reflection: Topology, Least-Context Scoping & Collective Intelligence

> **Central Question:** *How does the "Least-Context Principle" paired with deterministic State-Graph Reducers and 3-Color DFS Wait-For cycle detection mitigate context pollution, lost updates, and circular deadlocks compared to flat multi-agent group chats—and why does adding more homogeneous LLM agents often degrade rather than improve decision accuracy?*

A flat group chat gives every agent every message, lets any agent write anywhere, and lets any agent wait for any other. Each of those freedoms is a failure we measured. Passing everything means a specialist reads, and can be misled by, history it has no use for: a two-key whitelist removed 92 % of the global context at 4 chat turns and 99.5 % at 200, and the scoped context did not grow at all. Letting anyone write means concurrent writers overwrite each other: five parallel workers kept one finding and 100 of 500 units on a plain shared state, and all five and 500 through reducer channels, while eight threads on an unversioned key kept 101 of 800 increments and 800 of 800 with a version check. Letting anyone wait means a ring of three specialists can freeze the ship, because nothing in a chat forbids a cycle. The typed graph, the scoped envelope and the versioned channel remove those freedoms by construction, and a 3-colour DFS over the declared waits finds the cycle in time proportional to the graph before anything is dispatched: no false alarm on 300 diamond-heavy graphs, every planted cycle found, and the frozen run finished once one edge was pre-empted. These guarantees come from structure. They do not depend on an agent behaving well, and that is why they hold when an agent does not.

More agents help only if they bring independent information, and homogeneous clones do not. The Condorcet table shows what independence is worth: seven voters of accuracy 0.70 are right 87.4 % of the time. Clones of one model make the same mistakes, so correlation rho = 0.5 makes those seven worth 1.75 independent voters and rho = 1 makes the majority no better than one. In our simulation a calibrated specialist (0.92) outvoted by two correlated clones (0.55) dropped the unweighted majority to 0.589, far below the specialist alone, which is how adding agents lowers accuracy. Weighting votes by log-odds reliability recovered 0.9225, and an error-history correlation of 0.87 between the clones is the signal to raise the echo-chamber alarm. The same logic applies to defence: a robust median-based firewall handled 5 colluding probes out of 11 and failed completely at 6, because redundancy only protects while the honest voters are the majority. What we have not shown is how a real language model behaves in these roles. Our agents were scripted, our clone correlation was set by construction, and the live-studio traces we could produce were run with a scripted model, so the claims above are about the coordination machinery, which is the part the course asks us to engineer.
