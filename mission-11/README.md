# Squadron Hinton — Mission 11: Andromeda Swarm Nexus — State Graphs, Handoff Contracts, Consensus & Deadlock Defence

**Course:** 14013 Conversational Artificial Intelligence: History, Application, Future — WiSe 26/27
**Mission:** ps11 — Andromeda Galaxy Core (SEC-11-M31), AURA Subsystem 11
**Instructor repo:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai) (`mission-11/`)

A swarm of agents fails in four ways: one overloaded context, parallel writers that overwrite each other, clones that vote together and are wrong together, and circular waits. This mission builds the four safeguards in plain Python, and then runs the instructor's two showcase tracks as far as possible without a Gemini key or Antigravity:

- **State graph** — fan-out/gather with per-channel reducers (`OVERWRITE`, `APPEND_LIST`, `SUM_NUMERIC`, `MAX_CONFIDENCE`), conditional routers, a step circuit breaker and an agentic Amdahl speedup model.
- **Handoff contracts** — typed envelopes, least-context projection, `SUBROUTINE_TOOL` against `PEER_TRANSFER` stack semantics, a self-counted hop budget and output contracts.
- **Blackboard and debate** — optimistic concurrency control, namespace ACLs, exact Condorcet accuracy, log-odds weighted quorum and echo-chamber detection.
- **Sentinel** — iterative 3-colour DFS wait-for cycle detection, deterministic deadlock pre-emption, a MAD Byzantine firewall and a swarm token breaker.

## Squad roster

| Station | Role | Officer | Deliverable |
|---|---|---|---|
| architect | Systems Architect | Mohid | [`swarm_state_graph_architect.py`](swarm_state_graph_architect.py) |
| protocol | Communications & Protocol Officer | Sharjeel | [`handoff_contract_protocol.py`](handoff_contract_protocol.py) |
| data | Data & Payload Specialist | Shayan | [`blackboard_debate_consensus.py`](blackboard_debate_consensus.py) |
| security | Security & Ethics Sentinel | Ahmed | [`deadlock_byzantine_sentinel.py`](deadlock_byzantine_sentinel.py) |
| all | Part 5 showcase and documentation | all four | [`showcase_adk_offline.py`](showcase_adk_offline.py), [`showcase_antigravity_local.py`](showcase_antigravity_local.py), [`mission-11-doc.md`](mission-11-doc.md), [`mission-11-doc.pdf`](mission-11-doc.pdf) |

## Folder layout

```
mission-11/
├── README.md                         roster, results, sign-offs, how to run
├── swarm_state_graph_architect.py    Systems Architect
├── handoff_contract_protocol.py      Protocol Officer
├── blackboard_debate_consensus.py    Data Specialist
├── deadlock_byzantine_sentinel.py    Security Sentinel
├── integration_demo.py               the four modules together; produces the report's measurements
├── andromeda_crew/                   Track A crew (instructor's, one bug fixed: see below)
├── andromeda_branch_swarm.py         Track B workflow script (instructor's, unchanged)
├── showcase_adk_offline.py           Track A run on the real ADK runtime with a scripted model
├── showcase_antigravity_local.py     Track B script run against a local runtime, real branch-isolated modules
├── mission-11-doc.md                 squadron documentation
└── mission-11-doc.pdf                the same, for the Moodle drop-box
```

## How to run

Parts 1–4 and the Track B local run need only the standard library (Python 3.11+):

```bash
# from the repository root
cd mission-11
python swarm_state_graph_architect.py
python handoff_contract_protocol.py
python blackboard_debate_consensus.py
python deadlock_byzantine_sentinel.py
python integration_demo.py          # about 15 s
python showcase_antigravity_local.py
```

The Track A offline run needs `google-adk` (2.x; we used 2.14.1). Install it in its own environment, since it pulls in many dependencies:

```bash
python -m venv .venv-adk && source .venv-adk/bin/activate
pip install google-adk
python showcase_adk_offline.py      # real ADK runtime, scripted model, no key, no network
```

### To produce the live-studio screenshot the briefing asks for (not done yet)

- **Track A:** with `google-adk` installed, set `GOOGLE_API_KEY` (a free Gemini key), run `adk web` from this folder, open `http://localhost:8000`, select `andromeda_crew`, send a `SEC-11-M31` query, and capture the **Events / Graph Trace** view.
- **Track B:** run `andromeda_branch_swarm.py` with Antigravity's `run_workflow` tool (or `/teamwork`) and capture the **Subagents Auxiliary Pane**.

Then embed the image in `mission-11-doc.md` and regenerate the PDF.

## What each script checks

| Script | Checks (all pass) |
|---|---|
| `swarm_state_graph_architect.py` | the instructor's graph (T_seq 1060, T_crit 560, S 1.89, findings 3, units 1260, best vector 0.93); identical final state for all 6 worker orderings; tie-break independent of order; overwrite conflicts reported or raised; conditional loop and runaway loop; real threads run 3 x 60 ms workers in about one sleep; Amdahl saturation |
| `handoff_contract_protocol.py` | the instructor's three handoffs; eta_comp = 1 - scoped/global; deep copy; nested stack unwinds in order; spoofed sender, stack mismatch and understated hop counts refused; contract violations including `None` fields |
| `blackboard_debate_consensus.py` | OCC conflict; ACL deny and default-deny; reads are copies; 8 threads x 100 increments (800 kept with OCC, about 100 without); exact Condorcet values and a Monte Carlo check; correlated majority; error-correlation estimate; log-odds weights; a tie is 0 |
| `deadlock_byzantine_sentinel.py` | diamond (no false positive), self-loop, 2-cycle, tail into a cycle; 300 random DAGs with 0 false alarms and 300 of 300 planted cycles found; a 200,000-vertex chain; priority tie-break; several cycles cleared; MAD against mean/std; the 50 % breakdown point; non-finite readings; latched token breaker |
| `integration_demo.py` | lost updates, real concurrency against the model, saturation; compression sweep, delegation stack, ping-pong; OCC under threads, Condorcet and correlation tables, specialist against clones; deadlock freeze and release, Byzantine breakdown table |
| `showcase_adk_offline.py` | the instructor's crew on ADK 2.14.1: two `AgentTool` calls and a `@node` gate that rejects a P1-only draft and routes it through the critic once |
| `showcase_antigravity_local.py` | the instructor's Track B script: 4 isolated branches, 4 passes, original folder hash unchanged, finish contracts validated |

## Key results

Full tables and analysis are in [`mission-11-doc.pdf`](mission-11-doc.pdf).

| Experiment | Result |
|---|---|
| Lost update, 5 parallel workers | plain shared state kept 1 of 5 findings and 100 of 500 units; reducer channels kept 5 of 5 and 500 |
| Amdahl: model against real threads | 1.88x modelled, 1.87x measured; with coordination cost growing with the number of workers the speedup saturates (12.8x at 64 workers, limit 17.7x) |
| Least-context compression | 92 % of the global context removed at 4 chat turns, 99.5 % at 200; the scoped context stays 154 characters |
| Ping-pong between two agents | 5 transfers with the hop gate, 50 without |
| 8 threads x 100 increments | 101 of 800 kept without OCC, 800 of 800 with OCC and jittered retry |
| Specialist (0.92) against two correlated clones (0.55) | unweighted majority 0.589, log-odds weighted 0.9225 (equal to the specialist alone); clone error correlation 0.87 |
| Ship Anomaly 11 | 0 of 3 agents finish without the monitor; all 3 finish after one pre-emption |
| DFS accuracy | 0 false alarms on 300 diamond-heavy DAGs, 300 of 300 planted cycles found |
| MAD firewall breakdown | 5 of 11 colluding probes all quarantined; 6 of 11: none caught, the consensus is the lie and the honest 5 are quarantined |

## Notes and limitations

- **No live-UI screenshot.** `adk web` needs a Gemini API key and Antigravity's `run_workflow` is needed for Track B; we had neither. The showcase scripts run the real ADK runtime with a scripted model, and the instructor's Track B script against a local runtime. Both are labelled in the report; the screenshot is still to be taken by the team (instructions above).
- **A bug in the instructor's Track A crew.** On `google-adk` 2.14.1 the `@node` gate binds its parameters from session state by default, so its first call raises `WorkflowDataError: Missing value for parameter "draft_plan"`. Our copy declares `parameter_binding="node_input"` and passes `{"draft_plan": ...}`.
- The agents are scripted and the clone correlation in the debate experiment is set by construction, so the results are about the coordination machinery, not about any model.
- The firewall's limits are tested and kept: it needs an honest majority, and when most readings are identical a lone dissenter is flagged whatever its size.
- Starter issues fixed: `compression_ratio` was the opposite ratio to the briefing's eta_comp; handoffs were not checked for a spoofed sender and the specialist received references into the shared state; `OVERWRITE` collisions in a parallel group were resolved silently by branch order and `MAX_CONFIDENCE` ties by arrival order; an exact tie in the quorum approved the trajectory; a recursive DFS would overflow the stack on deep graphs; with MAD = 0 any deviation was infinitely anomalous.
- The briefing's deadline for Moodle is 2027-01-10.

## Station sign-offs

Each officer ticks their own box once they have read every deliverable and agree the squad's submission is ready.

- [x] **architect** — Systems Architect
- [x] **protocol** — Communications & Protocol Officer
- [x] **data** — Data & Payload Specialist
- [x] **security** — Security & Ethics Sentinel
