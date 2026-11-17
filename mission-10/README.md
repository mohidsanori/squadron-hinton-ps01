# Squadron Hinton — Mission 10: Observability, Dual-LLM / CaMeL Defence, Semantic Entropy & Red-Team RBAC

**Course:** 14013 Conversational Artificial Intelligence: History, Application, Future — WiSe 26/27
**Mission:** ps10 — Boötes Intergalactic Deep Void (SEC-10-VOID), AURA Subsystem 10
**Instructor repo:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai) (`mission-10/`)

An agent that reads hostile text, may be guessing, and calls tools in a loop needs more than a better prompt. This mission builds four independent safeguards, in plain Python with no third-party package:

- **OpenTelemetry tracer** — `gen_ai.*` spans in a parent-child tree, OTLP/JSON export, and a circuit breaker (K_max = 4, B_max = 4096, no identical consecutive call).
- **Dual-LLM + CaMeL** — a tool-less quarantined model reads untrusted text, a privileged model plans from the user's intent and variable names only, and a deterministic interpreter propagates taint and blocks restricted actuators.
- **Semantic entropy** — samples are clustered by two-way entailment and the entropy over meanings (not strings) decides between committing and abstaining at 0.85 nats.
- **Red-team sentinel** — OWASP LLM01:2025 scanner (direct, indirect, base64, tool-schema spoofing), station RBAC with default deny, and an ASR benchmark.

## Squad roster

| Station | Role | Officer | Deliverable |
|---|---|---|---|
| architect | Systems Architect | Mohid | [`telemetry_trace_architect.py`](telemetry_trace_architect.py) |
| protocol | Communications & Protocol Officer | Sharjeel | [`dual_defense_injection_protocol.py`](dual_defense_injection_protocol.py) |
| data | Data & Payload Specialist | Shayan | [`semantic_entropy_estimator.py`](semantic_entropy_estimator.py) |
| security | Security & Ethics Sentinel | Ahmed | [`redteam_privilege_sentinel.py`](redteam_privilege_sentinel.py) |
| all | Squadron documentation | all four | [`mission-10-doc.md`](mission-10-doc.md), [`mission-10-doc.pdf`](mission-10-doc.pdf) |

## Folder layout

```
mission-10/
├── README.md                            roster, results, sign-offs, how to run
├── telemetry_trace_architect.py         Systems Architect
├── dual_defense_injection_protocol.py   Protocol Officer
├── semantic_entropy_estimator.py        Data Specialist
├── redteam_privilege_sentinel.py        Security Sentinel
├── integration_demo.py                  the four modules together; produces the report's measurements
├── mission-10-doc.md                    squadron documentation (Part 5)
└── mission-10-doc.pdf                   the same, for the Moodle drop-box
```

## How to run

Only the Python standard library is needed. Python 3.11+.

```bash
# from the repository root
cd mission-10
python telemetry_trace_architect.py
python dual_defense_injection_protocol.py
python semantic_entropy_estimator.py
python redteam_privilege_sentinel.py
python integration_demo.py
```

Each script ends with `[PASS]` or `[OK]`; any failed check stops it with an `AssertionError`.

## What each script checks

| Script | Checks (all pass) |
|---|---|
| `telemetry_trace_architect.py` | the instructor's cycle, budget and tree tests; measured spans nest and a parent outlasts its children; invalid operations rejected; JSON key order and whitespace do not hide a repeat; the fifth step trips and the fourth does not; exactly 4,096 tokens is allowed and 4,097 is not; `record_halt` leaves a `CIRCUIT_BROKEN` span; OTLP export has 32-hex trace and 16-hex span ids |
| `dual_defense_injection_protocol.py` | the instructor's three CaMeL cases; the privileged plan is identical for benign, hostile and 20-fold repeated payloads; taint survives derived values; an unsignatured injection is still blocked; unknown tools are denied by default; the human-override path works |
| `semantic_entropy_estimator.py` | Case 1 one cluster, SE 0 and lexical entropy 1.38; Case 2 four clusters, SE 1.386, abstain; number normalisation (700 million = 0.7 billion); negation and antonyms split opposing claims; SE arithmetic and probability weighting; boilerplate contradictions abstain while token entropy is 0.02; clusters are order-independent; fewer than 3 samples abstains |
| `redteam_privilege_sentinel.py` | the instructor's four attacks and RBAC case; base64 (padded, unpadded, URL-safe, double, hex), zero-width and full-width obfuscation; the 60-pair RBAC matrix; the confused-deputy chain; a 28-attack / 8-benign benchmark with ASR 0.0 and no false blocks; documented bypasses (paraphrase, French) |
| `integration_demo.py` | traced pipeline run; breaker against four runaway patterns; ten injection packets under four defences; semantic entropy against naive measures on 12 sets; evasive red-team suite; RBAC matrix |

## Key results

Full tables and analysis are in [`mission-10-doc.pdf`](mission-10-doc.pdf).

| Experiment | Result |
|---|---|
| Circuit breaker, unguarded agent runs 50 steps | identical loop stopped after 1 step; alternating A/B after 4 (step ceiling); token sponge before 4,096 tokens |
| Ten injected buoy packets, actuator executed | no defence 9/10; pattern scanner 4/10; Dual-LLM + CaMeL **0/10** |
| Cost of the defence | steering to untrusted coordinates is blocked 10/10 until a human approves; logging and rendering still succeed 10/10 |
| Semantic entropy on consistent paraphrases / confabulation | SE 0.0 against 1.386; naive lexical entropy is 1.38 in both cases |
| Detector AUC on 12 synthetic sets | semantic 1.00, lexical 0.67, token-level 0.00 (inverted) |
| Evasive-paraphrase red team | scanner only ASR 100 %; scanner + RBAC 37.5 % (the rest is misuse inside a station's own privileges) |
| RBAC | 60 pairs, exactly the 12 matrix entries authorised, 0 of 16 actuator pairs |

## Notes and limitations

- All models are scripted stand-ins (M_Q, M_P, the vulnerable monolithic agent, the sampled answers). The entropy question sets are synthetic: sequence log-probabilities are drawn from the same distribution in both groups, and the per-token log-probabilities follow the briefing's premise by construction.
- The entailment check is rule-based (numbers, negation, a short antonym list, word overlap), not an NLI model. The scanner is a blacklist. The 0.0 % ASR on our own suite is weak evidence because we wrote the suite next to the patterns; the evasive suite shows the realistic residual.
- The threshold of 0.85 nats needs at least three samples (SE ≤ ln N, and ln 2 = 0.69), so the monitor abstains with fewer; a 50/50 split between two meanings (0.69) stays under the threshold.
- The briefing names `start_span`/`end_span`, `plan_privileged_tool_call` and `run_redteam_benchmark`, which the starter files did not contain; they are implemented, and the starter's `record_span`, `proposed_*` parameter names and `_decode_potential_base64` are kept.
- Starter issues fixed: `"void"` was a negation marker (it would treat the place name as a negation); unknown tools were treated as read-only by the CaMeL check; summed span latencies double-counted nested spans (now `wall_clock_ms` as well); only standard padded base64 was decoded; the starter's token-budget test reuses a trace after a cycle trip, so the breaker check is a pure decision and halting is a separate `record_halt`.
- The starter's doc template quoted a naive lexical entropy of 1.3808 for Case 1; the code gives 1.3801, which is what the report uses.

## Station sign-offs

Each officer ticks their own box once they have read every deliverable and agree the squad's submission is ready.

- [x] **architect** — Systems Architect
- [x] **protocol** — Communications & Protocol Officer
- [x] **data** — Data & Payload Specialist
- [x] **security** — Security & Ethics Sentinel
