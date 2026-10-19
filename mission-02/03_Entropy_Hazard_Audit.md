# 03 — Entropy & Hazard Audit

**Station:** security — Security & Ethics Sentinel
**Officer:** Ahmed
**Status:** draft

> Directives (ps02): run 5 stress tests on the FST dialogue manager with non-canonical utterances (unexpected order, sarcasm, off-topic) and document state breakage; analyse Shannon channel capacity against semantic loss and hallucination risk under high noise; define hazard protocols for detecting when channel noise makes an AI response unsafe.

## 1. FST stress tests

Run with `python 04_fst_noisy_simulator.py --mode fst`. The tests are `STRESS_TESTS` in the script; change them freely and tell Mohid.

| # | Category | Utterances | Intended result | Actual result | State trace | Broke? |
|---|---|---|---|---|---|---|
| 1 | Unexpected order | | | | | |
| 2 | Self-correction | | | | | |
| 3 | Negation | | | | | |
| 4 | Sarcasm | | | | | |
| 5 | Off-topic | | | | | |

### Why the FST breaks

<!-- Root cause per failure: first-match extraction, keyword yes/no, no notion of negation or irony... -->

## 1b. Portal FST simulator (course website)

Part 5 asks us to stress-test the FST in the portal's simulator too. Its parser uses substring checks (read from the site code), so these inputs should break it. Run each one, screenshot or copy the result, and fill in the table.

| # | State | Input | Expected break |
|---|---|---|---|
| 1 | INIT | `this ship needs a course` | "hi" inside "this"/"ship" is taken as a greeting |
| 2 | GREETING | `Sirius please` | slot is filled as "Alpha Centauri", not Sirius |
| 3 | GREETING | `check the console first` | "sol" inside "console" sets sector = Sol System |
| 4 | COLLECT_SLOTS | `warp 99` | any integer accepted, no 1–9 range check |
| 5 | CONFIRMATION | `I do not confirm` | "confirm" substring locks the course |
| 6 | any | an off-topic line | goes to FAILURE with no recovery (no FALLBACK state) |

| # | Actual result | Broke? |
|---|---|---|
| | | |

## 2. Shannon entropy and semantic loss

<!-- Use the metrics the simulator prints: H(sent), H(received), symbol error rate p, q-ary capacity C = log2 q − H_b(p) − p·log2(q−1), content-word retention.
Compare at least two noise levels, e.g. --noise 0.1 and --noise 0.4. -->

| Noise rate | Symbol error rate | Capacity (bits/word) | Retention | FST slot accuracy |
|---|---|---|---|---|
| | | | | |

## 3. Hallucination risk under noise

<!-- When the receiver fills gaps with guesses. Example from an early test run: the interrogator read channel corruption ("Cosmological Cosmological a Cosmological") as evidence that the witness was a looping language model, i.e. it blamed the channel's noise on the speaker. -->

## 4. Hazard protocols

| Hazard | Detection signal | Response |
|---|---|---|
| | | |
