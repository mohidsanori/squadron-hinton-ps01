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
