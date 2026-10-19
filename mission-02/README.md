# Squadron Hinton — Mission 02: FST Dialogue Systems, Lovelace's Objection & Noisy Channels

**Course:** 14013 Conversational Artificial Intelligence: History, Application, Future — WiSe 26/27
**Mission:** ps02 — Sirius Binary System (SEC-02-CMA)
**Due:** 2026-10-25
**Instructor repo:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai) (`mission-02/`)

Plasma noise from Sirius B corrupts AURA's subspace transceiver, and Command suspects AURA only shuffles symbols without understanding. This mission tests her on two fronts:

- **Deterministic control** — an FST dialogue manager that plots a course through fixed states and slot memory, stress-tested with non-canonical input.
- **Probabilistic conversation under noise** — a Lovelace (Anti-Turing) probe for originality, and a 10-turn imitation game over a Shannon noisy channel (max 7 words per message, mandatory noise tokens).

## Squad roster

| Station | Role | Officer | Deliverable |
|---|---|---|---|
| data | Data & Payload Specialist | Shayan | [`01_Lovelace_Inquiry_Brief.md`](01_Lovelace_Inquiry_Brief.md) |
| protocol | Communications & Protocol Officer | Sharjeel | [`02_FST_and_Noisy_Protocol.md`](02_FST_and_Noisy_Protocol.md) |
| security | Security & Ethics Sentinel | Ahmed | [`03_Entropy_Hazard_Audit.md`](03_Entropy_Hazard_Audit.md) |
| architect | Systems Architect | Mohid | [`04_fst_noisy_simulator.py`](04_fst_noisy_simulator.py), [`05_Noisy_Interrogation_Transcript.md`](05_Noisy_Interrogation_Transcript.md) |

## Folder layout

```
mission-02/
├── README.md                          roster, sign-offs, how to run
├── 01_Lovelace_Inquiry_Brief.md       Data Specialist
├── 02_FST_and_Noisy_Protocol.md       Protocol Officer
├── 03_Entropy_Hazard_Audit.md         Security Sentinel
├── 04_fst_noisy_simulator.py          Systems Architect (simulator)
├── 04_fst_noisy_simulator_starter.py  same simulator, under the instructor README's file name
└── 05_Noisy_Interrogation_Transcript.md   10-turn noisy interrogation (full squad)
```

## How to run

```bash
# from the repository root
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add the key for your provider
cd mission-02
python 04_fst_noisy_simulator.py --mode fst          # FST + stress tests, no LLM needed
python 04_fst_noisy_simulator.py                     # everything
```

| Flag | Default | Purpose |
|---|---|---|
| `--provider` | `auto` | `claude`, `openai`, `gemini`, `claude-code` (Claude via the Claude Code CLI login, no API key), or `mock` (offline, for testing only) |
| `--mode` | `all` | `fst`, `fst-noisy`, `probe`, `interrogate` or `all` |
| `--turns` | 10 | interrogation turns (one question + one answer each) |
| `--noise` | 0.15 | probability that the channel corrupts each word |
| `--seed` | none | channel random seed, for reproducible noise |
| `--interactive` | off | you type the interrogator's questions and give the verdict |
| `--transcript FILE` | none | save the console log |
| `--interrogation-md FILE` | none | write the interrogation as Markdown (the 05 deliverable) |

### What each mode does

| Mode | Part | What happens |
|---|---|---|
| `fst` | 2, 3 | Canonical course-plotting dialogue with state trace, then 6 stress tests (unexpected order, self-correction, prompt injection, negation, sarcasm, off-topic) with an OK/BREAK verdict each. |
| `fst-noisy` | 4 | An LLM crew member plots a course with the FST; every utterance passes through the noisy channel first. Reports slot accuracy. |
| `probe` | 1 | The 5 Anti-Turing questions, asked without channel limits. |
| `interrogate` | 5 | An LLM interrogator questions AURA over the noisy channel, then judges human or machine. Both sides only see what the channel delivered. |

Channel metrics per message: words cut, words corrupted, entropy of sent and received message, content-word retention. Per run: symbol error rate p, transcript entropy, and the capacity of a q-ary symmetric channel, C = log₂q − H_b(p) − p·log₂(q−1).

## Station sign-offs

Each officer ticks their own box once they have read every deliverable and agree the squad's submission is ready.

- [ ] **architect** — Systems Architect — Mohid
- [ ] **protocol** — Communications & Protocol Officer — Sharjeel
- [ ] **data** — Data & Payload Specialist — Shayan
- [ ] **security** — Security & Ethics Sentinel — Ahmed
