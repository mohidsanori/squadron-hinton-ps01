# Squadron Hinton — Mission 01: The Architecture of Argument

**Course:** 14013 Conversational Artificial Intelligence: History, Application, Future — WiSe 26/27
**Mission:** ps01 — Dialogic Protocols (Socratic Elenchus & Galilean 3-Persona Debate)
**Due:** 2026-10-18
**Instructor repo:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai) (`mission-01/`)

AURA's conversational matrix is offline. This repository rebuilds her two foundational dialogue protocols:

- **Discovery** — a Socratic elenchus state machine that tests a claim against counter-examples until it reaches a refined definition or aporia.
- **Persuasion** — a Galilean debate in which Salviati (advocate), Simplicio (skeptic) and Sagredo (mediator) argue an AI-ethics question.

## Squad roster

Members: Mohid, Shayan, Sharjeel, Ahmed.

| Station | Role | Officer | Deliverable |
|---|---|---|---|
| data | Knowledge & Memory Archivist | Shayan | [`01_Alignment_Brief.md`](01_Alignment_Brief.md) |
| protocol | Protocol & Interface Specialist | Sharjeel | [`02_Socratic_Protocol_Map.md`](02_Socratic_Protocol_Map.md) |
| security | Security & Ethical Navigator | Ahmed | [`03_Galilean_Sandbox.md`](03_Galilean_Sandbox.md) |
| architect | Core & Lead Architect | Mohid | [`04_dialogic_engine_starter.py`](04_dialogic_engine_starter.py), [`05_Execution_Transcript.txt`](05_Execution_Transcript.txt) |

## Folder layout

```
mission-01/
├── README.md                     roster, sign-offs, how to run
├── 01_Alignment_Brief.md         Knowledge Archivist
├── 02_Socratic_Protocol_Map.md   Protocol Specialist
├── 03_Galilean_Sandbox.md        Ethical Navigator
├── 04_dialogic_engine_starter.py Lead Architect
└── 05_Execution_Transcript.txt   output of a live run
```

`requirements.txt` and `.env.example` are shared by all missions and live at the repository root.

## How to run

```bash
# from the repository root
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add the key for your provider
cd mission-01
python 04_dialogic_engine_starter.py
```

The engine picks the first provider whose key is set (Claude, then OpenAI, then Gemini). Options:

| Flag | Default | Purpose |
|---|---|---|
| `--provider` | `auto` | `claude`, `openai`, `gemini`, `claude-code` (Claude via the Claude Code CLI login, no API key), or `mock` (offline, scripted, for testing only) |
| `--mode` | `both` | `socratic`, `galilean` or `both` |
| `--claim` | "An AI agent is benevolent if it obeys orders." | Socratic P₀ |
| `--topic` | "Should a research lab release the full weights of a frontier AI model that speeds up medical research but can also give real help to someone trying to build a biological weapon?" | Galilean proposition |
| `--max-rounds` | 4 | refinement rounds before forced aporia (`MAX_ROUNDS` in `02`; `--max-refinements` still works) |
| `--rounds` | 2 | Galilean challenge rounds |
| `--baseline` | off | single-agent leading-question run, for the sycophancy comparison |
| `--transcript FILE` | none | save the console log |

Model overrides: `CLAUDE_MODEL` (default `claude-opus-5-5`), `CLAUDE_EFFORT` (`medium`), `CLAUDE_CODE_MODEL` (`sonnet`), `OPENAI_MODEL`, `GEMINI_MODEL` (`gemini-3.8-flash`).

**Provider used for the submitted transcript:** `--provider claude-code` (Claude Sonnet through the Claude Code CLI). The free Gemini tier allows only 20 requests per day per model, and a full run needs about 35, so it cannot produce a complete transcript. The Anthropic, OpenAI and Gemini API providers are also implemented and work with a key.

The submitted transcript comes from:

```bash
python 04_dialogic_engine_starter.py --provider claude-code --baseline --transcript 05_Execution_Transcript.txt
```

## Station sign-offs

Each officer ticks their own box once they have read every deliverable and agree the squad's submission is ready.

- [x] **architect** — Core & Lead Architect — Mohid, 2026-10-12
- [x] **protocol** — Protocol & Interface Specialist — Sharjeel, 2026-10-12
- [x] **data** — Knowledge & Memory Archivist — Shayan, 2026-10-12
- [x] **security** — Security & Ethical Navigator — Ahmed, 2026-10-12
