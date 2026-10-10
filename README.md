# Squadron Hinton — Mission 01: The Architecture of Argument

**Course:** 14013 Conversational Artificial Intelligence: History, Application, Future — WiSe 26/27
**Mission:** ps01 — Dialogic Protocols (Socratic Elenchus & Galilean 3-Persona Debate)
**Due:** 2026-10-18
**Instructor repo:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai) (`mission-01/`)

AURA's conversational matrix is offline. This repository rebuilds her two foundational dialogue protocols:

- **Discovery** — a Socratic elenchus state machine that tests a claim against counter-examples until it reaches a refined definition or aporia.
- **Persuasion** — a Galilean debate in which Salviati (advocate), Simplicio (skeptic) and Sagredo (mediator) argue an AI-ethics question.

## Squad roster

| Station | Role | Officer | Deliverable |
|---|---|---|---|
| data | Knowledge & Memory Archivist | _TBD_ | [`01_Alignment_Brief.md`](01_Alignment_Brief.md) |
| protocol | Protocol & Interface Specialist | _TBD_ | [`02_Socratic_Protocol_Map.md`](02_Socratic_Protocol_Map.md) |
| security | Security & Ethical Navigator | _TBD_ | [`03_Galilean_Sandbox.md`](03_Galilean_Sandbox.md) |
| architect | Core & Lead Architect | _TBD_ | [`04_dialogic_engine.py`](04_dialogic_engine.py), [`05_Execution_Transcript.txt`](05_Execution_Transcript.txt) |

## Repository layout

```
squadron-hinton-ps01/
├── README.md                     roster, sign-offs, how to run
├── 01_Alignment_Brief.md         Knowledge Archivist
├── 02_Socratic_Protocol_Map.md   Protocol Specialist
├── 03_Galilean_Sandbox.md        Ethical Navigator
├── 04_dialogic_engine.py         Lead Architect
├── 05_Execution_Transcript.txt   output of a live run
├── requirements.txt
└── .env.example                  API key names (copy to .env)
```

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add the key for your provider
python 04_dialogic_engine.py
```

## Station sign-offs

Each officer ticks their own box once they have read every deliverable and agree the squad's submission is ready.

- [ ] **architect** — Core & Lead Architect — _name, date_
- [ ] **protocol** — Protocol & Interface Specialist — _name, date_
- [ ] **data** — Knowledge & Memory Archivist — _name, date_
- [ ] **security** — Security & Ethical Navigator — _name, date_
