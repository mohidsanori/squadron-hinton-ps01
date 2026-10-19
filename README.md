# Squadron Hinton — Conversational AI Missions

**Course:** 14013 Conversational Artificial Intelligence: History, Application, Future — WiSe 26/27
**Instructor repo:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)

Our solutions, one folder per mission, matching the instructor repo's layout.

| Mission | Sector Waypoint | Focus | Folder | Due |
|---|---|---|---|---|
| **Mission 01** | *Alpha Centauri* | Socratic Elenchus & Galilean 3-Persona Debate | [`mission-01/`](./mission-01/) | 2026-10-18 |
| **Mission 02** | *Sirius Binary System* | FST Dialogue Systems, Lovelace's Objection & Shannon Noisy Channel | [`mission-02/`](./mission-02/) | 2026-10-25 |

Each mission folder has its own README with the roster, deliverables, run instructions and station sign-offs.

## Squad

Mohid, Shayan, Sharjeel, Ahmed.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add the key for your LLM provider; never commit .env
```

`requirements.txt` and `.env.example` are shared by all missions.
