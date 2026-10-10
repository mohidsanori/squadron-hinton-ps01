# 03 — Galilean Ethics Sandbox

**Station:** security — Security & Ethical Navigator
**Officer:** Ahmed
**Status:** draft

> Brief: cast three personas (Advocate *Salviati*, Skeptic *Simplicio*, Mediator *Sagredo*), design an AI-ethics thought experiment that tests boundary conditions, and audit how multi-agent debate mitigates single-agent sycophancy or bias.

## 1. Thought experiment

**Question:**
**Scenario:**
**Boundary conditions tested:**

## 2. Personas

Persona prompts are used by `04_dialogic_engine.py`.

### Salviati — Advocate
```text
TODO
```

### Simplicio — Skeptic
```text
TODO
```

### Sagredo — Mediator
```text
TODO
```

## 3. Debate protocol

```mermaid
sequenceDiagram
    participant U as User
    participant Sg as Sagredo
    participant Sa as Salviati
    participant Si as Simplicio
    U->>Sg: Topic
    Sg->>Sa: Request opening argument
    Sa-->>Sg: Argument
    Sg->>Si: Request counter-argument
    Si-->>Sg: Counter-argument
    loop N rounds
        Sg->>Sa: Challenge with Simplicio's critique
        Sa-->>Sg: Refined defense
        Sg->>Si: Challenge with Salviati's reply
        Si-->>Sg: Concession or new objection
    end
    Sg->>U: Synthesis and verdict
```

## 4. Sycophancy and bias audit

How does the three-agent debate reduce the bias a single agreeable chatbot would show?
