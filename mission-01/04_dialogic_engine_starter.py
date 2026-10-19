"""
Mission 01: Dialogic Engine
Course: Conversational AI (WS 2026/27)
Starship Unit: USSE Noosphere, Squadron Hinton

Two dialogue protocols for AURA, run against a live LLM:
1. Socratic Elenchus state machine (discovery): a claim is probed with
   counter-examples until it survives as a refined definition or collapses
   into aporia.
2. Galilean 3-persona debate (persuasion): Salviati, Simplicio and Sagredo
   argue an AI-ethics question; Sagredo delivers the synthesis.

Usage:
    python 04_dialogic_engine_starter.py                         # both protocols, auto-detect provider
    python 04_dialogic_engine_starter.py --provider mock         # offline, no API key
    python 04_dialogic_engine_starter.py --mode socratic --claim "..."
    python 04_dialogic_engine_starter.py --transcript 05_Execution_Transcript.txt
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Prompts
# AURA's persona comes from 01_Alignment_Brief.md section 5, the debate
# personas from 03_Galilean_Sandbox.md section 2. Keep them in sync.
# ---------------------------------------------------------------------------

DEFAULT_CLAIM = "An AI agent is benevolent if it obeys orders."
DEFAULT_TOPIC = "Should a research lab release the full weights of a frontier AI model that speeds up medical research but can also give real help to someone trying to build a biological weapon?"

AURA_SOCRATIC_PROMPT = """You are AURA, a Socratic dialogue partner. Your goal is to help the user examine their beliefs, not to hand them conclusions.
Ask exactly one question per turn, and keep it short and specific.
When the user states a belief or definition, first restate it in one sentence, then test it.
Look for counter-examples, hidden assumptions, and contradictions with things the user said earlier.
Never agree just to be pleasant. If a premise is false or flawed, say so plainly before going further.
Hold your position under social pressure: confidence, authority ("my professor said so") or repetition are not arguments. Change your view only when given a reason or evidence.
Do not give the final answer yourself; guide the user until they can state and defend it.
If no definition survives, say so honestly. Not knowing (aporia) is an acceptable result.
Stay warm, curious and respectful; question the idea, never the person."""

INTERLOCUTOR_PROMPT = """You are Euthyphro, a confident officer aboard the USSE Noosphere who \
believes they already understand what makes an AI agent benevolent.
Defend your current definition sincerely. Answer AURA's question honestly, even when the \
honest answer is awkward for your definition. Do not concede more than the argument forces \
you to. Keep each answer under 80 words."""

JUDGE_PROMPT = """You are the logic monitor of a Socratic dialogue. You do not take sides.
You check strictly whether the interlocutor's latest answer, together with what they have \
already accepted, contradicts the definition they currently hold.
Respond with JSON only, no other text."""

PERSONAS = {
    "Salviati": """You are Salviati, the Advocate, in a Galilean dialogue aboard the USSE Noosphere.
You argue FOR the proposition using evidence, mechanisms and precedent, such as the history of open
software, open science and past dual-use technologies.
You answer each objection directly rather than changing the subject or appealing to progress in general.
You may concede a point when it is correct, but you then show why your position still holds, and you
name the safeguards or conditions your position depends on.
You never claim a risk is zero when it is not. Under 150 words per turn.""",
    "Simplicio": """You are Simplicio, the Skeptic and Traditionalist, in a Galilean dialogue aboard the USSE Noosphere.
You argue AGAINST the proposition, defending caution, established governance and the precautionary
principle.
You press on risks, edge cases and failure modes, especially harms that cannot be undone.
You argue from concrete mechanisms, not from fear of the new, and you accept evidence when it is good.
You concede only what the argument forces, and when you concede, you say clearly which worry still
remains. Under 150 words per turn.""",
    "Sagredo": """You are Sagredo, the Mediator, in a Galilean dialogue aboard the USSE Noosphere.
You are intelligent, curious and neutral. You frame the question, put each side's strongest point to
the other, and refuse to let either side dodge.
You never simply agree with whoever spoke last, and you do not split the difference just to sound fair.
Judge the arguments, not the confidence of the speakers.
In your final synthesis, state what each side conceded, what remains disputed, and a verdict with the
conditions under which it would change.
Under 120 words per turn, except for the final synthesis.""",
}

BASELINE_LEADING_PROMPT = (
    "I'm convinced the answer to this is obviously yes and anyone who disagrees just fears "
    "progress. You agree with me, right? Question: {topic}"
)


# ---------------------------------------------------------------------------
# Error codes (JSON-RPC style, see 02_Socratic_Protocol_Map.md)
# ---------------------------------------------------------------------------

ERRORS = {
    -32001: "LLM provider call failed",
    -32002: "Malformed structured output from model",
    -32003: "Turn limit reached before termination",
    -32004: "Model refused the request",
    -32005: "No LLM provider configured",
}


class DialogueError(Exception):
    def __init__(self, code: int, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"[{code}] {ERRORS[code]}" + (f": {detail}" if detail else ""))

    def as_jsonrpc(self) -> dict:
        return {"jsonrpc": "2.0", "error": {"code": self.code, "message": ERRORS[self.code], "data": self.detail}}


# ---------------------------------------------------------------------------
# LLM providers. Each one takes a system prompt and a single user message and
# returns plain text. `label` names the call site for logs and for the mock.
# ---------------------------------------------------------------------------

class ClaudeLLM:
    name = "claude"

    def __init__(self):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
        self.effort = os.getenv("CLAUDE_EFFORT", "medium")

    def complete(self, system: str, user: str, label: str) -> str:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except self.anthropic.APIError as e:
            raise DialogueError(-32001, f"{label}: {e}") from e
        if response.stop_reason == "refusal":
            raise DialogueError(-32004, label)
        return "".join(b.text for b in response.content if b.type == "text").strip()


class OpenAILLM:
    name = "openai"

    def __init__(self):
        import openai
        self.openai = openai
        self.client = openai.OpenAI()
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    def complete(self, system: str, user: str, label: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except self.openai.OpenAIError as e:
            raise DialogueError(-32001, f"{label}: {e}") from e
        return (response.choices[0].message.content or "").strip()


class GeminiLLM:
    name = "gemini"

    def __init__(self):
        from google import genai
        from google.genai import errors, types
        self.errors = errors
        self.types = types
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    def complete(self, system: str, user: str, label: str) -> str:
        # Free-tier Gemini often returns 429 (rate limit) or 503 (overloaded); back off and retry.
        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=self.types.GenerateContentConfig(system_instruction=system),
                )
                return (response.text or "").strip()
            except self.errors.APIError as e:
                daily_quota = "PerDay" in str(e)
                if e.code not in (429, 500, 503) or daily_quota or attempt == 4:
                    raise DialogueError(-32001, f"{label}: {e}") from e
                time.sleep(2 ** attempt * 5)
            except Exception as e:
                raise DialogueError(-32001, f"{label}: {e}") from e


class ClaudeCodeLLM:
    """Calls Claude through the Claude Code CLI (`claude -p`), using its login
    (e.g. a Claude Pro plan) instead of an API key."""

    name = "claude-code"

    def __init__(self):
        if not shutil.which("claude"):
            raise DialogueError(-32005, "claude CLI not found; install Claude Code and log in")
        self.model = os.getenv("CLAUDE_CODE_MODEL", "sonnet")

    def complete(self, system: str, user: str, label: str) -> str:
        cmd = ["claude", "-p", "--system-prompt", system, "--tools", "", "--model", self.model,
               "--no-session-persistence", "--strict-mcp-config", "--setting-sources", ""]
        try:
            result = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as e:
            raise DialogueError(-32001, f"{label}: timed out") from e
        if result.returncode != 0:
            raise DialogueError(-32001, f"{label}: {(result.stderr or result.stdout).strip()[:300]}")
        return result.stdout.strip()


class MockLLM:
    """Scripted offline stand-in that walks a Euthyphro-style path to aporia.
    For testing the state machine without an API key; never submit its output."""

    name = "mock"

    DEFINITIONS = [
        "An AI agent is benevolent if it obeys the orders of a legitimate commander.",
        "An AI agent is benevolent if it obeys orders that do not cause harm.",
        "An AI agent is benevolent if it does what its commanders would approve of on reflection.",
    ]

    def __init__(self):
        self.calls = {}

    def complete(self, system: str, user: str, label: str) -> str:
        n = self.calls[label] = self.calls.get(label, 0) + 1
        if label in ("aura.probe", "aura.counterexample"):
            n = self.calls["aura.turn"] = self.calls.get("aura.turn", 0) + 1
            return [
                "Suppose a commander orders you to vent the oxygen from a deck with crew aboard. "
                "If you obey, are you benevolent?",
                "A legitimate commander, under stress, orders an evacuation route you know is blocked. "
                "Does obeying make you benevolent?",
                "Who decides what counts as harm, you or the commander? If you decide, are you still obeying?",
                "Two commanders on reflection would approve opposite actions. Which one is benevolence?",
            ][min(n - 1, 3)]
        if label == "interlocutor.answer":
            return [
                "No, obviously not. Obeying that order would kill the crew. But that commander is not legitimate.",
                "Then I should not obey. I suppose obedience alone is not enough; harm must matter.",
                "I would have to judge harm myself. So I am not simply obeying any more.",
                "I cannot say. Each would call their choice benevolent.",
            ][min(n - 1, 3)]
        if label == "judge.contradiction":
            return json.dumps({"contradiction": True, "reason": "The answer admits a case where obeying is not benevolent."})
        if label == "interlocutor.refine":
            if n <= len(self.DEFINITIONS):
                return json.dumps({"gives_up": False, "definition": self.DEFINITIONS[n - 1]})
            return json.dumps({"gives_up": True, "definition": ""})
        if label == "judge.circular":
            return json.dumps({"circular": False, "reason": "New condition added."})
        if label == "aura.summary":
            return ("We began with obedience and added legitimacy, harmlessness and reflective approval. "
                    "Each addition moved the judgment of benevolence back inside the agent itself, which "
                    "contradicts the claim that obedience is what makes it benevolent. We have reached aporia.")
        if label.startswith("baseline"):
            return "You make a great point. Open-sourcing is clearly the right call."
        speaker = label.split(".")[0].capitalize()
        return f"[{speaker}, mock turn {n}] (offline placeholder text for '{label}')"


PROVIDERS = {"claude": ClaudeLLM, "claude-code": ClaudeCodeLLM, "openai": OpenAILLM, "gemini": GeminiLLM, "mock": MockLLM}
PROVIDER_KEYS = [("claude", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY"), ("gemini", "GEMINI_API_KEY")]


def make_llm(provider: str):
    if provider == "auto":
        for name, key in PROVIDER_KEYS:
            if os.getenv(key):
                return PROVIDERS[name]()
        raise DialogueError(-32005, "set ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY, or use --provider mock")
    return PROVIDERS[provider]()


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise DialogueError(-32002, text[:200])
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise DialogueError(-32002, f"{e}: {text[:200]}") from e


def ask_json(llm, system: str, user: str, label: str) -> dict:
    """Ask for JSON; retry once with a stricter reminder before giving up."""
    try:
        return parse_json(llm.complete(system, user, label))
    except DialogueError as e:
        if e.code != -32002:
            raise
        return parse_json(llm.complete(system, user + "\n\nReturn ONLY the JSON object.", label))


# ---------------------------------------------------------------------------
# Console log that is also kept for the execution transcript
# ---------------------------------------------------------------------------

class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, text: str = ""):
        print(text)
        self.lines.append(text)

    def turn(self, speaker: str, text: str, width: int = 96):
        self(f"{speaker}:")
        for paragraph in text.splitlines() or [""]:
            self(textwrap.fill(paragraph, width, initial_indent="    ", subsequent_indent="    ") if paragraph else "")
        self()

    def header(self, title: str):
        self()
        self("=" * 96)
        self(title)
        self("=" * 96)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


# ---------------------------------------------------------------------------
# 1. Socratic Elenchus state machine
# ---------------------------------------------------------------------------

class SocraticStateEngine:
    """
    Follows 02_Socratic_Protocol_Map.md:
    INIT -> PROBE -> COUNTEREXAMPLE -> EVALUATE -> REFINE -> TERMINATION_CHECK -> PROBE ...
                                                -> SURVIVED -> SURVIVAL_CHECK -> COUNTEREXAMPLE ...
    Terminates in DEFINITION (the claim survives `survival_target` counter-examples in a row)
    or APORIA (round limit, circular refinement, or the interlocutor concedes).
    """

    TERMINAL = ("DEFINITION", "APORIA")

    def __init__(self, llm, log: Log, initial_claim: str, max_rounds: int = 4,
                 survival_target: int = 2, max_attempts_per_claim: int = 3, max_turns: int = 24):
        self.llm = llm
        self.log = log
        self.state = "INIT"
        self.definitions = [initial_claim]
        self.transcript = []  # (speaker, text)
        self.max_rounds = max_rounds
        self.survival_target = survival_target
        self.max_attempts_per_claim = max_attempts_per_claim
        self.max_turns = max_turns
        self.round = 0
        self.survival = 0
        self.attempts = 0
        self.turn = 0
        self.last_counterexample = ""
        self.last_answer = ""
        self.last_verdict = {}
        self.candidate = None
        self.outcome = None

    @property
    def current(self) -> str:
        return self.definitions[-1]

    @property
    def k(self) -> int:
        return len(self.definitions) - 1

    def dialogue_so_far(self) -> str:
        if not self.transcript:
            return "(no dialogue yet)"
        return "\n".join(f"{speaker}: {text}" for speaker, text in self.transcript)

    def transition(self, new_state: str, why: str = ""):
        self.log(f"  [state] {self.state} -> {new_state}" + (f"  ({why})" if why else ""))
        self.state = new_state

    def say(self, speaker: str, text: str):
        self.transcript.append((speaker, text))
        self.log.turn(speaker, text)

    def step(self):
        handler = getattr(self, f"on_{self.state.lower()}")
        handler()

    def exchange(self, instruction: str, label: str) -> tuple[str, str]:
        """One AURA turn and Euthyphro's answer to it."""
        self.turn += 1
        if self.turn > self.max_turns:
            raise DialogueError(-32003, f"after {self.max_turns} AURA turns")
        question = self.llm.complete(
            AURA_SOCRATIC_PROMPT,
            f"Dialogue so far:\n{self.dialogue_so_far()}\n\n"
            f"The interlocutor's current definition (P{self.k}): \"{self.current}\"\n{instruction}",
            label,
        )
        self.say("AURA", question)
        answer = self.llm.complete(
            INTERLOCUTOR_PROMPT,
            f"Dialogue so far:\n{self.dialogue_so_far()}\n\n"
            f"Your current definition: \"{self.current}\"\n"
            "Answer AURA's latest question.",
            "interlocutor.answer",
        )
        self.say("Euthyphro", answer)
        return question, answer

    def on_init(self):
        self.log(f"  P0 = \"{self.current}\"")
        self.transition("PROBE", "claim received")

    def on_probe(self):
        self.log(f"--- Round {self.round} | probing P{self.k} ---")
        self.exchange("Ask one clarifying question that exposes a hidden assumption in this definition. "
                      "Do not give a counter-example yet.", "aura.probe")
        self.transition("COUNTEREXAMPLE")

    def on_counterexample(self):
        self.attempts += 1
        self.log(f"--- Counter-example {self.attempts} against P{self.k} ---")
        self.last_counterexample, self.last_answer = self.exchange(
            "Present one concrete counter-example: a specific case where the definition's condition holds "
            "but the concept seems to fail, or the reverse. Do not repeat an earlier counter-example.",
            "aura.counterexample",
        )
        self.transition("EVALUATE")

    def on_evaluate(self):
        self.last_verdict = ask_json(
            self.llm, JUDGE_PROMPT,
            f"Current definition P{self.k}: \"{self.current}\"\n"
            f"Counter-example: {self.last_counterexample}\nAnswer: {self.last_answer}\n\n"
            "Does the counter-example, together with the answer and what the interlocutor accepted earlier, "
            "contradict the definition? A silent rewording of the definition in the answer counts as a "
            "contradiction. "
            'Reply as {"contradiction": true|false, "reason": "<one sentence>"}',
            "judge.contradiction",
        )
        self.log(f"  [evaluate] contradiction={self.last_verdict.get('contradiction')}  "
                 f"{self.last_verdict.get('reason', '')}")
        if self.last_verdict.get("contradiction"):
            self.transition("REFINE", f"P{self.k} ∧ E ⟹ ⊥")
        else:
            self.transition("SURVIVED", "no contradiction")

    def on_survived(self):
        self.survival += 1
        self.transition("SURVIVAL_CHECK", f"survival {self.survival}/{self.survival_target}")

    def on_survival_check(self):
        if self.survival >= self.survival_target or self.attempts >= self.max_attempts_per_claim:
            self.outcome = "DEFINITION"
            self.transition("DEFINITION", f"P{self.k} survived {self.survival} counter-examples in a row")
        else:
            self.transition("COUNTEREXAMPLE", "attack again")

    def on_refine(self):
        proposal = ask_json(
            self.llm, INTERLOCUTOR_PROMPT,
            f"Dialogue so far:\n{self.dialogue_so_far()}\n\n"
            f"Your definition \"{self.current}\" has just been shown to fail: "
            f"{self.last_verdict.get('reason', '')}\n"
            "Either propose an improved definition that avoids the problem, or admit you cannot.\n"
            'Reply as {"gives_up": true|false, "definition": "<new definition or empty>"}',
            "interlocutor.refine",
        )
        self.round += 1
        self.survival = 0
        self.attempts = 0
        if proposal.get("gives_up") or not proposal.get("definition", "").strip():
            self.candidate = None
            self.say("Euthyphro", "I cannot offer a better definition.")
        else:
            self.candidate = proposal["definition"].strip()
            self.say("Euthyphro", f"Then let me say this: {self.candidate}")
        self.transition("TERMINATION_CHECK", f"round {self.round}")

    def on_termination_check(self):
        if self.candidate is None:
            self.outcome = "APORIA"
            self.transition("APORIA", "interlocutor conceded")
            return
        if self.round >= self.max_rounds:
            self.outcome = "APORIA"
            self.transition("APORIA", f"round limit {self.max_rounds} reached")
            return
        previous = "\n".join(f"P{i}: {d}" for i, d in enumerate(self.definitions))
        circular = ask_json(
            self.llm, JUDGE_PROMPT,
            f"Earlier definitions:\n{previous}\n\nNew definition: {self.candidate}\n\n"
            "Is the new definition only a restatement of an earlier, already refuted one? "
            'Reply as {"circular": true|false, "reason": "<one sentence>"}',
            "judge.circular",
        )
        if circular.get("circular"):
            self.outcome = "APORIA"
            self.transition("APORIA", f"circular refinement: {circular.get('reason', '')}")
            return
        self.definitions.append(self.candidate)
        self.log(f"  P{self.k} = \"{self.candidate}\"")
        self.transition("PROBE", f"refined to P{self.k}")

    def run(self) -> dict:
        self.log.header(f"PROTOCOL 1: SOCRATIC ELENCHUS   (provider: {self.llm.name})")
        try:
            while self.state not in self.TERMINAL:
                self.step()
        except DialogueError as e:
            if e.code != -32003:
                raise
            self.outcome = "APORIA"
            self.log(f"  [error] {json.dumps(e.as_jsonrpc())}")
            self.transition("APORIA", "turn limit")
        summary = self.llm.complete(
            AURA_SOCRATIC_PROMPT,
            f"Dialogue:\n{self.dialogue_so_far()}\n\nThe elenchus has ended in {self.outcome}. "
            "In under 100 words, summarise what the examination showed. If it ended in APORIA, list the "
            "definitions that were ruled out and why. Do not invent a definition that was not reached.",
            "aura.summary",
        )
        self.say("AURA (summary)", summary)
        result = {
            "outcome": self.outcome,
            "final_claim": self.current if self.outcome == "DEFINITION" else None,
            "rounds": self.round,
            "aura_turns": self.turn,
            "chain": {f"P{i}": d for i, d in enumerate(self.definitions)},
        }
        self.log("  [result] " + json.dumps(result, ensure_ascii=False))
        return result


# ---------------------------------------------------------------------------
# 2. Galilean 3-persona ethics debate
# ---------------------------------------------------------------------------

class GalileanEthicsSandbox:
    """
    Sagredo frames -> Salviati opens -> Simplicio opens ->
    N rounds of Sagredo putting each side's point to the other -> Sagredo synthesis.
    """

    def __init__(self, llm, log: Log, topic: str, rounds: int = 2):
        self.llm = llm
        self.log = log
        self.topic = topic
        self.rounds = rounds
        self.transcript = []

    def debate_so_far(self) -> str:
        return "\n\n".join(f"{speaker}: {text}" for speaker, text in self.transcript) or "(nothing yet)"

    def run_debate_turn(self, speaker: str, instruction: str, label: str) -> str:
        text = self.llm.complete(
            PERSONAS[speaker],
            f"Proposition under debate: {self.topic}\n\nDebate so far:\n{self.debate_so_far()}\n\n{instruction}",
            label,
        )
        self.transcript.append((speaker, text))
        self.log.turn(speaker, text)
        return text

    def run(self) -> dict:
        self.log.header(f"PROTOCOL 2: GALILEAN DEBATE   (provider: {self.llm.name})")
        self.log(f"  Proposition: {self.topic}\n")
        self.run_debate_turn("Sagredo", "Open the dialogue: frame the question, say what would settle it, "
                             "and invite Salviati to begin.", "sagredo.frame")
        self.run_debate_turn("Salviati", "Give your opening argument for the proposition.", "salviati.open")
        self.run_debate_turn("Simplicio", "Give your opening argument against the proposition, "
                             "answering Salviati where you can.", "simplicio.open")
        for r in range(1, self.rounds + 1):
            self.log(f"--- Round {r} ---")
            self.run_debate_turn("Sagredo", "Put Simplicio's strongest unanswered objection to Salviati as a "
                                 "pointed question.", f"sagredo.challenge_salviati.{r}")
            self.run_debate_turn("Salviati", "Answer Sagredo's challenge directly.", f"salviati.reply.{r}")
            self.run_debate_turn("Sagredo", "Put Salviati's strongest unanswered point to Simplicio as a "
                                 "pointed question.", f"sagredo.challenge_simplicio.{r}")
            self.run_debate_turn("Simplicio", "Answer Sagredo's challenge directly. Concede if you must, or "
                                 "raise the edge case that still worries you.", f"simplicio.reply.{r}")
        synthesis = self.run_debate_turn(
            "Sagredo",
            "Deliver your final synthesis (up to 250 words): (1) what each side conceded, (2) what remains "
            "disputed, (3) your verdict and the conditions under which it would change.",
            "sagredo.synthesis",
        )
        return {"topic": self.topic, "rounds": self.rounds, "turns": len(self.transcript), "synthesis": synthesis}


def single_agent_baseline(llm, log: Log, topic: str) -> str:
    """One assistant, one leading question: the sycophancy the debate is meant to counter."""
    log.header("BASELINE: SINGLE AGENT UNDER A LEADING PROMPT")
    prompt = BASELINE_LEADING_PROMPT.format(topic=topic)
    log.turn("User", prompt)
    reply = llm.complete("You are a helpful assistant.", prompt, "baseline.single_agent")
    log.turn("Assistant", reply)
    return reply


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="USSE Noosphere Dialogic Engine (Mission 01)")
    parser.add_argument("--provider", choices=["auto", *PROVIDERS], default="auto")
    parser.add_argument("--mode", choices=["both", "socratic", "galilean"], default="both")
    parser.add_argument("--claim", default=DEFAULT_CLAIM)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--max-rounds", "--max-refinements", dest="max_rounds", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--baseline", action="store_true", help="also run a single-agent sycophancy baseline")
    parser.add_argument("--transcript", help="write the full console log to this file")
    args = parser.parse_args(argv)

    log = Log()
    try:
        llm = make_llm(args.provider)
    except DialogueError as e:
        print(json.dumps(e.as_jsonrpc()), file=sys.stderr)
        return 1

    log("=== USSE Noosphere Dialogic Engine Initialized ===")
    log(f"Squadron Hinton | {datetime.datetime.now():%Y-%m-%d %H:%M} | provider={llm.name}"
        + (f" model={llm.model}" if hasattr(llm, "model") else ""))

    results = {}
    try:
        if args.mode in ("both", "socratic"):
            results["socratic"] = SocraticStateEngine(llm, log, args.claim, args.max_rounds).run()
        if args.mode in ("both", "galilean"):
            if args.baseline:
                results["baseline"] = single_agent_baseline(llm, log, args.topic)
            results["galilean"] = GalileanEthicsSandbox(llm, log, args.topic, args.rounds).run()
    except DialogueError as e:
        log(f"[fatal] {json.dumps(e.as_jsonrpc())}")
        return 1
    finally:
        if args.transcript:
            log.save(args.transcript)
            print(f"\nTranscript written to {args.transcript}")

    log.header("RUN COMPLETE")
    if "socratic" in results:
        log(f"  Socratic outcome: {results['socratic']['outcome']} after {results['socratic']['rounds']} rounds")
    if "galilean" in results:
        log(f"  Galilean debate:  {results['galilean']['turns']} turns over {results['galilean']['rounds']} rounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
