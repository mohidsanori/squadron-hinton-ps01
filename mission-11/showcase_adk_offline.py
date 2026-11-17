"""
Mission 11 - Part 5, Track A, offline run: the real Google ADK 2.x runtime with a scripted model
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-11-M31 (Andromeda Galaxy Core), Subsystem 11

`andromeda_crew/agent.py` (the instructor's Track A crew) is loaded unchanged and executed by ADK's own
`Runner`, `InMemorySessionService`, `AgentTool` calls and `@node` workflow engine. The only substitution
is the language model: each agent's `model="gemini-flash-latest"` is replaced by `ScriptedLlm`, a
deterministic `BaseLlm` that returns fixed responses, so the run needs no API key and no network.

What is real: ADK's event stream, the root agent's tool calls into the two specialists (`AgentTool`), the
specialists' own agent runs, the deterministic `@node` gate and the bounded repair loop.
What is scripted: every model response. Nothing here says anything about Gemini's behaviour.

This is NOT the `adk web` screenshot the briefing asks for. To produce that, set GOOGLE_API_KEY, run
`adk web` from this folder, open http://localhost:8000, select `andromeda_crew` and capture the Events /
Graph Trace view.

Run with a Python that has google-adk installed:
    pip install google-adk
    python showcase_adk_offline.py
"""

import asyncio
import json
import sys
from typing import AsyncGenerator, List, Optional

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import andromeda_crew.agent as crew


class ScriptedLlm(BaseLlm):
    """A deterministic stand-in for Gemini: returns the next scripted response on every call."""
    model: str = "scripted-offline"
    script: List[types.Content] = []
    calls: int = 0

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        if self.calls >= len(self.script):
            raise RuntimeError(f"{self.model}: script exhausted after {self.calls} calls")
        content = self.script[self.calls]
        self.calls += 1
        yield LlmResponse(content=content)


def text(s: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=s)])


def call(name: str, **args) -> types.Content:
    return types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])


def describe(event) -> str:
    """One line per ADK event: who spoke and what it carried."""
    parts = []
    if event.content and event.content.parts:
        for p in event.content.parts:
            if p.function_call:
                parts.append(f"CALL {p.function_call.name}({json.dumps(dict(p.function_call.args or {}))[:90]})")
            elif p.function_response:
                parts.append(f"RESULT {p.function_response.name} -> {str(p.function_response.response)[:90]}")
            elif p.text:
                parts.append(f"TEXT {p.text.strip()[:110]}")
    if getattr(event, "output", None) is not None:
        parts.append(f"OUTPUT {str(event.output)[:110]}")
    state = getattr(getattr(event, "actions", None), "state_delta", None)
    if state:
        parts.append(f"STATE {json.dumps(state)[:110]}")
    return f"{event.author:<28} " + (" | ".join(parts) if parts else "(no content)")


async def run_runner(runner: Runner, session_id: str, message: str, state: Optional[dict] = None) -> List:
    events = []
    msg = types.Content(role="user", parts=[types.Part(text=message)])
    async for ev in runner.run_async(user_id="bridge", session_id=session_id, new_message=msg, state_delta=state):
        events.append(ev)
        print("  " + describe(ev))
    return events


async def part_a_coordinator() -> List:
    """The root coordinator calls AstrogationSpecialist and SafetyCritic through AgentTool."""
    print("\n[A] Root coordinator: AgentTool calls into the two specialists (ADK Runner, scripted model)")
    crew.astrogation_specialist.model = ScriptedLlm(script=[text(
        "Orbital plan around M31*: velocity_kms=204.5, inclination_deg=38.2, clearance_ly=0.42. "
        "Passes outside nucleus P1 apoapsis and crosses the P2 disk at periapsis with margin.")])
    crew.safety_critic.model = ScriptedLlm(script=[text(
        "Audit: tidal shear at M31* within hull tolerance; P1 and P2 clearance margins verified. Verdict: CLEARED.")])
    crew.root_agent.model = ScriptedLlm(script=[
        call("AstrogationSpecialist", request="Draft a P1/P2 trajectory for SEC-11-M31."),
        call("SafetyCritic", request="Audit the AstrogationSpecialist plan: velocity 204.5 km/s, inclination 38.2 deg, clearance 0.42 ly."),
        text("Bridge clearance issued: velocity 204.5 km/s, inclination 38.2 deg, clearance 0.42 ly. SafetyCritic verdict: CLEARED."),
    ])
    runner = Runner(app_name="andromeda_offline", agent=crew.root_agent, session_service=InMemorySessionService(), auto_create_session=True)
    events = await run_runner(runner, "session-A", "Plan a safe transit of the Andromeda double nucleus.")
    calls = [p.function_call.name for e in events if e.content and e.content.parts for p in e.content.parts if p.function_call]
    return [calls, crew.root_agent.model.calls, crew.astrogation_specialist.model.calls, crew.safety_critic.model.calls, len(events)]


async def part_b_dynamic_workflow() -> List:
    """The @node dynamic workflow: a draft that misses P2 fails the deterministic gate, the critic repairs it."""
    print("\n[B] @node dynamic workflow: deterministic gate + bounded repair loop")
    crew.astrogation_specialist.model = ScriptedLlm(script=[text(
        "Orbital plan: velocity_kms=204.5, inclination_deg=38.2, clearance_ly=0.42 around P1 only.")])
    crew.safety_critic.model = ScriptedLlm(script=[text(
        "Hardened plan: velocity_kms=204.5, inclination_deg=38.2, clearance_ly=0.42; clears P1 apoapsis and the P2 disk crossing.")])
    runner = Runner(app_name="andromeda_offline_wf", node=crew.andromeda_dynamic_workflow, session_service=InMemorySessionService(), auto_create_session=True)
    query = "Plan a safe transit of the Andromeda double nucleus."
    events = await run_runner(runner, "session-B", query, state={"mission_query": query})
    outputs = [str(e.output) for e in events if getattr(e, "output", None) is not None]
    return [outputs, crew.astrogation_specialist.model.calls, crew.safety_critic.model.calls, len(events)]


async def main() -> int:
    import importlib.metadata as md
    print(f"Google ADK {md.version('google-adk')} | python {sys.version.split()[0]} | model: ScriptedLlm (no network, no API key)")
    a = await part_a_coordinator()
    b = await part_b_dynamic_workflow()

    calls, root_calls, astro_calls, critic_calls, n_a = a
    outputs, astro_b, critic_b, n_b = b
    print("\n--- checks ---")
    assert calls == ["AstrogationSpecialist", "SafetyCritic"], calls
    assert root_calls == 3 and astro_calls == 1 and critic_calls == 1
    print(f"[A] root coordinator made 3 model calls and 2 AgentTool calls {calls}; each specialist ran once; {n_a} events")
    assert outputs and "P1" in outputs[-1] and "P2" in outputs[-1], outputs
    assert astro_b == 1 and critic_b == 1
    print(f"[B] the draft missing P2 was routed through the critic once; final output mentions P1 and P2; {n_b} events")
    print("[PASS] Part 5 Track A (offline): the instructor's ADK crew executed by the real ADK runtime.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
