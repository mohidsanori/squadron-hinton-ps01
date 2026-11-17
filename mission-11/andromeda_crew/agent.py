"""
Squadron Hinton copy of the instructor's Track A crew. One change: the deterministic gate uses
parameter_binding="node_input" and is called with {"draft_plan": ...}; the original (default state binding, bare draft)
raises WorkflowDataError on google-adk 2.14.1.

USSE Noosphere — Mission 11 Part 5 (Track A: Google ADK 2.0 `adk web` Studio Showcase)
Run from `mission-11/` directory after `pip install google-adk` and setting `GOOGLE_API_KEY`:
    adk web
Then open http://localhost:8000 and select `andromeda_crew`.
"""

from google.adk import Context, Event
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.workflow import node

astrogation_specialist = LlmAgent(
    name="AstrogationSpecialist",
    model="gemini-flash-latest",
    description="Computes relativistic P1/P2 double-nucleus orbital clearance vectors in SEC-11-M31.",
    instruction=(
        "You are the USSE Noosphere Astrogation Specialist at Sector SEC-11-M31 (Andromeda Core). "
        "Propose a safe orbital vector around M31* referencing both nuclei P1 and P2, including "
        "velocity_kms, inclination_deg, and clearance_ly."
    ),
)

safety_critic = LlmAgent(
    name="SafetyCritic",
    model="gemini-flash-latest",
    description="Audits candidate orbital plans for M31* tidal shear and circular delegation hazards.",
    instruction=(
        "You are the USSE Noosphere Safety Critic. Audit the proposed orbital plan for tidal shear "
        "around M31* (1.4e8 solar masses), verify both P1 and P2 clearance margins, and return a "
        "hardened bridge clearance verdict."
    ),
)


@node(name="deterministic_telemetry_gate", parameter_binding="node_input")
async def verify_orbital_bounds(ctx: Context, draft_plan: str) -> dict:
    """Deterministic non-LLM check interleaved inside an ADK 2.0 Dynamic Workflow."""
    missing = [kw for kw in ("P1", "P2") if kw not in str(draft_plan)]
    return {
        "findings": "" if not missing else f"Missing double-nucleus reference(s): {missing}"
    }


@node(rerun_on_resume=True)
async def andromeda_dynamic_workflow(ctx: Context, mission_query: str):
    """Bounded dynamic verification loop combining LLM specialists with a deterministic @node gate."""
    draft = await ctx.run_node(astrogation_specialist, mission_query)
    # The gate declares parameter_binding="node_input", so its parameters are read from a dict keyed by name and the
    # draft is passed as {"draft_plan": ...}. With the default binding (session state) the first call, made before
    # anything has written draft_plan to state, raises WorkflowDataError("Missing value for parameter draft_plan").
    gate = await ctx.run_node(verify_orbital_bounds, {"draft_plan": str(draft)})
    rounds = 0
    while gate["findings"] and rounds < 3:
        rounds += 1
        yield Event(
            state={"draft_plan": draft, "lint_findings": gate["findings"], "round": rounds}
        )
        draft = await ctx.run_node(
            safety_critic, {"draft_plan": draft, "lint_findings": gate["findings"]}
        )
        gate = await ctx.run_node(verify_orbital_bounds, {"draft_plan": str(draft)})
    yield Event(output=draft)


# Root Coordinator exposing specialists via AgentTool for interactive inspection in `adk web`
root_agent = LlmAgent(
    name="AndromedaNexusCoordinator",
    model="gemini-flash-latest",
    description="Root Fleet Coordinator for USSE Noosphere Subsystem 11 (SEC-11-M31).",
    instruction=(
        "Coordinate Andromeda M31 double-nucleus navigation. Invoke AstrogationSpecialist first "
        "to draft a P1/P2 trajectory, then invoke SafetyCritic to audit tidal shear and deadlock "
        "safety before issuing final bridge clearance."
    ),
    tools=[
        agent_tool.AgentTool(agent=astrogation_specialist),
        agent_tool.AgentTool(agent=safety_critic),
    ],
)
