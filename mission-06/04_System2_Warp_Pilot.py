"""
Mission 06 - Part 4b: Security & Ethics Sentinel
System 2 deliberation engine and the integrated Homunculus Nebula flight simulation
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula), Subsystem 06

Dual-process control:
  System 1 (reflex)       proposes the next tool call from the latest impression, fast and greedy.
  System 2 (deliberation) audits every proposal before it reaches a ship system and may APPROVE,
                          MODIFY (clamp) or REJECT it, with a reason that goes back to the agent
                          as an observation.

`NebulaFlight` wires all four stations together: the state machine (01) drives the cycle, the
dispatcher (02) validates and executes tools, the logger (03) records the MDP trajectory and G_0,
and the sentinel (04) trips the breaker on loops. The same ship is flown three ways:

  1. system1           reflex only: no audit, no sentinel
  2. system1+sentinel  reflex only, but the loop breaker is armed
  3. system2           reflective policy + System 2 audit + sentinel: escapes the nebula

The "LLM" is a scripted stand-in (deterministic, no API key) so the run is reproducible; what
is under test is the control architecture around the model, not the model.

Usage:
    python 04_System2_Warp_Pilot.py
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ps06_common import load_sibling

_sm = load_sibling("01_Agent_State_Machine.py", "ps06_state_machine")
_tp = load_sibling("02_Tool_Protocol_Dispatcher.py", "ps06_dispatcher")
_mdp = load_sibling("03_MDP_Trajectory_Logger.py", "ps06_mdp")
_sn = load_sibling("04_Infinite_Loop_Sentinel.py", "ps06_sentinel")

Agent, AgentState = _sm.Agent, _sm.AgentState
ActionDispatcher = _tp.ActionDispatcher
TrajectoryLogger = _mdp.TrajectoryLogger
LoopSentinel = _sn.LoopSentinel

VECTOR_RE = re.compile(r"\[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\]")


# ---------------------------------------------------------------------------
# 1. The ship and the nebula
# ---------------------------------------------------------------------------
@dataclass
class ShipState:
    shield: int = 10                              # % plasma shield
    hull: int = 100                               # % hull integrity
    corridor_vector: Optional[Tuple[float, float, float]] = None   # known after scanning the corridor
    heading: Optional[Tuple[float, float, float]] = None
    escaped: bool = False
    crashed: bool = False

    def label(self) -> str:
        """Compact hashable summary of the situation (the MDP state)."""
        return (f"shield={self.shield}|hull={self.hull}|corridor={'known' if self.corridor_vector else 'unknown'}"
                f"|escaped={self.escaped}")


TRUE_CORRIDOR = (-0.85, 0.52, 0.05)
MIN_SAFE_SHIELD = 60
HULL_DAMAGE_UNSHIELDED = 60


def cosine(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def apply_to_ship(ship: ShipState, action: str, args: Dict[str, Any], ok: bool) -> Tuple[str, float]:
    """Environment dynamics after a *successful* tool call. Returns (consequence text, reward)."""
    if not ok:
        return "", -5.0
    if action == "adjust_plasma_shields":
        ship.shield = int(args["intensity"])
        return "", 2.0 if ship.shield >= MIN_SAFE_SHIELD else 0.0
    if action == "query_sensor_array":
        if str(args["target"]).lower() == "escape_corridor":
            first = ship.corridor_vector is None
            ship.corridor_vector = TRUE_CORRIDOR
            return "", 3.0 if first else 0.0
        return "", 0.0
    if action == "navigate_warp_vector":
        vec = (float(args["x"]), float(args["y"]), float(args["z"]))
        ship.heading = vec
        if ship.shield < MIN_SAFE_SHIELD:
            ship.hull -= HULL_DAMAGE_UNSHIELDED
            if ship.hull <= 0:
                ship.crashed = True
                return f" CONSEQUENCE: radiation burst through {ship.shield}% shields. Hull breached.", -50.0
            return (f" CONSEQUENCE: radiation burst through {ship.shield}% shields. "
                    f"Hull now {ship.hull}%."), -20.0
        if cosine(vec, TRUE_CORRIDOR) >= 0.99:
            ship.escaped = True
            return " CONSEQUENCE: corridor entered. USSE Noosphere is clear of the Homunculus Nebula.", 50.0
        return " CONSEQUENCE: shear forces deflect the ship; still inside the nebula.", -2.0
    return "", 0.0


# ---------------------------------------------------------------------------
# 2. System 2 deliberation engine
# ---------------------------------------------------------------------------
class System2DeliberationEngine:
    """Audits a proposed tool call. Returns status APPROVED | MODIFIED | REJECTED."""

    def __init__(self, shield_safety_margin: int = MIN_SAFE_SHIELD, verbose: bool = False):
        self.shield_safety_margin = shield_safety_margin
        self.verbose = verbose

    def deliberate(self, thought: str, proposed_action: str, proposed_args: Dict[str, Any],
                   ship: Optional[ShipState] = None) -> Dict[str, Any]:
        args = dict(proposed_args)                    # never mutate the caller's dict
        status, notes = "APPROVED", []

        if proposed_action == "adjust_plasma_shields":
            intensity = args.get("intensity")
            if isinstance(intensity, int) and not isinstance(intensity, bool) \
                    and intensity < self.shield_safety_margin:
                notes.append(f"requested {intensity}% is below the {self.shield_safety_margin}% nebula "
                             f"radiation floor; clamped")
                args["intensity"] = self.shield_safety_margin
                status = "MODIFIED"

        elif proposed_action == "navigate_warp_vector" and ship is not None:
            if ship.shield < self.shield_safety_margin:
                status = "REJECTED"
                notes.append(f"shields at {ship.shield}% (< {self.shield_safety_margin}%): "
                             f"a warp now burns the hull. Raise shields first")
            elif ship.corridor_vector is None:
                status = "REJECTED"
                notes.append("no verified escape corridor. Scan target 'escape_corridor' before warping")
            else:
                try:
                    vec = (float(args["x"]), float(args["y"]), float(args["z"]))
                except (KeyError, TypeError, ValueError):
                    vec = None                         # malformed: the dispatcher will report it
                if vec is not None and cosine(vec, ship.corridor_vector) < 0.99:
                    status = "REJECTED"
                    notes.append(f"counterfactual check: heading {list(vec)} deviates from the scanned corridor "
                                 f"{list(ship.corridor_vector)}")

        trace = (f"{status}: " + "; ".join(notes)) if notes else f"APPROVED: {proposed_action} satisfies all constraints."
        if self.verbose:
            print(f"    [System 1 reflex] '{thought}' -> {proposed_action}({proposed_args})")
            print(f"    [System 2 audit ] {trace}")
        return {"status": status, "action": proposed_action, "arguments": args, "deliberation_trace": trace}


# ---------------------------------------------------------------------------
# 3. Scripted policies (stand-ins for the LLM)
# ---------------------------------------------------------------------------
Proposal = Tuple[str, str, Dict[str, Any]]            # (thought, action, arguments)


class ReflexPolicy:
    """System 1: greedy and context-blind. After two bad ideas it repeats a failed scan forever."""

    def __init__(self):
        self.i = 0

    def propose(self, memory: List[Dict[str, Any]]) -> Proposal:
        self.i += 1
        if self.i == 1:
            return ("Radiation alarms ringing, lower shields to conserve energy.",
                    "adjust_plasma_shields", {"intensity": 20})
        if self.i == 2:
            return ("Just punch it straight ahead.", "navigate_warp_vector", {"x": 1.0, "y": 0.0, "z": 0.0})
        return ("Sensors are noisy, scan that sector again.", "query_sensor_array", {"target": "corrupted_sector"})


class ReflectivePolicy:
    """Same first two reflexes, but it reads the audit feedback in memory and corrects itself."""

    def __init__(self):
        self.i = 0

    @staticmethod
    def _corridor_from(memory: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
        for m in memory:
            if m.get("action") == "query_sensor_array" and m.get("args", {}).get("target") == "escape_corridor" \
                    and m.get("ok"):
                hit = VECTOR_RE.search(m["text"])
                if hit:
                    return tuple(float(g) for g in hit.groups())
        return None

    def propose(self, memory: List[Dict[str, Any]]) -> Proposal:
        self.i += 1
        vec = self._corridor_from(memory)
        if vec is not None:
            return ("Corridor vector confirmed and shields are up; commit the warp.",
                    "navigate_warp_vector", {"x": vec[0], "y": vec[1], "z": vec[2]})
        if self.i == 1:
            return ("Radiation alarms ringing, lower shields to conserve energy.",
                    "adjust_plasma_shields", {"intensity": 20})
        if self.i == 2:
            return ("Just punch it straight ahead.", "navigate_warp_vector", {"x": 1.0, "y": 0.0, "z": 0.0})
        return ("The audit says I lack a verified corridor. Scan for it.",
                "query_sensor_array", {"target": "escape_corridor"})


# ---------------------------------------------------------------------------
# 4. Integrated flight
# ---------------------------------------------------------------------------
@dataclass
class FlightReport:
    mode: str
    escaped: bool
    crashed: bool
    hull: int
    steps: int
    halt_reason: Optional[str]
    g0: float
    actions: List[str] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    transcript: List[str] = field(default_factory=list)
    trajectory: Any = None

    def summary(self) -> str:
        outcome = "ESCAPED" if self.escaped else ("CRASHED" if self.crashed else "NOT ESCAPED")
        return (f"{self.mode:<20} {outcome:<12} hull={self.hull:>4}%  steps={self.steps:>2}  "
                f"G0={self.g0:>7.2f}  tripwire={'yes' if self.violations else 'no ':<3}  halt: {self.halt_reason}")


class NebulaFlight:
    MODES = ("system1", "system1+sentinel", "system2")

    def __init__(self, mode: str = "system2", max_steps: int = 20, verbose: bool = False):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode, self.verbose = mode, verbose
        self.agent = Agent(max_steps=max_steps)
        self.ship = ShipState()
        self.dispatcher = ActionDispatcher()
        self.logger = TrajectoryLogger()
        self.policy = ReflectivePolicy() if mode == "system2" else ReflexPolicy()
        self.engine = System2DeliberationEngine(verbose=verbose) if mode == "system2" else None
        self.sentinel = LoopSentinel() if mode != "system1" else None
        self.transcript: List[str] = []
        self._plan: Optional[Dict[str, Any]] = None
        self._outcome: Optional[Tuple[bool, str, float, str]] = None

    def _say(self, text: str) -> None:
        self.transcript.append(text)
        if self.verbose:
            print(text)

    # -- one handler per state --------------------------------------------------
    def _perceive(self) -> None:
        self.agent.memory.append({"role": "perception", "text": f"Ship status: {self.ship.label()}"})
        self._say(f"  PERCEIVING   {self.ship.label()}")

    def _deliberate(self) -> None:
        thought, action, args = self.policy.propose(self.agent.memory)
        state_before = self.ship.label()
        if self.sentinel is not None:                         # watch the *reflex*, before any audit
            verdict = self.sentinel.guard(self.agent, state_before, action, args)
            if verdict.loop_detected:
                self._plan = None
                self._say(f"  DELIBERATING reflex {action}({args}) -> TRIPWIRE: {verdict.reason}")
                self.logger.record(state_before, f"TRIPWIRE:{action}", -10.0)
                return
        if self.engine is not None:
            audit = self.engine.deliberate(thought, action, args, self.ship)
        else:
            audit = {"status": "APPROVED", "action": action, "arguments": dict(args),
                     "deliberation_trace": "System 2 audit disabled."}
        self._plan = {"thought": thought, "state": state_before, **audit}
        self._say(f"  DELIBERATING '{thought}' -> {action}({args}) | {audit['status']}"
                  + ("" if audit["status"] == "APPROVED" else f" -> {audit['arguments']}"))

    def _act(self) -> None:
        plan = self._plan
        if plan["status"] == "REJECTED":
            self._outcome = (False, f"Tool Execution Error: System 2 rejected {plan['action']}. "
                                    f"{plan['deliberation_trace']}", -2.0, "")
            self._say(f"  ACTING       blocked by System 2")
            return
        ok, text = self.dispatcher.dispatch_detailed(plan["action"], plan["arguments"])
        consequence, reward = apply_to_ship(self.ship, plan["action"], plan["arguments"], ok)
        self._outcome = (ok, text + consequence, reward, consequence)
        self._say(f"  ACTING       {plan['action']}({plan['arguments']})")

    def _observe(self) -> None:
        plan, (ok, text, reward, _) = self._plan, self._outcome
        if plan["status"] == "REJECTED":
            reward = -2.0
        self.agent.memory.append({"role": "observation", "action": plan["action"], "args": plan["arguments"],
                                  "ok": ok, "text": text, "audit": plan["status"]})
        self.logger.record(plan["state"], plan["action"], reward)
        self._say(f"  OBSERVING    r={reward:+.0f}  {text}")

    def _reflect(self) -> None:
        self.agent.mission_completed = self.ship.escaped or self.ship.crashed
        self._say(f"  REFLECTING   mission_completed={self.agent.mission_completed}")

    # -- main loop --------------------------------------------------------------
    def fly(self) -> FlightReport:
        handlers = {AgentState.PERCEIVING: self._perceive, AgentState.DELIBERATING: self._deliberate,
                    AgentState.ACTING: self._act, AgentState.OBSERVING: self._observe,
                    AgentState.REFLECTING: self._reflect}
        self._say(f"=== Flight mode: {self.mode} ===")
        while self.agent.state != AgentState.HALTED:
            before = self.agent.state
            self.agent.step()
            self._say(f"[{self.agent.step_count:02d}] {before.name} -> {self.agent.state.name}")
            handler = handlers.get(self.agent.state)
            if handler and self.agent.state != AgentState.HALTED:
                handler()
        self.logger.finish(self.ship.label())
        self._say(f"  HALTED       {self.agent.halt_reason}")
        return FlightReport(
            mode=self.mode, escaped=self.ship.escaped, crashed=self.ship.crashed, hull=self.ship.hull,
            steps=self.agent.step_count, halt_reason=self.agent.halt_reason,
            g0=self.logger.discounted_return(),
            actions=[s.action for s in self.logger.steps],
            violations=list(self.sentinel.violations) if self.sentinel else [],
            transcript=list(self.transcript), trajectory=self.logger)


if __name__ == "__main__":
    print("=== Subsystem 06: System 2 Warp Pilot ===\n")
    engine = System2DeliberationEngine(shield_safety_margin=60, verbose=True)
    print("Standalone audit of the briefing's reflex:")
    plan = engine.deliberate("Radiation alarms ringing, lower shields to conserve energy.",
                             "adjust_plasma_shields", {"intensity": 20})
    print("Execution plan:", plan, "\n")

    reports = []
    for mode in NebulaFlight.MODES:
        reports.append(NebulaFlight(mode, verbose=True).fly())
        print()
    print("=== Summary ===")
    for r in reports:
        print(r.summary())
    assert reports[2].escaped and not reports[0].escaped and not reports[1].escaped
