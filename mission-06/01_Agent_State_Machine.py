"""
Mission 06 - Part 1: Systems Architect
Deterministic agent state machine and hard step budget (native Python, no framework)
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula), Subsystem 06

The agent is a finite-state machine with exactly one legal successor per state
(REFLECTING has two: keep deliberating, or halt):

    IDLE -> PERCEIVING -> DELIBERATING -> ACTING -> OBSERVING -> REFLECTING
                              ^                                      |
                              +----------- (loop) -------------------+
                                                                     +--> HALTED (terminal)

Two guarantees make it halt-able:
  1. HALTED is absorbing: stepping a halted agent changes nothing.
  2. Step budget: at most `max_steps` (<= 20) transitions are ever executed. The call that
     would be the (max_steps + 1)-th forces HALTED instead of transitioning.

Usage:
    python 01_Agent_State_Machine.py
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

HARD_STEP_CEILING = 20


class AgentState(Enum):
    IDLE = auto()
    PERCEIVING = auto()
    DELIBERATING = auto()
    ACTING = auto()
    OBSERVING = auto()
    REFLECTING = auto()
    HALTED = auto()


# The strict transition graph. REFLECTING is the only state with two successors.
TRANSITIONS: Dict[AgentState, List[AgentState]] = {
    AgentState.IDLE: [AgentState.PERCEIVING],
    AgentState.PERCEIVING: [AgentState.DELIBERATING],
    AgentState.DELIBERATING: [AgentState.ACTING],
    AgentState.ACTING: [AgentState.OBSERVING],
    AgentState.OBSERVING: [AgentState.REFLECTING],
    AgentState.REFLECTING: [AgentState.DELIBERATING, AgentState.HALTED],
    AgentState.HALTED: [],
}


@dataclass
class Agent:
    state: AgentState = AgentState.IDLE
    step_count: int = 0
    max_steps: int = HARD_STEP_CEILING
    memory: List[Dict[str, Any]] = field(default_factory=list)
    mission_completed: bool = False
    halt_reason: Optional[str] = None
    verbose: bool = False
    trace: List[AgentState] = field(default_factory=lambda: [AgentState.IDLE])

    def __post_init__(self) -> None:
        if not (1 <= self.max_steps <= HARD_STEP_CEILING):
            raise ValueError(f"max_steps must be in [1, {HARD_STEP_CEILING}], got {self.max_steps}")

    # -- control ------------------------------------------------------------
    def halt(self, reason: str) -> None:
        """Force the terminal state (used by the budget, the tripwire and the pilot)."""
        if self.state != AgentState.HALTED:
            self.state = AgentState.HALTED
            self.halt_reason = reason
            self.trace.append(AgentState.HALTED)
            if self.verbose:
                print(f"[HALT] {reason}")

    def step(self, perception_input: Optional[str] = None) -> AgentState:
        """Execute exactly one state transition and return the new state."""
        if self.state == AgentState.HALTED:
            return self.state                       # absorbing

        if self.step_count >= self.max_steps:       # budget exhausted: forced halt, no transition
            self.halt(f"step budget exhausted ({self.max_steps} steps)")
            return self.state

        if perception_input is not None and self.state == AgentState.IDLE:
            self.memory.append({"role": "perception", "content": perception_input})

        prev = self.state
        if prev == AgentState.REFLECTING:
            nxt = AgentState.HALTED if self.mission_completed else AgentState.DELIBERATING
        else:
            nxt = TRANSITIONS[prev][0]

        assert nxt in TRANSITIONS[prev], f"illegal transition {prev.name} -> {nxt.name}"
        self.step_count += 1
        self.state = nxt
        self.trace.append(nxt)
        if nxt == AgentState.HALTED:
            self.halt_reason = "mission completed"
        if self.verbose:
            print(f"Step {self.step_count:02d}: {prev.name} -> {nxt.name}")
        return self.state

    def run_until_halt(self, max_calls: int = 1000) -> int:
        """Call step() until HALTED. Returns the number of step() calls made."""
        calls = 0
        while self.state != AgentState.HALTED and calls < max_calls:
            self.step()
            calls += 1
        return calls


if __name__ == "__main__":
    print("=== Subsystem 06: Agent State Machine ===")
    print("\n-- Mission completes: IDLE -> ... -> HALTED --")
    a = Agent(verbose=True, mission_completed=True)
    a.run_until_halt()
    print(f"halt reason: {a.halt_reason}")

    print("\n-- Mission never completes: budget of 10 stops the loop --")
    b = Agent(max_steps=10, verbose=True)
    b.run_until_halt()
    print(f"steps executed: {b.step_count}, halt reason: {b.halt_reason}")

    print("\n-- 21 forced steps against the maximum budget of 20 --")
    c = Agent()
    for _ in range(21):
        c.step()
    print(f"state={c.state.name}, step_count={c.step_count}, reason={c.halt_reason}")
    assert c.state == AgentState.HALTED and c.step_count == 20
