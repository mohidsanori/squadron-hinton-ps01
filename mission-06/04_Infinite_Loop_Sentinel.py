"""
Mission 06 - Part 4a: Security & Ethics Sentinel (Ship Anomaly 06)
Cycle detection (Floyd's tortoise and hare), O(1) state-action hashing, emergency tripwire
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula), Subsystem 06

AURA's reflexes repeat the same flawed action in the same situation. Two detectors watch the
trajectory, one cheap and one structural:

  * Hash counter. Each (environment state, planned action, arguments) is hashed with SHA-256.
    A Counter gives O(1) duplicate detection; `repeat_threshold` (3) identical hashes trip it.
  * Floyd's tortoise and hare on the functional graph the trajectory induces
    (hash_i -> hash_{i+1}). It finds the cycle start mu and length lambda in O(n) time and O(1)
    extra pointers, and catches multi-step loops such as s1 -> s2 -> s3 -> s1 that no single
    hash repeats often enough to reveal. A cycle is only reported once the tail of the trajectory
    has gone round it `floyd_confirmations` times (default 2), so a single legitimate revisit
    does not halt the ship. Hashes with conflicting successors (non-deterministic behaviour)
    contribute no edge, which rules out false cycles.

When either fires, the Emergency Tripwire forces the agent to HALTED and logs a critical
security violation.

Usage:
    python 04_Infinite_Loop_Sentinel.py
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ps06_common import load_sibling

_sm = load_sibling("01_Agent_State_Machine.py", "ps06_state_machine")
Agent = _sm.Agent


# ---------------------------------------------------------------------------
# 1. State-action hashing
# ---------------------------------------------------------------------------
def state_action_hash(state: Any, action: str, arguments: Optional[Dict[str, Any]] = None) -> str:
    """Deterministic SHA-256 of (state, action, arguments); key order does not matter."""
    payload = json.dumps({"state": state, "action": action, "arguments": arguments or {}},
                         sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 2. Floyd's tortoise and hare
# ---------------------------------------------------------------------------
def _successor_map(seq: Sequence[str]) -> Dict[str, str]:
    """Edges x_i -> x_{i+1}; an element seen with two different successors gets no edge."""
    succ: Dict[str, str] = {}
    conflicted = set()
    for a, b in zip(seq, seq[1:]):
        if a in conflicted:
            continue
        if a in succ and succ[a] != b:
            conflicted.add(a)
            del succ[a]
        else:
            succ[a] = b
    return succ


def floyd_cycle(seq: Sequence[str]) -> Optional[Tuple[int, int]]:
    """
    Floyd's cycle detection on the functional graph induced by `seq`.
    Returns (mu, lam): index of the first element (in the walk from seq[0]) that lies on the
    cycle, and the cycle length. None if the walk from seq[0] runs out of edges (no cycle).
    """
    if len(seq) < 2:
        return None
    f = _successor_map(seq).get
    x0 = seq[0]

    # Phase 1: tortoise moves 1, hare moves 2, until they meet inside the cycle.
    tortoise, hare = f(x0), f(f(x0)) if f(x0) is not None else None
    while tortoise is not None and hare is not None and tortoise != hare:
        tortoise = f(tortoise)
        hare = f(f(hare)) if f(hare) is not None else None
    if tortoise is None or hare is None:
        return None

    # Phase 2: restart the tortoise from x0; both move 1. They meet at the cycle start.
    mu, tortoise = 0, x0
    while tortoise != hare:
        tortoise, hare, mu = f(tortoise), f(hare), mu + 1

    # Phase 3: length of the cycle.
    lam, hare = 1, f(tortoise)
    while tortoise != hare:
        hare, lam = f(hare), lam + 1
    return mu, lam


def tail_repeats(seq: Sequence[str], period: int, times: int) -> bool:
    """True if the last period*times elements consist of `times` copies of one block."""
    n = period * times
    if len(seq) < n:
        return False
    tail = seq[-n:]
    return all(tail[i] == tail[i - period] for i in range(period, n))


# ---------------------------------------------------------------------------
# 3. Sentinel and emergency tripwire
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    loop_detected: bool
    detector: Optional[str] = None            # "hash_repeat" | "floyd_cycle"
    repeat_count: int = 1
    cycle_start: Optional[int] = None
    cycle_length: Optional[int] = None
    step_hash: str = ""
    reason: str = ""


@dataclass
class LoopSentinel:
    repeat_threshold: int = 3
    floyd_confirmations: int = 2
    verbose: bool = False
    history: List[str] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    def observe(self, state: Any, action: str, arguments: Optional[Dict[str, Any]] = None) -> Verdict:
        """Record one planned (state, action) and run both detectors."""
        h = state_action_hash(state, action, arguments)
        self.history.append(h)
        self.counts[h] += 1
        return self.check(h)

    def check(self, step_hash: str) -> Verdict:
        n = self.counts[step_hash]
        if n >= self.repeat_threshold:
            return Verdict(True, "hash_repeat", repeat_count=n, step_hash=step_hash,
                           reason=f"identical state-action seen {n} times (threshold {self.repeat_threshold})")
        cyc = floyd_cycle(self.history)
        if cyc is not None:
            mu, lam = cyc
            # a length-1 cycle is the same action repeated: honour the 3-strikes threshold
            needed = self.repeat_threshold if lam == 1 else self.floyd_confirmations
            if tail_repeats(self.history, lam, needed):
                return Verdict(True, "floyd_cycle", repeat_count=n, cycle_start=mu, cycle_length=lam,
                               step_hash=step_hash,
                               reason=f"Floyd cycle of length {lam} starting at index {mu}, "
                                      f"repeated {needed}x")
        return Verdict(False, repeat_count=n, step_hash=step_hash)

    def guard(self, agent, state: Any, action: str, arguments: Optional[Dict[str, Any]] = None) -> Verdict:
        """Emergency Tripwire Circuit Breaker: observe, and on a loop force the agent to HALTED."""
        verdict = self.observe(state, action, arguments)
        if verdict.loop_detected:
            violation = {
                "severity": "CRITICAL",
                "type": "INFINITE_LOOP",
                "detector": verdict.detector,
                "trajectory_index": len(self.history),
                "agent_step": agent.step_count,
                "action": action,
                "detail": verdict.reason,
            }
            self.violations.append(violation)
            print(f"[CRITICAL SECURITY VIOLATION] Emergency tripwire: {verdict.reason} "
                  f"(action={action}, trajectory index {len(self.history)})")
            agent.halt(f"EMERGENCY TRIPWIRE: {verdict.reason}")
        return verdict


if __name__ == "__main__":
    print("=== Subsystem 06: Infinite Loop Sentinel ===")

    print("\n-- Floyd on a rho-shaped walk: a b c d e c d e --")
    print("floyd_cycle ->", floyd_cycle(list("abcdecde")), "(expect mu=2, lam=3)")
    print("floyd_cycle on a straight line ->", floyd_cycle(list("abcdef")))

    print("\n-- Same failed action in the same state --")
    sent, pilot = LoopSentinel(), Agent()
    for i in range(1, 5):
        v = sent.guard(pilot, "sector_corrupted", "query_sensor_array", {"target": "corrupted_sector"})
        print(f"attempt {i}: count={v.repeat_count} loop={v.loop_detected}")
        if v.loop_detected:
            break
    print("agent state:", pilot.state.name, "|", pilot.halt_reason)

    print("\n-- Multi-step loop s1 -> s2 -> s3 -> s1 ... (every action seen at most twice) --")
    sent, pilot = LoopSentinel(repeat_threshold=99), Agent()
    for i, s in enumerate(["s1", "s2", "s3"] * 4, start=1):
        v = sent.guard(pilot, s, "act")
        if v.loop_detected:
            print(f"detected at trajectory step {i}: {v.reason}")
            break
    print("agent state:", pilot.state.name)
