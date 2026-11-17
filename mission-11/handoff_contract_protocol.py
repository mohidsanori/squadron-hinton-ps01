"""
Mission 11 - Part 2: Communications & Protocol Officer
Typed handoff envelopes, least-context scoping, and delegation call-stack semantics
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-11-M31 (Andromeda Galaxy Core), Subsystem 11

A handoff is a contract, not a chat message. The envelope names the sender and recipient, the exact
context keys the recipient may see, the output fields it must return, and a hop budget.

  Least-context principle. The recipient receives `global_state` projected onto `allowed_context_keys`
    (deep-copied, so a specialist that mutates its view cannot corrupt the shared state). Chat history,
    scratchpads and unrelated logs never reach it:
        eta_comp = 1 - |serialize(S_scoped)| / |serialize(S_global)|
  Delegation semantics (Google ADK):
    SUBROUTINE_TOOL   AgentTool: the caller is pushed on the return stack and control comes back
    PEER_TRANSFER     sub_agents transfer: the active agent pointer moves and nothing is pushed
  Hop budget. A chain of delegations may not exceed max_hops. The gate counts hops itself per chain, so a
    caller that understates `hop_count` cannot extend its budget; ping-pong between two agents is cut at
    hop max_hops + 1 with RuntimeError("HOP_LIMIT_EXCEEDED").
  Output contract. `validate_response` checks every required output field is present (and not None).

The gate also refuses a handoff whose sender is not the active agent (a spoofed handoff) and a return
that does not match the top of the return stack.

Usage:
    python handoff_contract_protocol.py
"""

import copy
import enum
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class HandoffMode(enum.Enum):
    PEER_TRANSFER = 1    # sub_agents transfer: the active agent pointer moves to the recipient
    SUBROUTINE_TOOL = 2  # AgentTool call: the caller stays on the return stack until the specialist returns


@dataclass
class HandoffEnvelope:
    envelope_id: str
    sender: str
    recipient: str
    mode: HandoffMode
    allowed_context_keys: List[str]
    payload: Dict[str, Any]
    required_output_fields: List[str]
    hop_count: int = 0
    max_hops: int = 5
    chain_id: str = "default"


@dataclass
class HandoffDispatchResult:
    envelope_id: str
    active_agent: str
    caller_stack: List[str]
    scoped_context: Dict[str, Any]
    compression_ratio: float          # eta_comp = 1 - |scoped| / |global|  (higher = more pruning)
    token_reduction_pct: float
    size_ratio: float = 0.0           # |scoped| / |global|
    effective_hops: int = 0


class ContextScopingGate:
    """Enforces typed handoff contracts, the least-context principle and delegation semantics."""

    def __init__(self, initial_agent: str = "AndromedaCoordinator", enforce_active_sender: bool = True):
        self.active_agent: str = initial_agent
        self.caller_stack: List[str] = []
        self.audit_log: List[HandoffDispatchResult] = []
        self.enforce_active_sender = enforce_active_sender
        self._hops_seen: Dict[str, int] = {}

    # -- measurement ---------------------------------------------------------------------
    @staticmethod
    def _serialized_length(obj: Dict[str, Any]) -> int:
        return len(json.dumps(obj, sort_keys=True, default=str))

    def compute_size_ratio(self, original: Dict[str, Any], compressed: Dict[str, Any]) -> float:
        """|scoped| / |global| in serialized characters (0.0 for an empty original)."""
        orig = self._serialized_length(original)
        return 0.0 if orig == 0 else self._serialized_length(compressed) / orig

    def compute_compression_ratio(self, original: Dict[str, Any], compressed: Dict[str, Any]) -> float:
        """eta_comp = 1 - |serialize(S_scoped)| / |serialize(S_global)|: the share of the global context removed."""
        if self._serialized_length(original) == 0:
            return 0.0
        return round(1.0 - self.compute_size_ratio(original, compressed), 4)

    # -- contract validation ------------------------------------------------------------------
    def validate_response(self, envelope: HandoffEnvelope, specialist_output: Dict[str, Any]) -> List[str]:
        """Missing (or None) required output fields; raises SCHEMA_CONTRACT_VIOLATION if there are any."""
        missing = [f for f in envelope.required_output_fields if specialist_output.get(f) is None]
        if missing:
            raise ValueError(f"SCHEMA_CONTRACT_VIOLATION: Specialist '{envelope.recipient}' omitted required output fields: {missing}")
        return missing

    # -- handoff ------------------------------------------------------------------------------
    def process_handoff(self, global_state: Dict[str, Any], envelope: HandoffEnvelope) -> Dict[str, Any]:
        """
        Check the hop budget and the sender, project the global state onto the whitelisted keys, and update
        the active agent and return stack according to the delegation mode. Returns the scoped context.
        """
        effective = max(envelope.hop_count, self._hops_seen.get(envelope.chain_id, 0) + 1)
        if effective > envelope.max_hops:
            raise RuntimeError(f"HOP_LIMIT_EXCEEDED: Envelope '{envelope.envelope_id}' at hop {effective} exceeds max_hops={envelope.max_hops}.")
        if self.enforce_active_sender and envelope.sender != self.active_agent:
            raise PermissionError(f"SENDER_NOT_ACTIVE: '{envelope.sender}' tried to hand off but '{self.active_agent}' is the active agent.")

        scoped: Dict[str, Any] = {k: copy.deepcopy(global_state[k]) for k in envelope.allowed_context_keys if k in global_state}
        scoped["_handoff_payload"] = copy.deepcopy(envelope.payload)

        if envelope.mode == HandoffMode.SUBROUTINE_TOOL:
            self.caller_stack.append(envelope.sender)
        self.active_agent = envelope.recipient
        self._hops_seen[envelope.chain_id] = effective

        size_ratio = self.compute_size_ratio(global_state, scoped)
        eta = self.compute_compression_ratio(global_state, scoped)
        self.audit_log.append(HandoffDispatchResult(envelope.envelope_id, self.active_agent, list(self.caller_stack), scoped,
                                                    eta, round(eta * 100.0, 2), round(size_ratio, 4), effective))
        return scoped

    def complete_subroutine_return(self, envelope: HandoffEnvelope, specialist_output: Dict[str, Any]) -> str:
        """Validate the output contract and, for a SUBROUTINE_TOOL call, pop the return stack and restore the caller."""
        self.validate_response(envelope, specialist_output)
        if envelope.mode == HandoffMode.SUBROUTINE_TOOL:
            if not self.caller_stack or self.caller_stack[-1] != envelope.sender:
                raise RuntimeError(f"STACK_MISMATCH: return to '{envelope.sender}' but the stack is {self.caller_stack}")
            self.active_agent = self.caller_stack.pop()
        return self.active_agent


if __name__ == "__main__":
    gate = ContextScopingGate(initial_agent="AndromedaCoordinator")

    bloated_global_state = {
        "sector_id": "SEC-11-M31",
        "p1_ephemeris": {"ra": "00h42m44.3s", "dec": "+41d16m09s", "vel_kms": -301.0},
        "full_chat_history": ["turn_" + str(i) + "_" + ("x" * 120) for i in range(12)],
        "raw_sensor_dump": "NOISE_" * 200,
        "unrelated_crew_logs": ["galley inventory", "hydroponics cycle 44"],
    }

    sub_env = HandoffEnvelope("ENV-SUB-01", "AndromedaCoordinator", "AstrogationSpecialist", HandoffMode.SUBROUTINE_TOOL,
                              ["sector_id", "p1_ephemeris"], {"objective": "compute_p1_p2_clearance"},
                              ["trajectory_id", "clearance_ly"], hop_count=1, max_hops=5)
    scoped = gate.process_handoff(bloated_global_state, sub_env)
    last_audit = gate.audit_log[-1]
    print(f"[Subroutine Handoff] active={gate.active_agent} | stack={gate.caller_stack} | compression_ratio={last_audit.compression_ratio} | "
          f"reduction={last_audit.token_reduction_pct}%")
    assert "full_chat_history" not in scoped and "raw_sensor_dump" not in scoped
    assert gate.caller_stack == ["AndromedaCoordinator"]
    assert last_audit.token_reduction_pct > 75.0

    restored = gate.complete_subroutine_return(sub_env, {"trajectory_id": "M31-SAFE-07", "clearance_ly": 0.42})
    assert restored == "AndromedaCoordinator" and gate.caller_stack == []

    peer_env = HandoffEnvelope("ENV-PEER-02", "AndromedaCoordinator", "SafetyCritic", HandoffMode.PEER_TRANSFER,
                               ["sector_id"], {"trajectory_id": "M31-SAFE-07"}, ["approved"], hop_count=2, max_hops=5)
    gate.process_handoff(bloated_global_state, peer_env)
    assert gate.active_agent == "SafetyCritic" and gate.caller_stack == []

    runaway_env = HandoffEnvelope("ENV-LOOP-99", "SafetyCritic", "AstrogationSpecialist", HandoffMode.PEER_TRANSFER,
                                  ["sector_id"], {}, [], hop_count=6, max_hops=5)
    try:
        gate.process_handoff(bloated_global_state, runaway_env)
        assert False, "Expected RuntimeError for HOP_LIMIT_EXCEEDED"
    except RuntimeError as exc:
        print(f"[Hop Guard Verified] Caught expected error: {exc}")
        assert "HOP_LIMIT_EXCEEDED" in str(exc)

    # ---- extra checks ----------------------------------------------------------------
    G = {"a": "x" * 80, "b": {"n": 1}, "noise": "y" * 400}
    g = ContextScopingGate("Boss")
    mk = lambda sender, recipient, mode, hops=0, **kw: HandoffEnvelope(f"E-{sender}-{recipient}", sender, recipient, mode, ["a"], {}, kw.get("req", []), hop_count=hops, max_hops=kw.get("max_hops", 5), chain_id=kw.get("chain", "c"))
    # 1. eta_comp = 1 - scoped/global (brief), and size_ratio is its complement
    r = g.compute_compression_ratio(G, {"a": G["a"]})
    assert abs(r - (1 - len(json.dumps({"a": G["a"]}, sort_keys=True)) / len(json.dumps(G, sort_keys=True))) ) < 1e-4 and 0.7 < r < 0.9
    assert g.compute_compression_ratio({}, {}) == 0.0 and g.compute_compression_ratio(G, G) == 0.0
    # 2. the specialist gets a deep copy: mutating its view does not touch the global state
    s = ContextScopingGate("Boss").process_handoff(G, HandoffEnvelope("E", "Boss", "W", HandoffMode.PEER_TRANSFER, ["b"], {}, []))
    s["b"]["n"] = 999
    assert G["b"]["n"] == 1
    # 3. nested subroutines: the stack grows and unwinds in order; a peer transfer in the middle does not touch it
    n = ContextScopingGate("A")
    n.process_handoff(G, mk("A", "B", HandoffMode.SUBROUTINE_TOOL))
    n.process_handoff(G, mk("B", "C", HandoffMode.SUBROUTINE_TOOL))
    n.process_handoff(G, mk("C", "D", HandoffMode.PEER_TRANSFER))
    assert n.active_agent == "D" and n.caller_stack == ["A", "B"]
    # 4. a return that does not match the stack top is refused; the right one unwinds
    try:
        n.complete_subroutine_return(mk("A", "D", HandoffMode.SUBROUTINE_TOOL), {}); raise SystemExit("stack mismatch must raise")
    except RuntimeError as exc:
        assert "STACK_MISMATCH" in str(exc)
    assert n.complete_subroutine_return(mk("B", "D", HandoffMode.SUBROUTINE_TOOL), {}) == "B" and n.caller_stack == ["A"]
    # 5. a spoofed handoff (sender is not the active agent) is refused
    try:
        ContextScopingGate("A").process_handoff(G, mk("Mallory", "B", HandoffMode.PEER_TRANSFER)); raise SystemExit("spoofed sender must raise")
    except PermissionError as exc:
        assert "SENDER_NOT_ACTIVE" in str(exc)
    # 6. ping-pong between two agents is cut at hop 6, even when every envelope claims hop 0
    pp, executed, who = ContextScopingGate("P"), 0, "P"
    try:
        for _ in range(50):
            other = "Q" if who == "P" else "P"
            pp.process_handoff(G, mk(who, other, HandoffMode.PEER_TRANSFER, hops=0, chain="pingpong"))
            executed, who = executed + 1, other
    except RuntimeError as exc:
        assert "HOP_LIMIT_EXCEEDED" in str(exc)
    assert executed == 5 and pp.audit_log[-1].effective_hops == 5
    # 7. output contract: a missing field, or one set to None, is a violation; extra fields are fine
    env = mk("A", "B", HandoffMode.PEER_TRANSFER, req=["trajectory_id", "clearance_ly"])
    assert g.validate_response(env, {"trajectory_id": "T", "clearance_ly": 0.4, "extra": 1}) == []
    for bad in ({"trajectory_id": "T"}, {"trajectory_id": "T", "clearance_ly": None}):
        try:
            g.validate_response(env, bad); raise SystemExit("contract violation must raise")
        except ValueError as exc:
            assert "SCHEMA_CONTRACT_VIOLATION" in str(exc)
    # 8. keys outside the whitelist and missing keys: only whitelisted, existing keys appear
    sc = ContextScopingGate("A").process_handoff(G, HandoffEnvelope("E", "A", "B", HandoffMode.PEER_TRANSFER, ["a", "missing"], {"k": 1}, []))
    assert set(sc) == {"a", "_handoff_payload"}
    print("[PASS] Part 2 Protocol: Least-Context Scoping & Handoff Envelopes Confirmed.")
