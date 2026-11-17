"""
Mission 10 - Part 1: Systems Architect
OpenTelemetry GenAI (gen_ai.*) span tree and a deterministic loop / token-budget circuit breaker
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-10-VOID (Booetes Deep Void), Subsystem 10

An agent that calls tools in a loop is opaque unless every step leaves a span. Spans follow the
OpenTelemetry GenAI semantic conventions: one trace per request, one span per agent run, model call
or tool call, nested by parent, carrying attributes such as

    gen_ai.operation.name        invoke_agent | chat | execute_tool
    gen_ai.request.model         the model id
    gen_ai.usage.input_tokens / gen_ai.usage.output_tokens
    gen_ai.tool.name / gen_ai.tool.call.arguments
    gen_ai.safety.semantic_entropy / gen_ai.safety.taint_level     (custom, from Parts 2-3)

The circuit breaker is checked BEFORE a proposed step runs and halts the trace when

    |S_tool| > K_max (4)          too many tool steps
    sum(in + out tokens) > B_max (4096)   token budget exhausted
    a_t = a_{t-1}                 the same (tool, arguments) twice in a row

Arguments are compared after JSON normalisation, so reordering keys or adding whitespace does not
hide a repeat. A check is a pure decision; the caller halts the trace with `record_halt`, after which
every further check on that trace is refused.

No OpenTelemetry package is needed: `export_otlp` produces the OTLP/JSON structure directly.

Usage:
    python telemetry_trace_architect.py
"""

import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

SpanOperation = Literal["invoke_agent", "chat", "execute_tool"]
SpanStatus = Literal["OK", "ERROR", "CIRCUIT_BROKEN"]
OPERATIONS = ("invoke_agent", "chat", "execute_tool")


@dataclass
class GenAISpan:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: SpanOperation
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    status: SpanStatus = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["GenAISpan"] = field(default_factory=list)
    start_unix_nano: int = 0
    end_unix_nano: Optional[int] = None

    @property
    def name(self) -> str:
        """Span name per the GenAI convention: '{operation} {target}'."""
        target = self.attributes.get("gen_ai.tool.name") if self.operation_name == "execute_tool" \
            else self.attributes.get("gen_ai.agent.name") or self.model_name
        return f"{self.operation_name} {target}".strip()

    @property
    def is_open(self) -> bool:
        return self.end_unix_nano is None


@dataclass
class CircuitBreakerDecision:
    tripped: bool
    reason: str
    total_tool_steps: int
    cumulative_tokens: int


def normalize_tool_args(args: Optional[str]) -> str:
    """Canonical form of tool arguments: parsed JSON re-serialised with sorted keys, else stripped text."""
    if args is None:
        return ""
    try:
        return json.dumps(json.loads(args), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return " ".join(str(args).split())


class OpenTelemetryGenAITracer:
    """Hierarchical gen_ai.* span recorder with a deterministic circuit breaker."""

    def __init__(self, max_tool_steps: int = 4, max_token_budget: int = 4096,
                 service_name: str = "aura-subsystem-10"):
        if max_tool_steps < 1 or max_token_budget < 1:
            raise ValueError("max_tool_steps and max_token_budget must be >= 1")
        self.max_tool_steps = max_tool_steps
        self.max_token_budget = max_token_budget
        self.service_name = service_name
        self.spans_by_trace: Dict[str, List[GenAISpan]] = {}
        self.tool_history_by_trace: Dict[str, List[Tuple[str, str]]] = {}
        self.halted: Dict[str, str] = {}
        self._clock = time.time_ns

    # -- traces and spans ----------------------------------------------------------
    def create_trace(self) -> str:
        trace_id = secrets.token_hex(16)                 # 32 hex characters, as in W3C trace context
        self.spans_by_trace[trace_id] = []
        self.tool_history_by_trace[trace_id] = []
        return trace_id

    def _require_trace(self, trace_id: str) -> None:
        if trace_id not in self.spans_by_trace:
            raise KeyError(f"unknown trace {trace_id!r}")

    def start_span(self, trace_id: str, operation_name: SpanOperation, model_name: str,
                   parent_span_id: Optional[str] = None, tool_name: Optional[str] = None,
                   tool_args: Optional[str] = None, agent_name: Optional[str] = None) -> GenAISpan:
        """Open a span. Tokens and status are filled in by `end_span`."""
        self._require_trace(trace_id)
        if operation_name not in OPERATIONS:
            raise ValueError(f"operation_name must be one of {OPERATIONS}")
        if parent_span_id is not None and parent_span_id not in {s.span_id for s in self.spans_by_trace[trace_id]}:
            raise ValueError(f"parent span {parent_span_id!r} is not in trace {trace_id}")
        attrs: Dict[str, Any] = {"gen_ai.operation.name": operation_name, "gen_ai.request.model": model_name,
                                 "gen_ai.safety.taint_level": "TRUSTED_SYSTEM"}
        if tool_name:
            attrs["gen_ai.tool.name"] = tool_name
        if tool_args is not None:
            attrs["gen_ai.tool.call.arguments"] = tool_args
        if agent_name:
            attrs["gen_ai.agent.name"] = agent_name
        span = GenAISpan(trace_id, secrets.token_hex(8), parent_span_id, operation_name, model_name,
                         attributes=attrs, start_unix_nano=self._clock())
        self.spans_by_trace[trace_id].append(span)
        return span

    def end_span(self, span: GenAISpan, input_tokens: int = 0, output_tokens: int = 0, status: SpanStatus = "OK",
                 semantic_entropy: Optional[float] = None, taint_level: Optional[str] = None,
                 latency_ms: Optional[float] = None, error: Optional[str] = None) -> GenAISpan:
        """Close a span and record usage and safety attributes. `latency_ms` overrides the measured time."""
        if not span.is_open:
            raise RuntimeError(f"span {span.span_id} is already closed")
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        span.end_unix_nano = self._clock()
        span.latency_ms = round((span.end_unix_nano - span.start_unix_nano) / 1e6, 3) if latency_ms is None else latency_ms
        span.input_tokens, span.output_tokens, span.status = input_tokens, output_tokens, status
        span.attributes["gen_ai.usage.input_tokens"] = input_tokens
        span.attributes["gen_ai.usage.output_tokens"] = output_tokens
        if semantic_entropy is not None:
            span.attributes["gen_ai.safety.semantic_entropy"] = round(semantic_entropy, 4)
        if taint_level is not None:
            span.attributes["gen_ai.safety.taint_level"] = taint_level
        if error is not None:
            span.attributes["error.type"] = error
            if status == "OK":
                span.status = "ERROR"
        if span.operation_name == "execute_tool" and span.attributes.get("gen_ai.tool.name"):
            self.tool_history_by_trace[span.trace_id].append(
                (span.attributes["gen_ai.tool.name"], normalize_tool_args(span.attributes.get("gen_ai.tool.call.arguments"))))
        return span

    def record_span(self, trace_id: str, parent_span_id: Optional[str], operation_name: SpanOperation, model_name: str,
                    input_tokens: int, output_tokens: int, latency_ms: float, tool_name: Optional[str] = None,
                    tool_args: Optional[str] = None, semantic_entropy: Optional[float] = None,
                    taint_level: str = "TRUSTED_SYSTEM", status: SpanStatus = "OK") -> GenAISpan:
        """Open and close a span in one call with an explicit latency (the starter API)."""
        span = self.start_span(trace_id, operation_name, model_name, parent_span_id, tool_name, tool_args)
        return self.end_span(span, input_tokens, output_tokens, status, semantic_entropy, taint_level, latency_ms)

    # -- circuit breaker ---------------------------------------------------------------
    def cumulative_tokens(self, trace_id: str) -> int:
        """Tokens of closed spans (an open span has not reported its usage yet)."""
        return sum(s.input_tokens + s.output_tokens for s in self.spans_by_trace.get(trace_id, []) if not s.is_open)

    def check_circuit_breaker(self, trace_id: str, new_tokens: Optional[int] = None, tool_name: Optional[str] = None,
                              tool_args: Optional[str] = None, *, proposed_tokens: Optional[int] = None,
                              proposed_tool_name: Optional[str] = None,
                              proposed_tool_args: Optional[str] = None) -> CircuitBreakerDecision:
        """
        Decide whether the proposed next step may run. `new_tokens`/`tool_name`/`tool_args` describe
        the step; the `proposed_*` spellings of the starter are accepted too. The check has no side
        effect; a trace that was halted with `record_halt` is refused outright.
        """
        self._require_trace(trace_id)
        tokens = new_tokens if new_tokens is not None else (proposed_tokens or 0)
        tool = tool_name if tool_name is not None else proposed_tool_name
        args = tool_args if tool_args is not None else proposed_tool_args
        if tokens < 0:
            raise ValueError("new_tokens must be non-negative")

        history = self.tool_history_by_trace.get(trace_id, [])
        next_tokens = self.cumulative_tokens(trace_id) + tokens
        next_steps = len(history) + (1 if tool else 0)

        def trip(reason: str) -> CircuitBreakerDecision:
            return CircuitBreakerDecision(True, reason, next_steps, next_tokens)

        if trace_id in self.halted:
            return trip(f"TRACE_ALREADY_HALTED ({self.halted[trace_id].split(':')[0].split(' ')[0]})")
        if tool and history and history[-1] == (tool, normalize_tool_args(args)):
            return trip(f"IDENTICAL_TOOL_CYCLE_DETECTED: {tool}({args})")
        if next_steps > self.max_tool_steps:
            return trip(f"MAX_TOOL_STEPS_EXCEEDED ({next_steps} > {self.max_tool_steps})")
        if next_tokens > self.max_token_budget:
            return trip(f"TOKEN_BUDGET_EXCEEDED ({next_tokens} > {self.max_token_budget})")
        return CircuitBreakerDecision(False, "WITHIN_OPERATIONAL_ENVELOPE", next_steps, next_tokens)

    def record_halt(self, trace_id: str, parent_span_id: Optional[str], decision: CircuitBreakerDecision,
                    model_name: str = "circuit-breaker") -> GenAISpan:
        """Halt the trace and leave a CIRCUIT_BROKEN span in it so the halt is visible to whoever reads it."""
        self.halted.setdefault(trace_id, decision.reason)
        span = self.start_span(trace_id, "invoke_agent", model_name, parent_span_id, agent_name="circuit_breaker")
        span.attributes["gen_ai.safety.circuit_breaker.reason"] = decision.reason
        return self.end_span(span, status="CIRCUIT_BROKEN", latency_ms=0.0)

    # -- trace tree ------------------------------------------------------------------------
    def build_trace_tree(self, trace_id: str) -> Dict[str, Any]:
        """Nested parent -> child execution tree with aggregated token and latency telemetry."""
        self._require_trace(trace_id)
        spans = self.spans_by_trace[trace_id]
        by_id = {s.span_id: s for s in spans}
        for s in spans:
            s.children = []
        roots: List[GenAISpan] = []
        for s in spans:
            if s.parent_span_id in by_id:
                by_id[s.parent_span_id].children.append(s)
            else:
                roots.append(s)

        def subtree_tokens(s: GenAISpan) -> int:
            return s.input_tokens + s.output_tokens + sum(subtree_tokens(c) for c in s.children)

        total_in, total_out = sum(s.input_tokens for s in spans), sum(s.output_tokens for s in spans)
        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_latency_ms": round(sum(s.latency_ms for s in spans), 2),     # sum over spans: nested time counts twice
            "wall_clock_ms": round(sum(r.latency_ms for r in roots), 2),         # root spans only: no double counting
            "tool_steps": len(self.tool_history_by_trace[trace_id]),
            "halted": self.halted.get(trace_id),
            "roots": roots,
            "subtree_tokens": {r.span_id: subtree_tokens(r) for r in roots},
        }

    def render_tree(self, trace_id: str) -> str:
        """Indented text view: operation, model or tool, tokens, latency, status and safety attributes."""
        tree = self.build_trace_tree(trace_id)
        lines: List[str] = []

        def walk(s: GenAISpan, depth: int) -> None:
            safety = [f"{k.split('.')[-1]}={v}" for k, v in s.attributes.items() if k.startswith("gen_ai.safety.") and v != "TRUSTED_SYSTEM"]
            lines.append(f"{'  ' * depth}{s.name:<34} tok={s.input_tokens + s.output_tokens:<5} {s.latency_ms:>8.2f} ms  {s.status}"
                         + (f"  [{', '.join(safety)}]" if safety else ""))
            for c in s.children:
                walk(c, depth + 1)
        for r in tree["roots"]:
            walk(r, 0)
        return "\n".join(lines)

    def export_otlp(self, trace_id: str) -> Dict[str, Any]:
        """The trace as an OTLP/JSON `resourceSpans` document (what an OpenTelemetry collector ingests)."""
        def attr(k: str, v: Any) -> Dict[str, Any]:
            val = ({"intValue": str(v)} if isinstance(v, int) and not isinstance(v, bool)
                   else {"doubleValue": v} if isinstance(v, float) else {"stringValue": str(v)})
            return {"key": k, "value": val}
        out = []
        for s in self.spans_by_trace[trace_id]:
            out.append({
                "traceId": s.trace_id, "spanId": s.span_id, "parentSpanId": s.parent_span_id or "",
                "name": s.name, "kind": 3 if s.operation_name == "chat" else 1,           # CLIENT for model calls, INTERNAL otherwise
                "startTimeUnixNano": str(s.start_unix_nano), "endTimeUnixNano": str(s.end_unix_nano or s.start_unix_nano),
                "attributes": [attr(k, v) for k, v in s.attributes.items()],
                "status": {"code": 1 if s.status == "OK" else 2, "message": s.status if s.status != "OK" else ""},
            })
        return {"resourceSpans": [{"resource": {"attributes": [attr("service.name", self.service_name)]},
                                   "scopeSpans": [{"scope": {"name": "squadron.hinton.genai"}, "spans": out}]}]}


# ==============================================================================
# Verification harness for Part 1 (Systems Architect)
# ==============================================================================
if __name__ == "__main__":
    tracer = OpenTelemetryGenAITracer(max_tool_steps=4, max_token_budget=4096)
    t_id = tracer.create_trace()

    root = tracer.record_span(trace_id=t_id, parent_span_id=None, operation_name="invoke_agent",
                              model_name="aura-void-navigator-v10", input_tokens=320, output_tokens=140,
                              latency_ms=42.5, semantic_entropy=0.21)
    chat_span = tracer.record_span(trace_id=t_id, parent_span_id=root.span_id, operation_name="chat",
                                   model_name="gemini-2.5-pro", input_tokens=640, output_tokens=210,
                                   latency_ms=185.0, semantic_entropy=0.19)
    tool_span = tracer.record_span(trace_id=t_id, parent_span_id=chat_span.span_id, operation_name="execute_tool",
                                   model_name="gemini-2.5-pro", input_tokens=120, output_tokens=80, latency_ms=34.0,
                                   tool_name="scan_void_sector", tool_args='{"sector": "SEC-10-VOID"}')

    cycle_check = tracer.check_circuit_breaker(trace_id=t_id, proposed_tokens=200, proposed_tool_name="scan_void_sector",
                                               proposed_tool_args='{"sector": "SEC-10-VOID"}')
    print(f"[Circuit Breaker Cycle Test] Tripped={cycle_check.tripped} | Reason={cycle_check.reason}")
    assert cycle_check.tripped is True and "IDENTICAL_TOOL_CYCLE_DETECTED" in cycle_check.reason

    t2 = tracer.create_trace()
    tracer.record_span(t2, None, "invoke_agent", "aura", 320, 140, 42.5)
    tracer.record_span(t2, None, "chat", "gemini-2.5-pro", 640, 210, 185.0)
    tracer.record_span(t2, None, "execute_tool", "gemini-2.5-pro", 120, 80, 34.0, tool_name="scan_void_sector", tool_args='{"sector": "SEC-10-VOID"}')
    budget_check = tracer.check_circuit_breaker(trace_id=t2, proposed_tokens=3500, proposed_tool_name="calibrate_gyro",
                                                proposed_tool_args='{"axis": "z"}')
    print(f"[Circuit Breaker Budget Test] Tripped={budget_check.tripped} | Reason={budget_check.reason}")
    assert budget_check.tripped is True and "TOKEN_BUDGET_EXCEEDED" in budget_check.reason

    tree = tracer.build_trace_tree(t_id)
    print(f"[OpenTelemetry Tree] trace_id={tree['trace_id']} | spans={tree['span_count']} | "
          f"total_tokens={tree['total_tokens']} | root_children={len(tree['roots'][0].children)}")
    assert tree["span_count"] == 3 and tree["total_tokens"] == 1510

    # ---- extra checks ----------------------------------------------------------------
    # 1. measured spans: start/end nest correctly, parents must exist, closed spans cannot be closed again
    t3 = tracer.create_trace()
    agent = tracer.start_span(t3, "invoke_agent", "aura", agent_name="navigator")
    chat = tracer.start_span(t3, "chat", "gemini-2.5-pro", agent.span_id)
    time.sleep(0.01)
    tracer.end_span(chat, 100, 50)
    tool = tracer.start_span(t3, "execute_tool", "gemini-2.5-pro", agent.span_id, tool_name="scan", tool_args='{"b": 2, "a": 1}')
    tracer.end_span(tool, 10, 5)
    tracer.end_span(agent, 0, 0)
    tr = tracer.build_trace_tree(t3)
    assert chat.latency_ms >= 9.0 and agent.latency_ms >= chat.latency_ms          # a parent outlasts its children
    assert abs(tr["wall_clock_ms"] - agent.latency_ms) < 0.01 and tr["total_latency_ms"] > tr["wall_clock_ms"]
    assert tr["total_tokens"] == 165 and tr["subtree_tokens"][agent.span_id] == 165
    assert [c.name for c in tr["roots"][0].children] == ["chat gemini-2.5-pro", "execute_tool scan"] and agent.name == "invoke_agent navigator"
    for bad in (lambda: tracer.end_span(chat), lambda: tracer.start_span(t3, "chat", "m", "nonexistent"),
                lambda: tracer.start_span(t3, "teleport", "m"), lambda: tracer.start_span("nope", "chat", "m"),
                lambda: tracer.end_span(tracer.start_span(tracer.create_trace(), "chat", "m"), -1, 0)):
        try:
            bad(); raise SystemExit("invalid span operation must raise")
        except (RuntimeError, ValueError, KeyError):
            pass
    # 2. the cycle check ignores key order and whitespace
    assert tracer.check_circuit_breaker(t3, 10, "scan", '{ "a": 1,  "b": 2 }').tripped is True
    # 3. a check has no side effect; after record_halt the trace refuses even a harmless step
    assert t3 not in tracer.halted and tracer.check_circuit_breaker(t3, 1, "other", "{}").tripped is False
    tracer.record_halt(t3, None, tracer.check_circuit_breaker(t3, 10, "scan", '{"a":1,"b":2}'))
    assert tracer.check_circuit_breaker(t3, 1, "other", "{}").reason.startswith("TRACE_ALREADY_HALTED")
    # 4. step ceiling: the fifth distinct tool step trips, the fourth does not; an A,B,A,B alternation is not
    #    "consecutive identical" but still hits the ceiling
    t4 = tracer.create_trace()
    decisions = []
    for i, name in enumerate(["A", "B", "A", "B", "A"]):
        d = tracer.check_circuit_breaker(t4, 10, name, "{}")
        decisions.append(d.tripped)
        if d.tripped:
            break
        tracer.record_span(t4, None, "execute_tool", "m", 5, 5, 1.0, tool_name=name, tool_args="{}")
    assert decisions == [False, False, False, False, True] and "MAX_TOOL_STEPS_EXCEEDED" in d.reason
    # 5. token budget is cumulative: exactly 4096 is allowed, 4097 is not; open spans are not counted
    t5 = tracer.create_trace()
    tracer.record_span(t5, None, "chat", "m", 2000, 1000, 5.0)
    assert tracer.check_circuit_breaker(t5, 1096).tripped is False
    assert tracer.check_circuit_breaker(t5, 1097).tripped is True
    # 6. record_halt leaves a visible CIRCUIT_BROKEN span; render and OTLP export work
    r = tracer.record_halt(t5, None, tracer.check_circuit_breaker(t5, 1097))
    assert r.status == "CIRCUIT_BROKEN" and "gen_ai.safety.circuit_breaker.reason" in r.attributes
    otlp = tracer.export_otlp(t3)
    sp = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(sp[0]["traceId"]) == 32 and len(sp[0]["spanId"]) == 16 and sp[1]["parentSpanId"] == sp[0]["spanId"]
    assert any(a["key"] == "gen_ai.usage.input_tokens" and a["value"] == {"intValue": "100"} for a in sp[1]["attributes"])
    json.dumps(otlp)
    print("\n" + tracer.render_tree(t3))
    for bad in ({"max_tool_steps": 0}, {"max_token_budget": 0}):
        try:
            OpenTelemetryGenAITracer(**bad); raise SystemExit("invalid config must raise")
        except ValueError:
            pass
    print("[PASS] Part 1 OpenTelemetry GenAI Tracer & Circuit Breaker verified successfully!")
