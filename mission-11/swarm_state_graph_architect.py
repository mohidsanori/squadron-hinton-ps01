"""
Mission 11 - Part 1: Systems Architect
Hybrid state graph: sequential edges, conditional routers, parallel fan-out/gather, channel reducers
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-11-M31 (Andromeda Galaxy Core), Subsystem 11

When several workers run at the same time and write to one shared state, a plain dict loses updates:
the last writer wins and the others vanish silently. LangGraph and Google ADK 2.0 solve this with
channels, each with a reducer that merges concurrent writes. Here a channel is one of

    OVERWRITE       last writer wins (only safe when one branch writes the key)
    APPEND_LIST     list concatenation (associative)
    SUM_NUMERIC     numeric accumulation (associative and commutative)
    MAX_CONFIDENCE  keep the (confidence, payload) pair with the highest confidence; ties are broken
                    by the payload so that the result does not depend on branch order

`OVERWRITE` is the one reducer that is NOT order-independent, so a superstep in which two parallel
branches write different values to an OVERWRITE key is reported as a `ReducerConflict` (or raised in
strict mode) instead of being resolved silently by arrival order.

Parallel groups run as a superstep: every worker reads the same snapshot, writes a delta, and the
deltas are merged at a barrier in declared order. With `concurrent=True` the workers really run in a
thread pool; the merged state is the same either way.

Latency model (agentic Amdahl). With coordination overhead l_coord and worker latencies l_w:

    T_seq  = l_coord + sum_w l_w         S = T_seq / T_crit
    T_crit = l_coord + max_w l_w

Usage:
    python swarm_state_graph_architect.py
"""

import enum
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


class ReducerMode(enum.Enum):
    OVERWRITE = 1       # last-writer-wins scalar channel
    APPEND_LIST = 2     # associative list concatenation across concurrent workers
    SUM_NUMERIC = 3     # commutative numeric accumulation (token/cost counters)
    MAX_CONFIDENCE = 4  # (confidence, payload) with the highest confidence


@dataclass
class AgentNode:
    name: str
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    simulated_latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ParallelGroup:
    source: str
    workers: List[str]
    gather_target: str


@dataclass
class ReducerConflict:
    key: str
    writers: List[str]
    values: List[Any]


@dataclass
class ExecutionReport:
    visited_order: List[str]
    step_count: int
    total_tokens: int
    sequential_latency_ms: float
    critical_path_latency_ms: float
    amdahl_speedup: float
    final_state: Dict[str, Any]
    conflicts: List[ReducerConflict] = field(default_factory=list)
    wall_clock_ms: float = 0.0


def amdahl_speedup(l_coord: float, worker_latencies: List[float]) -> float:
    """S = (l_coord + sum l_w) / (l_coord + max l_w)."""
    if not worker_latencies:
        return 1.0
    t_crit = l_coord + max(worker_latencies)
    return (l_coord + sum(worker_latencies)) / t_crit if t_crit > 0 else 1.0


class SwarmStateGraph:
    """State-graph engine with deterministic reducers, fan-out/gather groups and a step circuit breaker."""

    def __init__(self, max_steps: int = 20, strict_overwrite: bool = False, concurrent: bool = False):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.max_steps = max_steps
        self.strict_overwrite = strict_overwrite
        self.concurrent = concurrent
        self.nodes: Dict[str, AgentNode] = {}
        self.state: Dict[str, Any] = {}
        self.reducers: Dict[str, ReducerMode] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_routers: Dict[str, Callable[[Dict[str, Any]], Optional[str]]] = {}
        self.parallel_groups: Dict[str, ParallelGroup] = {}
        self.conflicts: List[ReducerConflict] = []

    # -- construction -------------------------------------------------------------------
    def set_reducer(self, key: str, mode: ReducerMode) -> None:
        self.reducers[key] = mode

    def add_node(self, node: AgentNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"node {node.name!r} already exists")
        self.nodes[node.name] = node

    def add_edge(self, source: str, target: str) -> None:
        self.edges[source] = target

    def add_conditional_edges(self, source: str, router_fn: Callable[[Dict[str, Any]], Optional[str]]) -> None:
        self.conditional_routers[source] = router_fn

    def add_parallel_group(self, source: str, worker_names: List[str], gather_target: str) -> None:
        if not worker_names:
            raise ValueError("a parallel group needs at least one worker")
        self.parallel_groups[source] = ParallelGroup(source, list(worker_names), gather_target)

    # -- reducers -------------------------------------------------------------------------
    def apply_reduction(self, key: str, value: Any) -> None:
        """Merge `value` into state[key] with the channel's reducer (default OVERWRITE)."""
        mode = self.reducers.get(key, ReducerMode.OVERWRITE)
        if mode == ReducerMode.OVERWRITE:
            self.state[key] = value
        elif mode == ReducerMode.APPEND_LIST:
            existing = list(self.state.get(key, []))
            existing.extend(value if isinstance(value, list) else [value])
            self.state[key] = existing
        elif mode == ReducerMode.SUM_NUMERIC:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"SUM_NUMERIC channel '{key}' needs a number, got {type(value).__name__}")
            self.state[key] = self.state.get(key, 0) + value
        elif mode == ReducerMode.MAX_CONFIDENCE:
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(f"MAX_CONFIDENCE reducer for '{key}' expects a (confidence, payload) tuple.")
            best = self.state.get(key)
            # higher confidence wins; an exact tie is broken by repr(payload) so the result is order-independent
            if best is None or (float(value[0]), repr(value[1])) > (float(best[0]), repr(best[1])):
                self.state[key] = value

    def merge_delta(self, delta: Dict[str, Any]) -> None:
        if not isinstance(delta, dict):
            raise TypeError(f"a node must return a dict of state updates, got {type(delta).__name__}")
        for k, v in delta.items():
            self.apply_reduction(k, v)

    def _detect_overwrite_conflicts(self, worker_names: List[str], deltas: List[Dict[str, Any]]) -> List[ReducerConflict]:
        writes: Dict[str, List[Tuple[str, Any]]] = {}
        for name, delta in zip(worker_names, deltas):
            for k, v in delta.items():
                if self.reducers.get(k, ReducerMode.OVERWRITE) == ReducerMode.OVERWRITE:
                    writes.setdefault(k, []).append((name, v))
        found = []
        for k, ws in writes.items():
            if len(ws) > 1 and len({repr(v) for _, v in ws}) > 1:
                found.append(ReducerConflict(k, [n for n, _ in ws], [v for _, v in ws]))
        return found

    # -- latency model -----------------------------------------------------------------------
    def calculate_amdahl_speedup(self, seq_nodes: List[str], parallel_branch: List[str]) -> float:
        """S = (l_coord + sum l_w) / (l_coord + max l_w) with l_coord the sum over `seq_nodes`."""
        l_coord = sum(self.nodes[n].simulated_latency_ms for n in seq_nodes)
        return round(amdahl_speedup(l_coord, [self.nodes[w].simulated_latency_ms for w in parallel_branch]), 4)

    # -- execution ----------------------------------------------------------------------------
    def _node(self, name: Optional[str], context: str) -> AgentNode:
        if name not in self.nodes:
            raise KeyError(f"unknown node {name!r} ({context})")
        return self.nodes[name]

    def execute_graph(self, entry_node: str, initial_state: Optional[Dict[str, Any]] = None) -> ExecutionReport:
        """
        Run from `entry_node`. A node with a parallel group fans out to its workers (one superstep, merged at
        a barrier), then continues at the gather target; otherwise a conditional router, then a plain edge,
        decides the next node. More than `max_steps` node executions raises RuntimeError("MAX_STEPS_EXCEEDED").
        """
        self.state = dict(initial_state or {})
        self.conflicts = []
        visited: List[str] = []
        steps = total_tokens = 0
        seq_latency = crit_latency = 0.0
        started = time.perf_counter()
        current: Optional[str] = entry_node

        while current is not None:
            steps += 1
            if steps > self.max_steps:
                raise RuntimeError(f"MAX_STEPS_EXCEEDED: Graph exceeded {self.max_steps} execution transitions.")
            node = self._node(current, "entry/edge target")
            visited.append(node.name)
            self.merge_delta(node.handler(dict(self.state)))
            total_tokens += node.input_tokens + node.output_tokens
            seq_latency += node.simulated_latency_ms
            crit_latency += node.simulated_latency_ms

            if current in self.parallel_groups:
                group = self.parallel_groups[current]
                workers = [self._node(w, f"worker of {current}") for w in group.workers]
                steps += len(workers)
                if steps > self.max_steps:
                    raise RuntimeError(f"MAX_STEPS_EXCEEDED during parallel group from '{current}'.")
                snapshot = dict(self.state)
                if self.concurrent and len(workers) > 1:
                    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
                        deltas = list(pool.map(lambda w: w.handler(dict(snapshot)), workers))
                else:
                    deltas = [w.handler(dict(snapshot)) for w in workers]
                visited.extend(w.name for w in workers)
                total_tokens += sum(w.input_tokens + w.output_tokens for w in workers)
                found = self._detect_overwrite_conflicts([w.name for w in workers], deltas)
                if found and self.strict_overwrite:
                    raise RuntimeError(f"OVERWRITE_CONFLICT on {[c.key for c in found]}: parallel branches wrote different values")
                self.conflicts.extend(found)
                for d in deltas:                                         # barrier merge in declared order
                    self.merge_delta(d)
                lat = [w.simulated_latency_ms for w in workers]
                seq_latency += sum(lat)
                crit_latency += max(lat)
                current = group.gather_target
            elif current in self.conditional_routers:
                current = self.conditional_routers[current](self.state)
            else:
                current = self.edges.get(current)

        return ExecutionReport(visited, steps, total_tokens, round(seq_latency, 2), round(crit_latency, 2),
                               round(seq_latency / crit_latency, 4) if crit_latency > 0 else 1.0, dict(self.state),
                               list(self.conflicts), round((time.perf_counter() - started) * 1000, 2))


# ==============================================================================
# Verification harness for Part 1 (Systems Architect)
# ==============================================================================
if __name__ == "__main__":
    graph = SwarmStateGraph(max_steps=15)
    graph.set_reducer("mission_brief", ReducerMode.OVERWRITE)
    graph.set_reducer("orbital_findings", ReducerMode.APPEND_LIST)
    graph.set_reducer("compute_units", ReducerMode.SUM_NUMERIC)
    graph.set_reducer("best_vector", ReducerMode.MAX_CONFIDENCE)

    graph.add_node(AgentNode("Coordinator", lambda s: {"mission_brief": "SEC-11-M31 Double Nucleus Transit"}, 120.0, 180, 90))
    graph.add_node(AgentNode("Astrogation_P1", lambda s: {"orbital_findings": "P1 eccentric apoapsis clearance verified",
                                                          "compute_units": 420, "best_vector": (0.84, "VEC-P1-OUTER")}, 260.0, 240, 180))
    graph.add_node(AgentNode("Astrogation_P2", lambda s: {"orbital_findings": "P2 periapsis disk crossing stable",
                                                          "compute_units": 390, "best_vector": (0.93, "VEC-P2-INCLINED-38DEG")}, 300.0, 230, 160))
    graph.add_node(AgentNode("TidalShear_Scanner", lambda s: {"orbital_findings": "M31* relativistic tidal shear within hull tolerance",
                                                              "compute_units": 450, "best_vector": (0.88, "VEC-M31-POLAR")}, 240.0, 260, 190))
    graph.add_node(AgentNode("TrajectorySynthesizer", lambda s: {"approved_route": s["best_vector"][1],
                                                                 "finding_count": len(s["orbital_findings"])}, 140.0, 310, 120))
    graph.add_parallel_group("Coordinator", ["Astrogation_P1", "Astrogation_P2", "TidalShear_Scanner"], "TrajectorySynthesizer")

    report = graph.execute_graph("Coordinator")
    expected_speedup = graph.calculate_amdahl_speedup(["Coordinator", "TrajectorySynthesizer"],
                                                      ["Astrogation_P1", "Astrogation_P2", "TidalShear_Scanner"])
    print(f"[StateGraph Execution] steps={report.step_count} | tokens={report.total_tokens} | T_seq={report.sequential_latency_ms}ms | "
          f"T_crit={report.critical_path_latency_ms}ms | Amdahl Speedup={report.amdahl_speedup}x")
    print(f"[Reducer Channels] findings={len(report.final_state['orbital_findings'])} | "
          f"compute_units={report.final_state['compute_units']} | best_vector={report.final_state['best_vector']}")
    assert len(report.final_state["orbital_findings"]) == 3
    assert report.final_state["compute_units"] == 1260
    assert report.final_state["best_vector"] == (0.93, "VEC-P2-INCLINED-38DEG")
    assert report.amdahl_speedup == expected_speedup and report.amdahl_speedup > 1.85

    # ---- extra checks ----------------------------------------------------------------
    # 1. hand-computed numbers: T_seq = 120+260+300+240+140 = 1060, T_crit = 120+300+140 = 560, S = 1.8929
    assert report.sequential_latency_ms == 1060.0 and report.critical_path_latency_ms == 560.0 and abs(report.amdahl_speedup - 1060 / 560) < 1e-3
    assert report.visited_order == ["Coordinator", "Astrogation_P1", "Astrogation_P2", "TidalShear_Scanner", "TrajectorySynthesizer"]
    # 2. reducers: the same final state for every ordering of the parallel branches (the merge is order-independent)
    import itertools
    names = ["Astrogation_P1", "Astrogation_P2", "TidalShear_Scanner"]
    for perm in itertools.permutations(names):
        g = SwarmStateGraph(max_steps=15); g.reducers, g.nodes = dict(graph.reducers), dict(graph.nodes)
        g.add_parallel_group("Coordinator", list(perm), "TrajectorySynthesizer")
        s = g.execute_graph("Coordinator").final_state
        assert s["compute_units"] == 1260 and s["best_vector"] == (0.93, "VEC-P2-INCLINED-38DEG") and sorted(s["orbital_findings"]) == sorted(report.final_state["orbital_findings"])
    # 3. a MAX_CONFIDENCE tie is broken by payload, not by arrival order
    t = SwarmStateGraph(); t.set_reducer("v", ReducerMode.MAX_CONFIDENCE)
    for val in ((0.9, "B"), (0.9, "A")):
        t.apply_reduction("v", val)
    t2 = SwarmStateGraph(); t2.set_reducer("v", ReducerMode.MAX_CONFIDENCE)
    for val in ((0.9, "A"), (0.9, "B")):
        t2.apply_reduction("v", val)
    assert t.state == t2.state
    # 4. the lost-update race: a plain shared dict keeps 1 of 3 findings, the APPEND_LIST channel keeps all 3
    naive: Dict[str, Any] = {}
    for w in ("a", "b", "c"):
        naive.update({"findings": [w]})
    assert naive["findings"] == ["c"]
    # 5. OVERWRITE written differently by two parallel branches is reported (or raised), not silently resolved
    def conflict_graph(strict: bool) -> SwarmStateGraph:
        g = SwarmStateGraph(strict_overwrite=strict)
        g.add_node(AgentNode("S", lambda s: {}, 1)); g.add_node(AgentNode("A", lambda s: {"verdict": "go"}, 1))
        g.add_node(AgentNode("B", lambda s: {"verdict": "no-go"}, 1)); g.add_node(AgentNode("E", lambda s: {}, 1))
        g.add_parallel_group("S", ["A", "B"], "E")
        return g
    rc = conflict_graph(False).execute_graph("S")
    assert [c.key for c in rc.conflicts] == ["verdict"] and rc.conflicts[0].writers == ["A", "B"]
    try:
        conflict_graph(True).execute_graph("S"); raise SystemExit("strict mode must raise")
    except RuntimeError as exc:
        assert "OVERWRITE_CONFLICT" in str(exc)
    # 6. conditional router with a loop, and the max_steps breaker against a runaway loop
    loop = SwarmStateGraph(max_steps=10); loop.set_reducer("tries", ReducerMode.SUM_NUMERIC)
    loop.add_node(AgentNode("Retry", lambda s: {"tries": 1}, 5))
    loop.add_conditional_edges("Retry", lambda s: "Retry" if s["tries"] < 4 else None)
    assert loop.execute_graph("Retry").final_state["tries"] == 4
    runaway = SwarmStateGraph(max_steps=10); runaway.add_node(AgentNode("Spin", lambda s: {}, 5))
    runaway.add_conditional_edges("Spin", lambda s: "Spin")
    try:
        runaway.execute_graph("Spin"); raise SystemExit("runaway loop must be stopped")
    except RuntimeError as exc:
        assert "MAX_STEPS_EXCEEDED" in str(exc)
    # 7. real concurrency: 3 workers sleeping 60 ms finish in about one sleep, with the same state as the sequential run
    def sleeper(tag):
        return lambda s: (time.sleep(0.06), {"done": tag, "n": 1})[1]
    def timed(concurrent):
        g = SwarmStateGraph(concurrent=concurrent); g.set_reducer("done", ReducerMode.APPEND_LIST); g.set_reducer("n", ReducerMode.SUM_NUMERIC)
        g.add_node(AgentNode("S", lambda s: {}, 1)); g.add_node(AgentNode("E", lambda s: {}, 1))
        for tag in "abc":
            g.add_node(AgentNode(tag, sleeper(tag), 60))
        g.add_parallel_group("S", ["a", "b", "c"], "E")
        return g.execute_graph("S")
    seq, par = timed(False), timed(True)
    assert par.final_state == seq.final_state and par.wall_clock_ms < seq.wall_clock_ms * 0.7, (seq.wall_clock_ms, par.wall_clock_ms)
    print(f"Real threads: 3 x 60 ms workers took {seq.wall_clock_ms} ms sequentially and {par.wall_clock_ms} ms concurrently.")
    # 8. Amdahl: balanced workers give a speedup of about K / (1 + K*c); with a growing coordination cost it saturates
    assert abs(amdahl_speedup(0, [10, 10, 10, 10]) - 4.0) < 1e-12 and amdahl_speedup(100, []) == 1.0
    sat = [amdahl_speedup(50 + 20 * k, [100] * k) for k in (1, 2, 4, 8, 16, 64, 1024)]
    assert all(b >= a for a, b in zip(sat, sat[1:])) and sat[-1] < 1 + 100 / 20 and sat[-1] > 5.5   # tends to 1 + w/c1 = 6
    # 9. bad input
    for bad in (lambda: SwarmStateGraph(max_steps=0), lambda: graph.execute_graph("Nope"), lambda: SwarmStateGraph().add_parallel_group("x", [], "y"),
                lambda: (lambda g: (g.set_reducer("n", ReducerMode.SUM_NUMERIC), g.apply_reduction("n", "text")))(SwarmStateGraph()),
                lambda: (lambda g: (g.set_reducer("v", ReducerMode.MAX_CONFIDENCE), g.apply_reduction("v", 0.5)))(SwarmStateGraph()),
                lambda: (lambda g: (g.add_node(AgentNode("x", lambda s: [], 1)), g.execute_graph("x")))(SwarmStateGraph())):
        try:
            bad(); raise SystemExit("invalid input must raise")
        except (ValueError, KeyError, TypeError):
            pass
    print("[PASS] Part 1 Architect: State-Graph, Parallel Reducers & Amdahl Speedup Confirmed.")
