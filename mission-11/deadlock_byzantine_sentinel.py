"""
Mission 11 - Part 4: Security & Ethics Sentinel
Wait-for-graph deadlock detection (3-colour DFS), deterministic pre-emption, Byzantine telemetry firewall
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-11-M31 (Andromeda Galaxy Core), Subsystem 11

Ship Anomaly 11: three specialists wait on each other in a ring (Astrogation -> ProtocolValidator ->
PayloadBuilder -> Astrogation) and nothing moves. That is a Coffman circular wait, and it shows up as a
cycle in the wait-for graph G_W = (V, E_W), an edge u -> v meaning "u waits for v".

3-colour DFS finds a cycle in O(|V| + |E|):
    WHITE  not visited        GRAY  on the current recursion stack        BLACK  fully explored
An edge u -> v with v GRAY is a back edge and closes a cycle (the path from v on the stack, then back to v).
An edge to a BLACK vertex is a cross or forward edge: two branches that merely meet again (a diamond) are
NOT a deadlock. The DFS here is iterative, so a deep graph cannot overflow the Python stack.

`SupervisorDeadlockArbiter` breaks a cycle by pre-empting its waiting edge with the lowest priority (ties
by waiter, then holder) and re-checks; `resolve_until_clear` repeats until the graph is acyclic.

`ByzantineSwarmFirewall` quarantines readings from a compromised worker with the modified z-score
    z = 0.6745 * |x - median| / MAD  >  3.5
(robust: the median and MAD ignore a minority of outliers, unlike mean and standard deviation, whose
largest possible z-score in a sample of n is (n - 1) / sqrt(n), under 3 for n = 5). Its breakdown point is
50 %: once colluding outliers are the majority the median moves to them. When MAD = 0 (more than half the
readings identical) it falls back to the mean absolute deviation, z = |x - median| / (1.2533 * meanAD).
A swarm-wide token budget trips a latched circuit breaker.

Usage:
    python deadlock_byzantine_sentinel.py
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

WHITE = 0  # unvisited
GRAY = 1   # on the active recursion stack (an ancestor)
BLACK = 2  # fully explored (a safe cross / forward edge target)


@dataclass
class WaitEdge:
    waiter: str
    holder: str
    priority: int = 5  # lower = lower priority (pre-empted first)


@dataclass
class DeadlockResolution:
    detected_cycle: List[str]
    preempted_edge: Optional[WaitEdge]
    post_resolution_cycle: List[str]


class WaitForGraphMonitor:
    """Wait-for digraph with O(|V| + |E|) 3-colour DFS cycle detection."""

    def __init__(self):
        self.adj: Dict[str, List[WaitEdge]] = {}

    def add_wait_edge(self, u: str, v: str, priority: int = 5) -> None:
        self.adj.setdefault(u, []).append(WaitEdge(u, v, priority))
        self.adj.setdefault(v, [])

    def remove_wait_edge(self, u: str, v: str) -> None:
        if u in self.adj:
            self.adj[u] = [e for e in self.adj[u] if e.holder != v]

    @property
    def edge_count(self) -> int:
        return sum(len(es) for es in self.adj.values())

    def detect_cycle(self) -> List[str]:
        """
        3-colour DFS over all vertices (roots and neighbours in sorted order, so the answer is deterministic).
        Returns the closed cycle [v, ..., u, v] if a back edge u -> v to a GRAY vertex exists, else [].
        """
        color: Dict[str, int] = {n: WHITE for n in self.adj}
        for root in sorted(self.adj):
            if color[root] != WHITE:
                continue
            path: List[str] = [root]
            index_on_path = {root: 0}
            neighbours = {root: iter(sorted(e.holder for e in self.adj[root]))}
            color[root] = GRAY
            while path:
                u = path[-1]
                advanced = False
                for v in neighbours[u]:
                    c = color.get(v, WHITE)
                    if c == GRAY:                                  # back edge: cycle from v on the stack, back to v
                        return path[index_on_path[v]:] + [v]
                    if c == WHITE:                                 # tree edge: descend
                        color[v] = GRAY
                        index_on_path[v] = len(path)
                        path.append(v)
                        neighbours[v] = iter(sorted(e.holder for e in self.adj.get(v, [])))
                        advanced = True
                        break
                    # BLACK: cross / forward edge in a diamond, not a deadlock
                if not advanced:
                    color[u] = BLACK
                    path.pop()
                    del index_on_path[u]
        return []


class SupervisorDeadlockArbiter:
    """Breaks wait-for cycles deterministically by pre-empting the lowest-priority edge on the cycle."""

    @staticmethod
    def _cycle_edges(monitor: WaitForGraphMonitor, cycle: List[str]) -> List[WaitEdge]:
        edges: List[WaitEdge] = []
        for u, v in zip(cycle, cycle[1:]):
            candidates = [e for e in monitor.adj.get(u, []) if e.holder == v]
            edges.append(min(candidates, key=lambda e: (e.priority, e.waiter, e.holder)))
        return edges

    @staticmethod
    def resolve_deadlock(monitor: WaitForGraphMonitor) -> DeadlockResolution:
        """Detect one cycle, remove its lowest-priority edge, and report whether a cycle remains."""
        cycle = monitor.detect_cycle()
        if not cycle:
            return DeadlockResolution([], None, [])
        victim = min(SupervisorDeadlockArbiter._cycle_edges(monitor, cycle), key=lambda e: (e.priority, e.waiter, e.holder))
        monitor.remove_wait_edge(victim.waiter, victim.holder)
        return DeadlockResolution(cycle, victim, monitor.detect_cycle())

    @staticmethod
    def resolve_until_clear(monitor: WaitForGraphMonitor, max_rounds: int = 1000) -> List[DeadlockResolution]:
        """Pre-empt one edge per round until the graph is acyclic (every round removes at least one edge)."""
        rounds: List[DeadlockResolution] = []
        for _ in range(max_rounds):
            r = SupervisorDeadlockArbiter.resolve_deadlock(monitor)
            if not r.detected_cycle:
                return rounds
            rounds.append(r)
        raise RuntimeError("DEADLOCK_NOT_CLEARED")


class ByzantineSwarmFirewall:
    """Robust outlier quarantine and a swarm-wide token circuit breaker."""

    def __init__(self, z_threshold: float = 3.5, max_swarm_tokens: int = 16000):
        if z_threshold <= 0 or max_swarm_tokens <= 0:
            raise ValueError("z_threshold and max_swarm_tokens must be positive")
        self.z_threshold = z_threshold
        self.max_swarm_tokens = max_swarm_tokens
        self.cumulative_tokens = 0
        self.tokens_by_agent: Dict[str, int] = {}
        self.tripped = False

    @staticmethod
    def modified_z(x: float, median: float, mad: float, mean_ad: float = 0.0) -> float:
        """0.6745 |x - median| / MAD, or |x - median| / (1.2533 meanAD) when MAD = 0; 0 when the readings are all identical."""
        if mad > 0:
            return 0.6745 * abs(x - median) / mad
        if mean_ad > 0:
            return abs(x - median) / (1.253314 * mean_ad)
        return 0.0 if x == median else math.inf

    def check_z_score(self, telemetry_val: float, median: float, mad: float) -> bool:
        """True (quarantine) if the modified z-score exceeds the threshold. Non-finite readings always do."""
        if not math.isfinite(telemetry_val):
            return True
        return self.modified_z(telemetry_val, median, mad) > self.z_threshold

    def filter_byzantine_telemetry(self, readings: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
        """Split readings into (trusted, quarantined ids). With fewer than 3 finite readings nothing can be judged."""
        finite = {k: v for k, v in readings.items() if math.isfinite(v)}
        quarantined: List[str] = [k for k in readings if k not in finite]
        if len(finite) < 3:
            return dict(finite), quarantined
        values = list(finite.values())
        med = statistics.median(values)
        deviations = [abs(v - med) for v in values]
        mad, mean_ad = statistics.median(deviations), sum(deviations) / len(deviations)
        trusted: Dict[str, float] = {}
        for agent_id, val in finite.items():
            if self.modified_z(val, med, mad, mean_ad) > self.z_threshold:
                quarantined.append(agent_id)
            else:
                trusted[agent_id] = val
        order = {k: i for i, k in enumerate(readings)}
        return trusted, sorted(quarantined, key=order.get)

    def record_and_check_budget(self, step_tokens: int, agent_id: Optional[str] = None) -> bool:
        """Add tokens; True once cumulative tokens exceed the budget. The breaker stays tripped."""
        if step_tokens < 0:
            raise ValueError("step_tokens must be non-negative")
        self.cumulative_tokens += step_tokens
        if agent_id is not None:
            self.tokens_by_agent[agent_id] = self.tokens_by_agent.get(agent_id, 0) + step_tokens
        self.tripped = self.tripped or self.cumulative_tokens > self.max_swarm_tokens
        return self.tripped

    def heaviest_consumer(self) -> Optional[str]:
        return max(self.tokens_by_agent, key=self.tokens_by_agent.get) if self.tokens_by_agent else None


if __name__ == "__main__":
    diamond = WaitForGraphMonitor()
    for u, v in (("Root", "BranchA"), ("Root", "BranchB"), ("BranchA", "MergeSink"), ("BranchB", "MergeSink")):
        diamond.add_wait_edge(u, v)
    assert diamond.detect_cycle() == []
    print("[Diamond DAG Check] BLACK cross-edge accepted with 0 false positives.")

    anomaly11 = WaitForGraphMonitor()
    anomaly11.add_wait_edge("Astrogation", "ProtocolValidator", priority=9)
    anomaly11.add_wait_edge("ProtocolValidator", "PayloadBuilder", priority=7)
    anomaly11.add_wait_edge("PayloadBuilder", "Astrogation", priority=2)   # lowest priority

    resolution = SupervisorDeadlockArbiter.resolve_deadlock(anomaly11)
    print(f"[Ship Anomaly 11 Resolved] cycle={resolution.detected_cycle} | "
          f"preempted=({resolution.preempted_edge.waiter} -> {resolution.preempted_edge.holder}) | post_cycle={resolution.post_resolution_cycle}")
    assert len(resolution.detected_cycle) == 4
    assert resolution.preempted_edge is not None
    assert resolution.preempted_edge.waiter == "PayloadBuilder"
    assert resolution.post_resolution_cycle == []

    firewall = ByzantineSwarmFirewall(z_threshold=3.5, max_swarm_tokens=10000)
    sensor_reports = {"Probe_Alpha": -301.2, "Probe_Beta": -300.8, "Probe_Gamma": -301.5, "Probe_Delta": -300.9, "Rogue_Spoofer": 999.0}
    clean, quarantined = firewall.filter_byzantine_telemetry(sensor_reports)
    print(f"[Byzantine Firewall] clean={list(clean.keys())} | quarantined={quarantined}")
    assert quarantined == ["Rogue_Spoofer"] and "Rogue_Spoofer" not in clean
    assert firewall.record_and_check_budget(6000) is False
    assert firewall.record_and_check_budget(5500) is True

    # ---- extra checks ----------------------------------------------------------------
    import random, time
    # 1. cycle shapes: self-loop, two-cycle, a cycle hanging off a tail, and the exact cycle path
    m = WaitForGraphMonitor(); m.add_wait_edge("A", "A")
    assert m.detect_cycle() == ["A", "A"]
    m = WaitForGraphMonitor(); m.add_wait_edge("A", "B"); m.add_wait_edge("B", "A")
    assert m.detect_cycle() == ["A", "B", "A"]
    m = WaitForGraphMonitor()
    for u, v in (("T1", "T2"), ("T2", "X"), ("X", "Y"), ("Y", "Z"), ("Z", "X")):
        m.add_wait_edge(u, v)
    assert m.detect_cycle() == ["X", "Y", "Z", "X"]                              # the tail T1, T2 is not part of the cycle
    assert WaitForGraphMonitor().detect_cycle() == []
    # 2. a larger diamond lattice has many cross edges and zero cycles; adding one back edge makes exactly that cycle appear
    lat = WaitForGraphMonitor()
    for layer in range(6):
        for i in range(4):
            for j in range(4):
                lat.add_wait_edge(f"L{layer}_{i}", f"L{layer + 1}_{j}")
    assert lat.detect_cycle() == []
    lat.add_wait_edge("L6_0", "L2_3")
    cyc = lat.detect_cycle()
    assert cyc and cyc[0] == cyc[-1] and all(any(e.holder == b for e in lat.adj[a]) for a, b in zip(cyc, cyc[1:]))
    # 3. 300 random DAGs (edges only point forward): no false positives; the same graphs with one planted back edge: always detected
    rng = random.Random(5)
    fp = tp = 0
    for _ in range(300):
        n = rng.randint(5, 40)
        g = WaitForGraphMonitor()
        for i in range(n):
            g.add_wait_edge(f"n{i:02d}", f"n{min(n - 1, i + 1):02d}") if i + 1 < n else None
        for _ in range(rng.randint(n, 3 * n)):
            a, b = sorted(rng.sample(range(n), 2))
            g.add_wait_edge(f"n{a:02d}", f"n{b:02d}")
        fp += bool(g.detect_cycle())
        g.add_wait_edge(f"n{n - 1:02d}", "n00")                                  # n00 reaches n_{n-1} along the chain, so this closes a cycle
        tp += bool(g.detect_cycle())
    assert fp == 0 and tp == 300
    # 4. linear time and no recursion limit: a 200,000-vertex chain, then closed into a ring
    def chain(n):
        g = WaitForGraphMonitor()
        for i in range(n - 1):
            g.add_wait_edge(f"v{i:07d}", f"v{i + 1:07d}")
        return g
    timings = {}
    for n in (25000, 100000, 200000):
        g = chain(n); t0 = time.perf_counter(); assert g.detect_cycle() == []; timings[n] = time.perf_counter() - t0
    g = chain(200000); g.add_wait_edge("v0199999", "v0000000")
    ring = g.detect_cycle()
    assert len(ring) == 200001
    assert timings[200000] / timings[25000] < 16                                 # 8x the vertices, far less than 64x the time
    print(f"3-colour DFS on chains of 25k / 100k / 200k vertices: {timings[25000] * 1000:.0f} / {timings[100000] * 1000:.0f} / {timings[200000] * 1000:.0f} ms (no recursion limit hit).")
    # 5. arbiter: priority ties break by waiter then holder; several independent cycles are all cleared
    tie = WaitForGraphMonitor()
    tie.add_wait_edge("b", "c", 3); tie.add_wait_edge("c", "a", 3); tie.add_wait_edge("a", "b", 3)
    assert SupervisorDeadlockArbiter.resolve_deadlock(tie).preempted_edge.waiter == "a"
    multi = WaitForGraphMonitor()
    for ring_nodes in (("a", "b", "c"), ("d", "e"), ("f", "g", "h", "i")):
        for u, v in zip(ring_nodes, ring_nodes[1:] + ring_nodes[:1]):
            multi.add_wait_edge(u, v, 5)
    multi.add_wait_edge("c", "d", 9)
    rounds = SupervisorDeadlockArbiter.resolve_until_clear(multi)
    assert len(rounds) == 3 and multi.detect_cycle() == []
    assert SupervisorDeadlockArbiter.resolve_deadlock(WaitForGraphMonitor()).preempted_edge is None
    # 6. MAD against mean/std z-scores: a classical z-score can never flag one outlier among five readings (max z = 1.79)
    vals = list(sensor_reports.values()); mean, sd = statistics.mean(vals), statistics.pstdev(vals)
    assert max(abs(v - mean) / sd for v in vals) < 2.0 and quarantined == ["Rogue_Spoofer"]
    # 7. breakdown point: colluding outliers are caught while they are a minority; once they are the majority
    #    the median IS the lie. Identical rogues make MAD zero (nobody is flagged, the consensus is the lie);
    #    slightly different rogues keep MAD positive and the honest minority is the one quarantined.
    def attack(n_total, n_rogue, jitter=0.0):
        readings = {f"h{i}": -301.0 + 0.1 * (i % 5) for i in range(n_total - n_rogue)}
        readings.update({f"r{i}": 999.0 + jitter * i for i in range(n_rogue)})
        trusted, q = ByzantineSwarmFirewall().filter_byzantine_telemetry(readings)
        return sum(k.startswith("r") for k in q), sum(k.startswith("h") for k in q), statistics.median(trusted.values())
    caught, harmed, consensus = attack(11, 5)
    assert (caught, harmed) == (5, 0) and consensus < -300                       # 5 of 11 rogue: all caught, consensus is honest
    caught, harmed, consensus = attack(11, 6)
    assert (caught, harmed) == (0, 0) and consensus == 999.0                     # 6 identical rogues: MAD = 0, nobody flagged, consensus is the lie
    caught, harmed, consensus = attack(11, 6, jitter=0.05)
    assert caught == 0 and harmed == 5 and consensus > 900                       # 6 jittered rogues: the honest minority is quarantined
    # 8. identical readings (MAD = 0): no outlier; one slightly different honest reading is not flagged by an epsilon MAD
    f = ByzantineSwarmFirewall()
    t, q = f.filter_byzantine_telemetry({"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.0}); assert q == []
    # with no spread to measure against, a lone dissenter's z is n / 1.2533 whatever its size: 5 readings -> 3.99 > 3.5 (flagged),
    # 4 readings -> 3.19 (kept). Identical readings carry no scale information, so this is the best a robust test can do.
    t, q = f.filter_byzantine_telemetry({"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.0, "e": 5.4}); assert q == ["e"]
    t, q = f.filter_byzantine_telemetry({"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.4}); assert q == []
    t, q = f.filter_byzantine_telemetry({"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.0, "e": 500.0}); assert q == ["e"]
    t, q = f.filter_byzantine_telemetry({"a": 1.0, "b": float("nan"), "c": 1.1, "d": 0.9, "e": float("inf")}); assert q == ["b", "e"] and set(t) == {"a", "c", "d"}
    assert f.filter_byzantine_telemetry({"a": 1.0, "b": 900.0}) == ({"a": 1.0, "b": 900.0}, [])                       # two readings: nothing can be judged
    # 9. token breaker is latched, tracks who spends, and rejects negatives
    bf = ByzantineSwarmFirewall(max_swarm_tokens=100)
    assert bf.record_and_check_budget(60, "w1") is False and bf.record_and_check_budget(50, "w2") is True
    assert bf.record_and_check_budget(0, "w1") is True and bf.heaviest_consumer() == "w1"
    for bad in (lambda: bf.record_and_check_budget(-1), lambda: ByzantineSwarmFirewall(z_threshold=0)):
        try:
            bad(); raise SystemExit("invalid input must raise")
        except ValueError:
            pass
    print("[PASS] Part 4 Security: 3-Color DFS Deadlock Arbiter & Byzantine Firewall Confirmed.")
