"""
Mission 11 - integration demo and experiments (all four stations together)
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-11-M31 (Andromeda Galaxy Core), Subsystem 11

Everything printed is measured by running the four modules. The agents are scripted stand-ins (they do
arithmetic, not language), so the experiments test the coordination machinery, not any model.

  Experiment 1  State graph: lost updates, real concurrency against the Amdahl model, runaway loops
  Experiment 2  Handoff contracts: context compression, delegation stack, ping-pong
  Experiment 3  Blackboard and debate: OCC under threads, the Condorcet table, correlated clones
  Experiment 4  Deadlock and Byzantine defence: wait-for graphs, pre-emption, the MAD breakdown point

Usage:
    python integration_demo.py
"""

import random
import statistics
import threading
import time
from typing import Dict, List, Tuple

from blackboard_debate_consensus import AgentVote, DebateConsensusAggregator, VersionedBlackboard
from deadlock_byzantine_sentinel import ByzantineSwarmFirewall, SupervisorDeadlockArbiter, WaitForGraphMonitor
from handoff_contract_protocol import ContextScopingGate, HandoffEnvelope, HandoffMode
from swarm_state_graph_architect import AgentNode, ReducerMode, SwarmStateGraph, amdahl_speedup


# ---------------------------------------------------------------------------
def experiment_1() -> Dict:
    print("=" * 78 + "\nExperiment 1: state graph, reducers and the Amdahl model\n" + "=" * 78)
    out: Dict = {}

    # lost updates: five parallel workers each report one finding and spend 100 compute units
    def workers(g):
        for i in range(5):
            g.add_node(AgentNode(f"w{i}", (lambda i: lambda s: {"findings": f"finding-{i}", "units": 100})(i), 50))
    def build(reducers: bool):
        g = SwarmStateGraph()
        if reducers:
            g.set_reducer("findings", ReducerMode.APPEND_LIST); g.set_reducer("units", ReducerMode.SUM_NUMERIC)
        g.add_node(AgentNode("start", lambda s: {}, 10)); g.add_node(AgentNode("end", lambda s: {}, 10))
        workers(g); g.add_parallel_group("start", [f"w{i}" for i in range(5)], "end")
        return g
    naive, reduced = build(False).execute_graph("start"), build(True).execute_graph("start")
    kept_naive = 1 if isinstance(naive.final_state["findings"], str) else len(naive.final_state["findings"])
    out["lost"] = {"naive_findings": kept_naive, "naive_units": naive.final_state["units"], "reduced_findings": len(reduced.final_state["findings"]),
                   "reduced_units": reduced.final_state["units"], "conflicts": len(naive.conflicts)}
    print(f"  5 parallel workers, each reports 1 finding and 100 units:")
    print(f"    plain overwrite channels : {kept_naive} of 5 findings kept, units = {naive.final_state['units']} (expected 500), "
          f"{len(naive.conflicts)} overwrite conflicts reported")
    print(f"    reducer channels         : {len(reduced.final_state['findings'])} of 5 findings kept, units = {reduced.final_state['units']}")

    # real concurrency against the model (latencies scaled to milliseconds of real sleep)
    def timed(concurrent: bool):
        g = SwarmStateGraph(concurrent=concurrent)
        lat = {"coord": 25, "a": 52, "b": 60, "c": 48, "synth": 28}
        for name, ms in lat.items():
            g.add_node(AgentNode(name, (lambda ms: lambda s: (time.sleep(ms / 1000), {})[1])(ms), ms))
        g.add_parallel_group("coord", ["a", "b", "c"], "synth")
        return g.execute_graph("coord")
    seq_run = min((timed(False) for _ in range(3)), key=lambda r: r.wall_clock_ms)
    par_run = min((timed(True) for _ in range(3)), key=lambda r: r.wall_clock_ms)
    modelled = amdahl_speedup(25 + 28, [52, 60, 48])
    measured = seq_run.wall_clock_ms / par_run.wall_clock_ms
    out["amdahl"] = {"modelled": modelled, "measured": measured, "seq_ms": seq_run.wall_clock_ms, "par_ms": par_run.wall_clock_ms}
    print(f"\n  Real threads, workers of 52 / 60 / 48 ms around 25 + 28 ms of coordination:")
    print(f"    sequential {seq_run.wall_clock_ms:.1f} ms | concurrent {par_run.wall_clock_ms:.1f} ms | measured speedup {measured:.2f}x "
          f"| model T_seq/T_crit = {modelled:.2f}x")

    # saturation: coordination cost grows with the number of workers (the coordinator must read every result)
    print("\n  Agentic Amdahl with coordination cost l_coord = 120 + 15K ms and workers of 250 ms:")
    print(f"    {'K':>4}{'T_seq':>9}{'T_crit':>9}{'speedup':>9}{'efficiency S/K':>16}")
    rows = {}
    for k in (1, 2, 4, 8, 16, 32, 64):
        lc = 120 + 15 * k
        s = amdahl_speedup(lc, [250] * k)
        rows[k] = s
        print(f"    {k:>4}{lc + 250 * k:>9}{lc + 250:>9}{s:>9.2f}{s / k:>16.2f}")
    out["sat"] = rows
    print(f"    limit as K grows: 1 + 250/15 = {1 + 250 / 15:.1f}x")

    # runaway loop
    g = SwarmStateGraph(max_steps=20); g.add_node(AgentNode("spin", lambda s: {}, 1)); g.add_conditional_edges("spin", lambda s: "spin")
    try:
        g.execute_graph("spin")
        out["runaway"] = None
    except RuntimeError as exc:
        out["runaway"] = str(exc).split(":")[0]
    print(f"\n  A conditional edge that always loops back: stopped with {out['runaway']} after 20 steps (max_steps = 20).")
    return out


# ---------------------------------------------------------------------------
def experiment_2() -> Dict:
    print("\n" + "=" * 78 + "\nExperiment 2: handoff contracts and the least-context principle\n" + "=" * 78)
    out: Dict = {"compress": {}}
    print(f"  {'chat turns in global state':>28}{'global chars':>14}{'scoped chars':>14}{'eta_comp':>10}")
    for turns in (4, 12, 50, 200):
        g = {"sector_id": "SEC-11-M31", "p1_ephemeris": {"ra": "00h42m44.3s", "dec": "+41d16m09s", "vel_kms": -301.0},
             "full_chat_history": [f"turn_{i}_" + "x" * 120 for i in range(turns)], "raw_sensor_dump": "NOISE_" * 200,
             "unrelated_crew_logs": ["galley inventory", "hydroponics cycle 44"]}
        gate = ContextScopingGate("Coordinator")
        gate.process_handoff(g, HandoffEnvelope("E", "Coordinator", "Astro", HandoffMode.SUBROUTINE_TOOL, ["sector_id", "p1_ephemeris"],
                                                {"objective": "clearance"}, ["trajectory_id"]))
        r = gate.audit_log[-1]
        glob, scoped = gate._serialized_length(g), gate._serialized_length(r.scoped_context)
        out["compress"][turns] = r.compression_ratio
        print(f"  {turns:>28}{glob:>14,}{scoped:>14,}{r.compression_ratio:>10.3f}")
    print("  The scoped context stays the same size, so the saving grows with the history that never has to be passed on.")

    # call-stack semantics: a realistic chain
    gate = ContextScopingGate("Coordinator")
    G = {"sector_id": "S", "noise": "n" * 300}
    mk = lambda s, r, m, i: HandoffEnvelope(f"E{i}", s, r, m, ["sector_id"], {}, [], chain_id="trace")
    steps = [("Coordinator", "Astrogation", HandoffMode.SUBROUTINE_TOOL), ("Astrogation", "TidalScanner", HandoffMode.SUBROUTINE_TOOL),
             ("TidalScanner", "SafetyCritic", HandoffMode.PEER_TRANSFER)]
    print(f"\n  {'handoff':<44}{'mode':<17}{'active agent':<15}{'return stack'}")
    for i, (s, r, m) in enumerate(steps):
        gate.process_handoff(G, mk(s, r, m, i))
        print(f"  {s + ' -> ' + r:<44}{m.name:<17}{gate.active_agent:<15}{gate.caller_stack}")
    out["stack_after"] = list(gate.caller_stack)
    for env_sender, label in (("TidalScanner", "TidalScanner"), ("Astrogation", "Astrogation")):
        gate.caller_stack = list(gate.caller_stack)
    unwound = []
    while gate.caller_stack:
        top = gate.caller_stack[-1]
        gate.complete_subroutine_return(HandoffEnvelope("R", top, gate.active_agent, HandoffMode.SUBROUTINE_TOOL, [], {}, []), {})
        unwound.append(gate.active_agent)
    out["unwound"] = unwound
    print(f"  returning unwinds the stack in order: {unwound}")

    # ping-pong
    def pingpong(max_hops: int, guard: bool) -> int:
        gate = ContextScopingGate("P")
        who, n = "P", 0
        for _ in range(50):                                  # an unguarded pair would simply keep going
            other = "Q" if who == "P" else "P"
            if guard:
                try:
                    gate.process_handoff(G, HandoffEnvelope("E", who, other, HandoffMode.PEER_TRANSFER, ["sector_id"], {}, [], hop_count=0, max_hops=max_hops, chain_id="pp"))
                except RuntimeError:
                    return n
            n, who = n + 1, other
        return n
    out["pingpong"] = {"guarded": pingpong(5, True), "unguarded": pingpong(5, False)}
    print(f"\n  Two agents that keep handing the request back and forth (every envelope claims hop 0): "
          f"{out['pingpong']['guarded']} transfers with the gate (max_hops = 5), {out['pingpong']['unguarded']} without it (stopped only by our 50-step cap).")
    return out


# ---------------------------------------------------------------------------
def experiment_3() -> Dict:
    print("\n" + "=" * 78 + "\nExperiment 3: blackboard concurrency and debate consensus\n" + "=" * 78)
    out: Dict = {}
    # OCC under threads
    def run(use_occ: bool):
        board, blind, gave_up = VersionedBlackboard(), {"n": 0}, []
        board.write_optimistic("count", 0, 0)
        def worker():
            for _ in range(100):
                if use_occ:
                    try:
                        board.update("count", lambda v: (time.sleep(0.0001), v + 1)[1])  # same read-modify-write window as the blind variant
                    except RuntimeError:
                        gave_up.append(1)                                                # loud failure, never a silent loss
                else:
                    v = blind["n"]; time.sleep(0.0001); blind["n"] = v + 1
        ts = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in ts]; [t.join() for t in ts]
        return (board.read("count")[0], sum(a["result"] == "CONCURRENCY_CONFLICT" for a in board.audit), len(gave_up)) if use_occ else (blind["n"], 0, 0)
    occ_total, conflicts, gave = run(True)
    blind_total, _, _ = run(False)
    out["occ"] = {"occ": occ_total, "blind": blind_total, "conflicts": conflicts, "gave_up": gave}
    print(f"  8 threads x 100 read-modify-write increments (expected 800):")
    print(f"    without OCC : {blind_total} kept ({800 - blind_total} lost updates)")
    print(f"    with OCC    : {occ_total} kept; {conflicts} writes were refused with CONCURRENCY_CONFLICT and retried; {gave} increments given up on")

    agg = DebateConsensusAggregator()
    print("\n  Condorcet majority accuracy P_maj(N, p) for independent voters:")
    ps, ns = (0.55, 0.70, 0.90), (1, 3, 5, 7, 9, 15, 25)
    print(f"    {'N':>4}" + "".join(f"{'p=' + str(p):>10}" for p in ps))
    table = {}
    for n in ns:
        table[n] = [agg.compute_condorcet_probability(n, p) for p in ps]
        print(f"    {n:>4}" + "".join(f"{v:>10.4f}" for v in table[n]))
    out["condorcet"] = table

    print("\n  The same jury of N = 7 voters with p = 0.70 when they are correlated (common-shock model, Monte Carlo check):")
    print(f"    {'rho':>5}{'N_eff':>8}{'P_maj formula':>15}{'Monte Carlo':>13}")
    rng = random.Random(3)
    corr = {}
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        wins = 0
        for _ in range(30000):
            if rng.random() < rho:
                wins += rng.random() < 0.7
            else:
                wins += sum(rng.random() < 0.7 for _ in range(7)) >= 4
        mc = wins / 30000
        corr[rho] = (agg.compute_correlated_majority(7, 0.7, rho), mc, agg.effective_jury_size(7, rho))
        print(f"    {rho:>5.2f}{corr[rho][2]:>8.2f}{corr[rho][0]:>15.4f}{mc:>13.4f}")
    out["corr"] = corr

    # specialist plus two correlated clones, 20,000 simulated questions
    rng = random.Random(8)
    trials = 20000
    right = {"specialist": 0, "unweighted": 0, "weighted": 0}
    err = [[], [], []]
    for t in range(trials):
        spec = rng.random() < 0.92
        a = rng.random() < 0.55
        b = a if rng.random() < 0.85 else rng.random() < 0.55          # clone B mostly copies clone A
        votes = [AgentVote("spec", +1 if spec else -1, 0.92, 0.95), AgentVote("A", +1 if a else -1, 0.55, 0.6), AgentVote("B", +1 if b else -1, 0.55, 0.6)]
        d = agg.aggregate_quorum(votes)
        right["specialist"] += spec
        right["unweighted"] += d.unweighted_verdict == +1
        right["weighted"] += d.weighted_verdict == +1
        if t < 400:
            err[0].append(int(not spec)); err[1].append(int(not a)); err[2].append(int(not b))
    acc = {k: v / trials for k, v in right.items()}
    rho_clones = agg.estimate_error_correlation([err[1], err[2]])
    rho_all = agg.estimate_error_correlation(err)
    out["panel"] = {"acc": acc, "rho_clones": rho_clones, "rho_all": rho_all}
    print(f"\n  One calibrated specialist (p = 0.92) and two correlated clones (p = 0.55), {trials:,} simulated questions:")
    print(f"    specialist alone       {acc['specialist']:.4f}")
    print(f"    unweighted majority    {acc['unweighted']:.4f}")
    print(f"    log-odds weighted      {acc['weighted']:.4f}")
    print(f"    error correlation estimated from the first 400 questions: clone pair {rho_clones:.2f}, all three agents {rho_all:.2f}; "
          f"echo alarm for the clone pair: {agg.detect_echo_chamber([], rho_clones)[0]}")
    return out


# ---------------------------------------------------------------------------
def experiment_4() -> Dict:
    print("\n" + "=" * 78 + "\nExperiment 4: deadlock detection and the Byzantine firewall\n" + "=" * 78)
    out: Dict = {}

    def simulate(edges: List[Tuple[str, str, int]], preempt: bool, cap: int = 30) -> Tuple[List[str], int, List[str]]:
        """Agents finish when everything they wait for has finished. Returns (finish order, rounds used, pre-empted edges)."""
        mon = WaitForGraphMonitor()
        for u, v, p in edges:
            mon.add_wait_edge(u, v, p)
        preempted = []
        if preempt:
            for r in SupervisorDeadlockArbiter.resolve_until_clear(mon):
                preempted.append(f"{r.preempted_edge.waiter}->{r.preempted_edge.holder}")
        done: List[str] = []
        rounds = 0
        while len(done) < len(mon.adj) and rounds < cap:
            rounds += 1
            ready = sorted(n for n in mon.adj if n not in done and all(e.holder in done for e in mon.adj[n]))
            done.extend(ready)
            if not ready:
                continue
        return done, rounds, preempted
    anomaly = [("Astrogation", "ProtocolValidator", 9), ("ProtocolValidator", "PayloadBuilder", 7), ("PayloadBuilder", "Astrogation", 2)]
    d0, r0, _ = simulate(anomaly, False)
    d1, r1, pre = simulate(anomaly, True)
    out["anomaly"] = {"frozen_done": len(d0), "frozen_rounds": r0, "freed_done": len(d1), "preempted": pre, "order": d1}
    print(f"  Ship Anomaly 11 (Astrogation -> ProtocolValidator -> PayloadBuilder -> Astrogation):")
    print(f"    no monitor : {len(d0)} of 3 agents finished after {r0} rounds (frozen; cut off by our 30-round cap)")
    print(f"    with monitor: pre-empted {pre}; all {len(d1)} finished in order {d1}")

    # accuracy and scaling of the DFS
    rng = random.Random(5)
    fp = tp = 0
    for _ in range(300):
        n = rng.randint(5, 40)
        g = WaitForGraphMonitor()
        for i in range(n - 1):
            g.add_wait_edge(f"n{i:02d}", f"n{i + 1:02d}")
        for _ in range(rng.randint(n, 3 * n)):
            a, b = sorted(rng.sample(range(n), 2)); g.add_wait_edge(f"n{a:02d}", f"n{b:02d}")
        fp += bool(g.detect_cycle())
        g.add_wait_edge(f"n{n - 1:02d}", "n00"); tp += bool(g.detect_cycle())
    print(f"\n  300 random wait-for DAGs full of diamonds: {fp} false alarms; the same graphs with one back edge added: {tp} of 300 detected")
    times = {}
    for n in (25000, 100000, 200000):
        g = WaitForGraphMonitor()
        for i in range(n - 1):
            g.add_wait_edge(f"v{i:07d}", f"v{i + 1:07d}")
        t0 = time.perf_counter(); g.detect_cycle(); times[n] = (time.perf_counter() - t0) * 1000
    print(f"  Detection time on acyclic chains of 25k / 100k / 200k vertices: {times[25000]:.0f} / {times[100000]:.0f} / {times[200000]:.0f} ms "
          f"(vertex ratio 8x, time ratio {times[200000] / times[25000]:.1f}x)")
    out.update(fp=fp, tp=tp, times=times)

    # several independent deadlocks
    mon = WaitForGraphMonitor()
    for ring in (("a", "b", "c"), ("d", "e"), ("f", "g", "h", "i")):
        for u, v in zip(ring, ring[1:] + ring[:1]):
            mon.add_wait_edge(u, v, 5)
    rounds = SupervisorDeadlockArbiter.resolve_until_clear(mon)
    out["rings"] = len(rounds)
    print(f"  Three independent circular waits (3, 2 and 4 agents): cleared in {len(rounds)} pre-emptions, graph acyclic: {mon.detect_cycle() == []}")

    # Byzantine breakdown
    print("\n  Byzantine telemetry: 11 probes read about -301 km/s; r of them collude on 999 km/s")
    print(f"    {'r':>3}{'rogues caught':>15}{'honest wrongly quarantined':>28}{'consensus (median of trusted)':>32}{'classical z flags rogue?':>26}")
    bz = {}
    for r in (1, 3, 5, 6):
        readings = {f"h{i}": -301.0 + 0.1 * (i % 5) for i in range(11 - r)}
        readings.update({f"r{i}": 999.0 + 0.05 * i for i in range(r)})
        trusted, q = ByzantineSwarmFirewall().filter_byzantine_telemetry(readings)
        vals = list(readings.values()); mean, sd = statistics.mean(vals), statistics.pstdev(vals)
        classical = sum(1 for k, v in readings.items() if k.startswith("r") and abs(v - mean) / sd > 3.5)
        caught, harmed = sum(k.startswith("r") for k in q), sum(k.startswith("h") for k in q)
        cons = statistics.median(trusted.values())
        bz[r] = (caught, harmed, cons, classical)
        print(f"    {r:>3}{caught:>11}/{r:<3}{harmed:>24}/{11 - r:<3}{cons:>30.1f}{classical:>22}/{r}")
    out["byz"] = bz

    # token breaker
    fw = ByzantineSwarmFirewall(max_swarm_tokens=16000)
    spend = {"Astrogation": 900, "Protocol": 700, "Payload": 800, "Safety": 650, "Rogue": 2600}
    rounds_to_trip = 0
    while not fw.tripped and rounds_to_trip < 50:
        rounds_to_trip += 1
        for a, t in spend.items():
            fw.record_and_check_budget(t, a)
    out["tokens"] = (rounds_to_trip, fw.cumulative_tokens, fw.heaviest_consumer())
    print(f"\n  Swarm token budget 16,000: tripped in round {rounds_to_trip} at {fw.cumulative_tokens:,} tokens; heaviest consumer: {fw.heaviest_consumer()}")
    return out


if __name__ == "__main__":
    e1 = experiment_1()
    e2 = experiment_2()
    e3 = experiment_3()
    e4 = experiment_4()

    # the experiments must show what the report claims
    L = e1["lost"]
    assert L["naive_findings"] == 1 and L["naive_units"] == 100 and L["conflicts"] >= 1 and L["reduced_findings"] == 5 and L["reduced_units"] == 500
    assert e1["amdahl"]["measured"] > 1.3 and abs(e1["amdahl"]["modelled"] - 213 / 113) < 0.01
    assert all(b >= a for a, b in zip(list(e1["sat"].values()), list(e1["sat"].values())[1:])) and e1["sat"][64] < 1 + 250 / 15 and e1["runaway"] == "MAX_STEPS_EXCEEDED"
    c = e2["compress"]
    assert c[4] < c[12] < c[50] < c[200] and c[200] > 0.98 and e2["stack_after"] == ["Coordinator", "Astrogation"] and e2["unwound"] == ["Astrogation", "Coordinator"]
    assert e2["pingpong"] == {"guarded": 5, "unguarded": 50}
    assert e3["occ"]["occ"] + e3["occ"]["gave_up"] == 800 and e3["occ"]["gave_up"] == 0 and e3["occ"]["blind"] < 800
    for n, vals in e3["condorcet"].items():
        assert abs(vals[1] - 0.70) < 1e-9 or n > 1
    assert abs(e3["condorcet"][7][1] - 0.873964) < 1e-6 and e3["condorcet"][25][0] > e3["condorcet"][3][0] > 0.55
    for rho, (f, mc, neff) in e3["corr"].items():
        assert abs(f - mc) < 0.015
    assert e3["corr"][1.0][0] == 0.7 and e3["corr"][0.5][0] < e3["corr"][0.0][0]
    pa = e3["panel"]
    assert pa["acc"]["weighted"] > pa["acc"]["unweighted"] + 0.05 and pa["acc"]["weighted"] > 0.9 and pa["rho_clones"] > 0.5
    an = e4["anomaly"]
    assert an["frozen_done"] == 0 and an["freed_done"] == 3 and an["preempted"] == ["PayloadBuilder->Astrogation"]
    assert e4["fp"] == 0 and e4["tp"] == 300 and e4["rings"] == 3 and e4["times"][200000] / e4["times"][25000] < 16
    bz = e4["byz"]
    assert bz[1][:2] == (1, 0) and bz[5][:2] == (5, 0) and bz[6][0] == 0 and bz[6][2] > 900 and all(v[3] == 0 for v in bz.values())
    assert e4["tokens"][2] == "Rogue" and e4["tokens"][1] > 16000
    print("\n[OK] Integration experiments passed.")
