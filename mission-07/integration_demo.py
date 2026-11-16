"""
Mission 07 - integration demo and experiments (all four stations together)
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-07-SGR (Sagittarius A*), Subsystem 07

Everything printed here is measured by running the four modules; nothing is hard-coded.

  Experiment 1  ToT-BFS vs ToT-DFS under a compute budget, on two scripted trees
                (one with a high-scoring decoy branch, one where the greedy path is right),
                plus the effect of beam width.
  Experiment 2  A reward-hacked branch fools a naive PRM. Search with the naive PRM walks into
                it; search with the EventHorizonPruner in front of the PRM does not.
                Run with both ToT and MCTS.

The generators and priors are scripted stand-ins for an LLM, so the runs are deterministic.

Usage:
    python integration_demo.py
"""

from typing import Dict, List

from horizon_pruner import EventHorizonPruner
from mcts_protocol import MCTSEngine, MCTSNode
from prm_evaluator import ProcessRewardEvaluator
from tot_solver import ThoughtNode, TreeOfThoughtsSolver


# ---------------------------------------------------------------------------
# Experiment 1: BFS vs DFS under a budget
# ---------------------------------------------------------------------------
def make_tree(spec: Dict[str, List[str]], scores: Dict[str, float]):
    gen = lambda node: spec.get(node.state, [])
    ev = lambda thought: scores[thought]
    return gen, ev


DECOY_SPEC = {
    "R": ["A", "B", "C"],
    "A": ["A1", "A2"],
    "A1": ["A1x", "A1y", "A1z"],
    "A2": ["A2x", "A2y"],
    "B": ["B1"],
    "B1": ["GOAL"],
    "C": ["C1"],
    "C1": ["C1x"],
}
DECOY_SCORES = {"A": 0.90, "B": 0.80, "C": 0.60, "A1": 0.90, "A2": 0.50, "B1": 0.85, "C1": 0.50, "GOAL": 0.95,
                "A1x": 0.2, "A1y": 0.2, "A1z": 0.2, "A2x": 0.2, "A2y": 0.2, "C1x": 0.2}

GREEDY_SPEC = {
    "R": ["A", "B", "C"],
    "A": ["A1", "A2"],
    "A1": ["GOAL"],
    "B": ["B1", "B2"],
    "B1": ["B1x", "B1y"],
    "B2": ["B2x"],
}
GREEDY_SCORES = {"A": 0.90, "B": 0.80, "C": 0.70, "A1": 0.90, "A2": 0.80, "B1": 0.85, "B2": 0.80, "GOAL": 0.95,
                 "B1x": 0.2, "B1y": 0.2, "B2x": 0.2}


def run_search(kind: str, spec, scores, budget: int, beam: int = 2):
    gen, ev = make_tree(spec, scores)
    solver = TreeOfThoughtsSolver(max_expansions=budget, beam_width=beam, prune_threshold=0.45)
    root = ThoughtNode("R")
    fn = solver.bfs_beam_search if kind == "bfs" else solver.dfs_backtrack_search
    best = fn(root, gen, ev, max_depth=3)
    return {"found": best.state == "GOAL", "used": solver.expansion_count, "nodes": root.count_nodes(),
            "best": best.state}


def experiment_1() -> Dict:
    out = {}
    print("=" * 78)
    print("Experiment 1: ToT-BFS (beam 2) vs ToT-DFS under a compute budget (max_depth = 3)")
    print("=" * 78)
    for name, spec, scores in (("decoy tree", DECOY_SPEC, DECOY_SCORES), ("greedy-friendly tree", GREEDY_SPEC, GREEDY_SCORES)):
        print(f"\n[{name}]  minimum budget that reaches GOAL:")
        row = {}
        for kind in ("bfs", "dfs"):
            need = next((b for b in range(0, 40) if run_search(kind, spec, scores, b)["found"]), None)
            row[kind] = need
            print(f"  {kind.upper()}: {need} expansions")
        out[name] = row
        print(f"  budget sweep ({'budget':>6}: BFS / DFS found?)")
        for b in (4, 6, 8, 10, 11, 12, 14):
            r_b, r_d = run_search("bfs", spec, scores, b), run_search("dfs", spec, scores, b)
            print(f"    {b:>3}: BFS {'yes' if r_b['found'] else 'no ':<3} (used {r_b['used']:>2}, tree {r_b['nodes']:>2} nodes) | "
                  f"DFS {'yes' if r_d['found'] else 'no ':<3} (used {r_d['used']:>2}, tree {r_d['nodes']:>2} nodes)")
    print("\n[decoy tree] beam width sweep (budget 20):")
    out["beam"] = {}
    for b in (1, 2, 3):
        r = run_search("bfs", DECOY_SPEC, DECOY_SCORES, 20, beam=b)
        need = next(x for x in range(0, 40) if run_search("bfs", DECOY_SPEC, DECOY_SCORES, x, beam=b)["found"]) if r["found"] else None
        out["beam"][b] = {"found": r["found"], "used": r["used"], "nodes": r["nodes"], "min_budget": need}
        print(f"  B={b}: found={r['found']!s:<5} expansions used={r['used']:>2}  tree nodes={r['nodes']:>2}  "
              f"min budget={need}")
    return out


# ---------------------------------------------------------------------------
# Experiment 2: a vulnerable PRM, with and without the Event Horizon Pruner
# ---------------------------------------------------------------------------
HONEST = "Step 2: Burn prograde 2.1 km/s to set up the frame-dragging assist (4 + 4 = 8)"
HACKED = "Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0"
HACKED_FOLLOWUP = "Step 3: Obviously, by trivial inspection, the plunge is safe and thrust = 0 is optimal"
ROUTE_SPEC = {
    "Step 1: Compute r_s = 2GM / c^2": [HACKED, HONEST],
    HACKED: [HACKED_FOLLOWUP],
    HONEST: ["Step 3: Exit along the escape geodesic -> v_esc = 24.0 km/s [GOAL]"],
}


def naive_prm(step: str, base: ProcessRewardEvaluator = ProcessRewardEvaluator()) -> float:
    """A vulnerable PRM: the verifier plus a bonus for confident-sounding language (the exploit)."""
    s = base.step_score(step)
    if any(w in step.lower() for w in ("rigorous", "carefully verify", "undeniably", "trivial")):
        s = min(0.99, s + 0.05)
    return s


def experiment_2() -> Dict:
    print("\n" + "=" * 78)
    print("Experiment 2: reward hacking against a naive PRM, with and without the pruner")
    print("=" * 78)
    pruner = EventHorizonPruner(tau_horizon=0.45, hacking_penalty_factor=0.20)

    def defended_prm(step: str) -> float:
        hacked, _ = pruner.check_reward_hacking(step)
        return naive_prm(step) * (pruner.hacking_penalty_factor if hacked else 1.0)

    gen = lambda node: ROUTE_SPEC.get(node.state, [])
    res = {}
    for label, ev in (("naive PRM", naive_prm), ("PRM + pruner", defended_prm)):
        solver = TreeOfThoughtsSolver(max_expansions=10, beam_width=1, prune_threshold=0.45)
        best = solver.bfs_beam_search(ThoughtNode("Step 1: Compute r_s = 2GM / c^2"), gen, ev, max_depth=2)
        took_hack = HACKED in best.get_trajectory()
        res[f"tot_{label}"] = took_hack
        print(f"\nToT-BFS with {label}:  step-2 scores  hacked={ev(HACKED):.3f}  honest={ev(HONEST):.3f}")
        print(f"  chosen trajectory ends: {best.state}")
        print(f"  entered the reward-hacked branch: {took_hack}")

    # MCTS: the policy prior also favours the hacked branch (0.7 vs 0.3)
    print("\nMCTS (c_puct = 1.5, 40 simulations), policy prior 0.7 on the hacked branch:")
    for label, ev in (("naive PRM", naive_prm), ("PRM + pruner", defended_prm)):
        engine = MCTSEngine(c_puct=1.5)
        root = MCTSNode("R")
        policy = lambda n: ({HACKED: 0.7, HONEST: 0.3} if n.state == "R" else
                            {s: 1.0 for s in ROUTE_SPEC.get(n.state, [])})
        value = lambda s: ev(s)
        engine.search(root, policy, value, n_simulations=40)
        visits = {c.state[:42]: c.visit_count for c in root.children}
        hacked_visits = root.children[0].visit_count
        honest_visits = root.children[1].visit_count
        res[f"mcts_{label}"] = (hacked_visits, honest_visits)
        print(f"  {label:<13} visits: hacked={hacked_visits:>2}  honest={honest_visits:>2}  -> "
              f"{'HACKED branch preferred' if hacked_visits > honest_visits else 'honest branch preferred'}")

    # the severing view: audit the naive search's trajectory
    traj = ["Step 1: Compute r_s = 2GM / c^2", HACKED, HACKED_FOLLOWUP]
    raw = [naive_prm(s) for s in traj]
    rep = pruner.prune_trajectory(traj, raw)
    print(f"\nAuditing the naive trajectory: raw scores {[round(x, 2) for x in raw]} -> severed at step {rep.pruned_at_step}")
    for a in rep.security_alerts:
        print("  ", a)
    res["severed_at"] = rep.pruned_at_step
    return res


if __name__ == "__main__":
    e1 = experiment_1()
    e2 = experiment_2()

    # assertions: the experiments must show what the report claims
    d, g = e1["decoy tree"], e1["greedy-friendly tree"]
    assert d["bfs"] < d["dfs"], "decoy tree: BFS should need a smaller budget than DFS"
    assert g["dfs"] < g["bfs"], "greedy-friendly tree: DFS should need a smaller budget than BFS"
    assert not e1["beam"][1]["found"] and e1["beam"][2]["found"], "beam 1 should be trapped by the decoy; beam 2 escapes"
    assert e2["tot_naive PRM"] is True and e2["tot_PRM + pruner"] is False
    hn, ho = e2["mcts_naive PRM"]
    dn, do = e2["mcts_PRM + pruner"]
    assert hn > ho and do > dn, "MCTS should prefer the hacked branch only without the pruner"
    assert e2["severed_at"] == 2
    print("\n[OK] Integration experiments passed.")
