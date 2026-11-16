"""
Mission 07 - Part 2: Communications & Protocol Officer
Four-phase Monte Carlo Tree Search (MCTS) with PUCT selection
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-07-SGR (Sagittarius A*), Subsystem 07

The four phases, repeated for a fixed number of simulations:
  1. Selection      walk down from the root, always taking the child with the highest PUCT score
  2. Expansion      attach children for candidate next thoughts, with LLM policy priors P(a|s)
  3. Simulation     value the new node in [0, 1] with a rollout or a PRM
  4. Backpropagation add that value to every ancestor and count the visit

    PUCT(s, a) = Q(s, a) + c_puct * P(a | s) * sqrt(N(s)) / (1 + N(s, a))
    Q(s, a)    = W(s, a) / N(s, a)  (0 when N = 0),   N(s) = max(1, N(parent))

Q exploits what has paid off; the second term explores branches that the policy likes but that
have few visits. A larger c_puct shifts weight toward the second term.

Usage:
    python mcts_protocol.py
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class MCTSNode:
    """A node in the Monte Carlo reasoning tree."""
    state: str
    prior_prob: float = 1.0
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        """Empirical mean action-value Q(s, a) = W(s, a) / N(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def trajectory(self) -> List[str]:
        path: List[str] = []
        curr: Optional[MCTSNode] = self
        while curr is not None:
            path.append(curr.state)
            curr = curr.parent
        return list(reversed(path))


class MCTSEngine:
    """The four MCTS phases for LLM reasoning, guided by policy priors and PUCT."""

    def __init__(self, c_puct: float = 1.5):
        if c_puct < 0:
            raise ValueError("c_puct must be >= 0")
        self.c_puct = c_puct

    # -- Phase 1: selection -------------------------------------------------------
    def puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """PUCT(s, a) = Q + c_puct * P * sqrt(max(1, N(s))) / (1 + N(s, a))."""
        parent_visits = max(1, parent.visit_count)
        exploration = self.c_puct * child.prior_prob * math.sqrt(parent_visits) / (1 + child.visit_count)
        return child.q_value + exploration

    def select_best_child(self, node: MCTSNode) -> MCTSNode:
        """Child with the highest PUCT score (ties go to the earliest child)."""
        if not node.children:
            raise ValueError("Cannot select a child from a leaf node with no children.")
        return max(node.children, key=lambda child: self.puct_score(node, child))

    def select_leaf(self, root: MCTSNode) -> MCTSNode:
        """Walk down by PUCT until a leaf (no children) or terminal node."""
        curr = root
        while curr.children and not curr.is_terminal:
            curr = self.select_best_child(curr)
        return curr

    # -- Phase 2: expansion -------------------------------------------------------
    def expand(self, node: MCTSNode, action_priors: Dict[str, float]) -> List[MCTSNode]:
        """Attach one child per (thought, prior) pair. Priors must lie in [0, 1]."""
        for state_str, prior_prob in action_priors.items():
            if not 0.0 <= prior_prob <= 1.0:
                raise ValueError(f"prior for {state_str!r} must be in [0, 1], got {prior_prob}")
            node.children.append(MCTSNode(state=state_str, prior_prob=prior_prob, parent=node))
        return node.children

    # -- Phase 3: simulation ------------------------------------------------------
    def simulate_rollout(self, node: MCTSNode, rollout_evaluator: Callable[[str], float]) -> float:
        """Value of `node` in [0, 1] from the rollout evaluator (clamped)."""
        reward = float(rollout_evaluator(node.state))
        return max(0.0, min(1.0, reward))

    # -- Phase 4: backpropagation -------------------------------------------------
    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Add one visit and `reward` to `node` and every ancestor up to the root."""
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += reward
            curr = curr.parent

    # -- full search --------------------------------------------------------------
    def search(
        self,
        root: MCTSNode,
        policy_fn: Callable[[MCTSNode], Dict[str, float]],
        rollout_evaluator: Callable[[str], float],
        n_simulations: int = 50,
        is_terminal_fn: Optional[Callable[[str], bool]] = None,
    ) -> MCTSNode:
        """
        Run `n_simulations` select -> expand -> simulate -> backpropagate iterations and return
        the most-visited root child (robust to a single lucky rollout).
        """
        for _ in range(n_simulations):
            leaf = self.select_leaf(root)                              # phase 1
            if is_terminal_fn is not None and is_terminal_fn(leaf.state):
                leaf.is_terminal = True
            if not leaf.is_terminal:
                priors = policy_fn(leaf)
                if priors:                                             # phase 2
                    self.expand(leaf, priors)
                    leaf = self.select_best_child(leaf)
                    if is_terminal_fn is not None and is_terminal_fn(leaf.state):
                        leaf.is_terminal = True
                else:
                    leaf.is_terminal = True                            # no continuations: dead end
            self.backpropagate(leaf, self.simulate_rollout(leaf, rollout_evaluator))   # phases 3-4
        if not root.children:
            return root
        return max(root.children, key=lambda c: (c.visit_count, c.q_value))

    def principal_variation(self, root: MCTSNode) -> List[MCTSNode]:
        """Most-visited path from the root."""
        path, curr = [], root
        while curr.children:
            curr = max(curr.children, key=lambda c: (c.visit_count, c.q_value))
            path.append(curr)
        return path


# ==============================================================================
# Verification harness for Part 2 (Communications & Protocol Officer)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 2): MCTS & PUCT Protocol Verification ===")

    engine = MCTSEngine(c_puct=1.5)
    root = MCTSNode(state="Root: Sagittarius A* Ergosphere Geodesic Entry", visit_count=16)

    branch_a = MCTSNode(state="Branch A: Prograde Frame-Dragging Assist", prior_prob=0.50, parent=root)
    branch_a.visit_count = 5
    branch_a.value_sum = 0.75 * 5  # Q(s, A) = 0.75

    branch_b = MCTSNode(state="Branch B: Retrograde Penrose Extraction", prior_prob=0.40, parent=root)
    branch_b.visit_count = 1
    branch_b.value_sum = 0.55 * 1  # Q(s, B) = 0.55

    root.children = [branch_a, branch_b]

    puct_a = engine.puct_score(root, branch_a)
    puct_b = engine.puct_score(root, branch_b)
    selected = engine.select_best_child(root)

    print(f"Branch A -> Q={branch_a.q_value:.2f}, N={branch_a.visit_count}, P={branch_a.prior_prob:.2f} | PUCT = {puct_a:.4f}")
    print(f"Branch B -> Q={branch_b.q_value:.2f}, N={branch_b.visit_count}, P={branch_b.prior_prob:.2f} | PUCT = {puct_b:.4f}")
    print(f"Selected Next Node for Expansion: {selected.state}")

    assert abs(puct_a - 1.25) < 1e-6, f"Expected PUCT(A)=1.25, got {puct_a}"
    assert abs(puct_b - 1.75) < 1e-6, f"Expected PUCT(B)=1.75, got {puct_b}"
    assert selected is branch_b, "MCTS failed to select under-explored high-PUCT Branch B!"

    # Backpropagation
    engine.backpropagate(branch_b, reward=0.95)
    assert branch_b.visit_count == 2 and root.visit_count == 17, "Backpropagation failed to increment visit counts!"
    assert abs(branch_b.value_sum - 1.50) < 1e-9 and abs(root.value_sum - 0.95) < 1e-9

    # Expansion, simulation clamping, leaf selection
    fresh = MCTSNode(state="r")
    kids = engine.expand(fresh, {"x": 0.7, "y": 0.3})
    assert [k.state for k in kids] == ["x", "y"] and all(k.parent is fresh for k in kids)
    assert engine.simulate_rollout(kids[0], lambda s: 1.7) == 1.0
    assert engine.simulate_rollout(kids[0], lambda s: -0.2) == 0.0
    assert engine.select_leaf(fresh) in kids

    # c_puct sweep on the same statistics: where does the choice flip?
    print("\nc_puct sweep (same tree): selected branch")
    for c in (0.25, 0.5, 1.5, 2.5):
        e = MCTSEngine(c_puct=c)
        root.visit_count = 16
        branch_a.visit_count, branch_a.value_sum = 5, 3.75
        branch_b.visit_count, branch_b.value_sum = 1, 0.55
        print(f"  c_puct={c:<4}  PUCT(A)={e.puct_score(root, branch_a):.4f}  "
              f"PUCT(B)={e.puct_score(root, branch_b):.4f}  -> {e.select_best_child(root).state[:8]}")

    # full search on a toy tree: the high-prior but poor branch must lose to the better one
    TREE = {"R": {"good": 0.30, "decoy": 0.70}, "good": {"good-goal": 1.0}, "decoy": {"decoy-end": 1.0}}
    VALUE = {"R": 0.0, "good": 0.6, "decoy": 0.4, "good-goal": 0.95, "decoy-end": 0.10}
    search_engine = MCTSEngine(c_puct=1.5)
    search_root = MCTSNode(state="R")
    best = search_engine.search(search_root, lambda n: TREE.get(n.state, {}), lambda s: VALUE[s], n_simulations=60)
    print(f"\nSearch (60 simulations) most-visited first move: {best.state} (N={best.visit_count}, Q={best.q_value:.2f})")
    assert best.state == "good", "search should prefer the branch with the higher value over the higher prior"
    print("[OK] Part 2 (Communications & Protocol Officer) verification passed.")
