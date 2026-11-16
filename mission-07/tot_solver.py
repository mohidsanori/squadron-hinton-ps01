"""
Mission 07 - Part 1: Systems Architect
Tree-of-Thoughts (ToT) breadth-first beam search and depth-first backtracking solver
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-07-SGR (Sagittarius A*), Subsystem 07

A single left-to-right decode commits to every token; if each step fails with probability eps the
chance of at least one error over T steps is 1 - (1 - eps)^T. Tree search spends extra inference
compute on lookahead and backtracking instead. Both searches share the same budget rule: at most
`max_expansions` calls to the evaluator (one per generated candidate thought).

  * ToT-BFS keeps the best `beam_width` nodes per depth. Memory and token cost grow with B.
  * ToT-DFS follows the best child first and backtracks from dead ends. It is cheap per path but
    can burn the whole budget inside one wrong subtree.

A node's `cumulative_score` is the minimum step score along its path (weakest link), so one bad
step cannot be averaged away.

Usage:
    python tot_solver.py
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class ThoughtNode:
    """A discrete semantic reasoning step z_t in a Tree-of-Thoughts."""
    state: str
    depth: int = 0
    step_score: float = 1.0
    cumulative_score: float = 1.0
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = field(default_factory=list)
    is_pruned: bool = False

    def add_child(self, child_state: str, step_score: float) -> "ThoughtNode":
        """Create and attach a child; its cumulative score is the minimum link along the path."""
        child = ThoughtNode(
            state=child_state,
            depth=self.depth + 1,
            step_score=step_score,
            cumulative_score=min(self.cumulative_score, step_score),
            parent=self,
        )
        self.children.append(child)
        return child

    def get_trajectory(self) -> List[str]:
        """The chain of thoughts [z_0, z_1, ..., z_t] from the root to this node."""
        path: List[str] = []
        curr: Optional[ThoughtNode] = self
        while curr is not None:
            path.append(curr.state)
            curr = curr.parent
        return list(reversed(path))

    def count_nodes(self) -> int:
        """Size of the subtree rooted here (this node included)."""
        return 1 + sum(c.count_nodes() for c in self.children)


def _better(candidate: ThoughtNode, incumbent: ThoughtNode) -> bool:
    """Deeper wins; at equal depth the higher weakest-link score wins."""
    return candidate.depth > incumbent.depth or (
        candidate.depth == incumbent.depth and candidate.cumulative_score > incumbent.cumulative_score
    )


class TreeOfThoughtsSolver:
    """Deliberate System 2 search under a hard inference compute budget (`max_expansions`)."""

    def __init__(self, max_expansions: int = 16, beam_width: int = 2, prune_threshold: float = 0.45):
        if max_expansions < 0:
            raise ValueError("max_expansions must be >= 0")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if not 0.0 <= prune_threshold <= 1.0:
            raise ValueError("prune_threshold must be in [0, 1]")
        self.max_expansions = max_expansions
        self.beam_width = beam_width
        self.prune_threshold = prune_threshold
        self.expansion_count = 0

    @property
    def budget_exhausted(self) -> bool:
        return self.expansion_count >= self.max_expansions

    def _expand(self, node: ThoughtNode, generator_fn, evaluator_fn) -> List[ThoughtNode]:
        """Generate and score children within the budget. Returns the children that survive pruning."""
        survivors: List[ThoughtNode] = []
        for thought in generator_fn(node):
            if self.budget_exhausted:
                break
            self.expansion_count += 1
            child = node.add_child(thought, float(evaluator_fn(thought)))
            if child.step_score < self.prune_threshold:
                child.is_pruned = True
            else:
                survivors.append(child)
        return survivors

    def bfs_beam_search(
        self,
        root: ThoughtNode,
        generator_fn: Callable[[ThoughtNode], List[str]],
        evaluator_fn: Callable[[str], float],
        max_depth: int = 3,
    ) -> ThoughtNode:
        """ToT-BFS: breadth-first search keeping the top `beam_width` nodes at each depth."""
        self.expansion_count = 0
        beam: List[ThoughtNode] = [root]
        best_node = root

        for _ in range(max_depth):
            candidates: List[ThoughtNode] = []
            for node in beam:
                if self.budget_exhausted:
                    break
                candidates.extend(self._expand(node, generator_fn, evaluator_fn))
            if not candidates:
                break
            candidates.sort(key=lambda c: (c.cumulative_score, c.step_score), reverse=True)
            beam = candidates[: self.beam_width]
            if _better(beam[0], best_node):
                best_node = beam[0]
        return best_node

    def dfs_backtrack_search(
        self,
        root: ThoughtNode,
        generator_fn: Callable[[ThoughtNode], List[str]],
        evaluator_fn: Callable[[str], float],
        max_depth: int = 3,
    ) -> ThoughtNode:
        """ToT-DFS: follow the best child first; backtrack at dead ends and below-threshold steps."""
        self.expansion_count = 0
        best_node = root

        def _dfs(node: ThoughtNode) -> None:
            nonlocal best_node
            if _better(node, best_node):
                best_node = node
            if node.depth >= max_depth or self.budget_exhausted:
                return
            valid = self._expand(node, generator_fn, evaluator_fn)
            valid.sort(key=lambda c: c.step_score, reverse=True)
            for child in valid:                      # returning from a dead end = backtracking
                _dfs(child)

        _dfs(root)
        return best_node


# ==============================================================================
# Verification harness for Part 1 (Systems Architect)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 1): Tree-of-Thoughts Solver Verification ===")

    # Mock Sagittarius A* 24-vector thruster synthesis tree
    MOCK_BRANCHES = {
        "Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]": [
            "Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID]",
            "Step 1B: 10 + 13 = 23 (Remaining: [4, 9, 23]) [DEAD-END]",
        ],
        "Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID]": [
            "Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL]",
            "Step 2B: 4 + 4 = 8 (Remaining: [8, 10]) [DEAD-END]",
        ],
        "Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL]": [
            "Step 3A: 4 * 6 = 24.0 km/s -> Escape Geodesic Locked! [GOAL]",
        ],
    }

    def mock_generator(node: ThoughtNode) -> List[str]:
        return MOCK_BRANCHES.get(node.state, [])

    def mock_evaluator(thought: str) -> float:
        if "[GOAL]" in thought:
            return 0.98
        if "[OPTIMAL]" in thought:
            return 0.92
        if "[VALID]" in thought:
            return 0.85
        return 0.25  # dead-end branch below prune_threshold

    solver = TreeOfThoughtsSolver(max_expansions=12, beam_width=2, prune_threshold=0.45)

    root_bfs = ThoughtNode("Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]")
    best_bfs = solver.bfs_beam_search(root_bfs, mock_generator, mock_evaluator, max_depth=3)
    bfs_used = solver.expansion_count
    print(f"[ToT-BFS] Expansions Used: {bfs_used}/{solver.max_expansions}")
    print(f"[ToT-BFS] Best Terminal Score: {best_bfs.cumulative_score:.2f}")
    print(f"[ToT-BFS] Winning Trajectory: {' -> '.join(best_bfs.get_trajectory())}")

    root_dfs = ThoughtNode("Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]")
    best_dfs = solver.dfs_backtrack_search(root_dfs, mock_generator, mock_evaluator, max_depth=3)
    dfs_used = solver.expansion_count
    print(f"[ToT-DFS] Expansions Used: {dfs_used}/{solver.max_expansions}")
    print(f"[ToT-DFS] Winning Trajectory: {' -> '.join(best_dfs.get_trajectory())}")

    assert "[GOAL]" in best_bfs.state, "ToT-BFS failed to reach [GOAL] state!"
    assert "[GOAL]" in best_dfs.state, "ToT-DFS failed to reach [GOAL] state!"

    # extra checks: the budget is a hard cap, pruned nodes are marked, the beam is respected
    for budget in (0, 1, 3, 5):
        s = TreeOfThoughtsSolver(max_expansions=budget, beam_width=2)
        s.bfs_beam_search(ThoughtNode("Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]"), mock_generator, mock_evaluator)
        assert s.expansion_count <= budget
        s.dfs_backtrack_search(ThoughtNode("Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]"), mock_generator, mock_evaluator)
        assert s.expansion_count <= budget
    assert any(c.is_pruned for c in root_bfs.children), "dead-end child should be marked pruned"
    assert all(len(n.children) <= 2 for n in [root_bfs])
    print("[OK] Part 1 (Systems Architect) verification passed.")
