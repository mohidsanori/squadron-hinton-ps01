"""
Mission 07 - Part 3: Data & Payload Specialist
Process Reward Model (PRM) step evaluator, ORM comparison and Math-Shepherd supervision
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-07-SGR (Sagittarius A*), Subsystem 07

An Outcome Reward Model (ORM) grades only the final answer, so a proof whose errors cancel
(two sign flips) scores 1.0. A Process Reward Model scores every step z_t and aggregates:

    min             min_t r_t              weakest link
    prod            prod_t r_t             joint validity; shrinks with length
    geometric_mean  (prod_t r_t)^(1/T)     length-normalised quality

The PRM here is a transparent rule-based verifier, not a trained network. A step is scored by
  1. explicit flags the generator emits ([FLAWED], SIGN_ERROR, [MINOR_DRIFT]), and
  2. an arithmetic check: every "a op b = c" in the step is recomputed, so an unflagged
     "2 + 2 = 5" is still caught.

Math-Shepherd (Wang et al., 2024) labels a step without human annotation by sampling M
completions from it and checking how many reach the ground-truth answer y*:
    soft = (1/M) * sum_j 1[ans_j = y*]        hard = 1[soft > 0]

Usage:
    python prm_evaluator.py
"""

import math
import re
from typing import Dict, List

SCORE_VALID = 0.94
SCORE_DRIFT = 0.55
SCORE_FLAWED = 0.12

# a op b = c  (ops: + - * x / ; numbers may be signed or decimal; c may be followed by units)
_EQUATION = re.compile(
    r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*([+\-*/x×])\s*([+-]?\d+(?:\.\d+)?)\s*=\s*([+-]?\d+(?:\.\d+)?)"
)


def arithmetic_errors(step: str) -> List[str]:
    """Every 'a op b = c' in `step` whose stated result is wrong (|c - a op b| > 1e-6)."""
    errors = []
    for a, op, b, c in _EQUATION.findall(step):
        a, b, c = float(a), float(b), float(c)
        if op == "+":
            true = a + b
        elif op == "-":
            true = a - b
        elif op in ("*", "x", "×"):
            true = a * b
        else:
            if b == 0:
                errors.append(f"{a:g} / 0 = {c:g} (division by zero)")
                continue
            true = a / b
        if abs(true - c) > 1e-6:
            errors.append(f"{a:g} {op} {b:g} = {c:g} (should be {true:g})")
    return errors


class OutcomeRewardEvaluator:
    """ORM: grades a chain only by its final step. Fooled when intermediate errors cancel."""

    def __init__(self, expected_final_marker: str):
        self.expected_final_marker = expected_final_marker

    def score_trajectory(self, steps: List[str]) -> float:
        if not steps:
            return 0.0
        return 1.0 if self.expected_final_marker in steps[-1] else 0.0


class ProcessRewardEvaluator:
    """PRM: scores every intermediate step, aggregates, and computes Math-Shepherd labels."""

    REDUCTIONS = ("min", "prod", "geometric_mean")

    def step_score(self, step: str) -> float:
        """Score one reasoning step in (0, 1]."""
        if "[FLAWED]" in step or "SIGN_ERROR" in step:
            return SCORE_FLAWED
        if arithmetic_errors(step):
            return SCORE_FLAWED
        if "[MINOR_DRIFT]" in step:
            return SCORE_DRIFT
        return SCORE_VALID

    def score_all_steps(self, steps: List[str]) -> List[float]:
        return [self.step_score(s) for s in steps]

    def aggregate_trajectory_min(self, steps: List[str]) -> float:
        """Weakest link: min_t r_t."""
        return min(self.score_all_steps(steps)) if steps else 0.0

    def aggregate_trajectory_prod(self, steps: List[str]) -> float:
        """Joint validity: prod_t r_t."""
        if not steps:
            return 0.0
        return math.prod(self.score_all_steps(steps))

    def aggregate_trajectory_geometric_mean(self, steps: List[str]) -> float:
        """Length-normalised: (prod_t r_t)^(1/T), computed in log space."""
        if not steps:
            return 0.0
        scores = self.score_all_steps(steps)
        return math.exp(sum(math.log(r) for r in scores) / len(scores))

    def aggregate(self, steps: List[str], reduction: str = "min") -> float:
        if reduction not in self.REDUCTIONS:
            raise ValueError(f"reduction must be one of {self.REDUCTIONS}")
        return {"min": self.aggregate_trajectory_min,
                "prod": self.aggregate_trajectory_prod,
                "geometric_mean": self.aggregate_trajectory_geometric_mean}[reduction](steps)

    def math_shepherd_estimation(
        self,
        intermediate_step: str,
        rollout_final_answers: List[str],
        ground_truth_answer: str,
    ) -> Dict[str, float]:
        """Soft and hard Math-Shepherd labels for `intermediate_step` from M sampled completions."""
        if not rollout_final_answers:
            return {"soft_label": 0.0, "hard_label": 0.0}
        truth = ground_truth_answer.strip()
        matches = sum(1 for ans in rollout_final_answers if ans.strip() == truth)
        soft_label = matches / len(rollout_final_answers)
        return {"soft_label": soft_label, "hard_label": 1.0 if soft_label > 0 else 0.0}


# ==============================================================================
# Verification harness for Part 3 (Data & Payload Specialist)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 3): PRM vs. ORM & Math-Shepherd Verification ===")

    orm = OutcomeRewardEvaluator(expected_final_marker="v_esc = 24.0 km/s")
    prm = ProcessRewardEvaluator()

    # Trajectory 1: rigorous Kerr geodesic proof
    valid_chain = [
        "Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
        "Step 2: Integrate conserved orbital energy E and angular momentum L_z",
        "Step 3: Solve radial potential boundary -> v_esc = 24.0 km/s",
    ]

    # Trajectory 2: false-positive trap (double sign flip in Step 2 lands on the right scalar)
    false_positive_chain = [
        "Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
        "Step 2: [FLAWED] Flip sign of frame-dragging term twice (-1 * -1 = +1) via illegal tensor contraction",
        "Step 3: Accidental cancellation yields -> v_esc = 24.0 km/s",
    ]

    for label, chain in [("Rigorous Chain", valid_chain), ("False-Positive Trap", false_positive_chain)]:
        orm_score = orm.score_trajectory(chain)
        prm_min = prm.aggregate_trajectory_min(chain)
        prm_prod = prm.aggregate_trajectory_prod(chain)
        prm_geo = prm.aggregate_trajectory_geometric_mean(chain)
        print(
            f"{label:20s} | ORM={orm_score:.2f} | PRM(min)={prm_min:.3f} | "
            f"PRM(prod)={prm_prod:.3f} | PRM(geo)={prm_geo:.3f}"
        )

    assert orm.score_trajectory(false_positive_chain) == 1.0, "ORM should be fooled by the terminal answer!"
    assert prm.aggregate_trajectory_min(false_positive_chain) < 0.20, "PRM(min) must catch the flawed intermediate step!"

    # Math-Shepherd Monte Carlo step estimation
    shepherd = prm.math_shepherd_estimation(
        intermediate_step=valid_chain[1],
        rollout_final_answers=["24.0 km/s", "24.0 km/s", "22.5 km/s", "24.0 km/s"],
        ground_truth_answer="24.0 km/s",
    )
    print(f"Math-Shepherd Step Estimation (M=4 rollouts): {shepherd}")
    assert abs(shepherd["soft_label"] - 0.75) < 1e-6 and shepherd["hard_label"] == 1.0

    # Extra checks -----------------------------------------------------------------
    # 1. the arithmetic verifier catches an UNFLAGGED error that a marker-only PRM would miss
    unflagged = ["Step 1: 13 - 9 = 4", "Step 2: 4 + 4 = 9", "Step 3: 4 * 6 = 24.0 km/s"]
    assert arithmetic_errors("Step 2: 4 + 4 = 9") == ["4 + 4 = 9 (should be 8)"]
    assert prm.score_all_steps(unflagged) == [SCORE_VALID, SCORE_FLAWED, SCORE_VALID]
    assert arithmetic_errors("10 - 4 = 6 (Remaining: [4, 6])") == []
    assert arithmetic_errors("(-1 * -1 = +1)") == []                       # the cancellation itself is valid arithmetic
    # 2. reductions: min <= geometric mean <= max, prod <= min, and prod shrinks as the chain grows
    scores = prm.score_all_steps(valid_chain + valid_chain)
    assert prm.aggregate_trajectory_prod(valid_chain + valid_chain) < prm.aggregate_trajectory_prod(valid_chain)
    assert abs(prm.aggregate_trajectory_geometric_mean(valid_chain + valid_chain) - SCORE_VALID) < 1e-9
    assert prm.aggregate_trajectory_prod(false_positive_chain) <= prm.aggregate_trajectory_min(false_positive_chain)
    assert prm.aggregate([], "min") == 0.0
    # 3. Math-Shepherd: a single lucky rollout flips the hard label but barely moves the soft label
    lucky = prm.math_shepherd_estimation(false_positive_chain[1],
                                         ["19.8 km/s", "24.0 km/s", "31.2 km/s", "18.4 km/s"], "24.0 km/s")
    assert lucky == {"soft_label": 0.25, "hard_label": 1.0}
    assert prm.math_shepherd_estimation("s", ["x", "y"], "z") == {"soft_label": 0.0, "hard_label": 0.0}
    assert prm.math_shepherd_estimation("s", [], "z") == {"soft_label": 0.0, "hard_label": 0.0}
    print(f"Unflagged arithmetic slip scored {SCORE_FLAWED} by the verifier; lucky-rollout trap: {lucky}")
    print("[OK] Part 3 (Data & Payload Specialist) verification passed.")
