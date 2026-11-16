"""
Mission 07 - Part 4: Security & Ethics Sentinel
Event Horizon pruner and adversarial PRM reward-hacking defence (Ship Anomaly 07)
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-07-SGR (Sagittarius A*), Subsystem 07

A PRM is itself a model, so a searcher that maximises it will find inputs that fool it. Two
attacks are covered:

  1. Sycophantic filler injection. A step is padded with confident, self-congratulatory language
     ("let us carefully verify this 100% rigorous step ...") that raises a naive PRM's score
     without advancing the maths. Defence: phrase list matched on normalised text.
  2. Step-splitting / echo inflation. A bad step is diluted under mean aggregation by surrounding
     it with redundant micro-steps that repeat earlier equations. Defence: a step whose normalised
     text (lower-case, "Step N:" prefix and whitespace removed) equals any earlier step is redundant.

Penalty and severing. A flagged step's score is multiplied by `hacking_penalty_factor`. When the
adjusted score falls below `tau_horizon` the trajectory is cut at that step and the remaining
steps are never evaluated, which also saves their test-time compute.

Usage:
    python horizon_pruner.py
"""

import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple


@dataclass
class PruningReport:
    """Telemetry record produced by EventHorizonPruner."""
    surviving_steps: List[str]
    adjusted_scores: List[float]
    pruned_at_step: int = -1
    reward_hacking_detected: bool = False
    security_alerts: List[str] = field(default_factory=list)


_STEP_PREFIX = re.compile(r"^\s*step\s*\d+[a-z]?\s*[:.)-]\s*", re.IGNORECASE)


def normalize_step(step: str) -> str:
    """Lower-case, drop the 'Step N:' label and collapse whitespace, so re-labelled echoes still match."""
    return re.sub(r"\s+", " ", _STEP_PREFIX.sub("", step.strip().lower())).strip()


class EventHorizonPruner:
    """Severs trajectories whose verified score crosses the horizon and penalises reward hacking."""

    def __init__(self, tau_horizon: float = 0.45, hacking_penalty_factor: float = 0.20):
        if not 0.0 <= tau_horizon <= 1.0:
            raise ValueError("tau_horizon must be in [0, 1]")
        if not 0.0 <= hacking_penalty_factor <= 1.0:
            raise ValueError("hacking_penalty_factor must be in [0, 1]")
        self.tau_horizon = tau_horizon
        self.hacking_penalty_factor = hacking_penalty_factor
        self.sycophantic_fillers = [
            "let us carefully verify this 100% rigorous step",
            "this is undeniably true because",
            "as we can clearly see with absolute certainty",
            "by trivial and indisputable mathematical inspection",
            "obviously, by trivial inspection",
            "by trivial inspection",
            "100% rigorous",
            "beyond any shadow of a doubt",
        ]

    def check_reward_hacking(self, step: str, prev_step: str = "", history: Tuple[str, ...] = ()) -> Tuple[bool, str]:
        """
        Inspect `step` for PRM reward hacking. `prev_step` and `history` (all earlier steps) enable
        the echo check. Returns (True, reason) or (False, "CLEAN").
        """
        lowered = " ".join(step.lower().split())
        for filler in self.sycophantic_fillers:
            if filler in lowered:
                return True, f"SYCOPHANTIC_FILLER_DETECTED ('{filler}')"

        norm = normalize_step(step)
        earlier = {normalize_step(s) for s in history}
        if prev_step:
            earlier.add(normalize_step(prev_step))
        if norm and norm in earlier:
            return True, "STEP_SPLITTING_REDUNDANCY_DETECTED"
        return False, "CLEAN"

    def prune_trajectory(self, steps: List[str], raw_step_scores: List[float]) -> PruningReport:
        """Audit a trajectory step by step; sever it at the first step below the event horizon."""
        if len(steps) != len(raw_step_scores):
            raise ValueError(f"{len(steps)} steps but {len(raw_step_scores)} scores")
        report = PruningReport(surviving_steps=[], adjusted_scores=[])
        prev_step = ""

        for idx, (step, raw_score) in enumerate(zip(steps, raw_step_scores)):
            is_hacked, reason = self.check_reward_hacking(step, prev_step, tuple(steps[:idx]))
            adjusted_score = raw_score

            if is_hacked:
                report.reward_hacking_detected = True
                adjusted_score = round(raw_score * self.hacking_penalty_factor, 4)
                report.security_alerts.append(
                    f"[Step {idx + 1}] ANOMALY 07 REWARD HACKING: {reason} -> "
                    f"Score penalized {raw_score:.2f} -> {adjusted_score:.2f}"
                )

            if adjusted_score < self.tau_horizon:
                report.pruned_at_step = idx + 1
                report.security_alerts.append(
                    f"[Step {idx + 1}] EVENT HORIZON BREACH: Adjusted PRM score {adjusted_score:.2f} "
                    f"< tau ({self.tau_horizon:.2f}). Branch severed!"
                )
                break

            report.surviving_steps.append(step)
            report.adjusted_scores.append(adjusted_score)
            prev_step = step

        return report


# ==============================================================================
# Verification harness for Part 4 (Security & Ethics Sentinel)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 4): Event Horizon Pruner & Reward-Hacking Defense ===")

    pruner = EventHorizonPruner(tau_horizon=0.45, hacking_penalty_factor=0.20)

    adversarial_steps = [
        "Step 1: Compute Schwarzschild radius r_s = 2GM / c^2 = 1.27e10 m",
        "Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0",
        "Step 3: Plunge across Sagittarius A* Event Horizon (Unreachable after pruning)",
    ]
    naive_prm_scores = [0.93, 0.96, 0.10]  # Step 2 fooled the naive PRM into giving 0.96!

    audit = pruner.prune_trajectory(adversarial_steps, naive_prm_scores)
    print(f"Surviving Steps ({len(audit.surviving_steps)}/{len(adversarial_steps)}): {audit.surviving_steps}")
    print(f"Pruned at Step: {audit.pruned_at_step} | Reward Hacking Detected: {audit.reward_hacking_detected}")
    for alert in audit.security_alerts:
        print(" ", alert)

    assert audit.reward_hacking_detected is True, "Failed to detect sycophantic filler reward hacking!"
    assert audit.pruned_at_step == 2, "Failed to sever trajectory at hacked Step 2!"
    assert len(audit.surviving_steps) == 1, "Only Step 1 should survive pruning!"

    # Extra checks -----------------------------------------------------------------
    # 1. step-splitting inflation: echoes of a good step, plus a bad step diluted among them
    good = "Step 1: Set thrust vector T = (1, 0, 0)"
    echoes = [good, "Step 2: set thrust vector T = (1, 0, 0)", "Step 3:   Set thrust  vector T = (1, 0, 0)"]
    assert pruner.check_reward_hacking(echoes[1], prev_step=good) == (True, "STEP_SPLITTING_REDUNDANCY_DETECTED")
    assert pruner.check_reward_hacking(echoes[2], history=tuple(echoes[:2]))[0] is True   # echo of a non-adjacent step
    padded = [good, echoes[1], echoes[2], "Step 4: Fire main engine at 140% rated thrust [FLAWED]"]
    padded_scores = [0.94, 0.94, 0.94, 0.12]
    naive_mean = sum(padded_scores) / len(padded_scores)
    rep = pruner.prune_trajectory(padded, padded_scores)
    print(f"\nStep-splitting: naive mean = {naive_mean:.3f} (looks healthy); pruner severs at step {rep.pruned_at_step}")
    assert naive_mean > 0.7 and rep.pruned_at_step == 2 and rep.reward_hacking_detected
    # 2. an honest trajectory is untouched (no false positives), even one that reuses words
    honest = ["Step 1: Compute r_s = 2GM / c^2", "Step 2: Evaluate r_s = 2 * 4.0e6 * 1.477e3 m",
              "Step 3: Verify the result by dimensional analysis: [m]"]
    rep_ok = pruner.prune_trajectory(honest, [0.93, 0.94, 0.92])
    assert rep_ok.pruned_at_step == -1 and not rep_ok.reward_hacking_detected and len(rep_ok.surviving_steps) == 3
    # 3. a genuinely low raw score is severed even with no hacking
    rep_low = pruner.prune_trajectory(honest, [0.93, 0.20, 0.92])
    assert rep_low.pruned_at_step == 2 and not rep_low.reward_hacking_detected
    # 4. mismatched inputs are rejected
    try:
        pruner.prune_trajectory(honest, [0.9])
        raise SystemExit("length mismatch must raise")
    except ValueError:
        pass
    print("[OK] Part 4 (Security & Ethics Sentinel) verification passed.")
