"""
Mission 06 - Part 3: Data & Payload Specialist
MDP trajectory logger, empirical discounted return G_0 and transition-frequency matrix
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-06-CAR (Eta Carinae & Homunculus Nebula), Subsystem 06

An episode is a trajectory   tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T).
Its empirical discounted return is
    G_0 = sum_{t=0}^{T-1} gamma^t r_t        (gamma = 0.95)
(one reward per action; s_T is terminal and carries no reward). Past jumps give an empirical
transition model
    P_hat(s' | s, a) = N(s, a, s') / N(s, a)
by maximum likelihood.

Usage:
    python 03_MDP_Trajectory_Logger.py
"""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

GAMMA = 0.95


# ---------------------------------------------------------------------------
# 1. Trajectory logging
# ---------------------------------------------------------------------------
@dataclass
class Step:
    state: Any
    action: Any
    reward: float


@dataclass
class TrajectoryLogger:
    """Records (s_t, a_t, r_t) triples and the terminal state s_T of one episode."""
    steps: List[Step] = field(default_factory=list)
    final_state: Any = None
    finished: bool = False

    def record(self, state: Any, action: Any, reward: float) -> None:
        if self.finished:
            raise RuntimeError("episode already finished; start a new TrajectoryLogger")
        self.steps.append(Step(state, action, float(reward)))

    def finish(self, final_state: Any) -> None:
        self.final_state = final_state
        self.finished = True

    @property
    def rewards(self) -> List[float]:
        return [s.reward for s in self.steps]

    def as_tuple(self) -> Tuple:
        """tau = (s_0, a_0, r_0, ..., s_T) as a flat tuple."""
        flat: List[Any] = []
        for s in self.steps:
            flat.extend([s.state, s.action, s.reward])
        flat.append(self.final_state)
        return tuple(flat)

    def discounted_return(self, gamma: float = GAMMA) -> float:
        return discounted_return(self.rewards, gamma)

    def transitions(self) -> List[Tuple[Any, Any, Any]]:
        """(s_t, a_t, s_{t+1}) triples for the transition model."""
        nxt = [s.state for s in self.steps[1:]] + [self.final_state]
        return [(s.state, s.action, n) for s, n in zip(self.steps, nxt)]


# ---------------------------------------------------------------------------
# 2. Discounted return
# ---------------------------------------------------------------------------
def discounted_return(rewards: Sequence[float], gamma: float = GAMMA) -> float:
    """G_0 = sum_t gamma^t r_t, summed explicitly."""
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    return sum((gamma ** t) * r for t, r in enumerate(rewards))


def discounted_return_recursive(rewards: Sequence[float], gamma: float = GAMMA) -> float:
    """Same quantity via G_t = r_t + gamma G_{t+1}, computed backwards (an independent check)."""
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


# ---------------------------------------------------------------------------
# 3. Transition-frequency matrix P(s' | s, a)
# ---------------------------------------------------------------------------
def transition_frequency(jumps: Iterable[Tuple[Any, Any, Any]]) -> Dict[Tuple[Any, Any], Dict[Any, float]]:
    """Maximum-likelihood estimate of P(s' | s, a) from (s, a, s') records."""
    counts: Dict[Tuple[Any, Any], Counter] = defaultdict(Counter)
    for s, a, s2 in jumps:
        counts[(s, a)][s2] += 1
    return {sa: {s2: n / sum(c.values()) for s2, n in sorted(c.items(), key=lambda kv: str(kv[0]))}
            for sa, c in counts.items()}


def success_probability(model: Dict[Tuple[Any, Any], Dict[Any, float]], safe_states: Iterable[Any]
                        ) -> Dict[Tuple[Any, Any], float]:
    """P(next state is in `safe_states` | s, a) for every observed (s, a)."""
    safe = set(safe_states)
    return {sa: sum(p for s2, p in dist.items() if s2 in safe) for sa, dist in model.items()}


# Nebula sectors, actions and the *ground truth* used to generate the historical jump log.
SECTORS = ["stable_lane", "shear_zone", "dust_cloud", "flare_front", "escape_corridor", "adrift"]
SAFE_SECTORS = ["stable_lane", "escape_corridor"]
ACTIONS = ["warp_shields_high", "warp_shields_low"]

_TRUE_P = {
    ("stable_lane", "warp_shields_high"): {"stable_lane": 0.30, "shear_zone": 0.30, "escape_corridor": 0.40},
    ("stable_lane", "warp_shields_low"): {"shear_zone": 0.40, "flare_front": 0.40, "adrift": 0.20},
    ("shear_zone", "warp_shields_high"): {"stable_lane": 0.45, "dust_cloud": 0.35, "shear_zone": 0.20},
    ("shear_zone", "warp_shields_low"): {"adrift": 0.50, "flare_front": 0.50},
    ("dust_cloud", "warp_shields_high"): {"stable_lane": 0.50, "dust_cloud": 0.30, "flare_front": 0.20},
    ("dust_cloud", "warp_shields_low"): {"adrift": 0.70, "flare_front": 0.30},
    ("flare_front", "warp_shields_high"): {"shear_zone": 0.50, "stable_lane": 0.30, "adrift": 0.20},
    ("flare_front", "warp_shields_low"): {"adrift": 1.00},
}


def historical_jump_dataset(n_per_pair: int = 400, seed: int = 6) -> List[Tuple[str, str, str]]:
    """
    SYNTHETIC historical jump log, generated from a fixed ground-truth kernel with a seeded RNG.
    It stands in for the instructor's "provided dataset" (none ships with the brief), so the
    estimated matrix can be checked against the kernel that produced it.
    """
    rng = random.Random(seed)
    jumps: List[Tuple[str, str, str]] = []
    for (s, a), dist in _TRUE_P.items():
        states, probs = zip(*dist.items())
        jumps.extend((s, a, rng.choices(states, probs)[0]) for _ in range(n_per_pair))
    rng.shuffle(jumps)
    return jumps


def format_matrix(model: Dict[Tuple[Any, Any], Dict[Any, float]], cols: Sequence[str] = SECTORS) -> str:
    head = f"{'(s, a)':<40}" + "".join(f"{c[:11]:>12}" for c in cols)
    rows = [head, "-" * len(head)]
    for (s, a), dist in sorted(model.items()):
        rows.append(f"{s + ' | ' + a:<40}" + "".join(f"{dist.get(c, 0.0):>12.3f}" for c in cols))
    return "\n".join(rows)


if __name__ == "__main__":
    print("=== Subsystem 06: MDP Trajectory Logger ===")
    log = TrajectoryLogger()
    for s, a, r in [("stable_lane", "warp_shields_high", 10), ("shear_zone", "warp_shields_high", 0),
                    ("stable_lane", "warp_shields_low", -5), ("flare_front", "warp_shields_high", 20)]:
        log.record(s, a, r)
    log.finish("escape_corridor")
    print("tau =", log.as_tuple())
    g = log.discounted_return()
    print(f"G_0 (gamma={GAMMA}) = {g:.6f}   (backward recursion: {discounted_return_recursive(log.rewards):.6f})")
    assert abs(g - 22.635) < 1e-9

    data = historical_jump_dataset()
    model = transition_frequency(data)
    print(f"\nEmpirical P(s' | s, a) from {len(data)} synthetic historical jumps:\n")
    print(format_matrix(model))
    print("\nP(reach a safe sector | s, a):")
    for (s, a), p in sorted(success_probability(model, SAFE_SECTORS).items()):
        print(f"  {s:<12} {a:<18} {p:.3f}")
