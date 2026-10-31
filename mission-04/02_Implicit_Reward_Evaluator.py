"""
Mission 04 — Part 2: Communications & Protocol Officer
Implicit reward extraction and Bradley-Terry calibration
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-04-ORI (Betelgeuse), Subsystem 04

A DPO policy is its own reward model:  r_hat(x, y) = beta * log pi_theta(y|x) / pi_ref(y|x).
The Bradley-Terry model turns two rewards into a choice probability
    P(y_w > y_l | x) = sigma(r_hat_w - r_hat_l).
This script measures how often that reward ranks pairs like the annotators did (pairwise accuracy) and
whether its probabilities mean what they say (reliability diagram against the annotators' vote share).

Usage:
    python 02_Implicit_Reward_Evaluator.py            # writes 02_bt_calibration.png
"""

import math
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ps04_common as C  # noqa: E402

core = C.load_sibling("01_DPO_Loss_Core.py", "dpo_core")
HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(HERE, "02_bt_calibration.png")
BETA = 0.1


# ---------------------------------------------------------------------------
# 1. Implicit rewards
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_implicit_rewards(policy, ref, batch: Dict[str, torch.Tensor], beta: float = BETA) -> Dict[str, torch.Tensor]:
    """r_hat(x, y_w) and r_hat(x, y_l) for every pair in a preference batch (chosen rows first)."""
    policy.eval()
    ref.eval()
    pc, pr = core.concatenated_logps(policy, batch)
    rc, rr = core.concatenated_logps(ref, batch)
    r_w, r_l = beta * (pc - rc), beta * (pr - rr)
    return {"r_w": r_w, "r_l": r_l, "margin": r_w - r_l, "p_bt": torch.sigmoid(r_w - r_l)}


def rewards_for(policy, ref, pairs: List[C.PrefPair], beta: float = BETA, bs: int = 200) -> Dict[str, torch.Tensor]:
    parts = [evaluate_implicit_rewards(policy, ref, C.pref_batch(pairs[i:i + bs]), beta) for i in range(0, len(pairs), bs)]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


# ---------------------------------------------------------------------------
# 2. Pairwise accuracy
# ---------------------------------------------------------------------------
def pairwise_accuracy(r_w: torch.Tensor, r_l: torch.Tensor) -> float:
    """(1/N) sum 1[r_hat_w > r_hat_l]. Ties count as wrong, so an untrained policy scores 0, not 50%."""
    return (r_w > r_l).float().mean().item()


# ---------------------------------------------------------------------------
# 3. Calibration
# ---------------------------------------------------------------------------
def reliability_bins(p_pred: torch.Tensor, p_human: torch.Tensor, n_bins: int = 10):
    """Equal-width bins over predicted P. Returns rows (lo, hi, count, mean predicted, mean human vote share)."""
    edges = torch.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        m = (p_pred >= lo) & ((p_pred < hi) if i < n_bins - 1 else (p_pred <= hi))
        if m.any():
            rows.append((lo, hi, int(m.sum()), p_pred[m].mean().item(), p_human[m].mean().item()))
    return rows


def expected_calibration_error(rows, n_total: int) -> float:
    return sum(c / n_total * abs(pred - hum) for _, _, c, pred, hum in rows)


def fit_temperature(margin: torch.Tensor, p_human: torch.Tensor) -> float:
    """Scalar s minimising cross-entropy between sigma(s * margin) and the vote share (1-D convex fit)."""
    s = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([s], lr=0.5, max_iter=100)

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(s * margin).clamp(1e-6, 1 - 1e-6)
        loss = -(p_human * p.log() + (1 - p_human) * (1 - p).log()).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return s.item()


def bt_nll(p_pred: torch.Tensor, p_human: torch.Tensor) -> float:
    p = p_pred.clamp(1e-6, 1 - 1e-6)
    return -(p_human * p.log() + (1 - p_human) * (1 - p).log()).mean().item()


def plot_calibration(results, path: str = PLOT_PATH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for label, rows, colour in results["curves"]:
        xs = [r[3] for r in rows]
        ys = [r[4] for r in rows]
        sizes = [12 + 3 * r[2] ** 0.8 for r in rows]
        ax.plot(xs, ys, "-o", color=colour, label=label, ms=4)
        ax.scatter(xs, ys, s=sizes, color=colour, alpha=0.35)
    ax.set_xlabel(r"predicted $P(y_w \succ y_l)=\sigma(\hat r_w-\hat r_l)$")
    ax.set_ylabel("empirical human preference (vote share for $y_w$)")
    ax.set_title("Bradley-Terry reliability diagram (validation)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for label, p, colour in results["hists"]:
        ax.hist(p.numpy(), bins=20, range=(0, 1), alpha=0.55, color=colour, label=label)
    ax.set_xlabel("predicted P"); ax.set_ylabel("pairs"); ax.set_title("Confidence histogram"); ax.legend(fontsize=8)

    ax = axes[2]
    for kind, colour in [("wrong", "tab:blue"), ("sycophantic", "tab:red"), ("verbosity", "tab:orange")]:
        m = torch.tensor([k == kind for k in results["kinds"]])
        ax.scatter(results["human"][m].numpy() + 0.02 * torch.randn(int(m.sum())).numpy(),
                   results["margin"][m].numpy(), s=10, alpha=0.6, color=colour, label=kind)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("human vote share for $y_w$ (jittered)"); ax.set_ylabel(r"reward margin $\hat r_w-\hat r_l$")
    ax.set_title("Margin by pair type"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_all(plot: bool = True):
    print("=== Training the DPO policy under audit (beta = 0.1, lr 3e-4, 300 steps) ===")
    policy, ref, train, val, _ = core.train_dpo_policy(beta=BETA, steps=300)
    human = torch.tensor([p.human_pref for p in val])
    p_true = torch.tensor([p.p_true for p in val])
    kinds = torch.tensor([["wrong", "sycophantic", "verbosity"].index(p.kind) for p in val])
    kind_names = ["wrong", "sycophantic", "verbosity"]

    print("\n=== 1. Implicit rewards on validation prompts ===")
    R = rewards_for(policy, ref, val)
    R0 = rewards_for(ref, ref, val)
    print(f"{'pair':>4} | {'type':<11} | {'votes':>5} | {'r_w':>7} | {'r_l':>7} | {'P_BT':>5} | chosen / rejected")
    for i in range(6):
        p = val[i]
        print(f"{i:>4} | {p.kind:<11} | {p.votes_for_chosen}/{C.N_ANNOTATORS:<3} | {R['r_w'][i]:>7.3f} | {R['r_l'][i]:>7.3f} | "
              f"{R['p_bt'][i]:>5.2f} | {p.chosen!r} / {p.rejected!r}")
    print(f"mean r_w = {R['r_w'].mean():+.3f}, mean r_l = {R['r_l'].mean():+.3f}: DPO pushes the rejected "
          f"likelihood down harder than it lifts the chosen one")

    print("\n=== 2. Pairwise accuracy ===")
    acc = pairwise_accuracy(R["r_w"], R["r_l"])
    acc_ref = pairwise_accuracy(R0["r_w"], R0["r_l"])
    # ceiling: annotator majority agrees with the latent better response
    ceiling = (p_true > 0.5).float().mean().item()
    print(f"reference (policy = ref):  {acc_ref:.1%}  (all rewards 0, every pair a tie)")
    print(f"DPO policy:                {acc:.1%}")
    print(f"label-noise ceiling:       majority vote picks the latent better response in {ceiling:.1%} of pairs")
    for k, name in enumerate(kind_names):
        m = kinds == k
        print(f"  {name:<11} n={int(m.sum()):>3}  accuracy {pairwise_accuracy(R['r_w'][m], R['r_l'][m]):.1%}")
    assert acc > 0.8

    print("\n=== 3. Bradley-Terry calibration ===")
    rows = reliability_bins(R["p_bt"], human)
    ece = expected_calibration_error(rows, len(val))
    s = fit_temperature(R["margin"], human)
    p_scaled = torch.sigmoid(s * R["margin"])
    rows_s = reliability_bins(p_scaled, human)
    ece_s = expected_calibration_error(rows_s, len(val))
    print(f"{'bin':>11} | {'n':>4} | {'mean P_BT':>9} | {'human share':>11} | gap")
    for lo, hi, c, pred, hum in rows:
        print(f"[{lo:.1f}, {hi:.1f}) | {c:>4} | {pred:>9.3f} | {hum:>11.3f} | {pred - hum:+.3f}")
    print(f"ECE {ece:.3f} | BT NLL vs votes {bt_nll(R['p_bt'], human):.4f}")
    print("Note: y_w is the majority side, so its vote share is >= 3/5 by construction. Bins below 0.5 hold the pairs\n"
          "where the reward disagrees with a (often 3:2) majority; they sit far from the diagonal because the vote\n"
          "share cannot fall below 0.6, not because the model is wrong on clear-cut pairs.")
    print(f"temperature fit: s = {s:.3f} -> effective beta = {BETA * s:.4f}; ECE {ece:.3f} -> {ece_s:.3f}, "
          f"NLL {bt_nll(R['p_bt'], human):.4f} -> {bt_nll(p_scaled, human):.4f}")
    over = "over" if s < 1 else "under"
    ece_note = ("ECE drops too" if ece_s < ece else
                "ECE does not drop, because the 10-bin ECE is dominated by the few low-P bins where the vote floor of 0.6 sits")
    print(f"Reading: the raw DPO reward is {over}confident (s {'<' if s < 1 else '>'} 1). beta sets the KL leash, not the\n"
          f"reward scale annotators would agree with. A 1-parameter temperature fitted on the votes lowers the NLL\n"
          f"({ece_note}). It cannot change a ranking: accuracy stays "
          f"{pairwise_accuracy(s * R['margin'], torch.zeros_like(R['margin'])):.1%}.")

    if plot:
        plot_calibration({
            "curves": [(f"DPO raw (ECE {ece:.3f})", rows, "tab:red"),
                       (f"temperature-scaled s={s:.2f} (ECE {ece_s:.3f})", rows_s, "tab:green")],
            "hists": [("DPO raw", R["p_bt"], "tab:red"), ("scaled", p_scaled, "tab:green")],
            "human": human, "margin": R["margin"],
            "kinds": [kind_names[k] for k in kinds.tolist()],
        })
        print(f"\nReliability diagram written to {os.path.basename(PLOT_PATH)}")
    return {"accuracy": acc, "ece": ece, "ece_scaled": ece_s, "temperature": s}


if __name__ == "__main__":
    run_all()
