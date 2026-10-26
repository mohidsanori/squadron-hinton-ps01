"""
Mission 03 — Part 2: Communications & Protocol Officer
Rotary Position Embeddings (RoPE): complex-plane rotation, relative invariance, decay vs ALiBi
Course: Conversational AI (WS 2026/27)
Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-03-M45 (Pleiades Open Cluster), Subsystem 03

RoPE (Su et al., 2021) splits a head vector x in R^d into d/2 pairs and reads each
pair as a complex number z_i = x_{2i} + j x_{2i+1}. At position m, pair i is rotated
by the angle m * theta_i:

    theta_i = base^(-2(i-1)/d),   i = 1 .. d/2,   base = 10000
    RoPE(x, m)_i = z_i * e^(j m theta_i)

Why the dot product only depends on m - n. For complex pairs, the real dot product
is <a, b> = Re(a * conj(b)). Summing over the pairs:

    <RoPE(q, m), RoPE(k, n)> = sum_i Re( q_i e^(j m theta_i) * conj(k_i e^(j n theta_i)) )
                             = sum_i Re( q_i conj(k_i) e^(j (m - n) theta_i) )
                             = g(q, k, m - n)

The absolute positions cancel, so shifting both tokens by delta changes nothing:
<RoPE(q, m+delta), RoPE(k, n+delta)> = g(q, k, m - n). Rotations also keep the norm,
so RoPE never changes |q| or |k|, only the angle between them.

Checks run by `python 02_RoPE_Positional_Engine.py`:
1. apply_rope equals multiplication by the explicit block-diagonal rotation matrix R_m.
2. Norm preservation.
3. Relative invariance for 1000 random (q, k, m, n, delta), plus the starter's single case.
4. Position 0 is the identity.
5. Long-range decay: RoPE's Su et al. upper bound and q = k score curves vs ALiBi's
   linear penalty, with attention heatmaps -> 02_rope_vs_alibi_decay.png
6. Context extrapolation: which frequency pairs see rotation angles at inference
   that they never saw in training, for plain RoPE, position interpolation and NTK scaling.
"""

import math
import os
from typing import Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(HERE, "02_rope_vs_alibi_decay.png")


# ---------------------------------------------------------------------------
# RoPE engine
# ---------------------------------------------------------------------------

def rope_frequencies(dim: int, theta: float = 10000.0) -> torch.Tensor:
    """theta_i = base^(-2(i-1)/d) for i = 1 .. d/2 (shape (dim // 2,))."""
    assert dim % 2 == 0, "head dimension must be even for RoPE 2D pairing"
    return 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, scale: float = 1.0) -> torch.Tensor:
    """
    cis(m * theta_i) = e^(j m theta_i) for positions m = 0 .. end-1. Shape (end, dim // 2), complex64.

    scale < 1 is linear position interpolation (Chen et al., 2023): position m is
    treated as m * scale, so a longer context is squeezed into the trained angle range.
    """
    freqs = rope_frequencies(dim, theta)
    t = torch.arange(end, dtype=torch.float32) * scale
    angles = torch.outer(t, freqs)                                # (end, dim // 2)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Rotate every consecutive pair of x by its position's angle.

    x:         (batch, heads, seq_len, head_dim), real
    freqs_cis: (seq_len, head_dim // 2), complex, row m = positions of x along seq_len
    """
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))   # (..., seq, d/2)
    rotated = x_c * freqs_cis.view(1, 1, *freqs_cis.shape)                  # complex multiply = 2D rotation
    return torch.view_as_real(rotated).flatten(-2).type_as(x)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Starter interface: rotate queries and keys with the same frequencies."""
    return apply_rope(xq, freqs_cis), apply_rope(xk, freqs_cis)


def rotation_matrix(dim: int, m: int, theta: float = 10000.0) -> torch.Tensor:
    """Explicit real R^d_{Theta,m}: block-diagonal 2x2 rotations [[cos, -sin], [sin, cos]]."""
    R = torch.zeros(dim, dim, dtype=torch.float64)
    for i, f in enumerate(rope_frequencies(dim, theta).double()):
        c, s = math.cos(m * f), math.sin(m * f)
        R[2 * i:2 * i + 2, 2 * i:2 * i + 2] = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    return R


def rope_score(q: torch.Tensor, k: torch.Tensor, m: int, n: int, freqs_cis: torch.Tensor) -> float:
    """<RoPE(q, m), RoPE(k, n)> for single vectors q, k of shape (dim,)."""
    qm = apply_rope(q.view(1, 1, 1, -1), freqs_cis[m:m + 1])
    kn = apply_rope(k.view(1, 1, 1, -1), freqs_cis[n:n + 1])
    return (qm * kn).sum().item()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_rotation_matrix(dim=16, positions=(0, 1, 7, 300)):
    print("=== 1. Complex multiply == explicit rotation matrix R_m ===")
    torch.manual_seed(0)
    freqs_cis = precompute_freqs_cis(dim, 512)
    x = torch.randn(dim, dtype=torch.float64)
    for m in positions:
        ours = apply_rope(x.float().view(1, 1, 1, -1), freqs_cis[m:m + 1]).view(-1).double()
        ref = rotation_matrix(dim, m) @ x
        err = (ours - ref).abs().max().item()
        print(f"m = {m:>3}: max |apply_rope - R_m x| = {err:.2e}")
        assert err < 1e-4
    print("[PASS] apply_rope is the block-diagonal rotation R^d_{Theta,m}.\n")


def verify_norm_preservation(dim=64, seq_len=256):
    print("=== 2. Norm preservation ===")
    torch.manual_seed(1)
    x = torch.randn(2, 4, seq_len, dim)
    y = apply_rope(x, precompute_freqs_cis(dim, seq_len))
    err = (x.norm(dim=-1) - y.norm(dim=-1)).abs().max().item()
    print(f"max | |x| - |RoPE(x)| | over {2 * 4 * seq_len} vectors = {err:.2e}")
    assert err < 1e-4
    print("[PASS] RoPE only rotates; vector lengths are unchanged.\n")


def verify_relative_invariance(dim=64, max_seq_len=4096, trials=1000):
    print("=== 3. Relative invariance <RoPE(q,m), RoPE(k,n)> = g(q, k, m-n) ===")
    torch.manual_seed(42)
    freqs_cis = precompute_freqs_cis(dim, max_seq_len)

    # The starter's case: m = 10, n = 25, delta = 40
    q, k = torch.randn(dim), torch.randn(dim)
    a, b = rope_score(q, k, 10, 25, freqs_cis), rope_score(q, k, 50, 65, freqs_cis)
    print(f"<RoPE(q,10), RoPE(k,25)> = {a:.6f}   <RoPE(q,50), RoPE(k,65)> = {b:.6f}   diff {abs(a - b):.2e}")
    assert abs(a - b) < 1e-4

    # 1000 random cases, compared against g(q, k, m-n) computed directly from the formula
    worst, worst_rel = 0.0, 0.0
    freqs = rope_frequencies(dim).double()
    for _ in range(trials):
        q, k = torch.randn(dim), torch.randn(dim)
        m, n = torch.randint(0, max_seq_len // 2, (2,)).tolist()
        delta = torch.randint(0, max_seq_len // 2, (1,)).item()
        s1 = rope_score(q, k, m, n, freqs_cis)
        s2 = rope_score(q, k, m + delta, n + delta, freqs_cis)
        qc = torch.view_as_complex(q.double().view(-1, 2))
        kc = torch.view_as_complex(k.double().view(-1, 2))
        g = (qc * kc.conj() * torch.polar(torch.ones_like(freqs), (m - n) * freqs)).real.sum().item()
        worst = max(worst, abs(s1 - s2))
        worst_rel = max(worst_rel, abs(s1 - g))
    print(f"{trials} random (q, k, m, n, delta) up to position {max_seq_len}: "
          f"max |shifted - original| = {worst:.2e}, max |score - g(q,k,m-n)| = {worst_rel:.2e}")
    # float32 angles m*theta lose precision at large m, so the tolerance grows with position
    assert worst < 1e-3 and worst_rel < 1e-3
    print("[PASS] Attention scores depend only on relative distance, for any translation delta.\n")
    return worst


def verify_identity_at_zero(dim=32):
    print("=== 4. Position 0 is the identity ===")
    x = torch.randn(1, 1, 1, dim)
    assert torch.allclose(apply_rope(x, precompute_freqs_cis(dim, 1)), x)
    print("[PASS] RoPE(x, 0) = x.\n")


# ---------------------------------------------------------------------------
# Long-range decay: RoPE vs ALiBi
# ---------------------------------------------------------------------------

def rope_decay_bound(dim: int, distances: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """
    Su et al. (2021), Sec. 3.4.3: with h_i = q_i conj(k_i) and S_j = sum_{l<j} e^(j r theta_l),
    |g(q, k, r)| <= (max_i |h_{i+1} - h_i|) * sum_j |S_j|. The relative bound
    (1 / (d/2)) * sum_j |S_j| is what decays with distance r.
    """
    freqs = rope_frequencies(dim, theta).double()
    phases = torch.polar(torch.ones(len(distances), len(freqs), dtype=torch.float64),
                         distances.double().unsqueeze(1) * freqs)
    S = torch.cumsum(phases, dim=1)
    return S.abs().mean(dim=1)


def rope_score_curve(dim: int, distances: torch.Tensor, trials: int = 2000, rho: float = 0.8) -> torch.Tensor:
    """
    Expected RoPE score E[g(q, k, r)] for correlated q, k (corr rho per coordinate, i.e. a key
    that is relevant to the query), normalised to 1 at r = 0. For q = k exactly (rho = 1) this is
    (2/d) * sum_i cos(r theta_i).
    """
    torch.manual_seed(7)
    q = torch.randn(trials, dim, dtype=torch.float64)
    k = rho * q + math.sqrt(1 - rho ** 2) * torch.randn(trials, dim, dtype=torch.float64)
    h = torch.view_as_complex(q.view(trials, -1, 2)) * torch.view_as_complex(k.view(trials, -1, 2)).conj()
    freqs = rope_frequencies(dim).double()
    rot = torch.polar(torch.ones(len(distances), len(freqs), dtype=torch.float64), distances.double().unsqueeze(1) * freqs)
    scores = (h.unsqueeze(0) * rot.unsqueeze(1)).real.sum(-1).mean(-1)     # (len(distances),)
    return scores / scores[0]


def alibi_slopes(num_heads: int) -> torch.Tensor:
    """Press et al. (2022): geometric slopes 2^(-8/H), 2^(-16/H), ..., 2^(-8)."""
    return torch.tensor([2 ** (-8 * (h + 1) / num_heads) for h in range(num_heads)])


def causal_attention_map(logits: torch.Tensor) -> torch.Tensor:
    n = logits.size(-1)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    return torch.softmax(logits.masked_fill(mask, float("-inf")), dim=-1)


def plot_decay(dim=128, max_dist=1024, heat_len=64, path=PLOT_PATH):
    print("=== 5. Long-range decay: RoPE vs ALiBi ===")
    from matplotlib.figure import Figure  # no pyplot: leaves a notebook's backend alone

    r = torch.arange(0, max_dist + 1)
    bound = rope_decay_bound(dim, r)
    curve = rope_score_curve(dim, r)
    slopes = alibi_slopes(8)

    fig = Figure(figsize=(17, 4.8))
    axes = fig.subplots(1, 3)
    ax = axes[0]
    ax.plot(r, bound / bound[0], label="RoPE relative upper bound (Su et al.)", color="#1f77b4")
    ax.plot(r, curve, label="RoPE E[score], correlated q,k (rho=0.8)", color="#2ca02c", alpha=0.8)
    ax.set_title(f"RoPE score vs distance (d_head = {dim})")
    ax.set_xlabel("relative distance |m - n|")
    ax.set_ylabel("normalised score (1 at distance 0)")
    ax.set_xscale("symlog", linthresh=10)
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend(fontsize=8)

    ax = axes[1]
    for h in (0, 3, 7):
        ax.plot(r, -slopes[h] * r, label=f"ALiBi head {h + 1}: slope 2^-{8 * (h + 1) // 8}")
    ax.set_title("ALiBi bias vs distance (8 heads, linear axes)")
    ax.set_xlabel("relative distance |m - n|")
    ax.set_ylabel("additive logit bias")
    ax.legend(fontsize=8)

    # Attention maps for the same correlated content, positional signal only
    n = heat_len
    pos = torch.arange(n)
    rope_logits = rope_score_curve(dim, (pos.view(-1, 1) - pos.view(1, -1)).abs().flatten()).view(n, n)
    rope_logits = rope_logits * math.sqrt(dim) * 0.8        # typical score scale of a relevant key
    alibi_logits = -slopes[3] * (pos.view(-1, 1) - pos.view(1, -1)).abs().float()
    ax = axes[2]
    rope_map = causal_attention_map(rope_logits.float())
    alibi_map = causal_attention_map(alibi_logits)
    combined = torch.cat([rope_map, torch.full((n, 2), float("nan")), alibi_map], dim=1)
    from matplotlib.colors import LogNorm
    im = ax.imshow(combined.numpy(), cmap="viridis", aspect="auto", norm=LogNorm(vmin=1e-4, vmax=1))
    ax.set_title(f"Causal attention weights (log colour), {n} tokens:\nRoPE (left) | ALiBi head 4 (right)", fontsize=10)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    ax.set_xticks([0, n - 1, n + 2, 2 * n + 1], ["0", str(n - 1), "0", str(n - 1)])
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)

    print(f"{'distance':>8} | {'RoPE bound':>10} | {'RoPE E[score]':>13} | {'ALiBi h1':>8} | {'ALiBi h8':>8}")
    for d in (0, 1, 4, 16, 64, 256, 1024):
        print(f"{d:>8} | {bound[d] / bound[0]:>10.3f} | {curve[d]:>13.3f} | {-slopes[0] * d:>8.1f} | {-slopes[7] * d:>8.2f}")
    print("RoPE decays smoothly and non-monotonically (it oscillates, never forces a score to 0);")
    print("ALiBi subtracts a fixed amount per token, so distant tokens are always penalised.")
    print(f"[SAVED] {os.path.basename(path)}\n")
    return path


# ---------------------------------------------------------------------------
# Context extrapolation
# ---------------------------------------------------------------------------

def extrapolation_report(dim=128, train_len=4096, test_len=32768, theta=10000.0):
    """
    During training, pair i only ever sees angles up to train_len * theta_i. Pairs with
    train_len * theta_i < 2 pi never complete a full turn, so at test_len they meet
    angles they never saw (out of distribution). Two fixes:
      - Position interpolation (PI): scale positions by train_len / test_len.
      - NTK-aware scaling: raise the base to theta * s^(d / (d - 2)), s = test_len / train_len,
        which stretches the low frequencies and leaves the high ones almost unchanged.
    """
    print(f"=== 6. Context extrapolation: trained at {train_len}, run at {test_len} (d_head = {dim}) ===")
    s = test_len / train_len
    variants = {
        "plain RoPE": (rope_frequencies(dim, theta), 1.0),
        "position interpolation": (rope_frequencies(dim, theta), 1.0 / s),
        "NTK-aware base": (rope_frequencies(dim, theta * s ** (dim / (dim - 2))), 1.0),
    }
    train_max = train_len * rope_frequencies(dim, theta)          # max angle seen in training, per pair
    print(f"{'variant':<24} | {'OOD pairs':>9} | {'lowest-freq angle':>17} | {'highest-freq step':>17}")
    print("-" * 78)
    rows = {}
    for name, (freqs, scale) in variants.items():
        test_max = test_len * scale * freqs
        ood = ((test_max > train_max * 1.0001) & (train_max < 2 * math.pi)).sum().item()
        rows[name] = ood
        print(f"{name:<24} | {ood:>4} / {dim // 2:<3} | {test_max[-1].item():>11.2f} rad   | "
              f"{scale * freqs[0].item():>11.4f} rad")
    print(f"(trained max angle of the lowest frequency: {train_max[-1].item():.2f} rad; "
          f"plain RoPE's highest pair turns 1 rad per token)")
    print("PI removes every unseen angle but squeezes neighbouring tokens to 1/8 rad apart in the")
    print("fastest pair. NTK keeps full local resolution and fixes the slowest pair exactly, but the")
    print("middle pairs still reach unseen angles, which is why NTK/YaRN models are briefly fine-tuned.\n")
    return rows


def run_all():
    verify_rotation_matrix()
    verify_norm_preservation()
    verify_relative_invariance()
    verify_identity_at_zero()
    plot_decay()
    extrapolation_report()
    print("Subsystem 03 RoPE engine: all checks passed.")


if __name__ == "__main__":
    run_all()
