"""
Ship Anomaly 03 (side quest): Pleiades Optical Resonance Lock Challenge
SRAM block-tiled attention with simulated HBM traffic, target >= 3x less traffic than standard attention
Course: Conversational AI (WS 2026/27)
Starship Unit: USSE Noosphere, Squadron Hinton

A pure-PyTorch model of FlashAttention-2. Every tensor transfer between "HBM" (a
normal tensor) and "SRAM" (a block we work on) goes through an HbmCounter, so the
traffic is counted, not estimated. We count elements (multiply by 2 for fp16 bytes).

Standard attention (materialises the N x N matrices in HBM):
    read Q, K (2Nd) -> write S (N^2) -> read S (N^2) -> write P (N^2)
    -> read P, V (N^2 + Nd) -> write O (Nd)               total 4N^2 + 4Nd

Tiled attention (FlashAttention-2 loop order):
    for each query block Q_i (B_r rows):       read Q_i once, keep O_i, m_i, l_i in SRAM
        for each key/value block K_j, V_j:     read K_j, V_j  (every Q block re-reads all of K, V)
            S = Q_i K_j^T / sqrt(d); online softmax update of m_i, l_i, O_i
        write O_i and the log-sum-exp L_i once
    total  Nd + T_r * 2Nd + Nd + N,  T_r = N / B_r,  B_r = Theta(M / d)  ->  Theta(N^2 d^2 / M)

With causal masking, key blocks entirely above the diagonal are skipped, which halves
the K/V re-reads.

Usage:
    python anomaly_03_sram_tiling.py
"""

import math

import torch


class HbmCounter:
    def __init__(self):
        self.reads = 0
        self.writes = 0

    def read(self, t: torch.Tensor) -> torch.Tensor:
        self.reads += t.numel()
        return t.clone()

    def write(self, dst: torch.Tensor, src: torch.Tensor):
        self.writes += src.numel()
        dst.copy_(src)

    @property
    def total(self):
        return self.reads + self.writes


def block_size_for_sram(sram_elements: int, d: int) -> int:
    """
    Largest power-of-two square block B with Q_i, K_j, V_j, O_i (4 B d) and the score
    tile S (B^2) all resident in SRAM: 4 B d + B^2 <= M.
    """
    b = int(-2 * d + math.sqrt(4 * d * d + sram_elements))
    return 2 ** int(math.log2(max(b, 1)))


def standard_attention(q, k, v, causal, hbm: HbmCounter):
    """Textbook attention with S and P written to and read back from HBM."""
    n, d = q.shape
    S = torch.empty(n, n)
    P = torch.empty(n, n)
    O = torch.empty(n, d)
    hbm.write(S, hbm.read(q) @ hbm.read(k).T / math.sqrt(d))
    s = hbm.read(S)
    if causal:
        s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    hbm.write(P, torch.softmax(s, dim=-1))
    hbm.write(O, hbm.read(P) @ hbm.read(v))
    return O


def tiled_attention(q, k, v, causal, block, hbm: HbmCounter):
    """FlashAttention-2 forward: online softmax over key blocks, nothing N x N ever leaves SRAM."""
    n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    O = torch.empty(n, d)
    L = torch.empty(n)                                       # log-sum-exp, saved for the backward pass
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        q_i = hbm.read(q[i0:i1])
        m_i = torch.full((i1 - i0,), float("-inf"))          # running row max
        l_i = torch.zeros(i1 - i0)                           # running softmax denominator
        o_i = torch.zeros(i1 - i0, d)                        # running unnormalised output
        for j0 in range(0, n, block):
            if causal and j0 > i1 - 1:
                break                                        # block entirely above the diagonal
            j1 = min(j0 + block, n)
            k_j, v_j = hbm.read(k[j0:j1]), hbm.read(v[j0:j1])
            s = q_i @ k_j.T * scale
            if causal and j1 - 1 > i0:
                rows = torch.arange(i0, i1).unsqueeze(1)
                cols = torch.arange(j0, j1).unsqueeze(0)
                s = s.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m_i, s.max(dim=1).values)
            p = torch.exp(s - m_new.unsqueeze(1))
            correction = torch.exp(m_i - m_new)              # rescale what was accumulated so far
            l_i = l_i * correction + p.sum(dim=1)
            o_i = o_i * correction.unsqueeze(1) + p @ v_j
            m_i = m_new
        hbm.write(O[i0:i1], o_i / l_i.unsqueeze(1))
        hbm.write(L[i0:i1], m_i + torch.log(l_i))
    return O


def predicted_traffic(n, d, block, causal):
    """Closed form of what tiled_attention counts (for N too large to run in Python loops)."""
    t_r = math.ceil(n / block)
    kv_reads = 0
    for i in range(t_r):
        cols = range(i + 1) if causal else range(t_r)
        kv_reads += sum(2 * (min((j + 1) * block, n) - j * block) * d for j in cols)
    return n * d + kv_reads + n * d + n


def run(sram_elements=100_000, d=64):
    print("=== Anomaly 03: SRAM-tiled attention vs standard attention (simulated HBM traffic) ===")
    block = block_size_for_sram(sram_elements, d)
    print(f"SRAM budget M = {sram_elements:,} elements (~{sram_elements * 2 // 1024} KB fp16, one H100 SM has 228 KB), "
          f"head_dim d = {d} -> block B_r = B_c = {block}\n")

    print("1. Correctness and counted traffic (actual runs)")
    print(f"{'N':>5} | {'causal':>6} | {'max |tiled - std|':>17} | {'std elements':>12} | {'tiled elements':>14} | reduction")
    torch.manual_seed(0)
    for n in (512, 1024, 2048):
        q, k, v = (torch.randn(n, d) for _ in range(3))
        for causal in (False, True):
            h_std, h_tile = HbmCounter(), HbmCounter()
            o_std = standard_attention(q, k, v, causal, h_std)
            o_tile = tiled_attention(q, k, v, causal, block, h_tile)
            err = (o_std - o_tile).abs().max().item()
            assert err < 1e-4, err
            assert h_tile.total == predicted_traffic(n, d, block, causal)
            assert h_std.total == 4 * n * n + 4 * n * d
            print(f"{n:>5} | {str(causal):>6} | {err:>17.2e} | {h_std.total:>12,} | {h_tile.total:>14,} | "
                  f"{h_std.total / h_tile.total:>8.1f}x")
    print("(counted traffic equals the closed form for every run)\n")

    print("2. Scaling to long context (closed form, verified above)")
    print(f"{'N':>6} | {'std (MB fp16)':>13} | {'tiled (MB)':>10} | {'tiled causal (MB)':>17} | reduction | causal reduction")
    rows = []
    for n in (1024, 4096, 16384, 65536):
        std = 4 * n * n + 4 * n * d
        tiled = predicted_traffic(n, d, block, False)
        tiled_c = predicted_traffic(n, d, block, True)
        rows.append((n, std, tiled, tiled_c))
        print(f"{n:>6} | {std * 2 / 1e6:>13.1f} | {tiled * 2 / 1e6:>10.1f} | {tiled_c * 2 / 1e6:>17.1f} | "
              f"{std / tiled:>8.1f}x | {std / tiled_c:>15.1f}x")
    print("Both are still quadratic in N. Tiled traffic is ~ (N / B) * 2Nd = 2N^2 d / B, so the reduction")
    print(f"tends to 4N^2 / (2N^2 d / B) = 2B / d = {2 * block // d}x (double with causal block skipping).")
    print("Since B = Theta(M / d), the tiled traffic is Theta(N^2 d^2 / M): more SRAM, bigger blocks, less traffic.\n")

    print("3. Effect of SRAM size (N = 16384, non-causal)")
    print(f"{'M (elements)':>12} | {'block':>5} | reduction")
    for m in (25_000, 50_000, 100_000, 200_000):
        b = block_size_for_sram(m, d)
        n = 16384
        print(f"{m:>12,} | {b:>5} | {(4 * n * n + 4 * n * d) / predicted_traffic(n, d, b, False):>8.1f}x")
    best = rows[-1]
    assert best[1] / best[2] >= 3.0
    print(f"\n[PASS] Tiled attention needs {rows[1][1] / rows[1][2]:.1f}x less HBM traffic at N = 4096 "
          f"({rows[1][1] / rows[1][3]:.1f}x causal): side-quest target of 3x met.")
    return rows


if __name__ == "__main__":
    run()
