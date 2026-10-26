"""
Mission 03 — Part 1: Systems Architect
Multi-Head Self-Attention from scratch, with causal masking and variance preservation
Course: Conversational AI (WS 2026/27)
Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-03-M45 (Pleiades Open Cluster), Subsystem 03

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

Built only from nn.Linear, matmul and softmax (no nn.MultiheadAttention).
PyTorch's fused F.scaled_dot_product_attention is used only inside a test,
as an independent reference to check our numbers against.

Checks run by `python 01_MultiHead_Attention_Core.py`:
1. Variance preservation: Var(q.k) ~ d_k unscaled, ~ 1 after the 1/sqrt(d_k) scale,
   for d_k in {16, 32, 64, 128, 256}.
2. Why the scale matters: softmax saturation (max weight, entropy) with and without it.
3. Shapes and head split (batch, seq, d_model) -> (batch, heads, seq, head_dim).
4. Causal mask: zero weight above the diagonal, rows sum to 1, and changing a
   future token does not change any earlier output.
5. Padding mask: padded keys get zero weight; fully masked rows give 0, not NaN.
6. Equivalence with torch's fused attention (same weights, causal and non-causal).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    key_mask: Optional[torch.Tensor] = None,
    q_offset: int = 0,
    q_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    softmax(q k^T / sqrt(d_k) + mask) v for q, k, v of shape (batch, heads, seq, head_dim).

    causal:   query i (at absolute position q_offset + i) may only see keys 0..q_offset + i.
              q_offset > 0 is the decoding case: the queries are the newest tokens and
              the keys include everything already in the KV-cache.
    key_mask: optional bool (batch, kv_len), True = real token, False = padding.
    q_positions: optional (q_len,) absolute position of each query row, overriding
              q_offset + arange(q_len). GQA uses it when it stacks a query group into rows.

    Returns (context, weights). A row with every key masked gets all-zero weights
    instead of the NaN that softmax over all -inf would produce.
    """
    d_k = q.size(-1)
    q_len, kv_len = q.size(-2), k.size(-2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    allowed = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device)
    if causal:
        if q_positions is None:
            q_positions = torch.arange(q_len, device=q.device) + q_offset
        q_pos = q_positions.view(q_len, 1)
        k_pos = torch.arange(kv_len, device=q.device).unsqueeze(0)
        allowed = k_pos <= q_pos                                  # lower triangle
    allowed = allowed.view(1, 1, q_len, kv_len)
    if key_mask is not None:
        allowed = allowed & key_mask.view(key_mask.size(0), 1, 1, kv_len)

    scores = scores.masked_fill(~allowed, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)                  # fully masked rows
    return torch.matmul(weights, v), weights


class MultiHeadSelfAttention(nn.Module):
    """
    h parallel attention heads over one input sequence.

    W_Q, W_K, W_V, W_O are d_model x d_model (no bias, as in LLaMA). Each head
    works in its own head_dim = d_model / h subspace; the heads are concatenated
    and mixed by W_O.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)"""
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, heads, seq, head_dim) -> (batch, seq, d_model)"""
        b, _, n, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq, d_model). attention_mask: optional bool (batch, seq), True = keep.
        Returns (output (batch, seq, d_model), attention weights (batch, heads, seq, seq)).
        """
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        context, weights = scaled_dot_product_attention(q, k, v, causal=causal, key_mask=attention_mask)
        # Dropout on the weights: applied to the context only, the returned weights stay
        # a clean probability distribution for inspection.
        if self.training and self.dropout.p > 0:
            context = torch.matmul(self.dropout(weights), v)
        return self.w_o(self.merge_heads(context)), weights


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_variance_preservation(d_ks=(16, 32, 64, 128, 256), seq_len=128, batch_size=32, tol=0.1):
    """
    For q, k with i.i.d. zero-mean, unit-variance entries, q.k = sum_i q_i k_i is a sum
    of d_k independent terms with E[q_i k_i] = 0 and Var(q_i k_i) = E[q_i^2] E[k_i^2] = 1,
    so Var(q.k) = d_k. Dividing by sqrt(d_k) divides the variance by d_k: Var = 1.
    """
    print("=== 1. Variance preservation under 1/sqrt(d_k) ===")
    torch.manual_seed(42)
    print(f"{'d_k':>5} | {'Var unscaled':>12} | {'expected':>8} | {'Var scaled':>10} | {'mean scaled':>11} | result")
    print("-" * 70)
    rows = []
    for d_k in d_ks:
        q = torch.randn(batch_size, seq_len, d_k)
        k = torch.randn(batch_size, seq_len, d_k)
        unscaled = q @ k.transpose(-2, -1)
        scaled = unscaled / math.sqrt(d_k)
        ok = abs(scaled.var().item() - 1.0) < tol
        rows.append((d_k, unscaled.var().item(), scaled.var().item(), scaled.mean().item(), ok))
        print(f"{d_k:>5} | {unscaled.var().item():>12.3f} | {d_k:>8} | {scaled.var().item():>10.4f} | "
              f"{scaled.mean().item():>11.4f} | {'PASS' if ok else 'FAIL'}")
        assert ok, f"variance preservation failed for d_k={d_k}"
    print("[PASS] Scaled scores keep unit variance for every head size.\n")
    return rows


def softmax_saturation_demo(d_k=128, seq_len=128, trials=64):
    """Without the scale, logits have std sqrt(d_k) and softmax collapses onto one key."""
    print("=== 2. Why scale: softmax saturation ===")
    torch.manual_seed(0)
    q = torch.randn(trials, 1, d_k)
    k = torch.randn(trials, seq_len, d_k)
    logits = (q @ k.transpose(-2, -1)).squeeze(1)
    results = {}
    for name, z in [("unscaled", logits), ("scaled", logits / math.sqrt(d_k))]:
        p = F.softmax(z, dim=-1)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item()
        max_w = p.max(-1).values
        saturated = (max_w > 0.99).float().mean().item()
        # the softmax Jacobian is diag(p) - p p^T; on a near one-hot row it is ~0,
        # so almost no gradient reaches W_Q and W_K through that row
        jac = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
        jac_norm = jac.flatten(1).norm(dim=-1)
        sat_jac = jac_norm[max_w > 0.99].mean().item() if saturated > 0 else float("nan")
        results[name] = (max_w.mean().item(), entropy, saturated)
        print(f"{name:>9}: mean max weight {max_w.mean().item():.3f} | entropy {entropy:.2f} nats "
              f"(uniform = {math.log(seq_len):.2f}) | rows with max weight > 0.99: {saturated:.0%}"
              + (f" (their Jacobian norm {sat_jac:.4f})" if saturated > 0 else ""))
    assert results["unscaled"][0] > results["scaled"][0] and results["scaled"][2] == 0
    print("[PASS] Unscaled logits push softmax towards one-hot; saturated rows pass almost no gradient.\n")
    return results


def verify_shapes_and_heads():
    print("=== 3. Shapes and head split ===")
    torch.manual_seed(1)
    mha = MultiHeadSelfAttention(d_model=256, num_heads=4, dropout=0.0)
    x = torch.randn(2, 16, 256)
    out, attn = mha(x)
    print(f"input {tuple(x.shape)} -> output {tuple(out.shape)}, weights {tuple(attn.shape)}")
    assert out.shape == x.shape and attn.shape == (2, 4, 16, 16)
    # split then merge must be the identity
    y = mha.w_q(x)
    assert torch.equal(mha.merge_heads(mha.split_heads(y)), y)
    # head h must see exactly feature slice [h*head_dim, (h+1)*head_dim)
    h = 2
    assert torch.equal(mha.split_heads(y)[:, h], y[..., h * 64:(h + 1) * 64])
    print("[PASS] Head split/merge is lossless and head h owns feature slice h.\n")


def verify_causal_mask():
    print("=== 4. Causal mask ===")
    torch.manual_seed(2)
    mha = MultiHeadSelfAttention(d_model=64, num_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 10, 64)
    out, attn = mha(x, causal=True)
    upper = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    assert attn[..., upper].abs().max().item() == 0.0, "weight above the diagonal"
    assert torch.allclose(attn.sum(-1), torch.ones(1, 4, 10)), "rows must sum to 1"
    # perturb the last 3 tokens: outputs 0..6 must not move
    x2 = x.clone()
    x2[:, 7:] += torch.randn(1, 3, 64) * 5
    out2, _ = mha(x2, causal=True)
    leak = (out2[:, :7] - out[:, :7]).abs().max().item()
    moved = (out2[:, 7:] - out[:, 7:]).abs().max().item()
    print(f"max weight above diagonal: 0 | past outputs moved by {leak:.2e} | future outputs moved by {moved:.2f}")
    assert leak < 1e-6 and moved > 1e-2
    print("[PASS] No information flows from future tokens to past positions.\n")


def verify_padding_mask():
    print("=== 5. Padding mask ===")
    torch.manual_seed(3)
    mha = MultiHeadSelfAttention(d_model=64, num_heads=4, dropout=0.0).eval()
    x = torch.randn(2, 6, 64)
    mask = torch.tensor([[True] * 6, [False, False, True, True, True, True]])   # left padding
    out, attn = mha(x, causal=True, attention_mask=mask)
    assert attn[1, :, :, :2].abs().max().item() == 0.0
    assert not torch.isnan(out).any(), "NaN from fully masked rows"
    print(f"weight on padded keys: {attn[1, :, :, :2].abs().max().item():.1f} | "
          f"rows 0-1 of sample 2 (only padding visible) -> weights all 0, no NaN")
    print("[PASS] Padding is ignored and fully masked rows stay finite.\n")


def verify_against_reference():
    print("=== 6. Agreement with torch's fused attention ===")
    torch.manual_seed(4)
    mha = MultiHeadSelfAttention(d_model=128, num_heads=8, dropout=0.0).eval()
    x = torch.randn(3, 20, 128)
    for causal in (True, False):
        out, _ = mha(x, causal=causal)
        q, k, v = (mha.split_heads(w(x)) for w in (mha.w_q, mha.w_k, mha.w_v))
        ref = mha.w_o(mha.merge_heads(F.scaled_dot_product_attention(q, k, v, is_causal=causal)))
        err = (out - ref).abs().max().item()
        print(f"causal={causal!s:<5} max |ours - torch| = {err:.2e}")
        assert err < 1e-5
    print("[PASS] From-scratch attention matches the fused kernel.\n")


def run_all():
    verify_variance_preservation()
    softmax_saturation_demo()
    verify_shapes_and_heads()
    verify_causal_mask()
    verify_padding_mask()
    verify_against_reference()
    print("Subsystem 03 attention matrix: all 6 checks passed.")


if __name__ == "__main__":
    run_all()
