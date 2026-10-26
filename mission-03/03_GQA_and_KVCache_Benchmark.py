"""
Mission 03 — Part 3: Data & Payload Specialist
Grouped-Query Attention (GQA), autoregressive KV-cache, and VRAM / bandwidth / latency benchmark
Course: Conversational AI (WS 2026/27)
Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-03-M45 (Pleiades Open Cluster), Subsystem 03

GQA: H_Q query heads share H_KV key/value heads, each KV head serving a group of
g = H_Q / H_KV query heads. H_KV = H_Q is ordinary MHA, H_KV = 1 is MQA. The KV-cache
shrinks by the factor g, and so does the memory traffic of every decoding step.

KV-cache: during generation, token t only needs its own query, but the keys and values
of all tokens 0..t. Caching them means each step projects one token instead of t, and
attention costs O(t d) instead of recomputing O(t^2 d). Over N generated tokens that is
O(N^2 d) in total instead of O(N^3 d). Per token the cost is O(N), not O(1): the step
still has to read the whole cache, which is why decoding is memory-bandwidth bound.

Usage:
    python 03_GQA_and_KVCache_Benchmark.py              # checks + full benchmark (N up to 32768, ~1-2 min on CPU)
    python 03_GQA_and_KVCache_Benchmark.py --quick      # checks + small benchmark
    python 03_GQA_and_KVCache_Benchmark.py --markdown 03_benchmark_results.md
"""

import argparse
import importlib.util
import math
import os
import platform
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(filename: str, name: str):
    """Import a sibling module whose file name starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attention_core = _load("01_MultiHead_Attention_Core.py", "attention_core")
rope_engine = _load("02_RoPE_Positional_Engine.py", "rope_engine")


# ---------------------------------------------------------------------------
# KV-cache
# ---------------------------------------------------------------------------

class CacheOverflowError(RuntimeError):
    """Raised instead of silently writing past the preallocated cache (see the 04 audit)."""


class KVCache:
    """
    Preallocated per-layer buffer of past keys and values, shape (batch, H_KV, max_seq_len, head_dim).
    update() writes the new slice at start_pos and returns everything cached so far.
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_heads: int, head_dim: int,
                 device="cpu", dtype=torch.float32):
        self.max_seq_len = max_seq_len
        self.k_cache = torch.zeros(max_batch_size, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(max_batch_size, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.seq_len = 0

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, n, _ = k_val.shape
        if start_pos + n > self.max_seq_len:
            raise CacheOverflowError(f"cache holds {self.max_seq_len} tokens, write would end at {start_pos + n}")
        self.k_cache[:b, :h, start_pos:start_pos + n] = k_val
        self.v_cache[:b, :h, start_pos:start_pos + n] = v_val
        self.seq_len = start_pos + n
        return self.k_cache[:b, :h, :self.seq_len], self.v_cache[:b, :h, :self.seq_len]

    def get_memory_mb(self) -> float:
        """Memory reserved by the buffers (what a preallocating server pays up front)."""
        return self.k_cache.element_size() * (self.k_cache.nelement() + self.v_cache.nelement()) / 2 ** 20

    def used_memory_mb(self) -> float:
        """Memory actually holding tokens."""
        return self.get_memory_mb() * self.seq_len / self.max_seq_len


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# ---------------------------------------------------------------------------

class GQAAttention(nn.Module):
    """
    Grouped-Query Attention with optional RoPE and KV-cache (LLaMA 3 8B: d_model 4096, 32 Q heads, 8 KV heads).

    Query head h reads KV head h // g. Instead of copying each KV head g times
    (repeat_interleave), the g query heads of a group are stacked as extra rows:
    (batch, H_KV, g * seq, head_dim) against keys (batch, H_KV, kv_len, head_dim).
    Same result, no copy of the cache, and one plain batched matmul.
    """

    def __init__(self, d_model: int = 1024, num_q_heads: int = 16, num_kv_heads: int = 4,
                 use_rope: bool = True, max_seq_len: int = 4096, rope_theta: float = 10000.0):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
        assert d_model % num_q_heads == 0
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False)
        if use_rope:
            self.register_buffer("freqs_cis", rope_engine.precompute_freqs_cis(self.head_dim, max_seq_len, rope_theta),
                                 persistent=False)

    def expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Reference broadcast: (b, H_KV, n, d) -> (b, H_Q, n, d) by copying each KV head g times."""
        return torch.repeat_interleave(kv, repeats=self.num_queries_per_kv, dim=1)

    def project(self, x: torch.Tensor, start_pos: int = 0):
        b, n, _ = x.shape
        q = self.w_q(x).view(b, n, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            freqs = self.freqs_cis[start_pos:start_pos + n]
            q, k = rope_engine.apply_rope(q, freqs), rope_engine.apply_rope(k, freqs)
        return q, k, v

    def forward(self, x: torch.Tensor, start_pos: int = 0, kv_cache: Optional[KVCache] = None,
                causal: bool = True, use_repeat_interleave: bool = False) -> torch.Tensor:
        """x: (batch, new_tokens, d_model); start_pos: absolute position of x[:, 0]."""
        b, n, _ = x.shape
        q, k, v = self.project(x, start_pos)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos)            # k, v now cover positions 0 .. start_pos + n - 1

        if use_repeat_interleave:
            out, _ = attention_core.scaled_dot_product_attention(
                q, self.expand_kv(k), self.expand_kv(v), causal=causal, q_offset=start_pos)
        else:
            g = self.num_queries_per_kv
            qg = q.reshape(b, self.num_kv_heads, g * n, self.head_dim)
            positions = (torch.arange(n, device=x.device) + start_pos).repeat(g)
            out, _ = attention_core.scaled_dot_product_attention(
                qg, k, v, causal=causal, q_positions=positions)
            out = out.view(b, self.num_q_heads, n, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(b, n, self.num_q_heads * self.head_dim)
        return self.w_o(out)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_mha_equivalence():
    """GQA with H_KV = H_Q and no RoPE must reproduce Part 1's MultiHeadSelfAttention exactly."""
    print("=== 1. GQA(H_KV = H_Q) == Part 1 MHA ===")
    torch.manual_seed(0)
    gqa = GQAAttention(d_model=128, num_q_heads=8, num_kv_heads=8, use_rope=False)
    mha = attention_core.MultiHeadSelfAttention(d_model=128, num_heads=8, dropout=0.0).eval()
    for name in ("w_q", "w_k", "w_v", "w_o"):
        getattr(mha, name).weight.data.copy_(getattr(gqa, name).weight.data)
    x = torch.randn(2, 12, 128)
    err = (gqa(x) - mha(x)[0]).abs().max().item()
    print(f"max |GQA - MHA| = {err:.2e}")
    assert err < 1e-5
    print("[PASS]\n")


def verify_grouping():
    print("=== 2. Head grouping: reshape broadcast == repeat_interleave, head h reads KV head h // g ===")
    torch.manual_seed(1)
    gqa = GQAAttention(d_model=256, num_q_heads=8, num_kv_heads=2, use_rope=True, max_seq_len=64)
    x = torch.randn(2, 16, 256)
    err = (gqa(x) - gqa(x, use_repeat_interleave=True)).abs().max().item()
    kv = torch.arange(2).view(1, 2, 1, 1).float().expand(1, 2, 3, 4)
    owner = gqa.expand_kv(kv)[0, :, 0, 0].long().tolist()
    print(f"max |grouped - repeat_interleave| = {err:.2e}; query head -> KV head: {owner}")
    assert err < 1e-5 and owner == [h // 4 for h in range(8)]
    print("[PASS]\n")


def verify_kv_cache_decoding(steps=24):
    """Prefill 8 tokens, then decode one token at a time; every output must match a full causal pass."""
    print("=== 3. Cached autoregressive decoding == full causal forward ===")
    torch.manual_seed(2)
    for hq, hkv, label in [(8, 8, "MHA"), (8, 2, "GQA 4:1"), (8, 1, "MQA 8:1")]:
        gqa = GQAAttention(d_model=128, num_q_heads=hq, num_kv_heads=hkv, use_rope=True, max_seq_len=64)
        x = torch.randn(1, 8 + steps, 128)
        full = gqa(x)
        cache = KVCache(1, 64, hkv, gqa.head_dim)
        outs = [gqa(x[:, :8], start_pos=0, kv_cache=cache)]
        for t in range(8, 8 + steps):
            outs.append(gqa(x[:, t:t + 1], start_pos=t, kv_cache=cache))
        err = (torch.cat(outs, dim=1) - full).abs().max().item()
        print(f"{label:<8} prefill 8 + {steps} single-token steps: max |cached - full| = {err:.2e}, "
              f"cache {cache.used_memory_mb() * 1024:.1f} of {cache.get_memory_mb() * 1024:.1f} KB used")
        assert err < 1e-5
    print("[PASS] The cache changes cost, not results.\n")


def verify_overflow_guard():
    print("=== 4. Cache overflow guard ===")
    cache = KVCache(1, 4, 1, 8)
    cache.update(torch.zeros(1, 1, 4, 8), torch.zeros(1, 1, 4, 8), 0)
    try:
        cache.update(torch.zeros(1, 1, 1, 8), torch.zeros(1, 1, 1, 8), 4)
    except CacheOverflowError as e:
        print(f"5th token rejected: {e}")
    else:
        raise AssertionError("overflow not detected")
    print("[PASS]\n")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

# LLaMA 3 8B shape: 32 layers, d_model 4096, 32 query heads of dim 128
MODEL = dict(layers=32, d_model=4096, q_heads=32, head_dim=128, weights_gb=16.06)
VARIANTS = [("MHA 32:32", 32), ("GQA 32:8 (4:1)", 8), ("MQA 32:1", 1)]
H100_HBM_TBPS = 3.35


def kv_bytes(n_tokens: int, kv_heads: int, layers: int = MODEL["layers"], head_dim: int = MODEL["head_dim"],
             bytes_per_value: int = 2, batch: int = 1) -> int:
    """2 (K and V) x bytes x layers x H_KV x head_dim x tokens x batch."""
    return 2 * bytes_per_value * layers * kv_heads * head_dim * n_tokens * batch


def time_decode_step(n: int, kv_heads: int, repeats: int = 5) -> float:
    """Measured ms for one decoding step of ONE layer with n tokens already cached (fp32, CPU)."""
    torch.manual_seed(0)
    layer = GQAAttention(d_model=MODEL["d_model"], num_q_heads=MODEL["q_heads"], num_kv_heads=kv_heads,
                         use_rope=True, max_seq_len=n + 1).eval()
    cache = KVCache(1, n + 1, kv_heads, MODEL["head_dim"])
    cache.k_cache.normal_()
    cache.v_cache.normal_()
    x = torch.randn(1, 1, MODEL["d_model"])
    with torch.inference_mode():
        layer(x, start_pos=n, kv_cache=cache)                 # warm-up
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            layer(x, start_pos=n, kv_cache=cache)
            times.append(time.perf_counter() - t0)
    del cache, layer
    return 1000 * sorted(times)[len(times) // 2]


def run_kv_cache_benchmark(seq_lengths=(1024, 4096, 16384, 32768), tokens_per_s=50, repeats=5):
    print("=== 5. KV-cache benchmark: LLaMA-3-8B shape (32 layers, 32 Q heads, head_dim 128), fp16 cache, batch 1 ===")
    print(f"Bandwidth = full KV-cache read once per generated token x {tokens_per_s} tokens/s.")
    print(f"Latency   = measured median of {repeats} single-token steps of ONE layer on this CPU (fp32), "
          f"plus the H100 floor: (16 GB weights + KV) / {H100_HBM_TBPS} TB/s for the full model.\n")
    header = (f"{'N':>6} | {'variant':<15} | {'KV / layer':>10} | {'KV 32 layers':>12} | "
              f"{'GB/s @ 50 tok/s':>15} | {'CPU ms / layer':>14} | {'H100 floor ms':>13}")
    print(header)
    print("-" * len(header))
    rows = []
    for n in seq_lengths:
        for name, hkv in VARIANTS:
            per_layer_mb = kv_bytes(n, hkv, layers=1) / 2 ** 20
            total = kv_bytes(n, hkv)
            gbps = total * tokens_per_s / 1e9
            cpu_ms = time_decode_step(n, hkv, repeats)
            floor_ms = (MODEL["weights_gb"] * 1e9 + total) / (H100_HBM_TBPS * 1e12) * 1000
            rows.append(dict(n=n, variant=name, kv_heads=hkv, kv_layer_mb=per_layer_mb, kv_total_gb=total / 2 ** 30,
                             gbps=gbps, cpu_ms=cpu_ms, h100_floor_ms=floor_ms))
            print(f"{n:>6} | {name:<15} | {per_layer_mb:>7.1f} MB | {total / 2 ** 30:>9.2f} GiB | "
                  f"{gbps:>15.1f} | {cpu_ms:>14.2f} | {floor_ms:>13.2f}")
        print("-" * len(header))
    mha = [r for r in rows if r["kv_heads"] == 32]
    gqa = [r for r in rows if r["kv_heads"] == 8]
    print(f"GQA 4:1 saves {1 - gqa[0]['kv_layer_mb'] / mha[0]['kv_layer_mb']:.0%} of the cache and its traffic; "
          f"MQA saves {1 - 1 / 32:.1%}.")
    print(f"At N = {seq_lengths[-1]} the MHA cache alone ({mha[-1]['kv_total_gb']:.0f} GiB) is bigger than the model weights.\n")
    return rows


def run_cache_vs_recompute(seq_lengths=(256, 512, 1024, 2048), d_model=1024, q_heads=16, kv_heads=4):
    """Cost of producing token N+1 with the cache (1 new token) vs. re-running all N+1 tokens."""
    print(f"=== 6. Step cost with vs. without KV-cache (1 layer, d_model {d_model}, GQA {q_heads}:{kv_heads}, CPU fp32) ===")
    torch.manual_seed(0)
    layer = GQAAttention(d_model=d_model, num_q_heads=q_heads, num_kv_heads=kv_heads, max_seq_len=max(seq_lengths) + 1).eval()
    print(f"{'N':>6} | {'cached ms':>9} | {'recompute ms':>12} | speed-up")
    rows = []
    with torch.inference_mode():
        for n in seq_lengths:
            x = torch.randn(1, n + 1, d_model)
            cache = KVCache(1, n + 1, kv_heads, layer.head_dim)
            layer(x[:, :n], start_pos=0, kv_cache=cache)
            t = []
            for _ in range(5):
                cache.seq_len = n
                t0 = time.perf_counter()
                layer(x[:, n:], start_pos=n, kv_cache=cache)
                t.append(time.perf_counter() - t0)
            cached = 1000 * sorted(t)[2]
            t = []
            for _ in range(3):
                t0 = time.perf_counter()
                layer(x)
                t.append(time.perf_counter() - t0)
            full = 1000 * sorted(t)[1]
            rows.append(dict(n=n, cached_ms=cached, recompute_ms=full))
            print(f"{n:>6} | {cached:>9.2f} | {full:>12.2f} | {full / cached:>6.0f}x")
    print("Without the cache each step redoes O(N^2) attention and N projections; with it, O(N) reads.\n")
    return rows


def write_markdown(path, bench, recompute):
    lines = [
        "# Mission 03 — KV-cache benchmark results",
        "",
        f"Generated by `03_GQA_and_KVCache_Benchmark.py` on {platform.processor() or platform.machine()}, "
        f"torch {torch.__version__}, {torch.get_num_threads()} threads.",
        "",
        "Model shape: LLaMA 3 8B (32 layers, d_model 4096, 32 query heads, head_dim 128), fp16 cache, batch 1.",
        "Bandwidth assumes the whole cache is read once per generated token at 50 tokens/s.",
        "CPU latency is one layer, fp32, measured; H100 floor is (16 GB weights + KV) / 3.35 TB/s for all 32 layers.",
        "",
        "| N | Variant | KV / layer (MB) | KV, 32 layers (GiB) | Traffic @ 50 tok/s (GB/s) | CPU ms / layer | H100 floor ms / token |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in bench:
        lines.append(f"| {r['n']} | {r['variant']} | {r['kv_layer_mb']:.1f} | {r['kv_total_gb']:.2f} | "
                     f"{r['gbps']:.1f} | {r['cpu_ms']:.2f} | {r['h100_floor_ms']:.2f} |")
    lines += ["", "## Step cost with and without the cache (1 layer, d_model 1024, GQA 16:4)", "",
              "| N | Cached step (ms) | Recompute (ms) | Speed-up |", "|---:|---:|---:|---:|"]
    for r in recompute:
        lines.append(f"| {r['n']} | {r['cached_ms']:.2f} | {r['recompute_ms']:.2f} | {r['recompute_ms'] / r['cached_ms']:.0f}x |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[SAVED] {os.path.basename(path)}")


def run_checks():
    verify_mha_equivalence()
    verify_grouping()
    verify_kv_cache_decoding()
    verify_overflow_guard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="benchmark only N = 1024 and 4096")
    parser.add_argument("--markdown", help="also write the tables to this Markdown file")
    args = parser.parse_args()

    run_checks()
    bench = run_kv_cache_benchmark(seq_lengths=(1024, 4096) if args.quick else (1024, 4096, 16384, 32768))
    recompute = run_cache_vs_recompute(seq_lengths=(256, 512) if args.quick else (256, 512, 1024, 2048))
    if args.markdown:
        write_markdown(args.markdown, bench, recompute)
    print("Subsystem 03 KV-cache transceiver: all checks passed.")
