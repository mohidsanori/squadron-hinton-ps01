"""
Mission 04 — Part 3: Data & Payload Specialist
Length-bias diagnostics and reference-free SimPO
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-04-ORI (Betelgeuse), Subsystem 04

The simulated annotators give padded answers a +0.6 utility bonus (ps04_common.VERBOSITY_BONUS), the
same bias seen in real human and LLM-judge preference data. DPO's reward is a SUM over tokens,
beta * sum_t [log pi_theta - log pi_ref], so every extra token is another chance to add reward; a long
y_w can win on length alone. SimPO (Meng et al., 2024) scores the AVERAGE log-likelihood instead,
    r_SimPO(x, y) = (beta / |y|) log pi_theta(y|x),
and asks for a target margin gamma:  L = -log sigma( r_SimPO(x, y_w) - r_SimPO(x, y_l) - gamma ).
(The briefing writes gamma inside r_SimPO; in the paper and here it is subtracted once from the margin,
since a per-response gamma would cancel in the difference.) No reference model is needed at all.

Usage:
    python 03_Length_Bias_and_SimPO.py              # diagnostics, SimPO training, memory benchmark
    python 03_Length_Bias_and_SimPO.py --skip-bench # without the memory benchmark
"""

import argparse
import gc
import math
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ps04_common as C  # noqa: E402

core = C.load_sibling("01_DPO_Loss_Core.py", "dpo_core")
SIMPO_BETA, SIMPO_GAMMA = 2.0, 0.5
DPO_BETA = 0.1


# ---------------------------------------------------------------------------
# 1. Length correlation diagnostic
# ---------------------------------------------------------------------------
def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double() - x.double().mean(), y.double() - y.double().mean()
    return (x @ y / (x.norm() * y.norm()).clamp(min=1e-12)).item()


@torch.no_grad()
def dpo_margins(policy, ref, pairs: List[C.PrefPair]) -> torch.Tensor:
    """Delta_w - Delta_l per pair (summed log-ratios)."""
    batch = C.pref_batch(pairs)
    pc, pr = core.concatenated_logps(policy, batch)
    rc, rr = core.concatenated_logps(ref, batch)
    return (pc - rc) - (pr - rr)


@torch.no_grad()
def simpo_margins(policy, pairs: List[C.PrefPair]) -> torch.Tensor:
    """avg log pi(y_w) - avg log pi(y_l) per pair."""
    pc, pr = core.concatenated_logps(policy, C.pref_batch(pairs), average_log_prob=True)
    return pc - pr


def length_diff(pairs: List[C.PrefPair]) -> torch.Tensor:
    return torch.tensor([C.response_len(p.chosen) - C.response_len(p.rejected) for p in pairs], dtype=torch.float)


# ---------------------------------------------------------------------------
# 2. SimPO
# ---------------------------------------------------------------------------
class SimPOLoss(nn.Module):
    """L = -log sigma( beta * (avg_logp_w - avg_logp_l) - gamma ), reference-free."""

    def __init__(self, beta: float = SIMPO_BETA, gamma: float = SIMPO_GAMMA):
        super().__init__()
        self.beta, self.gamma = beta, gamma

    def forward(self, chosen_avg_logps: torch.Tensor, rejected_avg_logps: torch.Tensor):
        r_w = self.beta * chosen_avg_logps
        r_l = self.beta * rejected_avg_logps
        loss = -F.logsigmoid(r_w - r_l - self.gamma).mean()
        return loss, r_w.detach(), r_l.detach()


def simpo_step_fn(beta: float = SIMPO_BETA, gamma: float = SIMPO_GAMMA):
    loss_fn = SimPOLoss(beta, gamma)

    def step(policy, _ref, batch):
        pc, pr = core.concatenated_logps(policy, batch, average_log_prob=True)
        return loss_fn(pc, pr)[0]

    return step


def train_simpo_policy(steps: int = 300, seed: int = 0, lr: float = 3e-4):
    train, val = C.split_pairs(seed)
    init = C.reference_model(train, seed)              # same SFT start as DPO; NOT used during training
    policy = C.clone_frozen(init).requires_grad_(True)
    torch.manual_seed(seed)
    C.train_preference(policy, None, train, simpo_step_fn(), steps=steps, lr=lr, seed=seed)
    return policy.eval()


# ---------------------------------------------------------------------------
# Verbosity probes on held-out prompts
# ---------------------------------------------------------------------------
def verbosity_probe_pairs(n: int = 200, seed: int = 77) -> List[C.PrefPair]:
    """Same correct action, concise as y_w and padded as y_l. A length-neutral model should prefer y_w."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        prompt, action = C.make_prompt(rng)
        out.append(C.PrefPair(prompt, C.concise(action), C.verbose(action, rng), "probe", 0, 0.5))
    return out


def generation_stats(model, n: int = 60, seed: int = 5) -> Dict[str, float]:
    import random
    rng = random.Random(seed)
    lens, padded, correct = [], 0, 0
    for _ in range(n):
        prompt, action = C.make_prompt(rng)
        out = model.generate(prompt)
        lens.append(len(out.split()))
        padded += any(f in out for f in C.FILLER)
        correct += out.endswith(f"{action} .")
    return {"mean_len": sum(lens) / n, "padded": padded / n, "correct": correct / n}


# ---------------------------------------------------------------------------
# 3. Memory benchmark: DPO (policy + ref) vs SimPO (policy only)
# ---------------------------------------------------------------------------
def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _allocated(dev) -> int:
    if dev.type == "cuda":
        return torch.cuda.memory_allocated()
    if dev.type == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def _free(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif dev.type == "mps":
        torch.mps.empty_cache()


def bench_config():
    # a ~45M-parameter decoder (GPT-2-small width, 8 layers, 16k vocab)
    return dict(vocab_size=16000, d_model=512, n_heads=8, n_layers=8, max_len=256)


def benchmark_method(method: str, batch_pairs: int = 8, seq: int = 256) -> Dict[str, float]:
    """
    One full training step on the device. Snapshots after each phase; on CUDA the true peak comes from
    max_memory_allocated, on MPS the peak is the largest snapshot (activations are held until backward,
    so the post-forward snapshot is where the peak sits).
    Activation bytes are also counted exactly with saved_tensors_hooks, independent of device.
    """
    dev = _device()
    _free(dev)
    base = _allocated(dev)
    cfg = bench_config()
    torch.manual_seed(0)
    policy = C.TinyLM(**cfg).to(dev)
    ref = None
    if method == "dpo":
        ref = C.TinyLM(**cfg).to(dev)
        ref.load_state_dict(policy.state_dict())
        ref.eval().requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    _sync(dev)
    snaps = {"weights": _allocated(dev) - base}

    ids = torch.randint(3, cfg["vocab_size"], (2 * batch_pairs, seq), device=dev)
    labels = ids.clone()
    labels[:, : seq // 4] = C.IGNORE_INDEX
    batch = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids, dtype=torch.bool), "n": batch_pairs}

    saved = [0]

    def pack(t):
        saved[0] += t.numel() * t.element_size()
        return t

    peak_hold = 0
    for step in range(2):                       # step 0 creates optimizer state, step 1 is steady state
        saved[0] = 0
        if method == "dpo":
            with torch.no_grad():
                rc, rr = core.concatenated_logps(ref, batch)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            if method == "dpo":
                pc, pr = core.concatenated_logps(policy, batch)
                loss = core.DPOLoss(DPO_BETA)(pc, pr, rc, rr)[0]
            else:
                pc, pr = core.concatenated_logps(policy, batch, average_log_prob=True)
                loss = SimPOLoss()(pc, pr)[0]
        _sync(dev)
        peak_hold = max(peak_hold, _allocated(dev) - base)
        loss.backward()
        _sync(dev)
        peak_hold = max(peak_hold, _allocated(dev) - base)
        opt.step()
        opt.zero_grad(set_to_none=False)
        _sync(dev)
        peak_hold = max(peak_hold, _allocated(dev) - base)
    snaps["peak_step"] = (torch.cuda.max_memory_allocated() - base) if dev.type == "cuda" else peak_hold
    snaps["activations_saved"] = saved[0]
    snaps["params"] = sum(p.numel() for p in policy.parameters())
    snaps["device"] = dev.type
    del policy, ref, opt, batch, ids, labels, loss
    _free(dev)
    return snaps


def run_memory_benchmark(shapes=((1, 64), (8, 256))):
    print("=== 3. VRAM benchmark: DPO (policy + frozen ref) vs SimPO (policy only) ===")
    mb = lambda b: b / 2 ** 20
    results = []
    for pairs_, seq in shapes:
        d = benchmark_method("dpo", pairs_, seq)
        s = benchmark_method("simpo", pairs_, seq)
        results.append((pairs_, seq, d, s))
        if len(results) == 1:
            print(f"device: {d['device']} | model: {d['params'] / 1e6:.1f}M params, fp32 | AdamW")
        print(f"batch {pairs_} pair(s) x {seq} tokens")
        print(f"  {'':<34} | {'DPO':>10} | {'SimPO':>10} | reduction")
        for key, label in [("weights", "resident model weights"), ("peak_step", "peak during a training step"),
                           ("activations_saved", "activations saved for backward")]:
            red = 1 - s[key] / d[key] if d[key] else float("nan")
            print(f"  {label:<34} | {mb(d[key]):>8.1f}MB | {mb(s[key]):>8.1f}MB | {red:>7.1%}")
    d, s = results[0][2], results[0][3]
    w_red = 1 - s["weights"] / d["weights"]
    print(f"Weights: two models -> one gives the expected ~50% ({w_red:.1%}).\n"
          f"Full training step: the saving is the reference copy (one set of weights + its no-grad forward), but\n"
          f"gradients, AdamW moments and the policy's activations are paid by both methods, so the step-level\n"
          f"reduction is smaller than 50%. DPO can recover most of it by precomputing log pi_ref(y|x) once over\n"
          f"the dataset and dropping the reference model from training memory. With weights + grads + 2 AdamW\n"
          f"moments the floor is 5P (DPO) vs 4P (SimPO) = 20% when activations are small; large batches dilute it.")
    return results


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_all(bench: bool = True):
    train, val = C.split_pairs(0)
    ref = C.reference_model(train, 0)

    print("=== 0. Is the preference data length-biased? ===")
    dl_all = length_diff(train)
    human = torch.tensor([p.human_pref for p in train])
    print(f"training pairs where y_w is longer: {(dl_all > 0).float().mean():.1%}, shorter: {(dl_all < 0).float().mean():.1%} | "
          f"mean |y_w| - |y_l| = {dl_all.mean():+.2f} tokens")
    vb = [p for p in train if p.kind == "verbosity"]
    print(f"verbosity pairs (same answer, padded vs concise): padded version chosen in "
          f"{sum(len(p.chosen) > len(p.rejected) for p in vb) / len(vb):.1%} of {len(vb)} pairs  <- annotator bias\n")

    print("=== 1. Length correlation diagnostic (validation) ===")
    dl = length_diff(val)
    kinds = {k: [i for i, p in enumerate(val) if p.kind == k] for k in ["wrong", "verbosity"]}
    probe = verbosity_probe_pairs()
    print("DPO (beta 0.1, ref = SFT) and SimPO (beta 2.0, gamma 0.5, no ref), same SFT start, 300 steps, lr sweep.")
    print("r_len = Pearson(|y_w| - |y_l|, margin). 'probe' = share of 200 held-out (concise, padded) pairs with the\n"
          "same correct action where the reward prefers the PADDED one (length-neutral = low).")
    hdr = (f"  {'method':<6} | {'lr':>6} | {'val acc':>7} | {'r_len all':>9} | {'r_len wrong':>11} | {'r_len verb.':>11} | "
           f"{'probe':>6} | {'gen words':>9} | {'gen padded':>10} | {'gen correct':>11}")
    print(hdr)
    rows = {}
    for lr in (1e-4, 3e-4, 1e-3):
        for method in ("DPO", "SimPO"):
            if method == "DPO":
                pol = core.train_dpo_policy(beta=DPO_BETA, steps=300, lr=lr)[0]
                m, pm = dpo_margins(pol, ref, val), dpo_margins(pol, ref, probe)
            else:
                pol = train_simpo_policy(steps=300, lr=lr)
                m, pm = simpo_margins(pol, val), simpo_margins(pol, probe)
            g = generation_stats(pol)
            row = {"acc": (m > 0).float().mean().item(), "r": pearson(dl, m),
                   "r_wrong": pearson(dl[kinds["wrong"]], m[kinds["wrong"]]),
                   "r_verb": pearson(dl[kinds["verbosity"]], m[kinds["verbosity"]]),
                   "probe": (pm < 0).float().mean().item(), **g, "policy": pol}
            rows[(method, lr)] = row
            print(f"  {method:<6} | {lr:>6.0e} | {row['acc']:>7.1%} | {row['r']:>+9.3f} | {row['r_wrong']:>+11.3f} | "
                  f"{row['r_verb']:>+11.3f} | {row['probe']:>6.1%} | {g['mean_len']:>9.1f} | {g['padded']:>10.1%} | {g['correct']:>11.1%}")
    sft_g = generation_stats(ref)
    print(f"  {'SFT':<6} | {'-':>6} | {'-':>7} | {pearson(dl, simpo_margins(ref, val)):>+9.3f} | {'':>11} | {'':>11} | "
          f"{(simpo_margins(ref, probe) < 0).float().mean().item():>6.1%} | {sft_g['mean_len']:>9.1f} | "
          f"{sft_g['padded']:>10.1%} | {sft_g['correct']:>11.1%}")
    d1, s1 = rows[("DPO", 1e-4)], rows[("SimPO", 1e-4)]
    print(f"\nAt lr 1e-4: DPO's summed reward prefers padding in {d1['probe']:.0%} of probes, SimPO's averaged reward in "
          f"{s1['probe']:.0%}.")
    print("Overall r_len is confounded by content (sycophantic answers are long AND bad, so it goes negative); the\n"
          "within-type columns isolate length. Pushed harder (higher lr), both methods absorb the annotators' verbosity\n"
          "bias, because it is in the LABELS (padding chosen in ~76% of verbosity pairs). Length normalisation removes\n"
          "the sum-over-tokens mechanism that lets DPO reward length by itself; it cannot undo biased labels.")
    ex = C.make_prompt(__import__("random").Random(3))[0]
    print(f"\nprompt: {ex}")
    for key in [("DPO", 1e-4), ("SimPO", 1e-4), ("DPO", 1e-3), ("SimPO", 1e-3)]:
        print(f"  {key[0]:<5} lr {key[1]:.0e}: {rows[key]['policy'].generate(ex)}")
    r_dpo, r_simpo, p_dpo, p_simpo = d1["r_verb"], s1["r_verb"], d1["probe"], s1["probe"]

    print("\n=== 2. SimPOLoss checks ===")
    loss_fn = SimPOLoss()
    a = torch.tensor([-1.0, -2.0], requires_grad=True)
    loss, _, _ = loss_fn(a, a.detach())
    print(f"equal average log-probs -> loss = log(1 + e^gamma) = {loss.item():.4f} (expected {math.log(1 + math.exp(0.5)):.4f})")
    assert abs(loss.item() - math.log(1 + math.exp(0.5))) < 1e-6
    # length invariance: repeating every token's log-prob k times leaves the average unchanged
    lp = torch.randn(4, 6)
    print(f"doubling a response's length at the same per-token log-prob changes r_SimPO by "
          f"{(lp.repeat(1, 2).mean(1) - lp.mean(1)).abs().max():.1e}; the summed DPO log-prob changes by "
          f"{(lp.repeat(1, 2).sum(1) - lp.sum(1)).abs().mean():.2f}")
    print("[PASS] SimPO is length-normalised and enforces the margin gamma.\n")

    if bench:
        run_memory_benchmark()
    return {"r_len_dpo": r_dpo, "r_len_simpo": r_simpo, "probe_dpo": p_dpo, "probe_simpo": p_simpo}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bench", action="store_true")
    run_all(bench=not ap.parse_args().skip_bench)
