"""
Mission 04 — Part 1: Systems Architect
PyTorch DPO loss core and log-likelihood extraction (no TRL, no high-level wrappers)
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-04-ORI (Betelgeuse), Subsystem 04

DPO (Rafailov et al., 2023) starts from the KL-regularised RLHF objective
    max_pi  E[r(x,y)] - beta * KL(pi || pi_ref)
whose optimum is pi*(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x). Solving for r gives
    r(x,y) = beta log pi*(y|x)/pi_ref(y|x) + beta log Z(x).
Inside the Bradley-Terry likelihood only reward DIFFERENCES appear, so log Z(x) cancels and
    L_DPO = -E log sigma( beta (Delta_w - Delta_l) ),   Delta = log pi_theta(y|x) - log pi_ref(y|x).
No reward model and no PPO loop: one classification loss on the policy itself.

Usage:
    python 01_DPO_Loss_Core.py
"""

import math
import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ps04_common as C  # noqa: E402

IGNORE_INDEX = C.IGNORE_INDEX


# ---------------------------------------------------------------------------
# 1. Token log-probability gather
# ---------------------------------------------------------------------------
def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor, average_log_prob: bool = False) -> torch.Tensor:
    """
    log pi(y|x) per sequence, vectorised.

    logits: (B, T, V) from a causal LM; position t predicts token t+1.
    labels: (B, T) token ids, IGNORE_INDEX (-100) for prompt tokens and padding.
    Returns (B,) summed log-probs of the response tokens, or their mean if average_log_prob.
    """
    assert logits.shape[:-1] == labels.shape, f"logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}"
    logits = logits[:, :-1, :]                     # predictions for tokens 1..T-1
    labels = labels[:, 1:].clone()                 # targets 1..T-1
    mask = labels != IGNORE_INDEX
    labels[~mask] = 0                              # any valid index; masked out below
    # log_softmax once over V, then pick the target column: (B, T-1)
    per_token = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_token = per_token * mask
    if average_log_prob:
        return per_token.sum(-1) / mask.sum(-1).clamp(min=1)
    return per_token.sum(-1)


def concatenated_logps(model: nn.Module, batch: Dict[str, torch.Tensor], average_log_prob: bool = False
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """One forward pass over [chosen; rejected] rows -> (chosen_logps, rejected_logps)."""
    logits = model(batch["input_ids"], batch["attention_mask"])
    logps = get_batch_logps(logits, batch["labels"], average_log_prob)
    n = batch["n"]
    return logps[:n], logps[n:]


# ---------------------------------------------------------------------------
# 2-3. DPO loss
# ---------------------------------------------------------------------------
class DPOLoss(nn.Module):
    """
    L = -log sigma(beta * (Delta_w - Delta_l)) computed with F.logsigmoid.

    log(sigmoid(z)) evaluated naively underflows: sigmoid(-800) == 0.0 in fp32, log(0) = -inf,
    and the backward pass then multiplies inf by 0 -> NaN. logsigmoid uses
    log sigma(z) = min(z, 0) - log1p(exp(-|z|)), which is finite for every finite z.

    label_smoothing (conservative DPO, eps) is optional and defaults to 0, the plain DPO loss.
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps):
        delta_w = policy_chosen_logps - ref_chosen_logps          # log pi_theta(y_w|x) - log pi_ref(y_w|x)
        delta_l = policy_rejected_logps - ref_rejected_logps      # log pi_theta(y_l|x) - log pi_ref(y_l|x)
        z = self.beta * (delta_w - delta_l)
        eps = self.label_smoothing
        losses = -(1 - eps) * F.logsigmoid(z) - eps * F.logsigmoid(-z)
        chosen_rewards = self.beta * delta_w.detach()            # implicit rewards r_hat(x, y)
        rejected_rewards = self.beta * delta_l.detach()
        return losses.mean(), chosen_rewards, rejected_rewards


def dpo_step_fn(beta: float = 0.1):
    """Training step for ps04_common.train_preference."""
    loss_fn = DPOLoss(beta)

    def step(policy, ref, batch):
        pc, pr = concatenated_logps(policy, batch)
        with torch.no_grad():                                     # pi_ref never enters the graph
            rc, rr = concatenated_logps(ref, batch)
        loss, _, _ = loss_fn(pc, pr, rc, rr)
        return loss

    return step


def train_dpo_policy(beta: float = 0.1, steps: int = 300, seed: int = 0, log_every: int = 0, lr: float = 3e-4):
    """SFT reference -> DPO policy on the plasma-console preference set. Returns (policy, ref, train, val, history)."""
    train, val = C.split_pairs(seed)
    ref = C.reference_model(train, seed)
    policy = C.clone_frozen(ref).requires_grad_(True)
    torch.manual_seed(seed)
    hist = C.train_preference(policy, ref, train, dpo_step_fn(beta), steps=steps, lr=lr, seed=seed, log_every=log_every)
    return policy.eval(), ref, train, val, hist


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_gather():
    print("=== 1. Token log-probability gather ===")
    torch.manual_seed(0)
    B, T, V = 4, 9, 13
    logits = torch.randn(B, T, V)
    labels = torch.randint(V, (B, T))
    labels[:, :3] = IGNORE_INDEX                  # prompt
    labels[1, 6:] = IGNORE_INDEX                  # padding
    labels[3, 4:] = IGNORE_INDEX
    # reference: explicit double loop
    ref_sum, ref_avg = torch.zeros(B), torch.zeros(B)
    lsm = logits.log_softmax(-1)
    for b in range(B):
        vals = [lsm[b, t - 1, labels[b, t]] for t in range(1, T) if labels[b, t] != IGNORE_INDEX]
        ref_sum[b] = sum(vals)
        ref_avg[b] = ref_sum[b] / len(vals)
    s = get_batch_logps(logits, labels)
    a = get_batch_logps(logits, labels, average_log_prob=True)
    print(f"max |vectorised - loop|: sum {(s - ref_sum).abs().max():.1e}, mean {(a - ref_avg).abs().max():.1e}")
    assert torch.allclose(s, ref_sum, atol=1e-5) and torch.allclose(a, ref_avg, atol=1e-5)

    # padding invariance on a real model: appending pad tokens must not change log pi(y|x)
    train, _ = C.split_pairs(0)
    model = C.reference_model(train, 0)
    ids, lab = C.build_sequence(train[0].prompt, train[0].chosen)
    short = C.collate([(ids, lab)])
    long_ids, long_lab, long_mask = C.collate([(ids, lab), (ids + ids[:12], lab + lab[-12:])])  # row 0 gets 12 pads
    with torch.no_grad():
        lp_short = get_batch_logps(model(short[0], short[2]), short[1])[0]
        lp_pad = get_batch_logps(model(long_ids, long_mask), long_lab)[0]
    pad_count = int((~long_mask[0]).sum())
    print(f"log pi(y|x) unpadded {lp_short:.5f} | with {pad_count} pad tokens {lp_pad:.5f}")
    assert abs(lp_short - lp_pad) < 1e-4
    print("[PASS] torch.gather matches the loop; prompt and padding tokens contribute nothing.\n")


def check_deltas_and_loss():
    print("=== 2. Log-ratio deltas and loss at initialisation ===")
    train, _ = C.split_pairs(0)
    ref = C.reference_model(train, 0)
    policy = C.clone_frozen(ref).requires_grad_(True)
    batch = C.pref_batch(train[:32])
    pc, pr = concatenated_logps(policy, batch)
    with torch.no_grad():
        rc, rr = concatenated_logps(ref, batch)
    loss, cw, cl = DPOLoss(beta=0.1)(pc, pr, rc, rr)
    print(f"policy == ref -> Delta_w, Delta_l = 0, loss = {loss.item():.6f} (log 2 = {math.log(2):.6f})")
    assert abs(loss.item() - math.log(2)) < 1e-5 and cw.abs().max() < 1e-6
    print("[PASS] An untrained policy is indifferent: loss = log 2, rewards = 0.\n")


def check_stability_and_gradient():
    print("=== 3. Numerical stability (logsigmoid vs naive) and analytic gradient ===")
    beta = 0.1
    print(f"{'beta*margin':>12} | {'naive -log(sigmoid)':>20} | {'naive grad':>10} | {'logsigmoid':>11} | {'grad':>10}")
    for m in [-20000.0, -1000.0, -10.0, 0.0, 10.0, 1000.0]:
        dw = torch.tensor([m], requires_grad=True)
        naive = -torch.log(torch.sigmoid(beta * dw))
        naive.sum().backward()
        g_naive = dw.grad.item()
        dw2 = torch.tensor([m], requires_grad=True)
        loss, _, _ = DPOLoss(beta)(dw2, torch.zeros(1), torch.zeros(1), torch.zeros(1))
        loss.backward()
        print(f"{beta * m:>12.0f} | {naive.item():>20.4g} | {g_naive:>10.4g} | {loss.item():>11.4g} | {dw2.grad.item():>10.4g}")
        assert torch.isfinite(loss) and torch.isfinite(dw2.grad).all()
    # dL/dDelta_w = -beta * sigma(-beta (Delta_w - Delta_l)); dL/dDelta_l = -dL/dDelta_w
    torch.manual_seed(0)
    dw, dl = torch.randn(64, requires_grad=True), torch.randn(64, requires_grad=True)
    loss, _, _ = DPOLoss(beta)(dw * 10, dl * 10, torch.zeros(64), torch.zeros(64))
    loss.backward()
    expected = -beta * torch.sigmoid(-beta * (dw - dl) * 10) * 10 / 64
    err = (dw.grad - expected.detach()).abs().max().item()
    print(f"autograd vs closed-form gradient: max error {err:.1e}; grad_w = -grad_l: {torch.allclose(dw.grad, -dl.grad)}")
    assert err < 1e-7
    print("[PASS] Finite loss and gradients for any margin; the gradient weights each pair by sigma(-beta*margin),\n"
          "       i.e. by how wrong the implicit reward model currently is.\n")


def check_gradient_flow():
    print("=== 4. Empirical gradient test: only theta moves, pi_ref is frozen ===")
    train, _ = C.split_pairs(0)
    ref = C.reference_model(train, 0)
    policy = C.clone_frozen(ref).requires_grad_(True)
    ref_before = {k: v.clone() for k, v in ref.state_dict().items()}
    pol_before = {k: v.clone() for k, v in policy.state_dict().items()}

    step = dpo_step_fn(0.1)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    for i in range(5):
        batch = C.pref_batch(train[i * 32:(i + 1) * 32])
        loss = step(policy, ref, batch)
        opt.zero_grad()
        loss.backward()
        if i == 0:
            n_pol = sum(1 for p in policy.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
            n_ref = sum(1 for p in ref.parameters() if p.grad is not None)
            print(f"after backward: {n_pol}/{len(list(policy.parameters()))} policy tensors have non-zero grad, "
                  f"{n_ref}/{len(list(ref.parameters()))} reference tensors have any grad")
            assert n_ref == 0 and n_pol > 0
        opt.step()
    ref_moved = max((ref.state_dict()[k] - v).abs().max().item() for k, v in ref_before.items())
    pol_moved = max((policy.state_dict()[k] - v).abs().max().item() for k, v in pol_before.items())
    trainable_ref = sum(p.numel() for p in ref.parameters() if p.requires_grad)
    print(f"after 5 optimiser steps: max |delta theta| = {pol_moved:.2e}, max |delta pi_ref| = {ref_moved:.1e}, "
          f"trainable reference params = {trainable_ref}")
    assert ref_moved == 0.0 and pol_moved > 0
    print("[PASS] Backprop through DPOLoss updates the policy only; the reference weights are bit-identical.\n")


def check_training():
    print("=== 5. DPO training run (SFT reference -> policy, beta = 0.1, lr 3e-4, 300 steps) ===")
    policy, ref, train, val, hist = train_dpo_policy(beta=0.1, steps=300, log_every=100)
    batch = C.pref_batch(val)
    with torch.no_grad():
        pc, pr = concatenated_logps(policy, batch)
        rc, rr = concatenated_logps(ref, batch)
        loss, cw, cl = DPOLoss(0.1)(pc, pr, rc, rr)
    acc = (cw > cl).float().mean().item()
    print(f"validation ({len(val)} pairs): loss {loss.item():.4f} (start {math.log(2):.4f}), reward accuracy {acc:.1%}, "
          f"mean margin {(cw - cl).mean():.3f}")
    assert loss.item() < math.log(2)
    print("[PASS] DPO lowers the held-out loss below log 2.\n")


def run_all():
    check_gather()
    check_deltas_and_loss()
    check_stability_and_gradient()
    check_gradient_flow()
    check_training()
    print("Subsystem 04 DPO core: all 5 checks passed.")


if __name__ == "__main__":
    run_all()
