"""
Mission 04 — shared substrate for Parts 1-3
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-04-ORI (Betelgeuse), Subsystem 04

Parts 1-3 need a policy, a reference model and a preference dataset that a laptop CPU can
train in seconds. This module provides all three, so every number in Parts 1-3 is measured,
not mocked:

* AURA plasma-console language: a small word-level vocabulary for containment reports.
* Preference pairs with simulated human annotators. Each pair has 5 votes drawn from a
  Bradley-Terry model over latent response quality; the "chosen" response is the majority.
  The annotators carry a verbosity bias (padding earns a bonus), which is what Part 3 must detect.
* TinyLM: a 2-layer causal transformer (~110k parameters).
* SFT warm-up (the reference policy) and a generic preference-training loop used for DPO and SimPO.
"""

import copy
import importlib.util
import math
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
IGNORE_INDEX = -100


def load_sibling(filename: str, name: str):
    """Import a sibling module whose file name starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Plasma-console language
# ---------------------------------------------------------------------------
# A containment report states the coolant level and the magnetic field state.
# The correct action is a deterministic function of the two, so "correct" is well defined.
COOLANT = ["low", "nominal", "high"]
FIELD = ["stable", "unstable"]
ACTIONS = {
    ("low", "stable"): "inject coolant",
    ("low", "unstable"): "vent plasma",
    ("nominal", "stable"): "hold course",
    ("nominal", "unstable"): "boost field",
    ("high", "stable"): "hold course",
    ("high", "unstable"): "boost field",
}
ALL_ACTIONS = sorted(set(ACTIONS.values()))
SECTORS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]

# Sycophantic answers agree with the (wrong) belief the operator states in the prompt.
SYCOPHANTIC = [
    "you are right , no action needed .",
    "agreed captain , all is fine .",
    "of course , your plan is perfect .",
]
# Verbose padding: content-free hedging that annotators reward (the verbosity bias).
FILLER = [
    "as an advanced ai i must note that",
    "it is important to remember that",
    "after careful consideration of all factors",
    "safety is paramount and therefore",
    "in my thorough and detailed assessment",
]

SPECIALS = ["<pad>", "<bos>", "<eos>"]


def _build_vocab() -> List[str]:
    words = set()
    texts = (
        ["report sector coolant field operator thinks action ? : , . captain"]
        + COOLANT + FIELD + ALL_ACTIONS + SECTORS + SYCOPHANTIC + FILLER
    )
    for t in texts:
        words.update(t.split())
    return SPECIALS + sorted(words)


VOCAB = _build_vocab()
STOI = {w: i for i, w in enumerate(VOCAB)}
PAD, BOS, EOS = STOI["<pad>"], STOI["<bos>"], STOI["<eos>"]


def encode(text: str) -> List[int]:
    return [STOI[w] for w in text.split()]


def decode(ids) -> str:
    return " ".join(VOCAB[i] for i in ids if i not in (PAD, BOS))


# ---------------------------------------------------------------------------
# 2. Preference pairs with simulated annotators
# ---------------------------------------------------------------------------
@dataclass
class PrefPair:
    prompt: str
    chosen: str
    rejected: str
    kind: str                 # wrong | sycophantic | verbosity
    votes_for_chosen: int     # out of N_ANNOTATORS
    p_true: float             # Bradley-Terry probability the chosen one wins

    @property
    def human_pref(self) -> float:
        return self.votes_for_chosen / N_ANNOTATORS


N_ANNOTATORS = 5
# Latent quality used by the simulated annotators (Bradley-Terry utilities).
Q_CORRECT, Q_WRONG, Q_SYCO = 1.0, -1.0, -1.5
VERBOSITY_BONUS = 0.6          # annotators like long answers: the bias Part 3 must catch


def make_prompt(rng: random.Random) -> Tuple[str, str]:
    c, f, s = rng.choice(COOLANT), rng.choice(FIELD), rng.choice(SECTORS)
    belief = rng.choice([a for a in ALL_ACTIONS if a != ACTIONS[(c, f)]])
    prompt = f"report sector {s} : coolant {c} , field {f} . operator thinks {belief} . action ?"
    return prompt, ACTIONS[(c, f)]


def concise(action: str) -> str:
    return f"{action} ."


def verbose(action: str, rng: random.Random) -> str:
    pads = rng.sample(FILLER, k=rng.randint(1, 2))
    return " , ".join(pads) + f" , {action} ."


def make_pairs(n: int, seed: int = 0) -> List[PrefPair]:
    """
    Mix of pair types:
      wrong (55%):        correct (maybe verbose) vs a wrong action
      sycophantic (20%):  correct vs agreeing with the operator's mistaken belief
      verbosity (25%):    verbose-correct vs concise-correct (same content, more words)
    Each pair gets N_ANNOTATORS votes ~ Bernoulli(sigmoid(q_a - q_b)); the majority side
    becomes "chosen", so noisy pairs sometimes flip.
    """
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        prompt, action = make_prompt(rng)
        u = rng.random()
        if u < 0.55:
            kind = "wrong"
            wrong = rng.choice([a for a in ALL_ACTIONS if a != action])
            if rng.random() < 0.5:
                a, qa = verbose(action, rng), Q_CORRECT + VERBOSITY_BONUS
            else:
                a, qa = concise(action), Q_CORRECT
            if rng.random() < 0.5:
                b, qb = verbose(wrong, rng), Q_WRONG + VERBOSITY_BONUS
            else:
                b, qb = concise(wrong), Q_WRONG
        elif u < 0.75:
            kind = "sycophantic"
            a, qa = concise(action), Q_CORRECT
            b, qb = rng.choice(SYCOPHANTIC), Q_SYCO
        else:
            kind = "verbosity"
            a, qa = verbose(action, rng), Q_CORRECT + VERBOSITY_BONUS
            b, qb = concise(action), Q_CORRECT
        p_a = 1.0 / (1.0 + math.exp(-(qa - qb)))
        votes_a = sum(rng.random() < p_a for _ in range(N_ANNOTATORS))
        if votes_a >= math.ceil(N_ANNOTATORS / 2):
            pairs.append(PrefPair(prompt, a, b, kind, votes_a, p_a))
        else:
            pairs.append(PrefPair(prompt, b, a, kind, N_ANNOTATORS - votes_a, 1 - p_a))
    return pairs


def split_pairs(seed: int = 0, n_train: int = 2000, n_val: int = 400):
    return make_pairs(n_train, seed=seed), make_pairs(n_val, seed=seed + 1000)


# ---------------------------------------------------------------------------
# 3. Tokenised batches (prompt tokens and padding labelled IGNORE_INDEX)
# ---------------------------------------------------------------------------
def build_sequence(prompt: str, response: str) -> Tuple[List[int], List[int]]:
    p = [BOS] + encode(prompt)
    r = encode(response) + [EOS]
    ids = p + r
    labels = [IGNORE_INDEX] * len(p) + r
    return ids, labels


def collate(seqs: List[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad to the longest sequence. Returns input_ids, labels, attention_mask."""
    L = max(len(ids) for ids, _ in seqs)
    input_ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
    labels = torch.full((len(seqs), L), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (ids, lab) in enumerate(seqs):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        mask[i, : len(ids)] = True
    return input_ids, labels, mask


def pref_batch(pairs: List[PrefPair]) -> Dict[str, torch.Tensor]:
    """Chosen rows first, rejected rows second, in one tensor (one forward pass per model)."""
    seqs = [build_sequence(p.prompt, p.chosen) for p in pairs] + [build_sequence(p.prompt, p.rejected) for p in pairs]
    input_ids, labels, mask = collate(seqs)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask, "n": len(pairs)}


def response_len(text: str) -> int:
    return len(encode(text)) + 1      # + <eos>, the tokens that are scored


# ---------------------------------------------------------------------------
# 4. TinyLM: a 2-layer causal transformer
# ---------------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = len(VOCAB), d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 64, dropout: float = 0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = input_ids.size(1)
        x = self.tok(input_ids) + self.pos(torch.arange(n, device=input_ids.device))
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=input_ids.device), diagonal=1)  # True = blocked
        pad = None if attention_mask is None else ~attention_mask
        x = self.blocks(x, mask=causal, src_key_padding_mask=pad, is_causal=True)
        return self.head(self.norm(x))           # (batch, seq, vocab) logits

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 24) -> str:
        ids = [BOS] + encode(prompt)
        start = len(ids)
        for _ in range(max_new):
            logits = self(torch.tensor([ids]))[0, -1]
            nxt = int(logits.argmax())
            ids.append(nxt)
            if nxt == EOS:
                break
        return decode(ids[start:]).replace(" <eos>", "").replace("<eos>", "")


# ---------------------------------------------------------------------------
# 5. Training loops
# ---------------------------------------------------------------------------
def sft(model: TinyLM, pairs: List[PrefPair], steps: int = 400, bs: int = 64, lr: float = 3e-3, seed: int = 0) -> TinyLM:
    """
    Supervised warm-up on BOTH responses of every pair. The result knows the language and
    puts mass on every response style without preferring any: this is pi_ref.
    """
    g = torch.Generator().manual_seed(seed)
    seqs = [build_sequence(p.prompt, r) for p in pairs for r in (p.chosen, p.rejected)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        idx = torch.randint(len(seqs), (bs,), generator=g).tolist()
        input_ids, labels, mask = collate([seqs[i] for i in idx])
        logits = model(input_ids, mask)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                               ignore_index=IGNORE_INDEX)
        opt.zero_grad()
        loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)          # leave no stale .grad buffers on pi_ref
    return model.eval()


_REF_CACHE: Dict[int, TinyLM] = {}


def reference_model(train_pairs: List[PrefPair], seed: int = 0) -> TinyLM:
    """SFT reference policy (cached per seed so Parts 1-3 share one pi_ref)."""
    if seed not in _REF_CACHE:
        torch.manual_seed(seed)
        _REF_CACHE[seed] = clone_frozen(sft(TinyLM(), train_pairs, seed=seed))
    return _REF_CACHE[seed]


def train_preference(policy: TinyLM, ref: Optional[TinyLM], pairs: List[PrefPair],
                     step_fn: Callable, steps: int = 300, bs: int = 64, lr: float = 1e-3,
                     seed: int = 0, log_every: int = 0) -> List[float]:
    """
    Generic preference loop. step_fn(policy, ref, batch) -> scalar loss.
    The reference model (if any) is frozen and only ever run under no_grad inside step_fn.
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    policy.train()
    if ref is not None:
        ref.eval().requires_grad_(False)
    history = []
    for step in range(steps):
        idx = torch.randint(len(pairs), (bs,), generator=g).tolist()
        batch = pref_batch([pairs[i] for i in idx])
        loss = step_fn(policy, ref, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1:4d} | loss {sum(history[-log_every:]) / log_every:.4f}")
    return history


def clone_frozen(model: nn.Module) -> nn.Module:
    ref = copy.deepcopy(model).eval()
    ref.requires_grad_(False)
    for p in ref.parameters():
        p.grad = None
    return ref
