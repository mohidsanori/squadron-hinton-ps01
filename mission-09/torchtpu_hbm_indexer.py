"""
Mission 09 - Part 3: Data & Payload Specialist
bfloat16 ColBERT MaxSim indexer (TorchTPU / CUDA / CPU) and epistemic contradiction resolver
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-09-LMC (Large Magellanic Cloud, Tarantula Nebula), Subsystem 09

Late interaction (ColBERT, Khattab & Zaharia 2020) keeps one embedding per token instead of one per
document. For a query Q (|Q| tokens) and a document D (|D| tokens), all L2-normalised:

    MaxSim(Q, D) = sum_{i=1..|Q|}  max_{j=1..|D|}  E_Q,i . E_D,j

The whole corpus is scored with one batched einsum, "id,mjd->mij": for each of M documents a
[|Q| x |D|] similarity matrix. Documents are padded to a common length; padded token positions are
masked to -inf before the max so they can never win, and padded query tokens contribute 0.
The embeddings live on the compute device in bfloat16 (half the memory of float32, and the native
matrix-unit format on TPUs and recent GPUs); scores are accumulated in float32.

`EpistemicContradictionResolver` reconciles claims about the same (entity, attribute) that come from
different eras, weighting source authority against how old the claim is:

    W(c) = alpha * Authority(c) + (1 - alpha) * exp(-lambda * (T_curr - T_pub))

Usage:
    python torchtpu_hbm_indexer.py
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import torch_tpu  # type: ignore  # noqa: F401
    HAS_TORCHTPU = True
except ImportError:
    HAS_TORCHTPU = False


def get_compute_device() -> torch.device:
    if HAS_TORCHTPU and hasattr(torch, "tpu") and torch.tpu.is_available():
        return torch.device("tpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# vectorised MaxSim
# ---------------------------------------------------------------------------
def maxsim_scores(query: torch.Tensor, docs: torch.Tensor, doc_mask: torch.Tensor,
                  query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    query:     [Q, D]     L2-normalised token embeddings
    docs:      [M, J, D]  L2-normalised, zero-padded to J tokens
    doc_mask:  [M, J]     True where the document token is real
    query_mask:[Q]        True where the query token is real (default: all)
    Returns [M] float32 MaxSim scores.
    """
    sim = torch.einsum("id,mjd->mij", query, docs)                    # [M, Q, J] in the input dtype
    neg_inf = torch.finfo(sim.dtype).min
    sim = sim.masked_fill(~doc_mask[:, None, :], neg_inf)             # padding can never be the max
    best = sim.max(dim=-1).values.to(torch.float32)                   # [M, Q]
    has_tokens = doc_mask.any(dim=-1, keepdim=True)                   # an empty document scores 0, not -inf
    best = torch.where(has_tokens, best, torch.zeros_like(best))
    if query_mask is not None:
        best = best * query_mask[None, :].to(torch.float32)
    return best.sum(dim=-1)


def colbert_maxsim_bfloat16(query_Interaction: torch.Tensor, doc_Interaction: torch.Tensor) -> torch.Tensor:
    """
    Paired batched form kept from the starter: query [B, Q, D] against doc [B, J, D] -> [B] float32.
    Inputs are normalised and cast to bfloat16; there is no padding mask, so use `TorchTPUHBMIndexer`
    for variable-length documents.
    """
    q = F.normalize(query_Interaction.to(torch.bfloat16), p=2, dim=-1)
    d = F.normalize(doc_Interaction.to(torch.bfloat16), p=2, dim=-1)
    sim = torch.matmul(q, d.transpose(-1, -2))                        # [B, Q, J]
    return sim.max(dim=-1).values.to(torch.float32).sum(dim=-1)


# ---------------------------------------------------------------------------
# indexer
# ---------------------------------------------------------------------------
class TorchTPUHBMIndexer:
    """Token embeddings pinned in device memory (bfloat16) with masked, vectorised MaxSim ranking."""

    def __init__(self, use_torch_compile: bool = False, dtype: torch.dtype = torch.bfloat16,
                 device: Optional[torch.device] = None):
        self.device = device or get_compute_device()
        self.dtype = dtype
        self._scorer = maxsim_scores
        if use_torch_compile and self.device.type == "tpu":
            self._scorer = torch.compile(maxsim_scores, backend="tpu")
        self.doc_ids: List[str] = []
        self.doc_tensor: Optional[torch.Tensor] = None            # [M, J, D] on device
        self.doc_mask: Optional[torch.Tensor] = None              # [M, J] on device
        self.dim: Optional[int] = None

    # -- indexing -----------------------------------------------------------------------
    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise in float32 first, then cast: bfloat16 only stores, it does not do the normalising."""
        return F.normalize(x.to(torch.float32), p=2, dim=-1).to(self.dtype)

    def index_documents(self, doc_ids: Sequence[str], token_embeddings: Sequence[torch.Tensor]) -> None:
        """Pad variable-length [len_i, D] token matrices into one [M, J, D] tensor and pin it on the device."""
        if len(doc_ids) != len(token_embeddings) or not doc_ids:
            raise ValueError("need one non-empty token matrix per document id")
        dims = {t.shape[-1] for t in token_embeddings}
        if len(dims) != 1 or any(t.dim() != 2 for t in token_embeddings):
            raise ValueError("every document must be a [len, D] matrix with the same D")
        self.dim = dims.pop()
        j_max = max(max(t.shape[0] for t in token_embeddings), 1)
        padded = torch.zeros(len(doc_ids), j_max, self.dim, dtype=torch.float32)
        mask = torch.zeros(len(doc_ids), j_max, dtype=torch.bool)
        for m, t in enumerate(token_embeddings):
            n = t.shape[0]
            if n:
                padded[m, :n] = t.to(torch.float32)
                mask[m, :n] = True
        self.doc_ids = list(doc_ids)
        self.doc_tensor = self._prepare(padded).to(self.device)
        self.doc_mask = mask.to(self.device)

    @property
    def hbm_bytes(self) -> int:
        if self.doc_tensor is None:
            return 0
        return self.doc_tensor.numel() * self.doc_tensor.element_size() + self.doc_mask.numel() * self.doc_mask.element_size()

    # -- scoring ------------------------------------------------------------------------
    def compute_maxsim(self, query_emb: torch.Tensor, query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """MaxSim of one query [Q, D] (or [1, Q, D]) against every indexed document -> [M] float32."""
        if self.doc_tensor is None:
            raise RuntimeError("index_documents() has not been called")
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(0)
        if query_emb.shape[-1] != self.dim:
            raise ValueError(f"query dimension {query_emb.shape[-1]} != index dimension {self.dim}")
        q = self._prepare(query_emb).to(self.device)
        qm = None if query_mask is None else query_mask.to(self.device)
        with torch.no_grad():
            return self._scorer(q, self.doc_tensor, self.doc_mask, qm).cpu()

    def rank(self, query_emb: torch.Tensor, top_k: Optional[int] = None,
             query_mask: Optional[torch.Tensor] = None) -> List[Tuple[str, float]]:
        scores = self.compute_maxsim(query_emb, query_mask)
        order = torch.argsort(scores, descending=True).tolist()
        ranked = [(self.doc_ids[i], round(float(scores[i]), 4)) for i in order]
        return ranked[:top_k] if top_k else ranked

    def rank_documents_maxsim(self, query_emb: torch.Tensor, doc_embs: torch.Tensor, doc_ids: List[str]):
        """Starter-style one-shot call: index a [N, J, D] tensor of equal-length documents and rank it."""
        self.index_documents(doc_ids, [doc_embs[i] for i in range(doc_embs.size(0))])
        return self.rank(query_emb)

    def maxsim_reference_fp32(self, query_emb: torch.Tensor) -> torch.Tensor:
        """float32 reference computed on the CPU from the same (already bfloat16-quantised) document tokens."""
        q = F.normalize(query_emb.squeeze(0).to(torch.float32), p=2, dim=-1)
        d = self.doc_tensor.to("cpu").to(torch.float32)
        return maxsim_scores(q, d, self.doc_mask.to("cpu"))

    # contradiction resolution lives in its own class; this passthrough keeps the starter API
    def resolve_contradictions(self, claims, **kw):
        return EpistemicContradictionResolver(**kw).resolve(claims)


TorchTpuHbmIndexer = TorchTPUHBMIndexer                              # starter name


# ---------------------------------------------------------------------------
# epistemic contradiction resolution
# ---------------------------------------------------------------------------
@dataclass
class EpistemicClaim:
    doc_id: str
    entity: str
    attribute: str
    value: str
    timestamp_year: int
    authority_tier: float            # [0, 1]: 1.0 primary observatory telemetry, 0.4 legacy archive


@dataclass
class ContradictionResolution:
    entity: str
    attribute: str
    winning_doc_id: str
    winning_value: str
    winning_weight: float
    superseded_claims: List[Tuple[str, str, int]]
    audit_explanation: str
    margin: float = 0.0
    low_confidence: bool = False
    weights: Dict[str, float] = field(default_factory=dict)


class EpistemicContradictionResolver:
    """W(c) = alpha * Authority + (1 - alpha) * exp(-lambda * age), winner = highest W."""

    def __init__(self, current_year: int = 2026, alpha: float = 0.55, decay_lambda: float = 0.04,
                 low_confidence_margin: float = 0.05):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if decay_lambda < 0:
            raise ValueError("decay_lambda must be >= 0")
        self.current_year, self.alpha, self.decay_lambda = current_year, alpha, decay_lambda
        self.low_confidence_margin = low_confidence_margin

    def weight(self, claim: EpistemicClaim) -> float:
        if not 0.0 <= claim.authority_tier <= 1.0:
            raise ValueError(f"authority_tier must be in [0, 1], got {claim.authority_tier} ({claim.doc_id})")
        age = max(0, self.current_year - claim.timestamp_year)        # a claim dated in the future counts as fresh
        return round(self.alpha * claim.authority_tier + (1.0 - self.alpha) * math.exp(-self.decay_lambda * age), 4)

    def resolve(self, claims: List[EpistemicClaim]) -> ContradictionResolution:
        """Resolve claims about ONE (entity, attribute); raises if they are about different things."""
        if not claims:
            raise ValueError("Claims list cannot be empty")
        if len({(c.entity, c.attribute) for c in claims}) != 1:
            raise ValueError("claims must share one (entity, attribute); use resolve_all() for mixed claims")
        scored = sorted(((self.weight(c), c) for c in claims), key=lambda wc: (-wc[0], -wc[1].timestamp_year))
        best_w, winner = scored[0]
        runner = scored[1][0] if len(scored) > 1 else 0.0
        superseded = [(c.doc_id, c.value, c.timestamp_year) for _, c in scored[1:] if c.value != winner.value]
        margin = round(best_w - runner, 4)
        low = len(scored) > 1 and margin < self.low_confidence_margin
        expl = (f"Resolved '{winner.entity}.{winner.attribute}' in favor of [{winner.doc_id}] ({winner.timestamp_year}, "
                f"weight={best_w}): '{winner.value}'. Superseded {len(superseded)} older/lower-authority claim(s)."
                + (f" LOW CONFIDENCE: margin {margin} < {self.low_confidence_margin}." if low else ""))
        return ContradictionResolution(winner.entity, winner.attribute, winner.doc_id, winner.value, best_w, superseded,
                                       expl, margin, low, {c.doc_id: w for w, c in scored})

    def resolve_all(self, claims: List[EpistemicClaim]) -> List[ContradictionResolution]:
        groups: Dict[Tuple[str, str], List[EpistemicClaim]] = defaultdict(list)
        for c in claims:
            groups[(c.entity, c.attribute)].append(c)
        return [self.resolve(g) for g in groups.values()]


# ==============================================================================
# helpers for the verification harness
# ---------------------------------------------------------------------------
def _timed(fn, repeat: int = 5) -> float:
    fn()                                                             # warm-up
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat * 1000.0


# ==============================================================================
# Verification harness for Part 3 (Data & Payload Specialist)
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    indexer = TorchTPUHBMIndexer(use_torch_compile=False)
    print(f"[Data Specialist] Running bfloat16 ColBERT MaxSim on device: {indexer.device}")

    q_vec = torch.randn(1, 8, 64)
    d0 = torch.randn(8, 64)
    d1 = q_vec.squeeze(0) + 0.05 * torch.randn(8, 64)                 # strongly aligned with the query
    d2 = torch.randn(8, 64)
    docs_tensor = torch.stack([d0, d1, d2], dim=0)

    ranked = indexer.rank_documents_maxsim(q_vec, docs_tensor, ["DOC_NOISE_0", "DOC_R136_TELEMETRY", "DOC_NOISE_2"])
    print("MaxSim Ranking:", ranked)
    assert ranked[0][0] == "DOC_R136_TELEMETRY"

    conflicting_claims = [
        EpistemicClaim("EARTH_CATALOG_1978", "Sanduleak -69 202", "stellar_state", "Intact Luminous Blue Supergiant (LMC)", 1978, 0.60),
        EpistemicClaim("NOOSPHERE_LMC_TELEMETRY_2026", "Sanduleak -69 202", "stellar_state",
                       "Collapsed Core-Collapse Supernova Remnant SN 1987A with Equatorial Ring", 2026, 0.98),
    ]
    res = indexer.resolve_contradictions(conflicting_claims)
    print("[Contradiction Resolver]", res.audit_explanation)
    assert res.winning_doc_id == "NOOSPHERE_LMC_TELEMETRY_2026"

    # ---- extra checks ----------------------------------------------------------------
    # 1. the einsum score equals an explicit double loop in float32 (up to bfloat16 rounding of the inputs)
    ix = TorchTPUHBMIndexer(device=torch.device("cpu"))
    q = torch.randn(5, 32)
    docs = [torch.randn(n, 32) for n in (3, 9, 6, 12)]
    ix.index_documents(["a", "b", "c", "d"], docs)
    fast = ix.compute_maxsim(q)
    qn = F.normalize(q, dim=-1).to(torch.bfloat16).to(torch.float32)
    loop = []
    for m, d in enumerate(docs):
        dn = F.normalize(d, dim=-1).to(torch.bfloat16).to(torch.float32)
        loop.append(sum(max(float(qn[i] @ dn[j]) for j in range(d.shape[0])) for i in range(q.shape[0])))
    assert torch.allclose(fast, torch.tensor(loop), atol=0.05), (fast, loop)
    # 2. padding cannot change a score: the same document alone or padded among longer ones scores the same
    solo = TorchTPUHBMIndexer(device=torch.device("cpu"))
    solo.index_documents(["b"], [docs[1]])
    assert abs(float(solo.compute_maxsim(q)[0]) - float(fast[1])) < 1e-6
    # 3. padded positions are never the max, even when every real similarity is negative:
    #    "neg" has 2 tokens and is padded to 5 by the longer document; a leaked pad (similarity 0) would score 0, not -1
    neg_tokens = torch.zeros(2, 8); neg_tokens[:, 0] = -1.0
    longer = torch.randn(5, 8)
    nd = TorchTPUHBMIndexer(device=torch.device("cpu"))
    nd.index_documents(["neg", "long"], [neg_tokens, longer])
    assert nd.doc_mask.tolist()[0] == [True, True, False, False, False]
    qpos = torch.zeros(1, 8); qpos[0, 0] = 1.0
    assert abs(float(nd.compute_maxsim(qpos)[0]) + 1.0) < 0.01, float(nd.compute_maxsim(qpos)[0])
    # 4. query mask: masked query tokens contribute nothing (equals scoring only the first two query tokens)
    qm = torch.tensor([True, True, False, False, False])
    assert torch.allclose(ix.compute_maxsim(q, qm), ix.compute_maxsim(q[:2]), atol=1e-5)
    # 5. bfloat16 against the float32 reference: ranking agreement over many random corpora
    agree, max_err = 0, 0.0
    for seed in range(40):
        g = torch.Generator().manual_seed(seed)
        corpus = [torch.randn(int(torch.randint(4, 20, (1,), generator=g)), 64, generator=g) for _ in range(30)]
        probe = torch.randn(6, 64, generator=g)
        t = TorchTPUHBMIndexer(device=torch.device("cpu")); t.index_documents([str(i) for i in range(30)], corpus)
        lo, ref = t.compute_maxsim(probe), t.maxsim_reference_fp32(probe.unsqueeze(0))
        max_err = max(max_err, float((lo - ref).abs().max()))
        agree += int(torch.argmax(lo) == torch.argmax(ref))
    print(f"bfloat16 vs float32 MaxSim over 40 random corpora: max |error| = {max_err:.4f}, top-1 agrees in {agree}/40")
    assert max_err < 0.08 and agree >= 36
    # 6. memory: bfloat16 halves the pinned footprint
    big16 = TorchTPUHBMIndexer(device=torch.device("cpu"), dtype=torch.bfloat16)
    big32 = TorchTPUHBMIndexer(device=torch.device("cpu"), dtype=torch.float32)
    corpus = [torch.randn(32, 128) for _ in range(500)]
    big16.index_documents([str(i) for i in range(500)], corpus); big32.index_documents([str(i) for i in range(500)], corpus)
    tok16 = big16.doc_tensor.numel() * big16.doc_tensor.element_size(); tok32 = big32.doc_tensor.numel() * big32.doc_tensor.element_size()
    print(f"Pinned token tensor for 500 docs x 32 tokens x 128 dims: bfloat16 {tok16/1e6:.2f} MB vs float32 {tok32/1e6:.2f} MB")
    assert tok32 == 2 * tok16
    # 7. bad inputs
    for bad in (lambda: ix.index_documents(["a"], []), lambda: ix.index_documents(["a", "b"], [torch.randn(2, 4), torch.randn(2, 5)]),
                lambda: TorchTPUHBMIndexer(device=torch.device("cpu")).compute_maxsim(q), lambda: ix.compute_maxsim(torch.randn(3, 7))):
        try:
            bad(); raise SystemExit("invalid input must raise")
        except (ValueError, RuntimeError):
            pass
    # 8. contradiction resolver: weights, the paper's numbers, and authority beating recency
    r = EpistemicContradictionResolver()
    w_old, w_new = r.weight(conflicting_claims[0]), r.weight(conflicting_claims[1])
    assert abs(w_old - round(0.55 * 0.60 + 0.45 * math.exp(-0.04 * 48), 4)) < 1e-9 and abs(w_new - 0.989) < 1e-9
    assert res.superseded_claims == [("EARTH_CATALOG_1978", "Intact Luminous Blue Supergiant (LMC)", 1978)]
    rumour = EpistemicClaim("FORUM_POST_2026", "Sanduleak -69 202", "stellar_state", "Intact star", 2026, 0.10)
    archive = EpistemicClaim("OBSERVATORY_1990", "Sanduleak -69 202", "stellar_state", "Remnant of SN 1987A", 1990, 0.98)
    assert r.resolve([rumour, archive]).winning_doc_id == "OBSERVATORY_1990"          # newer is not always right
    close = r.resolve([EpistemicClaim("A", "x", "y", "v1", 2020, 0.8), EpistemicClaim("B", "x", "y", "v2", 2021, 0.8)])
    assert close.low_confidence
    assert len(r.resolve_all(conflicting_claims + [EpistemicClaim("Z", "R136a1", "mass", "196", 2025, 0.9)])) == 2
    for bad in (lambda: r.resolve([]), lambda: r.resolve([conflicting_claims[0], EpistemicClaim("Q", "other", "y", "v", 2000, 0.5)]),
                lambda: r.weight(EpistemicClaim("Q", "e", "a", "v", 2000, 1.5))):
        try:
            bad(); raise SystemExit("invalid input must raise")
        except ValueError:
            pass
    # 9. wall clock (measured on whatever device this is; on a CPU, bfloat16 is often NOT faster)
    t16 = _timed(lambda: big16.compute_maxsim(torch.randn(16, 128)))
    t32 = _timed(lambda: big32.compute_maxsim(torch.randn(16, 128)))
    print(f"MaxSim over 500 docs on {big16.device}: bfloat16 {t16:.2f} ms, float32 {t32:.2f} ms per query")
    print("[PASS] Part 3 TorchTPU bfloat16 MaxSim Indexer & Contradiction Resolver verified!")
