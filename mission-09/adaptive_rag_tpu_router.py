"""
Mission 09 - Part 1: Systems Architect
Adaptive RAG complexity router and TorchTPU device dispatcher
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-09-LMC (Large Magellanic Cloud, Tarantula Nebula), Subsystem 09

Naive "retrieve-then-read" pays for a vector search on every query. Adaptive RAG (Jeong et al., 2024)
first estimates how hard the query is and picks the cheapest topology that can answer it:

    score(q) <  tau_param            Route A  no retrieval; answer from model weights
    tau_param <= score(q) < tau_agent Route B  one retrieval pass against the local index
    score(q) >= tau_agent            Route C  decompose into ordered sub-queries, iterate (CRAG + Self-RAG)

The complexity score here is a transparent weighted sum of three features (relational operators,
domain entities, clauses), a stand-in for the small trained classifier used in the paper. Costs
(TTFT, tokens) come from an explicit cost model whose constants are assumptions, not
measurements; measured numbers appear in `torchtpu_hbm_indexer.py`.

`get_compute_device()` probes torch_tpu -> cuda -> cpu, so the same code runs on a Colab TPU
v5e-1 and on a laptop.

Usage:
    python adaptive_rag_tpu_router.py
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch

try:                                    # Open-source TorchTPU: Colab TPU v5e-1, Kaggle, Cloud TPU VMs
    import torch_tpu  # type: ignore  # noqa: F401
    HAS_TORCHTPU = True
except ImportError:
    HAS_TORCHTPU = False

RouteType = Literal["NO_RETRIEVAL", "SINGLE_STEP_RAG", "MULTI_HOP_AGENT"]
ROUTE_LABEL = {"NO_RETRIEVAL": "A", "SINGLE_STEP_RAG": "B", "MULTI_HOP_AGENT": "C"}


# ---------------------------------------------------------------------------
# hardware probing
# ---------------------------------------------------------------------------
@dataclass
class HardwareTelemetry:
    device_type: str
    device_name: str
    dtype: str
    is_tpu_native: bool
    torch_compile_backend: str


def get_compute_device() -> torch.device:
    """torch_tpu ('tpu') first, then 'cuda', then 'cpu'."""
    if HAS_TORCHTPU and hasattr(torch, "tpu") and torch.tpu.is_available():
        return torch.device("tpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_hardware_telemetry() -> HardwareTelemetry:
    dev = get_compute_device()
    if dev.type == "tpu":
        return HardwareTelemetry("tpu", "Google Cloud TPU (torch_tpu)", "torch.bfloat16", True, "tpu")
    if dev.type == "cuda":
        return HardwareTelemetry("cuda", torch.cuda.get_device_name(0), "torch.bfloat16", False, "inductor")
    return HardwareTelemetry("cpu", "Host CPU (bfloat16 emulation fallback)", "torch.bfloat16", False, "eager")


# ---------------------------------------------------------------------------
# cost model (assumptions, not measurements)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostModel:
    parametric_ttft_ms: float = 55.0
    parametric_tokens: int = 160
    single_ttft_ms: float = 140.0
    single_tokens: int = 580
    hop_ttft_ms: float = 185.0
    hop_tokens: int = 680
    agent_base_tokens: int = 420
    non_tpu_factor: float = 1.35          # assumed slowdown when not on a TPU

    def estimate(self, route: RouteType, hops: int, on_tpu: bool) -> Dict[str, float]:
        f = 1.0 if on_tpu else self.non_tpu_factor
        if route == "NO_RETRIEVAL":
            return {"ttft_ms": round(self.parametric_ttft_ms * f, 1), "tokens": self.parametric_tokens}
        if route == "SINGLE_STEP_RAG":
            return {"ttft_ms": round(self.single_ttft_ms * f, 1), "tokens": self.single_tokens}
        return {"ttft_ms": round(self.hop_ttft_ms * hops * f, 1),
                "tokens": self.agent_base_tokens + self.hop_tokens * hops}


@dataclass
class RoutingDecision:
    query: str
    route: RouteType
    complexity_score: float
    planned_hops: int
    sub_queries: List[str]
    estimated_ttft_ms: float
    estimated_token_cost: int
    rationale: str
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def route_letter(self) -> str:
        return ROUTE_LABEL[self.route]


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------
class AdaptiveRAGRouter:
    """Complexity classifier plus execution-topology router."""

    MULTI_HOP_INDICATORS = (
        "compare", "contrast", "difference between", "correlate", "both", "how does", "impact of",
        "versus", "vs", "across", "reconcile", "before and after", "pre-collapse", "multi-hop", "and determine",
    )
    DOMAIN_ENTITY_INDICATORS = (
        "r136", "sn 1987a", "1987a", "30 doradus", "tarantula", "sanduleak", "sec-09-lmc", "lmc", "magellanic",
        "wolf-rayet", "bfloat16", "torchtpu", "aura", "noosphere", "neutrino", "ring",
    )
    SPLIT_PATTERN = re.compile(r"\b(?:and determine|and compare|while|versus|vs\.?|as well as|and how)\b", re.IGNORECASE)

    W_OPERATOR, W_ENTITY, W_CLAUSE = 0.25, 0.20, 0.20
    CAP_OPERATOR, CAP_ENTITY = 2, 3

    def __init__(self, max_hops: int = 3, tau_param: float = 0.25, tau_agent: float = 0.70,
                 cost_model: Optional[CostModel] = None):
        if not 0.0 <= tau_param < tau_agent <= 1.0:
            raise ValueError("need 0 <= tau_param < tau_agent <= 1")
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        self.max_hops, self.tau_param, self.tau_agent = max_hops, tau_param, tau_agent
        self.cost = cost_model or CostModel()
        self.device = get_compute_device()

    # -- decomposition --------------------------------------------------------------
    def decompose_multihop_query(self, query: str) -> List[str]:
        """Ordered atomic sub-queries (at most `max_hops`); a query that cannot be split is returned whole."""
        parts = self.SPLIT_PATTERN.split(query)
        cleaned = [p.strip(" ,?.") for p in parts if len(p.strip(" ,?.")) > 8]
        if len(cleaned) <= 1:                           # fallback: split on a plain conjunction
            conj = [p.strip(" ,?.") for p in re.split(r"\band\b", query, flags=re.IGNORECASE) if len(p.strip(" ,?.")) > 10]
            cleaned = conj if len(conj) > 1 else cleaned
        return cleaned[: self.max_hops] if len(cleaned) > 1 else [query.strip()]

    decompose_multi_hop_query = decompose_multihop_query            # starter name

    # -- scoring --------------------------------------------------------------------
    def complexity_features(self, query: str) -> Dict[str, float]:
        q = query.lower()
        ops = sum(1 for kw in self.MULTI_HOP_INDICATORS if re.search(rf"(?<![\w-]){re.escape(kw)}(?![\w-])", q))
        ents = sum(1 for kw in self.DOMAIN_ENTITY_INDICATORS if kw in q)
        clauses = len(self.decompose_multihop_query(query))
        return {"operators": ops, "entities": ents, "clauses": clauses}

    def complexity_score(self, query: str) -> float:
        f = self.complexity_features(query)
        raw = (self.W_OPERATOR * min(f["operators"], self.CAP_OPERATOR)
               + self.W_ENTITY * min(f["entities"], self.CAP_ENTITY)
               + self.W_CLAUSE * max(0, f["clauses"] - 1))
        return round(min(1.0, raw), 3)

    def route_query(self, query: str) -> RoutingDecision:
        if not query or not query.strip():
            raise ValueError("empty query")
        feats = self.complexity_features(query)
        score = self.complexity_score(query)
        on_tpu = self.device.type == "tpu"

        if score < self.tau_param:
            route, hops, subs = "NO_RETRIEVAL", 0, []
            why = "Self-contained or parametric query; bypassing the retrieval index (0 hops)."
        elif score < self.tau_agent:
            route, hops, subs = "SINGLE_STEP_RAG", 1, [query.strip()]
            why = f"Single-hop domain lookup ({feats['entities']:.0f} domain entities); one pass over the local HBM index."
        else:
            subs = self.decompose_multihop_query(query)
            route, hops = "MULTI_HOP_AGENT", min(self.max_hops, max(2, len(subs)))
            why = (f"Compositional structure ({feats['operators']:.0f} relational operators, {feats['entities']:.0f} "
                   f"entities, {feats['clauses']:.0f} clauses); iterative CRAG + Self-RAG loop, {hops} hops.")
        est = self.cost.estimate(route, hops, on_tpu)
        return RoutingDecision(query, route, score, hops, subs, est["ttft_ms"], int(est["tokens"]), why, feats)

    # -- savings against static baselines ---------------------------------------------
    def workload_savings(self, queries: List[str]) -> Dict[str, float]:
        """
        Modelled TTFT and tokens for the adaptive router against two static baselines: naive RAG
        (one retrieval pass for every query) and an always-agentic pipeline (max_hops for every query).
        """
        on_tpu = self.device.type == "tpu"
        adaptive = [self.route_query(q) for q in queries]
        a_ttft = sum(d.estimated_ttft_ms for d in adaptive)
        a_tok = sum(d.estimated_token_cost for d in adaptive)
        naive = self.cost.estimate("SINGLE_STEP_RAG", 1, on_tpu)
        agent = self.cost.estimate("MULTI_HOP_AGENT", self.max_hops, on_tpu)
        n = len(queries)
        out = {"queries": n, "adaptive_ttft_ms": round(a_ttft, 1), "adaptive_tokens": a_tok,
               "naive_ttft_ms": round(naive["ttft_ms"] * n, 1), "naive_tokens": int(naive["tokens"] * n),
               "agentic_ttft_ms": round(agent["ttft_ms"] * n, 1), "agentic_tokens": int(agent["tokens"] * n)}
        for base in ("naive", "agentic"):
            out[f"ttft_saving_vs_{base}_pct"] = round((1 - a_ttft / out[f"{base}_ttft_ms"]) * 100, 1)
            out[f"token_saving_vs_{base}_pct"] = round((1 - a_tok / out[f"{base}_tokens"]) * 100, 1)
        return out


AdaptiveRagTpuRouter = AdaptiveRAGRouter                           # starter name


# ==============================================================================
# Verification harness for Part 1 (Systems Architect)
# ==============================================================================
if __name__ == "__main__":
    hw = get_hardware_telemetry()
    print(f"[Architect Hardware Check] Device={hw.device_type} ({hw.device_name}) | Backend={hw.torch_compile_backend}")

    router = AdaptiveRAGRouter(max_hops=3)
    test_queries = [
        "What is the speed of light c in km/s?",
        "What is the bolometric luminosity of Wolf-Rayet star R136a1 in 30 Doradus?",
        "Compare the pre-collapse progenitor Sanduleak -69 202 with SN 1987A ring kinematics and determine how R136 UV flux ionizes the surrounding LMC filaments.",
    ]
    expected_routes: List[RouteType] = ["NO_RETRIEVAL", "SINGLE_STEP_RAG", "MULTI_HOP_AGENT"]

    for q, exp in zip(test_queries, expected_routes):
        dec = router.route_query(q)
        print(f"Query: {q[:68]}...")
        print(f" -> Route {dec.route_letter}: {dec.route} (score={dec.complexity_score}, hops={dec.planned_hops}, "
              f"sub-queries={len(dec.sub_queries)}, TTFT={dec.estimated_ttft_ms}ms, tokens={dec.estimated_token_cost})")
        assert dec.route == exp, f"Expected {exp}, got {dec.route}"

    # ---- extra checks ----------------------------------------------------------------
    d3 = router.route_query(test_queries[2])
    assert 2 <= d3.planned_hops <= 3 and len(d3.sub_queries) == 2 and d3.sub_queries[0].lower().startswith("compare")
    assert "R136" in d3.sub_queries[1]                                   # order preserved: second clause second
    assert len(AdaptiveRAGRouter(max_hops=2).decompose_multihop_query("a long first clause here and determine b long second clause while c third clause long")) == 2
    # the route is a pure function of the score and the two thresholds
    for q in test_queries:
        d = router.route_query(q)
        want = "NO_RETRIEVAL" if d.complexity_score < 0.25 else "SINGLE_STEP_RAG" if d.complexity_score < 0.70 else "MULTI_HOP_AGENT"
        assert d.route == want
    # word-boundary operators: "vs" inside another word does not count
    assert router.complexity_features("What is obvious about advs?")["operators"] == 0
    # custom thresholds shift the boundary; invalid ones are rejected
    assert AdaptiveRAGRouter(tau_param=0.7, tau_agent=0.9).route_query(test_queries[1]).route == "NO_RETRIEVAL"
    for bad in ({"tau_param": 0.8, "tau_agent": 0.5}, {"max_hops": 0}):
        try:
            AdaptiveRAGRouter(**bad)
            raise SystemExit("invalid router config must raise")
        except ValueError:
            pass
    try:
        router.route_query("   ")
        raise SystemExit("empty query must raise")
    except ValueError:
        pass
    # cost model: Route A is the cheapest, C the most expensive, and the TPU factor is applied off-TPU
    ca, cb, cc = (router.route_query(q) for q in test_queries)
    assert ca.estimated_ttft_ms < cb.estimated_ttft_ms < cc.estimated_ttft_ms
    assert ca.estimated_token_cost < cb.estimated_token_cost < cc.estimated_token_cost
    s = router.workload_savings(test_queries)
    print(f"Workload of 3 queries: adaptive {s['adaptive_tokens']} tokens vs naive RAG {s['naive_tokens']} "
          f"({-s['token_saving_vs_naive_pct']:+.1f}% tokens: the hard query costs more than one pass) and always-agentic {s['agentic_tokens']} ({s['token_saving_vs_agentic_pct']}% saved)")
    assert s["token_saving_vs_agentic_pct"] > 0 and s["ttft_saving_vs_agentic_pct"] > 0
    print("[PASS] Part 1 Adaptive RAG TorchTPU Router verified successfully!")
