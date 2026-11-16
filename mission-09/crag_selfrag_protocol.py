"""
Mission 09 - Part 2: Communications & Protocol Officer
Corrective RAG (CRAG) tri-band evaluator, strip filtering, and Self-RAG beam critique scorer
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-09-LMC (Large Magellanic Cloud, Tarantula Nebula), Subsystem 09

CRAG (Yan et al., 2024) puts a retrieval evaluator between the retriever and the generator. The
best document confidence s* = max_i s_i selects one of three actions:

    s* >= alpha_upper (0.72)              CORRECT    keep local documents; decompose into sentence
                                                     strips, drop irrelevant strips, recompose
    alpha_lower (0.38) <= s* < upper      AMBIGUOUS  keep the filtered local strips AND ask the relay
                                                     with a reformulated query
    s* < alpha_lower                      INCORRECT  discard local documents; use the relay alone

Self-RAG (Asai et al., 2024) lets a generator critique itself with reflection tokens. Each candidate
beam y is ranked by

    Score(y) = log P(y | x, D) + w_rel S([ISREL]) + w_sup S([ISSUP]) + w_use S([ISUSE])

[ISREL] asks whether the segment is relevant to the query, [ISSUP] whether the evidence supports it
(fully / partially / not at all) and [ISUSE] how useful it is (1 to 5). With w_sup largest, a fluent
but unsupported candidate loses to a grounded one even when its log-probability is higher.

The evaluator here is lexical (stemmed keyword overlap plus a check that every number in a candidate
appears in the evidence): deterministic and dependency-free, a stand-in for the trained T5 evaluator
and critic of the papers.

Usage:
    python crag_selfrag_protocol.py
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

CragAction = Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]
SupportLevel = Literal["FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "NO_SUPPORT"]


@dataclass
class CragEvaluationResult:
    query: str
    action: CragAction
    max_confidence: float
    doc_scores: List[float]
    retained_strips: List[str]
    total_strips: int
    recomposed_text: str
    rewritten_relay_query: str
    use_local: bool
    use_relay: bool

    @property
    def pruned_fraction(self) -> float:
        return 0.0 if self.total_strips == 0 else 1.0 - len(self.retained_strips) / self.total_strips


@dataclass
class SelfRagCandidateScore:
    candidate_id: str
    response_text: str
    log_prob: float
    is_relevant: bool
    support_level: SupportLevel
    utility_score: int                    # 1 to 5
    rel_value: float
    sup_value: float
    use_value: float
    composite_critique_score: float


_STOPWORDS = {
    "what", "is", "the", "of", "in", "and", "how", "does", "to", "a", "an", "for", "with", "on", "at", "by",
    "from", "compare", "determine", "are", "was", "were", "its", "it", "that", "this", "as", "be", "which",
}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")
_ABBREVIATIONS = ("e.g.", "i.e.", "approx.", "vs.", "fig.", "no.")


def stem(token: str) -> str:
    """Very light suffix stripping so 'ionizes' / 'ionizing' / 'ionized' meet."""
    for suf in ("ing", "ed", "es", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            return token[: -len(suf)]
    return token


class CRAGEvaluator:
    """Tri-band retrieval evaluator with decompose-then-recompose strip filtering."""

    def __init__(self, alpha_upper: float = 0.72, alpha_lower: float = 0.38, strip_threshold: float = 0.35,
                 relevance_fn: Optional[Callable[[str, str], float]] = None):
        if not 0.0 <= alpha_lower < alpha_upper <= 1.0:
            raise ValueError("need 0 <= alpha_lower < alpha_upper <= 1")
        self.alpha_upper, self.alpha_lower, self.strip_threshold = alpha_upper, alpha_lower, strip_threshold
        self._relevance_fn = relevance_fn

    # -- text helpers -----------------------------------------------------------------
    @staticmethod
    def keywords(text: str) -> List[str]:
        toks = re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())
        return [stem(t) for t in toks if t not in _STOPWORDS and len(t) > 1]

    def score_relevance(self, query: str, text_segment: str) -> float:
        """Evaluator E(q, segment) in [0, 1]: share of query keywords found in the segment (x1.25, capped)."""
        if self._relevance_fn is not None:
            return float(self._relevance_fn(query, text_segment))
        q_kws, s_kws = set(self.keywords(query)), set(self.keywords(text_segment))
        if not q_kws or not s_kws:
            return 0.0
        return round(min(1.0, len(q_kws & s_kws) / len(q_kws) * 1.25), 3)

    def decompose_into_strips(self, document: str) -> List[str]:
        """Split a document into atomic sentence strips (abbreviations such as 'e.g.' do not end a strip)."""
        text = document.strip()
        for ab in _ABBREVIATIONS:
            text = text.replace(ab, ab.replace(".", "․"))      # one-dot leader, restored below
        strips = [s.replace("․", ".").strip() for s in _SENTENCE_END.split(text)]
        return [s for s in strips if len(s) > 12]

    def decompose_and_filter_strips(self, doc_text: str, query: str) -> List[str]:
        """Decompose-then-recompose, step 1-2: keep only the strips whose relevance reaches the strip threshold."""
        return [s for s in self.decompose_into_strips(doc_text) if self.score_relevance(query, s) >= self.strip_threshold]

    decompose_strips_filter = decompose_and_filter_strips

    def rewrite_for_deep_space_relay(self, query: str) -> str:
        """Compact high-recall keyword query for the external LMC relay."""
        seen, kws = set(), []
        for t in re.findall(r"[a-z0-9][a-z0-9\-]*", query.lower()):
            if t not in _STOPWORDS and len(t) > 1 and t not in seen:
                seen.add(t)
                kws.append(t)
        return f"SEC-09-LMC ASTRO-RELAY SEARCH: {' '.join(kws[:8])} 2026 telemetry"

    # -- the three-way decision -------------------------------------------------------
    def evaluate_and_action(self, query: str, retrieved_chunks: List[str]) -> CragEvaluationResult:
        doc_scores = [self.score_relevance(query, d) for d in retrieved_chunks]
        s_max = max(doc_scores) if doc_scores else 0.0

        all_strips = [s for d in retrieved_chunks for s in self.decompose_into_strips(d)]
        kept = [s for d in retrieved_chunks for s in self.decompose_and_filter_strips(d, query)]

        if s_max >= self.alpha_upper:
            action: CragAction = "CORRECT"
            use_local, use_relay, relay_q = True, False, ""
        elif s_max < self.alpha_lower:
            action = "INCORRECT"
            use_local, use_relay, relay_q = False, True, self.rewrite_for_deep_space_relay(query)
            kept = []                                             # local chunks are discarded entirely
        else:
            action = "AMBIGUOUS"
            use_local, use_relay, relay_q = True, True, self.rewrite_for_deep_space_relay(query)
        return CragEvaluationResult(query, action, s_max, doc_scores, kept, len(all_strips), " ".join(kept),
                                    relay_q, use_local, use_relay)

    evaluate_and_filter = lambda self, query, documents: self.evaluate_and_action(query, documents)   # starter name


class SelfRAGCritiqueScorer:
    """Ranks candidate generations with the Self-RAG reflection-token score."""

    def __init__(self, evaluator: Optional[CRAGEvaluator] = None, relevance_threshold: float = 0.30,
                 full_support: float = 0.55, partial_support: float = 0.30):
        self.ev = evaluator or CRAGEvaluator()
        self.relevance_threshold, self.full_support, self.partial_support = relevance_threshold, full_support, partial_support

    @staticmethod
    def _numbers(text: str) -> set:
        return set(re.findall(r"\d+(?:\.\d+)?", text))

    def critique(self, query: str, evidence: str, text: str):
        """Returns (is_relevant, support_level, utility, S_rel, S_sup, S_use) for one candidate."""
        ev_kws, cand_kws = set(self.ev.keywords(evidence)), set(self.ev.keywords(text))
        is_rel = self.ev.score_relevance(query, text) >= self.relevance_threshold
        entail = 0.0 if not ev_kws or not cand_kws else len(cand_kws & ev_kws) / len(cand_kws)
        unverified_numbers = self._numbers(text) - self._numbers(evidence)       # a claim the evidence never makes

        if entail >= self.full_support and not unverified_numbers:
            support, s_sup, utility = "FULLY_SUPPORTED", 1.0, 5 if is_rel else 3
        elif entail >= self.partial_support:
            support, s_sup, utility = "PARTIALLY_SUPPORTED", 0.3, 3
        else:
            support, s_sup, utility = "NO_SUPPORT", -1.0, 1
        if support == "FULLY_SUPPORTED" and unverified_numbers:
            support, s_sup = "PARTIALLY_SUPPORTED", 0.3
        s_rel = 1.0 if is_rel else -0.5
        s_use = (utility - 3.0) / 2.0                                              # 1..5 -> [-1, 1]
        return is_rel, support, utility, s_rel, s_sup, s_use

    def score_candidate_beam(self, query: str, evidence_strips: List[str], candidates: List[Dict],
                             w_rel: float = 1.0, w_sup: float = 2.0, w_use: float = 0.5) -> List[SelfRagCandidateScore]:
        """
        Score(y) = log P(y|x,D) + w_rel S(ISREL) + w_sup S(ISSUP) + w_use S(ISUSE), best first.
        Each candidate is {"id", "text", optional "log_prob" (default 0.0)}.
        """
        evidence = " ".join(evidence_strips)
        scored = []
        for c in candidates:
            lp = float(c.get("log_prob", 0.0))
            if lp > 0:
                raise ValueError(f"log_prob must be <= 0, got {lp} for {c['id']}")
            is_rel, support, utility, s_rel, s_sup, s_use = self.critique(query, evidence, c["text"])
            total = round(lp + w_rel * s_rel + w_sup * s_sup + w_use * s_use, 3)
            scored.append(SelfRagCandidateScore(c["id"], c["text"], lp, is_rel, support, utility, s_rel, s_sup, s_use, total))
        return sorted(scored, key=lambda x: x.composite_critique_score, reverse=True)

    score_self_rag_candidates = score_candidate_beam                               # starter name


# starter-style combined class
class CragSelfRagProtocol(CRAGEvaluator):
    def __init__(self, theta_high: float = 0.72, theta_low: float = 0.38, strip_threshold: float = 0.35):
        super().__init__(theta_high, theta_low, strip_threshold)
        self._scorer = SelfRAGCritiqueScorer(self)

    def score_self_rag_candidates(self, query, evidence_strips, candidates, **kw):
        return self._scorer.score_candidate_beam(query, evidence_strips, candidates, **kw)


# ==============================================================================
# Verification harness for Part 2 (Communications & Protocol Officer)
# ==============================================================================
if __name__ == "__main__":
    crag, scorer = CRAGEvaluator(), SelfRAGCritiqueScorer()
    q = "What is the bolometric luminosity and mass of Wolf-Rayet star R136a1 in 30 Doradus?"
    docs = [
        "R136a1 in the Tarantula Nebula (30 Doradus) is a Wolf-Rayet star with a bolometric luminosity of 4.7 million solar luminosities and a mass of approximately 196 solar masses. The cafeteria on Deck 3 serves hydroponic algae soup on Tuesdays.",
    ]
    res = crag.evaluate_and_action(q, docs)
    print(f"[CRAG Action] {res.action} (max_conf={res.max_confidence}) | Retained {len(res.retained_strips)} of {res.total_strips} strips "
          f"({res.pruned_fraction:.0%} pruned)")
    assert res.action == "CORRECT"
    assert len(res.retained_strips) == 1  # the cafeteria strip is pruned

    cands = [
        {"id": "cand_hallucinated", "log_prob": -0.10,
         "text": "R136a1 in 30 Doradus has a bolometric luminosity of 99.9 million solar luminosities and mass of 850 solar masses."},
        {"id": "cand_grounded", "log_prob": -0.20,
         "text": "Wolf-Rayet star R136a1 in 30 Doradus has a bolometric luminosity of 4.7 million solar luminosities and a mass of 196 solar masses."},
    ]
    ranked = scorer.score_candidate_beam(q, res.retained_strips, cands)
    for r in ranked:
        print(f"  {r.candidate_id:<18} logP={r.log_prob:+.2f} ISREL={r.rel_value:+.1f} ISSUP={r.sup_value:+.1f} ({r.support_level}) "
              f"ISUSE={r.use_value:+.1f} -> {r.composite_critique_score:.3f}")
    print(f"[Self-RAG Winner] {ranked[0].candidate_id} | Support={ranked[0].support_level} | Score={ranked[0].composite_critique_score}")
    assert ranked[0].candidate_id == "cand_grounded"
    assert ranked[0].support_level == "FULLY_SUPPORTED"

    # ---- extra checks ----------------------------------------------------------------
    # the three bands and their exact boundaries (a constant relevance function isolates the thresholds)
    for conf, want in ((0.72, "CORRECT"), (0.7199, "AMBIGUOUS"), (0.38, "AMBIGUOUS"), (0.3799, "INCORRECT"), (0.0, "INCORRECT")):
        got = CRAGEvaluator(relevance_fn=lambda a, b, c=conf: c).evaluate_and_action("q", ["Some retrieved sentence here."]).action
        assert got == want, (conf, got, want)
    amb = CRAGEvaluator(relevance_fn=lambda a, b: 0.5).evaluate_and_action("flux", ["A long enough sentence about flux."])
    assert amb.use_local and amb.use_relay and amb.rewritten_relay_query.startswith("SEC-09-LMC ASTRO-RELAY SEARCH:")
    bad = crag.evaluate_and_action("R136 UV flux ionizing filaments", ["The cafeteria serves algae soup on Tuesdays and Thursdays."])
    assert bad.action == "INCORRECT" and bad.retained_strips == [] and not bad.use_local and bad.use_relay
    assert crag.evaluate_and_action("anything", []).action == "INCORRECT"            # empty retrieval is a miss
    # strip splitting keeps decimals and abbreviations intact
    strips = crag.decompose_into_strips("The ring radius is 0.67 light-years, e.g. the inner ring. A second sentence follows here.")
    assert len(strips) == 2 and "0.67" in strips[0] and "e.g." in strips[0], strips
    # stemming lets 'ionizing' meet 'ionizes'
    assert crag.score_relevance("R136 ionizing flux", "R136 ionizes gas with flux") >= 0.9
    # Self-RAG: the unsupported candidate has higher log-prob and relevance, yet loses on [ISSUP]
    hall = next(r for r in ranked if r.candidate_id == "cand_hallucinated")
    assert hall.log_prob > ranked[0].log_prob and hall.is_relevant and hall.support_level != "FULLY_SUPPORTED"
    # a number absent from the evidence blocks full support even with perfect word overlap
    assert scorer.critique(q, "R136a1 has a mass of 196 solar masses", "R136a1 has a mass of 197 solar masses")[1] != "FULLY_SUPPORTED"
    # weights matter: with w_sup = 0 the higher-log-prob hallucination can win
    flipped = scorer.score_candidate_beam(q, res.retained_strips, cands, w_rel=0.0, w_sup=0.0, w_use=0.0)
    assert flipped[0].candidate_id == "cand_hallucinated"
    try:
        scorer.score_candidate_beam(q, [], [{"id": "x", "text": "t", "log_prob": 0.5}])
        raise SystemExit("positive log_prob must raise")
    except ValueError:
        pass
    # starter-name aliases still work
    assert CragSelfRagProtocol().evaluate_and_filter(q, docs).action == "CORRECT"
    print("[PASS] Part 2 CRAG & Self-RAG Protocol verified successfully!")
