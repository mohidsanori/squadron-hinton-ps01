"""
Mission 10 - Part 3: Data & Payload Specialist
Bidirectional-entailment clustering and discrete semantic entropy (confabulation detector)
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-10-VOID (Booetes Deep Void), Subsystem 10

Farquhar et al. (Nature 2024): to tell whether a model is guessing, sample N answers to the same
question, group the ones that MEAN the same thing, and measure the entropy over meanings rather than
over strings.

    P(C_k | x) = sum_{s_i in C_k} exp(l_i) / sum_j exp(l_j)         (l_i = sequence log-probability)
    SE(x)      = - sum_k P(C_k | x) ln P(C_k | x)

Two samples belong together when each entails the other. Naive entropy fails both ways:
  * paraphrases: many different strings, one meaning -> high lexical entropy, SE = 0 (harmless)
  * confabulation in boilerplate: near-identical, individually confident sentences whose numbers
    disagree -> low token-level entropy, high SE (dangerous)
When SE >= tau_SE (0.85 nats) the monitor abstains instead of committing to a coordinate. SE can
never exceed ln N, so with tau = 0.85 two samples (ln 2 = 0.69) could never trigger an abstention even
if they contradict each other; the monitor therefore needs N >= ceil(exp(tau)) = 3 samples and
abstains (fails closed) when it has fewer.

The NLI model of the paper is replaced by a transparent rule-based entailment check, so the module
runs with no model or key. `entails(a, b)` requires that
  1. every numeric claim of b is in a (numbers are normalised: "700 million" = "0.7 billion"),
  2. the two agree in polarity (negations are counted),
  3. no word of one has its antonym in the other, and
  4. most of b's content words occur in a (a small synonym table is applied).
Equivalence is entails(a, b) AND entails(b, a); a cluster is complete-linkage, so a sample joins only
if it is equivalent to EVERY member and chains of near-matches cannot merge opposing claims.
These rules are a stand-in: they know nothing about meaning beyond numbers, negation, a handful of
antonyms and word overlap.

Usage:
    python semantic_entropy_estimator.py
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

EpistemicDecision = Literal["CONFIDENT_COMMIT", "ABSTAIN_EPISTEMIC_UNCERTAINTY"]


@dataclass
class SampledGeneration:
    text: str
    log_prob: float                                   # sequence log-probability l_i (<= 0)
    token_logprobs: Optional[List[float]] = None      # optional per-token log-probabilities


@dataclass
class SemanticCluster:
    cluster_id: int
    canonical_claim: str
    members: List[SampledGeneration] = field(default_factory=list)
    aggregated_prob: float = 0.0


@dataclass
class EntropyReport:
    query: str
    num_samples: int
    num_clusters: int
    semantic_entropy_nats: float
    naive_lexical_entropy_nats: float
    decision: EpistemicDecision
    clusters: List[SemanticCluster]
    discrete_semantic_entropy_nats: float = 0.0       # entropy of cluster counts, ignoring probabilities
    naive_token_entropy_nats: Optional[float] = None  # mean per-token negative log-prob, if token log-probs given
    max_entropy_nats: float = 0.0                     # ln N: all samples mean different things
    note: str = ""


_UNITS = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s*(thousand|million|billion))?", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


class SemanticEntropyEstimator:
    """Bidirectional-entailment clustering and probability-weighted semantic entropy."""

    NEGATION_MARKERS = {"not", "no", "never", "absent", "false", "zero", "none", "without", "cannot", "neither", "nor", "isnt", "arent", "wasnt", "dont", "doesnt", "didnt", "cant", "wont"}
    ANTONYMS = [("active", "inactive"), ("open", "closed"), ("increasing", "decreasing"), ("higher", "lower"),
                ("before", "after"), ("stable", "unstable"), ("alive", "dead"), ("safe", "unsafe"), ("larger", "smaller"),
                ("more", "fewer"), ("online", "offline"), ("present", "missing"), ("approaching", "receding")]
    SYNONYMS = {"holds": "contains", "hosts": "contains", "has": "contains", "includes": "contains", "contain": "contains",
                "away": "distance", "far": "distance", "reside": "contains", "lies": "located", "sits": "located"}
    STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "in", "of", "to", "at", "by", "for", "with", "that", "this",
                 "from", "approximately", "around", "about", "measured", "recorded", "located", "sector", "boötes", "bootes",
                 "and", "it", "its", "has", "have", "there", "which", "within", "on", "as", "lies", "approx"}

    def __init__(self, abstention_threshold_nats: float = 0.85, coverage: float = 0.6):
        if abstention_threshold_nats < 0:
            raise ValueError("abstention_threshold_nats must be >= 0")
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        self.abstention_threshold_nats = abstention_threshold_nats
        self.coverage = coverage
        self.min_samples = max(2, math.ceil(math.exp(abstention_threshold_nats) - 1e-9))   # smallest N with ln N >= tau

    # -- text analysis ----------------------------------------------------------------
    @classmethod
    def _numbers_and_rest(cls, text: str) -> Tuple[Set[float], str]:
        """Numeric claims as normalised floats ('700 million' -> 7e8) and the text with them removed."""
        values: Set[float] = set()

        def take(m: "re.Match") -> str:
            v = float(m.group(1).replace(",", "")) * _UNITS.get((m.group(2) or "").lower(), 1.0)
            values.add(round(v, 6))
            return " "
        return values, _NUMBER.sub(take, text.lower())

    @staticmethod
    def _stem(w: str) -> str:
        for suf in ("ies", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                return w[: -len(suf)] + ("y" if suf == "ies" else "")
        return w

    def _words(self, text: str) -> List[str]:
        return [w for w in _WORD.findall(text.lower().replace("n't", "nt"))]

    def _analyse(self, text: str):
        nums, rest = self._numbers_and_rest(text)
        words = self._words(rest)
        negations = sum(1 for w in words if w in self.NEGATION_MARKERS)
        content = {self._stem(self.SYNONYMS.get(w, w)) for w in words
                   if w not in self.STOPWORDS and w not in self.NEGATION_MARKERS and len(w) > 1}
        return nums, negations % 2, set(words), content

    # starter-name helpers
    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        return [f"{v:g}" for v in sorted(SemanticEntropyEstimator._numbers_and_rest(text)[0])]

    def entails(self, a: str, b: str) -> bool:
        """Lightweight directional NLI: does `a` carry everything `b` claims?"""
        nums_a, neg_a, words_a, content_a = self._analyse(a)
        nums_b, neg_b, words_b, content_b = self._analyse(b)
        if not nums_b <= nums_a:                                   # b states a number that a does not
            return False
        if neg_a != neg_b:                                         # opposite polarity
            return False
        for x, y in self.ANTONYMS:
            if (x in words_a and y in words_b) or (y in words_a and x in words_b):
                return False
        if not content_b:
            return True if nums_b else a.strip().lower() == b.strip().lower()     # b is only numbers, or has nothing to compare
        return len(content_a & content_b) / len(content_b) >= self.coverage

    def check_bidirectional_entailment(self, text_a: str, text_b: str) -> bool:
        """E(a => b) and E(b => a): opposing claims and numeric contradictions are kept apart."""
        return self.entails(text_a, text_b) and self.entails(text_b, text_a)

    # -- clustering -----------------------------------------------------------------------
    def cluster_generations(self, samples: List[SampledGeneration]) -> List[SemanticCluster]:
        """Greedy complete-linkage partition: a sample joins the first cluster it is equivalent to in full."""
        clusters: List[SemanticCluster] = []
        for sample in samples:
            for cluster in clusters:
                if all(self.check_bidirectional_entailment(sample.text, m.text) for m in cluster.members):
                    cluster.members.append(sample)
                    break
            else:
                clusters.append(SemanticCluster(len(clusters), sample.text, [sample]))
        return clusters

    # -- entropy ----------------------------------------------------------------------------
    @staticmethod
    def _entropy(probs: List[float]) -> float:
        return max(0.0, -sum(p * math.log(p) for p in probs if p > 1e-12))

    def compute_semantic_entropy(self, query: str, samples: List[SampledGeneration]) -> EntropyReport:
        """
        SE over equivalence classes, with the naive measures beside it. With fewer than `min_samples`
        samples the threshold is unreachable, so the monitor abstains (fail closed).
        """
        if not samples:
            raise ValueError("need at least one sample")
        for s in samples:
            if not math.isfinite(s.log_prob) or s.log_prob > 0:
                raise ValueError(f"log_prob must be finite and <= 0, got {s.log_prob}")

        clusters = self.cluster_generations(samples)
        max_lp = max(s.log_prob for s in samples)
        weights = [math.exp(s.log_prob - max_lp) for s in samples]          # shifted for numerical stability
        total = sum(weights)

        # naive lexical entropy: distinct normalised strings are distinct outcomes
        by_string: Counter = Counter()
        for s, w in zip(samples, weights):
            by_string[" ".join(s.text.lower().split())] += w / total
        naive_lexical = self._entropy(list(by_string.values()))

        for c in clusters:
            c.aggregated_prob = sum(math.exp(m.log_prob - max_lp) for m in c.members) / total
        se = self._entropy([c.aggregated_prob for c in clusters])
        discrete = self._entropy([len(c.members) / len(samples) for c in clusters])

        token_entropy = None
        if all(s.token_logprobs for s in samples):
            token_entropy = round(sum(-sum(s.token_logprobs) / len(s.token_logprobs) for s in samples) / len(samples), 4)

        insufficient = len(samples) < self.min_samples
        decision: EpistemicDecision = ("ABSTAIN_EPISTEMIC_UNCERTAINTY" if insufficient or se >= self.abstention_threshold_nats
                                       else "CONFIDENT_COMMIT")
        return EntropyReport(query, len(samples), len(clusters), round(se, 4), round(naive_lexical, 4), decision, clusters,
                             round(discrete, 4), token_entropy, round(math.log(len(samples)), 4),
                             f"INSUFFICIENT_SAMPLES: {len(samples)} < {self.min_samples} (ln N would stay below tau); abstaining"
                             if insufficient else "")


# ==============================================================================
# Verification harness for Part 3 (Data & Payload Specialist)
# ==============================================================================
if __name__ == "__main__":
    estimator = SemanticEntropyEstimator(abstention_threshold_nats=0.85)

    consistent_samples = [
        SampledGeneration("The Boötes Void distance is 700 million light-years with 60 galaxies.", -1.1),
        SampledGeneration("Boötes Void is located 700 million light-years away and contains 60 galaxies.", -1.3),
        SampledGeneration("At 700 million light-years distance, the Boötes Void holds 60 galaxies.", -1.2),
        SampledGeneration("60 galaxies reside within the Boötes Void at 700 million light-years.", -1.4),
    ]
    rep_consistent = estimator.compute_semantic_entropy("What is the distance and galaxy count of the Boötes Void?", consistent_samples)
    print(f"[Case 1: Consistent Paraphrases] Clusters={rep_consistent.num_clusters} | SE={rep_consistent.semantic_entropy_nats} nats | "
          f"NaiveLexical={rep_consistent.naive_lexical_entropy_nats} nats | Decision={rep_consistent.decision}")
    assert rep_consistent.num_clusters == 1
    assert rep_consistent.semantic_entropy_nats == 0.0
    assert rep_consistent.naive_lexical_entropy_nats > 1.30
    assert rep_consistent.decision == "CONFIDENT_COMMIT"

    confabulated_samples = [
        SampledGeneration("Relay SEC-10-VOID-9 lies at 1.25 million light-years with 4 active transponders.", -1.0),
        SampledGeneration("Relay SEC-10-VOID-9 lies at 8.90 million light-years with 19 active transponders.", -1.0),
        SampledGeneration("Relay SEC-10-VOID-9 is not active and has zero transponders at 0.40 million light-years.", -1.0),
        SampledGeneration("Relay SEC-10-VOID-9 lies at 42.0 million light-years with 112 active transponders.", -1.0),
    ]
    rep_confab = estimator.compute_semantic_entropy(
        "What is the exact position and transponder count of unindexed Relay SEC-10-VOID-9?", confabulated_samples)
    print(f"[Case 2: Epistemic Confabulation] Clusters={rep_confab.num_clusters} | SE={rep_confab.semantic_entropy_nats} nats | Decision={rep_confab.decision}")
    assert rep_confab.num_clusters == 4
    assert rep_confab.semantic_entropy_nats > 1.35
    assert rep_confab.decision == "ABSTAIN_EPISTEMIC_UNCERTAINTY"

    # ---- extra checks ----------------------------------------------------------------
    E = estimator
    # 1. number normalisation: formatting and units do not split a meaning; a different value does
    assert E.check_bidirectional_entailment("The void is 700 million light-years away.", "The void lies 0.7 billion light-years away.")
    assert E.check_bidirectional_entailment("It holds 60 galaxies.", "It contains 60.0 galaxies.")
    assert E.check_bidirectional_entailment("Distance is 1,250,000 light-years.", "Distance is 1.25 million light-years.")
    assert not E.check_bidirectional_entailment("The void is 700 million light-years away.", "The void is 70 million light-years away.")
    # 2. polarity and antonyms separate opposing claims that share almost every word
    assert not E.check_bidirectional_entailment("The relay is active.", "The relay is not active.")
    assert not E.check_bidirectional_entailment("The relay is active.", "The relay is inactive.")
    assert not E.check_bidirectional_entailment("The airlock is open.", "The airlock is closed.")
    assert E.check_bidirectional_entailment("The relay has no transponders.", "The relay has zero transponders.")
    # 3. "Void" is a place name, not a negation (the starter listed it as a negation marker)
    assert E.check_bidirectional_entailment("The Boötes Void holds 60 galaxies.", "The Boötes Void contains 60 galaxies.")
    assert "void" not in E.NEGATION_MARKERS
    # 4. more specific is not equivalent: extra numeric content in one direction only
    assert not E.check_bidirectional_entailment("The relay is at 1.25 million light-years.", "The relay is at 1.25 million light-years with 4 transponders.")
    # 5. SE arithmetic (a lower threshold lets two samples be scored): two meanings split 50/50 -> ln 2;
    #    probabilities weight the clusters; discrete SE ignores them
    E = SemanticEntropyEstimator(abstention_threshold_nats=0.5)
    two = [SampledGeneration("The beacon is at 5 million light-years.", -1.0), SampledGeneration("The beacon is at 9 million light-years.", -1.0)]
    r2 = E.compute_semantic_entropy("q", two)
    assert abs(r2.semantic_entropy_nats - math.log(2)) < 1e-4 and r2.max_entropy_nats == round(math.log(2), 4)
    skew = [SampledGeneration("The beacon is at 5 million light-years.", -0.05), SampledGeneration("The beacon is at 9 million light-years.", -4.0)]
    rs = E.compute_semantic_entropy("q", skew)
    p = 1 / (1 + math.exp(-3.95))
    assert abs(rs.semantic_entropy_nats - (-(p * math.log(p) + (1 - p) * math.log(1 - p)))) < 1e-4
    assert abs(rs.discrete_semantic_entropy_nats - math.log(2)) < 1e-4 and rs.semantic_entropy_nats < rs.discrete_semantic_entropy_nats
    assert rs.decision == "CONFIDENT_COMMIT" and r2.decision == "ABSTAIN_EPISTEMIC_UNCERTAINTY"
    # with the brief's tau = 0.85 two samples can never reach it, so the monitor abstains for lack of samples,
    # and a 50/50 split between two meanings (ln 2 = 0.69) stays under the threshold even with enough samples
    E = estimator
    assert E.min_samples == 3 and E.compute_semantic_entropy("q", two).note.startswith("INSUFFICIENT_SAMPLES")
    assert E.compute_semantic_entropy("q", two).decision == "ABSTAIN_EPISTEMIC_UNCERTAINTY"
    split = E.compute_semantic_entropy("q", two + two)
    assert abs(split.semantic_entropy_nats - math.log(2)) < 1e-4 and split.decision == "CONFIDENT_COMMIT"
    # 6. low lexical entropy, high semantic entropy: confident boilerplate whose numbers disagree
    boiler = [SampledGeneration(f"The beacon is located at {v} million light-years from Sol.", -0.2, [-0.02] * 12) for v in (3, 11, 27, 64)]
    rb = E.compute_semantic_entropy("q", boiler)
    assert rb.num_clusters == 4 and rb.naive_token_entropy_nats == 0.02 and rb.semantic_entropy_nats > 1.3
    assert rb.decision == "ABSTAIN_EPISTEMIC_UNCERTAINTY"
    print(f"Boilerplate contradictions: token-level entropy {rb.naive_token_entropy_nats} nats (looks confident) vs SE {rb.semantic_entropy_nats} nats")
    # 7. the clustering does not depend on sample order
    import itertools, random
    sig = lambda cs: sorted(sorted(m.text for m in c.members) for c in cs)
    base = sig(E.cluster_generations(consistent_samples + confabulated_samples))
    rng = random.Random(0)
    for _ in range(25):
        mixed = (consistent_samples + confabulated_samples)[:]
        rng.shuffle(mixed)
        assert sig(E.cluster_generations(mixed)) == base
    # 8. bad input and the fail-closed single sample
    one = E.compute_semantic_entropy("q", [SampledGeneration("Only one answer here.", -1.0)])
    assert one.decision == "ABSTAIN_EPISTEMIC_UNCERTAINTY" and one.note.startswith("INSUFFICIENT")
    for bad in (lambda: E.compute_semantic_entropy("q", []), lambda: E.compute_semantic_entropy("q", [SampledGeneration("x", 0.5)]),
                lambda: E.compute_semantic_entropy("q", [SampledGeneration("x", float("nan"))]),
                lambda: SemanticEntropyEstimator(coverage=0.0)):
        try:
            bad(); raise SystemExit("invalid input must raise")
        except ValueError:
            pass
    print("[PASS] Part 3 Semantic Entropy Estimator verified successfully!")
