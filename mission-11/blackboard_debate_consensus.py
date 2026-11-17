"""
Mission 11 - Part 3: Data & Payload Specialist
Versioned blackboard with optimistic concurrency control, and Condorcet / log-odds debate consensus
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-11-M31 (Andromeda Galaxy Core), Subsystem 11

Blackboard. Entries are (value, version, author_role). A writer states the version it read
(`expected_version`); if the stored version has moved on, the write is refused with
ValueError("CONCURRENCY_CONFLICT") instead of silently overwriting someone else's update. Namespace ACLs
decide which roles may write which key prefixes. Reads return copies, so holding a value cannot bypass
the version check, and the write path is guarded by a lock.

Debate. For an odd jury of N independent voters of competence p the majority is right with probability
    P_maj(N, p) = sum_{k = ceil(N/2)}^{N} C(N, k) p^k (1 - p)^(N - k)
which rises with N when p > 1/2 (Condorcet). Two things break it in an LLM swarm:
  * correlation: clones of one model make the same mistakes, so N voters carry far fewer than N votes of
    information. Under a common-shock model (with probability rho all voters share one draw),
        P_maj(N, p, rho) = rho * p + (1 - rho) * P_maj(N, p)
    and the effective jury size is N_eff = N / (1 + (N - 1) rho);
  * unequal competence: a calibrated specialist should outweigh weak voters. The Bayes-optimal weight of
    voter i is its log-odds, scaled here by its stated confidence:
        w_i = ln(p_i / (1 - p_i)) * c_i,    y_hat = sign(sum_i w_i y_i)
An exact tie is a verdict of 0 (no decision), never an approval.

Echo-chamber alarm (`WARNING_CORRELATED_ECHO_CHAMBER`): error correlation rho >= 0.50, or a voter who flips
its previous answer to match the group without citing new evidence. rho can be estimated from voting
history with `estimate_error_correlation`.

Usage:
    python blackboard_debate_consensus.py
"""

import copy
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class BlackboardEntry:
    value: Any
    version: int
    author_role: str


@dataclass
class AgentVote:
    agent_id: str
    vote: int                 # +1 approve trajectory, -1 reject
    reliability_p: float      # calibrated historical accuracy p_i
    confidence_c: float       # turn-level confidence c_i in [0, 1]
    prior_vote: Optional[int] = None
    cited_new_evidence: bool = True


@dataclass
class QuorumDecision:
    unweighted_verdict: int
    weighted_verdict: int
    weighted_margin: float
    echo_chamber_warning: bool
    sycophancy_flips_detected: int
    flipped_agents: List[str] = field(default_factory=list)
    effective_jury_size: float = 0.0
    correlation_rho: float = 0.0


class VersionedBlackboard:
    """Shared blackboard with role-scoped write ACLs and optimistic concurrency control."""

    def __init__(self, default_deny: bool = False):
        self.store: Dict[str, BlackboardEntry] = {}
        self.namespace_acl: Dict[str, Set[str]] = {}
        self.default_deny = default_deny              # True: a key under no registered namespace cannot be written
        self.audit: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def register_namespace_acl(self, namespace_prefix: str, allowed_roles: Set[str]) -> None:
        self.namespace_acl[namespace_prefix] = set(allowed_roles)

    def read(self, key: str) -> Tuple[Optional[Any], int]:
        """(value copy, version); (None, 0) for a key that was never written."""
        with self._lock:
            entry = self.store.get(key)
            return (None, 0) if entry is None else (copy.deepcopy(entry.value), entry.version)

    def _acl_allows(self, key: str, role: str) -> bool:
        matching = [roles for prefix, roles in self.namespace_acl.items() if key.startswith(prefix)]
        if not matching:
            return not self.default_deny
        return all(role in roles for roles in matching)

    def write_optimistic(self, key: str, value: Any, expected_version: int, author_role: str = "architect") -> int:
        """Commit `value` at version `expected_version + 1` iff the role may write `key` and the stored version equals `expected_version`."""
        with self._lock:
            if not self._acl_allows(key, author_role):
                self.audit.append({"key": key, "role": author_role, "result": "ACL_DENIED"})
                raise PermissionError(f"ACL_DENIED: Role '{author_role}' cannot write to '{key}'.")
            current = self.store[key].version if key in self.store else 0
            if current != expected_version:
                self.audit.append({"key": key, "role": author_role, "result": "CONCURRENCY_CONFLICT"})
                raise ValueError(f"CONCURRENCY_CONFLICT on '{key}': expected v{expected_version}, found v{current}.")
            self.store[key] = BlackboardEntry(copy.deepcopy(value), current + 1, author_role)
            self.audit.append({"key": key, "role": author_role, "result": "OK", "version": current + 1})
            return current + 1

    def update(self, key: str, fn, author_role: str = "architect", max_retries: int = 200) -> int:
        """
        Read-modify-write with retry on conflict (the standard OCC loop), with randomised exponential backoff so
        contending writers do not collide in lock step. Returns the committed version; after `max_retries`
        conflicts it raises RuntimeError("OCC_RETRIES_EXHAUSTED") rather than dropping the update silently.
        """
        for attempt in range(max_retries):
            value, version = self.read(key)
            try:
                return self.write_optimistic(key, fn(value), version, author_role)
            except ValueError as exc:
                if "CONCURRENCY_CONFLICT" not in str(exc):
                    raise
                time.sleep(random.random() * min(0.0005 * 2 ** min(attempt, 6), 0.02))
        raise RuntimeError(f"OCC_RETRIES_EXHAUSTED on '{key}'")


class DebateConsensusAggregator:
    """Exact Condorcet majority accuracy, log-odds weighted quorum, and echo-chamber detection."""

    def __init__(self, correlation_alert_threshold: float = 0.50):
        if not 0.0 <= correlation_alert_threshold <= 1.0:
            raise ValueError("correlation_alert_threshold must be in [0, 1]")
        self.correlation_alert_threshold = correlation_alert_threshold

    # -- Condorcet ------------------------------------------------------------------------
    @staticmethod
    def compute_condorcet_probability(N: int, p: float, ties: str = "fail") -> float:
        """
        P(strict majority of N independent voters of accuracy p is right). For even N a tie is a failure
        (`ties="fail"`, default) or counts half (`ties="half"`).
        """
        if N <= 0:
            raise ValueError("Jury size N must be positive.")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if ties not in ("fail", "half"):
            raise ValueError("ties must be 'fail' or 'half'")
        prob = sum(math.comb(N, k) * p ** k * (1.0 - p) ** (N - k) for k in range(N // 2 + 1, N + 1))
        if N % 2 == 0 and ties == "half":
            prob += 0.5 * math.comb(N, N // 2) * (p * (1 - p)) ** (N // 2)
        return round(prob, 6)

    @staticmethod
    def effective_jury_size(N: int, rho: float) -> float:
        """N_eff = N / (1 + (N - 1) rho): how many independent voters N correlated voters are worth."""
        if N <= 0 or not 0.0 <= rho <= 1.0:
            raise ValueError("need N > 0 and rho in [0, 1]")
        return N / (1.0 + (N - 1) * rho)

    def compute_correlated_majority(self, N: int, p: float, rho: float) -> float:
        """Common-shock model: with probability rho everyone shares one draw (right with probability p)."""
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        return round(rho * p + (1.0 - rho) * self.compute_condorcet_probability(N, p), 6)

    # -- weighted quorum --------------------------------------------------------------------
    @staticmethod
    def compute_log_odds_weight(reliability_p: float, confidence_c: float) -> float:
        """w_i = ln(p_i / (1 - p_i)) * c_i, with p clamped to (1e-4, 1 - 1e-4) and c to [0, 1]."""
        p = min(max(reliability_p, 1e-4), 1.0 - 1e-4)
        c = min(max(confidence_c, 0.0), 1.0)
        return math.log(p / (1.0 - p)) * c

    @staticmethod
    def estimate_error_correlation(error_matrix: Sequence[Sequence[int]]) -> float:
        """
        Mean pairwise Pearson correlation of agents' error indicators (rows = agents, columns = questions,
        1 = the agent was wrong). Identical error patterns give 1.0; an agent with no variance is 0.0
        against anyone whose pattern differs and 1.0 against an identical one.
        """
        rows = [list(map(float, r)) for r in error_matrix]
        if len(rows) < 2 or len({len(r) for r in rows}) != 1 or not rows[0]:
            raise ValueError("need at least two agents with equal-length, non-empty histories")
        def corr(a: List[float], b: List[float]) -> float:
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
            if va == 0 or vb == 0:
                return 1.0 if a == b else 0.0
            return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)
        pairs = [corr(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]
        return sum(pairs) / len(pairs)

    def detect_echo_chamber(self, votes: List[AgentVote], correlation_rho: float) -> Tuple[bool, int]:
        """True if rho >= threshold or any voter flipped its prior vote without citing new evidence."""
        flips = sum(1 for v in votes if v.prior_vote is not None and v.prior_vote != v.vote and not v.cited_new_evidence)
        return (correlation_rho >= self.correlation_alert_threshold) or flips > 0, flips

    def aggregate_quorum(self, votes: List[AgentVote], correlation_rho: float = 0.0) -> QuorumDecision:
        if not votes:
            raise ValueError("need at least one vote")
        for v in votes:
            if v.vote not in (-1, 1):
                raise ValueError(f"vote of {v.agent_id} must be +1 or -1, got {v.vote}")
        sign = lambda x: (x > 0) - (x < 0)
        unweighted = sign(sum(v.vote for v in votes))
        margin = sum(self.compute_log_odds_weight(v.reliability_p, v.confidence_c) * v.vote for v in votes)
        echo, flips = self.detect_echo_chamber(votes, correlation_rho)
        flipped = [v.agent_id for v in votes if v.prior_vote is not None and v.prior_vote != v.vote and not v.cited_new_evidence]
        return QuorumDecision(unweighted, sign(round(margin, 12)), round(margin, 4), echo, flips, flipped,
                              round(self.effective_jury_size(len(votes), min(max(correlation_rho, 0.0), 1.0)), 4), correlation_rho)

    aggregate_weighted_quorum = aggregate_quorum


if __name__ == "__main__":
    bb = VersionedBlackboard()
    bb.register_namespace_acl("nav/", {"architect", "data"})
    v1 = bb.write_optimistic("nav/p1_vector", {"inclination_deg": 38.2}, expected_version=0, author_role="data")
    assert v1 == 1
    try:
        bb.write_optimistic("nav/p1_vector", {"inclination_deg": 12.0}, expected_version=0, author_role="architect")
        assert False, "Expected CONCURRENCY_CONFLICT!"
    except ValueError as exc:
        print(f"[OCC Race Caught] {exc}")
        assert "CONCURRENCY_CONFLICT" in str(exc)

    agg = DebateConsensusAggregator(correlation_alert_threshold=0.50)
    p_maj_3 = agg.compute_condorcet_probability(N=3, p=0.70)
    p_maj_7 = agg.compute_condorcet_probability(N=7, p=0.70)
    print(f"[Condorcet Jury] P_maj(N=3, p=0.70)={p_maj_3:.4f} | P_maj(N=7, p=0.70)={p_maj_7:.4f}")
    assert abs(p_maj_3 - 0.7840) < 1e-4 and p_maj_7 > p_maj_3

    panel = [
        AgentVote("AstrogationExpert", vote=+1, reliability_p=0.92, confidence_c=0.95),
        AgentVote("CloneWorker_A", vote=-1, reliability_p=0.55, confidence_c=0.60),
        AgentVote("CloneWorker_B", vote=-1, reliability_p=0.55, confidence_c=0.60, prior_vote=+1, cited_new_evidence=False),
    ]
    decision = agg.aggregate_quorum(panel, correlation_rho=0.82)
    print(f"[Quorum Decision] unweighted={decision.unweighted_verdict} | log_odds_weighted={decision.weighted_verdict} "
          f"(margin={decision.weighted_margin}) | echo_warning={decision.echo_chamber_warning} | flips={decision.sycophancy_flips_detected}")
    assert decision.unweighted_verdict == -1
    assert decision.weighted_verdict == +1
    assert decision.echo_chamber_warning is True and decision.sycophancy_flips_detected == 1

    # ---- extra checks ----------------------------------------------------------------
    # 1. exact values: P_maj(3, .7) = 3(.7^2)(.3) + .7^3 = 0.784; P_maj(1, p) = p; p = 1/2 gives 1/2 for odd N; p < 1/2 falls with N
    assert agg.compute_condorcet_probability(1, 0.8) == 0.8 and abs(agg.compute_condorcet_probability(5, 0.5) - 0.5) < 1e-9
    assert agg.compute_condorcet_probability(9, 0.45) < agg.compute_condorcet_probability(3, 0.45) < 0.45
    assert all(agg.compute_condorcet_probability(n + 2, 0.6) > agg.compute_condorcet_probability(n, 0.6) for n in (1, 3, 5, 7))
    assert abs(agg.compute_condorcet_probability(4, 0.7, ties="half") - 0.5 * (agg.compute_condorcet_probability(4, 0.7) + 1 - agg.compute_condorcet_probability(4, 0.3))) < 1e-5
    # 2. Monte Carlo agrees with the exact sum
    rng = random.Random(11)
    trials = 40000
    mc = sum(sum(rng.random() < 0.7 for _ in range(7)) >= 4 for _ in range(trials)) / trials
    assert abs(mc - agg.compute_condorcet_probability(7, 0.7)) < 0.01
    # 3. correlation destroys the Condorcet gain: N_eff shrinks and the majority is no better than one voter at rho = 1
    assert abs(agg.effective_jury_size(7, 0.0) - 7) < 1e-12 and abs(agg.effective_jury_size(7, 1.0) - 1) < 1e-12 and agg.effective_jury_size(7, 0.5) < 2
    assert abs(agg.compute_correlated_majority(7, 0.7, 1.0) - 0.7) < 1e-9 and abs(agg.compute_correlated_majority(7, 0.7, 0.0) - agg.compute_condorcet_probability(7, 0.7)) < 1e-9
    assert agg.compute_correlated_majority(7, 0.7, 0.5) < agg.compute_condorcet_probability(7, 0.7)
    # 4. error correlation from history: clones are perfectly correlated, independent agents are not
    clone = [1, 0, 0, 1, 1, 0, 1, 0]
    assert abs(agg.estimate_error_correlation([clone, clone, clone]) - 1.0) < 1e-12
    indep = agg.estimate_error_correlation([[1, 0, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1]])
    assert abs(indep) < 0.5
    # 5. weighted votes: a coin-flip voter (p = 0.5) has weight 0; p < 0.5 flips the sign; an exact tie is 0, not an approval
    assert agg.compute_log_odds_weight(0.5, 1.0) == 0.0 and agg.compute_log_odds_weight(0.3, 1.0) < 0 and agg.compute_log_odds_weight(0.9, 0.0) == 0.0
    tie = agg.aggregate_quorum([AgentVote("a", +1, 0.8, 1.0), AgentVote("b", -1, 0.8, 1.0)])
    assert tie.unweighted_verdict == 0 and tie.weighted_verdict == 0
    # 6. echo detection: clean independent panel passes; a flip that cites evidence is not sycophancy
    clean = agg.aggregate_quorum([AgentVote("a", +1, 0.9, 1.0), AgentVote("b", +1, 0.8, 0.9, prior_vote=-1, cited_new_evidence=True)], correlation_rho=0.1)
    assert clean.echo_chamber_warning is False and clean.sycophancy_flips_detected == 0
    assert decision.flipped_agents == ["CloneWorker_B"] and decision.effective_jury_size < 2
    # 7. blackboard: ACLs, copies, audit, OCC under real threads (no lost update with the retry loop, many with a blind write)
    ac = VersionedBlackboard(default_deny=True)
    ac.register_namespace_acl("nav/", {"data"})
    for role, key in (("architect", "nav/x"), ("data", "misc/y")):
        try:
            ac.write_optimistic(key, 1, 0, role); raise SystemExit("ACL must deny")
        except PermissionError:
            pass
    val, ver = ac.read("nav/x"); assert (val, ver) == (None, 0)
    ac.write_optimistic("nav/x", {"a": [1]}, 0, "data")
    got, _ = ac.read("nav/x"); got["a"].append(2)
    assert ac.read("nav/x")[0] == {"a": [1]}                                   # reads are copies
    assert [a["result"] for a in ac.audit] == ["ACL_DENIED", "ACL_DENIED", "OK"]
    board = VersionedBlackboard()
    board.write_optimistic("count", 0, 0)
    def worker():
        for _ in range(100):
            board.update("count", lambda v: v + 1)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert board.read("count") == (800, 801)                                    # 800 increments, none lost
    blind = {"count": 0}
    def blind_worker():
        for _ in range(100):
            v = blind["count"]; threading.Event().wait(0.0001); blind["count"] = v + 1
    threads = [threading.Thread(target=blind_worker) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    print(f"OCC with retry: 800/800 increments kept; blind read-modify-write without OCC kept {blind['count']}/800.")
    assert blind["count"] < 800
    # 8. input validation
    for bad in (lambda: agg.compute_condorcet_probability(0, 0.7), lambda: agg.compute_condorcet_probability(3, 1.2),
                lambda: agg.aggregate_quorum([]), lambda: agg.aggregate_quorum([AgentVote("a", 0, 0.9, 1.0)]),
                lambda: agg.estimate_error_correlation([[1, 0]]), lambda: DebateConsensusAggregator(2.0)):
        try:
            bad(); raise SystemExit("invalid input must raise")
        except ValueError:
            pass
    print("[PASS] Part 3 Data: Versioned Blackboard & Condorcet Debate Consensus Confirmed.")
