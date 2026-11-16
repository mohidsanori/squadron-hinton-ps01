"""
Mission 08 - Part 3: Data & Payload Specialist
MemGPT virtual context pager and Generative-Agents tri-factor retrieval
Course: Conversational AI (WS 2026/27) | Starship Unit: USSE Noosphere, Squadron Hinton
Sector: SEC-08-CYG (Cygnus X-1 & Veil Nebula), Subsystem 08

MemGPT (Packer et al., 2023) treats the LLM context window like RAM and external storage like disk:

    Main context (RAM)   system instructions + read/write core-memory blocks + FIFO message queue
    External context     recall storage (evicted raw messages) + archival storage (searchable nodes)

When active tokens reach tau_warn (0.70) of the budget the pager raises a memory-pressure
interrupt and flushes the oldest half of the FIFO queue into one summary node in archival
storage; it repeats until usage is back under tau_warn or the queue is empty.

Archival nodes are ranked with the Generative Agents score (Park et al., 2023):

    S(m, q) = a_rec * gamma^dt(m)  +  a_imp * I(m)/10  +  a_rel * cos(e_m, e_q)

Recency alone would let fresh chatter bury an old safety rule; relevance and importance keep
the rule retrievable. Nodes carry bi-temporal validity (t_valid, t_invalid, as in Zep), so a
superseded fact is invalidated instead of deleted and is excluded from search.

Usage:
    python memgpt_pager.py
"""

import hashlib
import math
import re
from typing import Any, Callable, Dict, List, Optional


class ArchivalMemoryNode:
    """A persistent record in archival storage."""

    def __init__(
        self,
        node_id: str,
        text: str,
        importance: float,
        hours_since_access: float,
        embedding: List[float],
        t_valid: int = 0,
        t_invalid: Optional[int] = None,
    ):
        if not 1.0 <= importance <= 10.0:
            raise ValueError(f"importance must be in [1, 10], got {importance}")
        if hours_since_access < 0:
            raise ValueError("hours_since_access must be >= 0")
        if not embedding:
            raise ValueError("embedding must not be empty")
        self.node_id = node_id
        self.text = text
        self.importance = importance                      # I(m) in [1, 10]
        self.hours_since_access = hours_since_access      # delta t since last access
        self.embedding = embedding
        self.t_valid = t_valid
        self.t_invalid = t_invalid                        # None = still valid

    def __repr__(self) -> str:
        return f"ArchivalMemoryNode({self.node_id!r}, I={self.importance}, dt={self.hours_since_access}h)"


class HashingEmbedder:
    """
    Deterministic bag-of-words embedding (signed feature hashing, L2-normalised). Not semantic:
    texts that share words are close, nothing more. It lets the pager build real embeddings for
    its own summary nodes and lets the demos run with no model or network.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def __call__(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9_]+", text.lower()):
            h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec


class MemGPTVirtualPager:
    """OS-inspired virtual context: core memory + FIFO queue in context, recall + archival outside."""

    def __init__(self, token_budget: int = 4096, tau_warn: float = 0.70, decay_gamma: float = 0.99,
                 embed_fn: Optional[Callable[[str], List[float]]] = None):
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if not 0.0 < tau_warn <= 1.0:
            raise ValueError("tau_warn must be in (0, 1]")
        if not 0.0 < decay_gamma <= 1.0:
            raise ValueError("decay_gamma must be in (0, 1]")
        self.token_budget = token_budget
        self.tau_warn = tau_warn
        self.decay_gamma = decay_gamma
        self.embed_fn = embed_fn
        self.core_memory: Dict[str, str] = {"persona": "", "task_state": ""}
        self.fifo_queue: List[Dict[str, Any]] = []
        self.recall_store: List[Dict[str, Any]] = []      # evicted raw messages, kept verbatim
        self.archival_store: List[ArchivalMemoryNode] = []
        self.system_interrupts: List[str] = []
        self._evict_seq = 0

    # -- core memory syscalls -----------------------------------------------------------
    def core_memory_append(self, block_key: str, content: str) -> None:
        if block_key not in self.core_memory:
            raise KeyError(f"Unknown core_memory block: {block_key}")
        sep = " " if self.core_memory[block_key] else ""
        self.core_memory[block_key] = f"{self.core_memory[block_key]}{sep}{content}".strip()

    def core_memory_replace(self, block_key: str, old_text: str, new_text: str) -> None:
        if block_key not in self.core_memory:
            raise KeyError(f"Unknown core_memory block: {block_key}")
        if old_text not in self.core_memory[block_key]:
            raise ValueError(f"Target substring '{old_text}' not found in core_memory['{block_key}']")
        self.core_memory[block_key] = self.core_memory[block_key].replace(old_text, new_text)

    # -- archival syscalls --------------------------------------------------------------
    def archival_memory_insert(self, node: ArchivalMemoryNode) -> None:
        if any(n.node_id == node.node_id for n in self.archival_store):
            raise ValueError(f"archival node id {node.node_id!r} already exists")
        self.archival_store.append(node)

    def invalidate(self, node_id: str, t_invalid: int) -> None:
        """Mark a fact as superseded at time `t_invalid` (it stays stored but leaves search results)."""
        for n in self.archival_store:
            if n.node_id == node_id:
                n.t_invalid = t_invalid
                return
        raise KeyError(node_id)

    # -- context accounting -------------------------------------------------------------
    def enqueue_message(self, role: str, text: str, tokens: Optional[int] = None) -> bool:
        """Append a turn to the FIFO queue; returns True if this triggered a pressure interrupt."""
        est = tokens if tokens is not None else len(text.split()) * 2
        self.fifo_queue.append({"role": role, "text": text, "tokens": est})
        return self.check_memory_pressure()

    def current_token_usage(self) -> int:
        core = sum(len(v.split()) * 2 for v in self.core_memory.values())
        fifo = sum(m.get("tokens", len(m["text"].split()) * 2) for m in self.fifo_queue)
        return core + fifo

    def check_memory_pressure(self) -> bool:
        """
        If usage / budget >= tau_warn: record a `memory_pressure_warning` interrupt and flush the
        oldest 50% of the FIFO queue (at least one message) into a consolidated archival summary
        node, repeating while usage is still at or above tau_warn and messages remain.
        Returns True if at least one flush happened.
        """
        flushed = False
        while self.fifo_queue and self.current_token_usage() / self.token_budget >= self.tau_warn:
            usage = self.current_token_usage()
            evict_count = max(1, len(self.fifo_queue) // 2)
            evicted, self.fifo_queue = self.fifo_queue[:evict_count], self.fifo_queue[evict_count:]
            self.recall_store.extend(evicted)

            self._evict_seq += 1
            summary_text = "[FIFO_FLUSH_SUMMARY]: " + " | ".join(m["text"] for m in evicted)
            embedding = self.embed_fn(summary_text) if self.embed_fn else [0.5, 0.5, 0.5]
            self.archival_store.append(ArchivalMemoryNode(
                node_id=f"evict_{self._evict_seq}", text=summary_text, importance=7.5,
                hours_since_access=0.0, embedding=embedding))
            self.system_interrupts.append(
                f"SYSTEM_INTERRUPT(memory_pressure_warning): usage={usage}/{self.token_budget} "
                f"({usage / self.token_budget:.1%}) >= {self.tau_warn:.0%}. "
                f"Evicted {evict_count} oldest FIFO messages to Archival Storage (evict_{self._evict_seq}).")
            flushed = True
        return flushed

    # -- tri-factor retrieval -----------------------------------------------------------
    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if len(vec_a) != len(vec_b):
            raise ValueError(f"embedding dimension mismatch: {len(vec_a)} vs {len(vec_b)}")
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        na, nb = math.sqrt(sum(a * a for a in vec_a)), math.sqrt(sum(b * b for b in vec_b))
        return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

    def score_breakdown(self, node: ArchivalMemoryNode, query_embedding: List[float],
                        alpha_rec: float = 1.0, alpha_imp: float = 1.0, alpha_rel: float = 1.0) -> Dict[str, float]:
        """The three components and the total S(m, q)."""
        recency = self.decay_gamma ** node.hours_since_access
        importance = node.importance / 10.0
        relevance = self._cosine_similarity(node.embedding, query_embedding)
        return {"recency": recency, "importance": importance, "relevance": relevance,
                "total": alpha_rec * recency + alpha_imp * importance + alpha_rel * relevance}

    def score_memory(self, node: ArchivalMemoryNode, query_embedding: List[float],
                     alpha_rec: float = 1.0, alpha_imp: float = 1.0, alpha_rel: float = 1.0) -> float:
        """S(m, q) = a_rec * gamma^dt + a_imp * I/10 + a_rel * cos(e_m, e_q)."""
        return self.score_breakdown(node, query_embedding, alpha_rec, alpha_imp, alpha_rel)["total"]

    def archival_memory_search(self, query_embedding: List[float], top_k: int = 3,
                               alpha_rec: float = 1.0, alpha_imp: float = 1.0, alpha_rel: float = 1.0,
                               touch_on_access: bool = False) -> List[ArchivalMemoryNode]:
        """Top-k currently valid nodes by S(m, q); optionally reset their recency clock."""
        active = [n for n in self.archival_store if n.t_invalid is None]
        ranked = sorted(active, key=lambda n: self.score_memory(n, query_embedding, alpha_rec, alpha_imp, alpha_rel),
                        reverse=True)
        selected = ranked[:max(0, top_k)]
        if touch_on_access:
            for n in selected:
                n.hours_since_access = 0.0
        return selected


# ==============================================================================
# Verification harness for Part 3 (Data & Payload Specialist)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 08 (Part 3): MemGPT Virtual Pager & Tri-Factor Retrieval Verification ===")

    pager = MemGPTVirtualPager(token_budget=1000, tau_warn=0.70, decay_gamma=0.99)
    pager.core_memory_append("persona", "AURA Tactical Navigator v8.0")
    pager.core_memory_append("task_state", "Orbit=Cygnus_X1_Outer_Corona")
    pager.core_memory_replace("task_state", "Outer_Corona", "Inner_Shock_Front")

    # archival memories with contrasting recency, importance and relevance
    pager.archival_memory_insert(ArchivalMemoryNode(
        node_id="mem_safety",
        text="CRITICAL DIRECTIVE: Never drop magnetic shielding below 42% near Cygnus X-1 X-ray flares.",
        importance=10.0, hours_since_access=48.0, embedding=[0.92, 0.35, 0.15]))
    pager.archival_memory_insert(ArchivalMemoryNode(
        node_id="mem_chatter",
        text="Bridge crew coffee dispenser recalibrated in Mess Hall Deck 3.",
        importance=1.5, hours_since_access=1.0, embedding=[0.10, 0.85, 0.45]))
    pager.archival_memory_insert(ArchivalMemoryNode(
        node_id="mem_telemetry",
        text="X-ray flare harmonic filter locked to 6.4 keV iron K-alpha line.",
        importance=8.5, hours_since_access=6.0, embedding=[0.88, 0.42, 0.20]))

    # trigger the 70% memory-pressure warning via FIFO queue turns
    for i in range(4):
        triggered = pager.enqueue_message("user", f"Shockwave telemetry packet {i}", tokens=200)
        if triggered:
            print(" ", pager.system_interrupts[-1])

    query_vec = [0.95, 0.30, 0.10]  # query about Cygnus X-1 X-ray shield safety
    top_memories = pager.archival_memory_search(query_vec, top_k=2)
    print("\nTop-2 Retrieved Archival Memories for X-Ray Shield Query:")
    for node in top_memories:
        s = pager.score_memory(node, query_vec)
        print(f"  [{node.node_id}] S(m,q)={s:.4f} | I={node.importance} | dt={node.hours_since_access}h | {node.text}")

    assert len(pager.system_interrupts) >= 1, "70% token-pressure interrupt failed to trigger!"
    assert top_memories[0].node_id in {"mem_safety", "mem_telemetry"}, "High-importance safety memory should outrank trivial recent chatter!"

    # ---- extra checks ----------------------------------------------------------------
    # 1. the score really is the sum of its three parts, matching the closed form
    b = pager.score_breakdown(pager.archival_store[0], query_vec)
    assert abs(b["total"] - (0.99 ** 48 + 1.0 + b["relevance"])) < 1e-12
    # 2. weights matter: recency alone lets chatter beat the safety rule; the tri-factor score does not
    top_recency = pager.archival_memory_search(query_vec, top_k=1, alpha_rec=1, alpha_imp=0, alpha_rel=0)[0]
    top_tri = pager.archival_memory_search(query_vec, top_k=1)[0]
    assert top_recency.node_id.startswith("evict_") or top_recency.node_id == "mem_chatter"
    assert top_tri.node_id == "mem_telemetry"
    # 3. eviction leaves usage under tau_warn, keeps raw messages in recall, and gives unique node ids
    assert pager.current_token_usage() / pager.token_budget < 0.70
    assert len(pager.recall_store) == 2 and len({n.node_id for n in pager.archival_store}) == len(pager.archival_store)
    # 4. repeated flushing: one oversized core block cannot leave the pager stuck above tau_warn
    p2 = MemGPTVirtualPager(token_budget=100, tau_warn=0.70)
    for i in range(6):
        p2.enqueue_message("user", f"m{i}", tokens=30)
    assert p2.current_token_usage() / 100 < 0.70 and len(p2.system_interrupts) >= 2
    # 5. embeddings for summary nodes come from embed_fn; mismatched dimensions are an error, not silent truncation
    p3 = MemGPTVirtualPager(token_budget=100, tau_warn=0.5, embed_fn=HashingEmbedder(16))
    p3.enqueue_message("user", "flux spike detected", tokens=60)
    assert len(p3.archival_store[0].embedding) == 16
    try:
        p3.archival_memory_search([1.0, 0.0, 0.0])
        raise SystemExit("dimension mismatch must raise")
    except ValueError:
        pass
    # 6. bi-temporal validity: an invalidated fact leaves search; touch_on_access resets recency
    pager.invalidate("mem_telemetry", t_invalid=10)
    assert "mem_telemetry" not in {n.node_id for n in pager.archival_memory_search(query_vec, top_k=5)}
    hit = pager.archival_memory_search(query_vec, top_k=1, touch_on_access=True)[0]
    assert hit.hours_since_access == 0.0
    # 7. input validation
    for bad in (lambda: ArchivalMemoryNode("x", "t", 11.0, 0.0, [1.0]), lambda: ArchivalMemoryNode("x", "t", 5.0, -1.0, [1.0])):
        try:
            bad()
            raise SystemExit("invalid node must raise")
        except ValueError:
            pass
    try:
        pager.archival_memory_insert(ArchivalMemoryNode("mem_safety", "dup", 5.0, 0.0, [1.0, 0.0, 0.0]))
        raise SystemExit("duplicate id must raise")
    except ValueError:
        pass
    print("Recency-only ranking would return:", top_recency.node_id, "| tri-factor returns:", top_tri.node_id)
    print("[OK] Part 3 (Data & Payload Specialist) verification passed.")
