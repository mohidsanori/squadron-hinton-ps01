"""Mission 05 — from-scratch BM25 (k1=1.5, b=.75) and RRF (k=60)."""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Sequence


def tokenize(text: str) -> list[str]:
    """Keep hyphenated serial numbers intact (e.g. NX-742-A)."""
    return re.findall(r'(?u)\b[\w]+(?:-[\w]+)*\b', text.lower())


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = .75):
        if not documents or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('Nonempty documents, k1>0, and 0<=b<=1 required')
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self.tokens = [tokenize(doc) for doc in documents]
        self.tf = [Counter(toks) for toks in self.tokens]
        self.avgdl = sum(map(len, self.tokens)) / len(self.tokens)
        self.df = Counter(word for doc in self.tf for word in doc)

    def score(self, query: str) -> list[float]:
        n = len(self.documents)
        scores = []
        for counts, toks in zip(self.tf, self.tokens):
            score = 0.0
            for word in set(tokenize(query)):
                freq = counts[word]
                if not freq:
                    continue
                df = self.df[word]
                idf = math.log(1 + (n - df + .5)/(df + .5))
                denom = freq + self.k1 * (1 - self.b + self.b * len(toks) / (self.avgdl or 1))
                score += idf * freq * (self.k1 + 1) / denom
            scores.append(score)
        return scores

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        return [(i, scores) for i, scores in sorted(enumerate(self.score(query)),
                     key=lambda x: (-x[1], x[0])) if scores > 0][:k]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    if k <= 0:
        raise ValueError('k must be positive')
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1/(k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def hybrid_search(query: str, bm25: BM25, hnsw_index, embed_query,
                  k: int = 5, candidate_k: int = 20) -> dict:
    """Embed query with same embedding model as index, then fuse rankings.

    Caller must ensure vector index insertion order equals BM25 document order.
    """
    dense = hnsw_index.search(embed_query(query), k=candidate_k)
    sparse = bm25.search(query, k=candidate_k)
    fused = reciprocal_rank_fusion([[doc for doc, _ in dense], [doc for doc, _ in sparse]])[:k]
    return {'dense': dense, 'sparse': sparse, 'fused': fused}


if __name__ == '__main__':
    corpus = ['AURA uses a dual path retrieval system for semantic grounding.',
              'The propulsion component serial code NX-742-A is in bay 7.',
              'GraphRAG summarizes graph communities to answer global questions.',
              'Hybrid retrieval combines semantic and exact keyword matching.']
    engine = BM25(corpus)
    for query in ('How do systems ground answers?', 'NX-742-A'):
        print(query, engine.search(query))
    print('RRF', reciprocal_rank_fusion([[0, 2, 3], [3, 1, 0]]))
