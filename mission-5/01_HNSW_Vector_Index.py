"""Mission 05 — pedagogical NumPy HNSW index (no external ANN library).

Implements probabilistic layers, bounded neighbor lists, greedy upper-layer
navigation and ef-search at the base. It is an educational implementation,
NOT an optimized production/vectorized ANN library. Benchmark claims are
measured rather than hard-coded. Python graph-building at N=25k can be slow.
"""
from __future__ import annotations
import argparse
import heapq
import math
import time
from collections import defaultdict
import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows or a single vector; reject zero-length vectors."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Zero-length embeddings are not allowed')
    return x / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine for raw vectors, returning a scalar or batched dot products."""
    a, b = normalize(a), normalize(b)
    return np.sum(a * b, axis=-1)


class HNSWIndex:
    """Incremental HNSW with bidirectional pruned connectivity.

    Node IDs are integer insertion IDs. Distances are 1 - cosine_similarity.
    ``search_layer`` returns (distance, node_id) pairs nearest-first.
    """
    def __init__(self, dimension: int, M: int = 16, ef_construction: int = 100,
                 ef_search: int = 64, seed: int = 42):
        if dimension < 1 or M < 2 or ef_construction < M or ef_search < 1:
            raise ValueError('Require dimension>=1, M>=2, ef_construction>=M, ef_search>=1')
        self.dimension = dimension
        self.M, self.ef_construction, self.ef_search = M, ef_construction, ef_search
        self.rng = np.random.default_rng(seed)
        self.mL = 1.0 / math.log(M)
        self.vectors: list[np.ndarray] = []
        self.levels: list[int] = []
        self.graph: dict[int, dict[int, set[int]]] = defaultdict(dict)
        self.entry_point: int | None = None
        self.max_level = -1

    def random_level(self) -> int:
        u = max(float(self.rng.random()), np.finfo(float).tiny)
        return int(math.floor(-math.log(u) * self.mL))

    def _distance(self, a: int, b: np.ndarray) -> float:
        return float(1.0 - np.dot(self.vectors[a], b))

    def _neighbors(self, node: int, layer: int) -> set[int]:
        return self.graph[layer].get(node, set())

    def _greedy(self, query: np.ndarray, ep: int, layer: int) -> int:
        best = self._distance(ep, query)
        while True:
            candidate = ep
            for neighbor in self._neighbors(ep, layer):
                d = self._distance(neighbor, query)
                if d < best:
                    best, candidate = d, neighbor
            if candidate == ep:
                return ep
            ep = candidate

    def search_layer(self, entry_point: int, query: np.ndarray, ef: int,
                     layer: int) -> list[tuple[float, int]]:
        """Best-first traversal with bounded candidate/result heaps."""
        if ef < 1 or entry_point not in self.graph[layer]:
            raise ValueError('Invalid ef or entry point for this layer')
        q = normalize(query)
        d0 = self._distance(entry_point, q)
        candidates = [(d0, entry_point)]             # min-heap
        results = [(-d0, entry_point)]               # max-distance via negatives
        visited = {entry_point}
        while candidates:
            d, node = heapq.heappop(candidates)
            worst = -results[0][0]
            if len(results) >= ef and d > worst:
                break
            for nb in self._neighbors(node, layer):
                if nb in visited:
                    continue
                visited.add(nb)
                dn = self._distance(nb, q)
                if len(results) < ef or dn < -results[0][0]:
                    heapq.heappush(candidates, (dn, nb))
                    heapq.heappush(results, (-dn, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted([(-negd, node) for negd, node in results])

    def _link(self, a: int, b: int, layer: int) -> None:
        if a == b:
            return
        self.graph[layer][a].add(b)
        self.graph[layer][b].add(a)
        for node in (a, b):
            cap = 2 * self.M if layer == 0 else self.M
            neighbors = self.graph[layer][node]
            if len(neighbors) > cap:
                keep = set(sorted(neighbors, key=lambda n: self._distance(n, self.vectors[node]))[:cap])
                for removed in neighbors - keep:
                    self.graph[layer][removed].discard(node)
                self.graph[layer][node] = keep

    def add(self, vector: np.ndarray) -> int:
        v = normalize(vector)
        if v.shape != (self.dimension,):
            raise ValueError(f'Expected shape ({self.dimension},), got {v.shape}')
        idx = len(self.vectors)
        level = self.random_level()
        self.vectors.append(v)
        self.levels.append(level)
        for layer in range(level + 1):
            self.graph[layer][idx] = set()
        if self.entry_point is None:
            self.entry_point, self.max_level = idx, level
            return idx
        ep = self.entry_point
        for layer in range(self.max_level, level, -1):
            ep = self._greedy(v, ep, layer)
        for layer in range(min(level, self.max_level), -1, -1):
            candidates = self.search_layer(ep, v, self.ef_construction, layer)
            cap = 2 * self.M if layer == 0 else self.M
            for _, neighbor in candidates[:cap]:
                self._link(idx, neighbor, layer)
            if candidates:
                ep = candidates[0][1]
        if level > self.max_level:
            self.entry_point, self.max_level = idx, level
        return idx

    def add_batch(self, vectors: np.ndarray) -> None:
        for vector in vectors:
            self.add(vector)

    def search(self, query: np.ndarray, k: int = 10, ef: int | None = None) -> list[tuple[int, float]]:
        if self.entry_point is None or k <= 0:
            return []
        q = normalize(query)
        if q.shape != (self.dimension,):
            raise ValueError('Query dimension mismatch')
        ep = self.entry_point
        for layer in range(self.max_level, 0, -1):
            ep = self._greedy(q, ep, layer)
        candidates = self.search_layer(ep, q, max(ef or self.ef_search, k), 0)
        return [(node, 1.0 - distance) for distance, node in candidates[:k]]


def flat_search(vectors: np.ndarray, query: np.ndarray, k: int = 10) -> list[int]:
    scores = vectors @ normalize(query)
    k = min(k, len(scores))
    return np.argsort(-scores)[:k].tolist() if k else []


def benchmark(n: int = 25_000, dim: int = 768, queries: int = 20,
              seed: int = 42, M: int = 16) -> dict:
    rng = np.random.default_rng(seed)
    vectors = normalize(rng.normal(size=(n, dim)).astype(np.float32))
    q_vectors = normalize(rng.normal(size=(queries, dim)).astype(np.float32))
    index = HNSWIndex(dim, M=M, ef_construction=80, ef_search=80, seed=seed)
    start = time.perf_counter()
    index.add_batch(vectors)
    build_s = time.perf_counter() - start
    # Warm up both paths to limit first-call effects.
    index.search(q_vectors[0]); flat_search(vectors, q_vectors[0])
    ann_ms, flat_ms, recalls = [], [], []
    for q in q_vectors:
        start = time.perf_counter()
        approx = [i for i, _ in index.search(q, k=10)]
        ann_ms.append((time.perf_counter() - start) * 1e3)
        start = time.perf_counter()
        exact = flat_search(vectors, q, k=10)
        flat_ms.append((time.perf_counter() - start) * 1e3)
        recalls.append(len(set(approx) & set(exact)) / len(exact))
    return {'N': n, 'dimension': dim, 'build_seconds': round(build_s, 2),
            'hnsw_mean_ms': round(float(np.mean(ann_ms)), 3),
            'flat_mean_ms': round(float(np.mean(flat_ms)), 3),
            'recall_at_10': round(float(np.mean(recalls)), 3),
            'speedup_flat_over_hnsw': round(float(np.mean(flat_ms)/np.mean(ann_ms)), 2)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000, help='Set 25000 for assignment-scale benchmark')
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--queries', type=int, default=10)
    args = parser.parse_args()
    print(benchmark(args.n, args.dim, args.queries))
