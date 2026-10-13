"""Independent cost checker.

Recomputes the cost of a placement from the raw link list with its own Dijkstra, sharing no code
with the solvers or with the scipy shortest-path routine they use.
"""

from __future__ import annotations

import heapq

import numpy as np

from .graph import Graph
from .workload import Flows


class Checker:
    def __init__(self, graph: Graph):
        self.num_nodes = graph.num_nodes
        self.candidates = {int(v) for v in graph.candidates}
        self._adj = [[] for _ in range(graph.num_nodes)]
        for (u, v), w in zip(graph.links.tolist(), graph.weights.tolist()):
            self._adj[u].append((v, w))
            self._adj[v].append((u, w))
        self._cache = {}

    def dist_from(self, source: int) -> np.ndarray:
        source = int(source)
        if source not in self._cache:
            dist = [float("inf")] * self.num_nodes
            dist[source] = 0.0
            heap = [(0.0, source)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                for v, w in self._adj[u]:
                    nd = d + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
            self._cache[source] = np.array(dist)
        return self._cache[source]

    def validate(self, placement) -> list[int]:
        placement = [int(v) for v in placement]
        if not placement:
            raise ValueError("empty placement")
        if len(set(placement)) != len(placement):
            raise ValueError(f"VNFs must be on distinct switches: {placement}")
        outside = [v for v in placement if v not in self.candidates]
        if outside:
            raise ValueError(f"not candidate switches: {outside}")
        return placement

    def cost(self, flows: Flows, placement, previous=None, mu: float = 0.0) -> float:
        """Eq. 1 cost of ``placement`` (node ids, in VNF order), plus mu times the migration
        distance from ``previous`` when given (Eq. 8)."""
        p = self.validate(placement)
        entry = float(self.dist_from(p[0])[flows.src] @ flows.rate)
        exit_ = float(self.dist_from(p[-1])[flows.dst] @ flows.rate)
        chain = sum(float(self.dist_from(a)[b]) for a, b in zip(p[:-1], p[1:]))
        total = entry + exit_ + flows.total_rate * chain
        if previous is not None:
            prev = [int(v) for v in previous]
            if len(prev) != len(p):
                raise ValueError("previous placement has a different length")
            total += mu * sum(float(self.dist_from(a)[b]) for a, b in zip(prev, p))
        return total

    def migration_distance(self, previous, placement) -> float:
        return sum(float(self.dist_from(int(a))[int(b)]) for a, b in zip(previous, placement))
