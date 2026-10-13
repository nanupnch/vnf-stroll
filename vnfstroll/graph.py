"""Undirected graphs with non-negative link weights."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class Graph:
    """Undirected graph on nodes 0..num_nodes-1.

    ``candidates`` are the nodes that may host a VNF (the switches V_s in the paper).
    Link weights are the per-unit costs w(u, v); unit weights give hop counts.
    """

    def __init__(self, num_nodes, links, weights=None, candidates=None, names=None):
        links = np.asarray(links, dtype=np.int64).reshape(-1, 2)
        if weights is None:
            weights = np.ones(len(links))
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(links),):
            raise ValueError("need one weight per link")
        if (weights <= 0).any():
            raise ValueError("link weights must be positive")
        if (links < 0).any() or (links >= num_nodes).any():
            raise ValueError("link endpoint out of range")

        self.num_nodes = int(num_nodes)
        self.links = links
        self.weights = weights
        self.unweighted = bool(np.all(weights == 1.0))
        self.names = list(names) if names is not None else [str(i) for i in range(num_nodes)]
        if candidates is None:
            candidates = np.arange(num_nodes)
        self.candidates = np.asarray(candidates, dtype=np.int64)

        adj = [set() for _ in range(num_nodes)]
        for u, v in links.tolist():
            adj[u].add(v)
            adj[v].add(u)
        self.adj = [sorted(a) for a in adj]

        rows = np.concatenate([links[:, 0], links[:, 1]])
        cols = np.concatenate([links[:, 1], links[:, 0]])
        self._csr = csr_matrix((np.concatenate([weights, weights]), (rows, cols)), shape=(num_nodes, num_nodes))
        self._hops = {}
        self._bipartite = None

    @property
    def num_links(self) -> int:
        return len(self.links)

    @property
    def bipartite(self) -> bool:
        if self._bipartite is None:
            self._bipartite = self._is_bipartite()
        return self._bipartite

    def distances(self, sources=None) -> np.ndarray:
        """Shortest-path costs from ``sources`` (default: all nodes) to every node."""
        indices = None if sources is None else np.asarray(sources, dtype=np.int64)
        return shortest_path(self._csr, method="D", directed=False, unweighted=self.unweighted, indices=indices)

    def hop_distances(self, source: int) -> list[int]:
        """Breadth-first hop counts from ``source`` (-1 if unreachable). Cached per source."""
        if source in self._hops:
            return self._hops[source]
        dist = [-1] * self.num_nodes
        dist[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for w in self.adj[u]:
                if dist[w] < 0:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        self._hops[source] = dist
        return dist

    def _is_bipartite(self) -> bool:
        color = [-1] * self.num_nodes
        for root in range(self.num_nodes):
            if color[root] >= 0:
                continue
            color[root] = 0
            queue = deque([root])
            while queue:
                u = queue.popleft()
                for w in self.adj[u]:
                    if color[w] < 0:
                        color[w] = 1 - color[u]
                        queue.append(w)
                    elif color[w] == color[u]:
                        return False
        return True
