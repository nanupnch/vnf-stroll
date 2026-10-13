"""Placement and migration of one SFC, written on the metric closure of the network.

For flows (s_i, t_i, lambda_i) and a placement p = (p_1, ..., p_n) of the chain f_1 -> ... -> f_n on
distinct switches, Eq. 1 of the paper is

    C(p) = A(p_1) + B(p_n) + Lambda * sum_j c(p_j, p_{j+1}),

with A(v) = sum_i lambda_i c(s_i, v), B(v) = sum_i lambda_i c(v, t_i) and Lambda = sum_i lambda_i.
TOP-1 is the case of a single flow. TOM (Eq. 8) adds mu * c(p_j, m_j) for moving f_j from its
current switch p_j to m_j, which is a per-position node cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import Graph
from .workload import Flows


@dataclass
class ChainInstance:
    D: np.ndarray  # (m, m) shortest-path costs between candidate switches
    A: np.ndarray  # (m,) entry cost of each candidate
    B: np.ndarray  # (m,) exit cost of each candidate
    total_rate: float  # Lambda
    n: int  # number of VNFs
    candidates: np.ndarray  # node id of each candidate
    node_cost: np.ndarray | None = None  # (n, m) extra cost of VNF j at candidate v

    @property
    def m(self) -> int:
        return len(self.A)

    def cost(self, idx) -> float:
        """Cost of a placement given as candidate indices."""
        idx = np.asarray(idx, dtype=np.int64)
        total = self.A[idx[0]] + self.B[idx[-1]] + self.total_rate * self.D[idx[:-1], idx[1:]].sum()
        if self.node_cost is not None:
            total += self.node_cost[np.arange(self.n), idx].sum()
        return float(total)

    def index_of(self, nodes) -> np.ndarray:
        position = {int(v): i for i, v in enumerate(self.candidates)}
        return np.array([position[int(v)] for v in nodes], dtype=np.int64)


@dataclass
class Solution:
    cost: float
    placement: tuple[int, ...]  # node ids of the switches hosting f_1, ..., f_n
    method: str
    optimal: bool
    bound: float | None = None  # proven lower bound when not optimal
    seconds: float = 0.0
    info: dict = field(default_factory=dict)


def chain_instance(graph: Graph, flows: Flows, n: int, dist: np.ndarray | None = None) -> ChainInstance:
    """Instance of TOP (TOP-1 for a single flow). ``dist`` may hold precomputed
    ``graph.distances(graph.candidates)`` to avoid recomputing shortest paths."""
    if n < 1:
        raise ValueError("need at least one VNF")
    if n > len(graph.candidates):
        raise ValueError("more VNFs than candidate switches")
    if dist is None:
        dist = graph.distances(graph.candidates)
    cand = graph.candidates
    return ChainInstance(
        D=dist[:, cand],
        A=dist[:, flows.src] @ flows.rate,
        B=dist[:, flows.dst] @ flows.rate,
        total_rate=flows.total_rate,
        n=n,
        candidates=cand,
    )


def migration_instance(graph: Graph, flows: Flows, current, mu: float, dist: np.ndarray | None = None) -> ChainInstance:
    """Instance of TOM: the chain currently sits on ``current`` (node ids); moving f_j costs
    mu * c(current_j, new_j). With mu = 0 this is TOP (Theorem 4 of the paper)."""
    inst = chain_instance(graph, flows, len(current), dist)
    pos = inst.index_of(current)
    inst.node_cost = mu * inst.D[pos, :]
    return inst
