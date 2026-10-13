"""Exact TOP on unweighted fat-trees by fixing the ingress and egress switches.

    C(p) = A(p_1) + B(p_n) + Lambda * H_{n-2}(p_1, p_n),

where H_m(u, v) is the fewest hops from switch u to switch v through m distinct switches other than
u and v. The paper's Algo. 3 uses the same decomposition with a heuristic inner stroll; here the
inner stroll is exact. H_m(u, v) depends only on the class of the ordered pair (layer of u, layer of v,
distance, same pod): 17 classes for k >= 4. ``InnerTable`` solves one pair per class and checks
other members of the class.
"""

from __future__ import annotations

import time

import numpy as np

from ..fattree import FatTree
from ..instance import ChainInstance, Solution
from .search import first_distinct, stroll_search


class InnerTable:
    def __init__(self, ft: FatTree, m_max: int, check: str = "sample", samples: int = 5, seed: int = 0):
        if not ft.unweighted:
            raise ValueError("the inner table assumes unit link weights")
        if check not in ("all", "sample", "none"):
            raise ValueError("check must be 'all', 'sample' or 'none'")
        self.ft = ft
        self.m_max = m_max
        S = ft.num_switches
        self.switch_mask = np.zeros(ft.num_nodes, dtype=bool)
        self.switch_mask[:S] = True

        dist = np.array([ft.hop_distances(u)[:S] for u in range(S)], dtype=np.int64)
        layer, pod = ft.layer[:S], ft.pod[:S]
        same_pod = (pod[:, None] == pod[None, :]) & (pod[:, None] >= 0)
        key = ((layer[:, None] * 3 + layer[None, :]) * 8 + dist) * 2 + same_pod
        np.fill_diagonal(key, -1)
        labels, cls = np.unique(key, return_inverse=True)
        cls = cls.reshape(S, S)
        diagonal_class = int(np.searchsorted(labels, -1))
        self.classes = [c for c in range(len(labels)) if c != diagonal_class]

        rng = np.random.default_rng(seed)
        values = np.full((len(labels), m_max + 1), np.inf)
        self.representatives = {}
        self.checked_pairs = 0
        for c in self.classes:
            members = np.argwhere(cls == c)
            u, v = (int(x) for x in members[0])
            self.representatives[c] = (u, v)
            for m in range(m_max + 1):
                values[c, m] = stroll_search(ft, u, v, m, counted=self.switch_mask)[0]
            if check == "all":
                others = members[1:]
            elif check == "sample":
                pick = rng.choice(len(members), size=min(samples, len(members)), replace=False)
                others = members[pick]
            else:
                others = members[:0]
            for a, b in others:
                for m in range(m_max + 1):
                    got = stroll_search(ft, int(a), int(b), m, counted=self.switch_mask)[0]
                    if got != values[c, m]:
                        raise AssertionError(f"pair ({a},{b}) m={m}: {got} != class value {values[c, m]}")
                self.checked_pairs += 1
        self.values = values
        self.cls = cls

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def matrix(self, m: int) -> np.ndarray:
        """H_m(u, v) for all ordered switch pairs, with +inf on the diagonal."""
        if not 0 <= m <= self.m_max:
            raise ValueError(f"m must be in 0..{self.m_max}")
        return self.values[self.cls, m]

    def rows(self):
        """(class label, representative pair, H_0..H_mmax) for reporting."""
        ft = self.ft
        out = []
        for c in self.classes:
            u, v = self.representatives[c]
            label = f"{ft.describe([u])}-{ft.describe([v])} d={int(self.values[c, 0])}"
            if ft.pod[u] >= 0 and ft.pod[u] == ft.pod[v]:
                label += " same pod"
            out.append((label, int((self.cls == c).sum()), [int(x) for x in self.values[c]]))
        return out


def solve_top_decomposition(ft: FatTree, inst: ChainInstance, table: InnerTable) -> Solution:
    start = time.perf_counter()
    S = ft.num_switches
    if inst.m != S or not np.array_equal(inst.candidates, np.arange(S)):
        raise ValueError("instance candidates must be the fat-tree's switches")
    if inst.node_cost is not None:
        raise ValueError("the decomposition solves TOP; use the MILP for TOM")
    n = inst.n
    if n == 1:
        v = int(np.argmin(inst.A + inst.B))
        placement = (v,)
    else:
        if n - 2 > table.m_max:
            raise ValueError(f"inner table only covers n <= {table.m_max + 2}")
        H = table.matrix(n - 2)
        total = inst.A[:, None] + inst.B[None, :] + inst.total_rate * H
        u, v = (int(x) for x in np.unravel_index(np.argmin(total), total.shape))
        length, walk = stroll_search(ft, u, v, n - 2, counted=table.switch_mask)
        if length != H[u, v]:
            raise AssertionError("inner stroll does not match the table")
        inner = first_distinct(walk[1:-1], table.switch_mask, n - 2, exclude=(u, v))
        placement = (u, *inner, v)
    return Solution(
        cost=inst.cost(np.array(placement)),
        placement=placement,
        method="decomposition",
        optimal=True,
        seconds=time.perf_counter() - start,
        info={"pattern": ft.describe(placement)},
    )
