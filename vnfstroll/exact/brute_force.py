"""Enumeration of all ordered placements, for checking the other solvers on small instances."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

from ..instance import ChainInstance, Solution


def solve_brute_force(inst: ChainInstance, max_placements: int = 3_000_000) -> Solution:
    start = time.perf_counter()
    m, n = inst.m, inst.n
    count = math.perm(m, n)
    if count > max_placements:
        raise ValueError(f"{count} placements exceed the limit of {max_placements}")
    P = np.fromiter(itertools.chain.from_iterable(itertools.permutations(range(m), n)), dtype=np.int32)
    P = P.reshape(count, n)
    cost = inst.A[P[:, 0]] + inst.B[P[:, -1]]
    for j in range(n - 1):
        cost = cost + inst.total_rate * inst.D[P[:, j], P[:, j + 1]]
    if inst.node_cost is not None:
        for j in range(n):
            cost = cost + inst.node_cost[j, P[:, j]]
    best = int(np.argmin(cost))
    return Solution(
        cost=inst.cost(P[best]),
        placement=tuple(int(v) for v in inst.candidates[P[best]]),
        method="brute-force",
        optimal=True,
        seconds=time.perf_counter() - start,
        info={"placements": count},
    )
