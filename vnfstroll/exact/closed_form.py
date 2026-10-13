"""Closed forms on unweighted fat-trees.

TOP-1. Hosts and aggregation switches form one side of a bipartition, edge and core switches the
other, so every host-to-host walk has even length, and a walk through n distinct switches needs at
least n + 1 hops. When both hosts hang off the same edge switch the walk leaves and re-enters that
switch, which costs one more position. These give ``top1_lower_bound``; on fat-trees with k >= 8 the
optimum equals it (checked exhaustively in experiments/r1_top1.py), while k = 4 and 6 have a few
exceptions because their pods are too small.

TOP. Every host is exactly 3 hops from every core switch. If no pod holds more than half of the
traffic, the optimal chain starts and ends at core switches (odd n); ``top_spread_cost`` gives the
resulting cost per unit of traffic.
"""

from __future__ import annotations

from ..fattree import FatTree
from ..workload import Flows, pod_shares

PAIR_DISTANCE = {"same-host": 0, "same-rack": 2, "same-pod": 4, "cross-pod": 6}


def parity_lower_bound(d: int, n: int) -> int:
    """max(d, n + 1), rounded up to the parity of d."""
    length = max(d, n + 1)
    return length + (length - d) % 2


def top1_lower_bound(pair_type: str, n: int) -> int:
    """Lower bound on the TOP-1 optimum (hops) for a host pair of the given type."""
    d = PAIR_DISTANCE[pair_type]
    bound = parity_lower_bound(d, n)
    if pair_type in ("same-host", "same-rack") and n >= 2:
        bound = max(bound, parity_lower_bound(d, n + 1))
    return bound


def top1_closed_form(k: int, pair_type: str, n: int) -> int:
    """TOP-1 optimum on an unweighted k-ary fat-tree with k >= 8."""
    if k < 8:
        raise ValueError("the closed form holds for k >= 8; k = 4 and 6 have exceptions")
    return top1_lower_bound(pair_type, n)


def top_spread_cost(ft: FatTree, flows: Flows, n: int) -> float | None:
    """TOP optimum divided by Lambda when no pod is the source of more than half of the traffic or the
    destination of more than half of it; None otherwise.

    Per unit of traffic, entering or leaving the chain at a core switch costs 3, at an aggregation
    switch of pod P 4 - 2 f(P), and at an edge switch 5 - 2 f(P) - 2 r, where f(P) and r are the shares
    of the traffic in that pod and rack. With every pod share at most 1/2, no switch beats a core.
    The chain needs at least n - 1 hops, and an even number of hops when both ends are on the same side
    of the bipartition (cores and edge switches on one side, aggregation switches on the other).
    Odd n: core to core, n + 5. Even n: one end on an aggregation switch of the busiest pod,
    n + 6 - 2 f_max.
    """
    f_src, f_dst = pod_shares(ft, flows)
    if f_src.max() > 0.5 or f_dst.max() > 0.5:
        return None
    if n == 1:
        return 6.0
    if n % 2:
        return n + 5.0
    return n + 6.0 - 2.0 * max(f_src.max(), f_dst.max())
