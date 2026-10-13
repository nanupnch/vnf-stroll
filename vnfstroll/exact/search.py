"""Exact n-stroll search on unweighted graphs.

The minimum cost of an s-t walk that visits at least n distinct candidate nodes equals the TOP-1
optimum: a placement's cost is the length of the walk that joins its switches by shortest paths,
and the first n distinct switches of any walk give a placement that costs no more than the walk.
"""

from __future__ import annotations

import time

import numpy as np

from ..graph import Graph
from ..instance import Solution


class SearchBudgetExceeded(RuntimeError):
    pass


def stroll_search(graph: Graph, s: int, t: int, n: int, counted=None, max_expansions: int = 50_000_000):
    """Fewest hops of an s-t walk that visits at least ``n`` distinct counted nodes other than s and t.

    Iterative deepening on the walk length L, starting from max(d(s,t), n+1) (rounded up to the parity
    of d(s,t) on bipartite graphs); each L is searched exhaustively by depth-first search with two
    valid prunings: the walk must still be able to reach t, and enough steps must remain to collect
    the missing nodes. ``counted`` is a boolean mask (default: the graph's candidates).
    Returns (L, walk) with walk = [s, ..., t].
    """
    if not graph.unweighted:
        raise ValueError("stroll_search needs unit link weights; use the MILP on weighted graphs")
    num = graph.num_nodes
    if counted is None:
        mask = np.zeros(num, dtype=bool)
        mask[graph.candidates] = True
    else:
        mask = np.asarray(counted, dtype=bool).copy()
    mask[s] = mask[t] = False
    if n > int(mask.sum()):
        raise ValueError(f"cannot visit {n} distinct nodes: only {int(mask.sum())} are available")
    counted = mask.tolist()

    adj = graph.adj
    dist_t = graph.hop_distances(t)
    if dist_t[s] < 0:
        raise ValueError("t is not reachable from s")
    if n == 0 and s == t:
        return 0, [s]

    t_neighbors = set(adj[t])
    t_leaf = len(adj[t]) == 1
    gate = adj[t][0] if t_leaf else -1
    gate_counts = t_leaf and counted[gate]
    # A non-counted leaf other than t can only be entered and left again, which never helps.
    skip = [(not counted[v]) and len(adj[v]) == 1 and v != t for v in range(num)]
    if t_leaf:
        skip[t] = True  # entering t before the last step is useless when t is a leaf

    visits = [0] * num
    walk = [s]
    expansions = 0

    def dfs(v: int, r: int, found: int) -> bool:
        nonlocal expansions
        if dist_t[v] > r:
            return False
        if r == 1:
            return found >= n and v in t_neighbors
        cap = r - 1  # interior positions still to come
        if gate_counts and visits[gate] > 0:
            cap -= 1  # the last interior node is t's only neighbour, already counted
        if found + cap < n:
            return False
        expansions += 1
        if expansions > max_expansions:
            raise SearchBudgetExceeded(f"more than {max_expansions} expansions")
        for w in adj[v]:
            if skip[w]:
                continue
            c = visits[w]
            visits[w] = c + 1
            walk.append(w)
            if dfs(w, r - 1, found + (1 if counted[w] and c == 0 else 0)):
                return True
            walk.pop()
            visits[w] = c
        return False

    step = 2 if graph.bipartite else 1
    length = max(dist_t[s], n + 1) if n > 0 else dist_t[s]
    if graph.bipartite and (length - dist_t[s]) % 2:
        length += 1
    while length <= 4 * num + n:
        if dfs(s, length, 0):
            return length, walk + [t]
        length += step
    raise RuntimeError("no walk found")  # unreachable for connected graphs


def first_distinct(walk, counted, n: int, exclude=()) -> list[int]:
    """The first ``n`` distinct counted nodes along ``walk``, in order of first visit."""
    seen, out = set(exclude), []
    for v in walk:
        if counted[v] and v not in seen:
            seen.add(v)
            out.append(int(v))
            if len(out) == n:
                break
    return out


def top1_search(graph: Graph, s: int, t: int, n: int) -> Solution:
    """Exact TOP-1 on an unweighted graph for a unit-rate flow; the cost is in hops."""
    start = time.perf_counter()
    length, walk = stroll_search(graph, s, t, n)
    mask = np.zeros(graph.num_nodes, dtype=bool)
    mask[graph.candidates] = True
    placement = first_distinct(walk[1:-1], mask, n, exclude=(s, t))
    return Solution(
        cost=float(length),
        placement=tuple(placement),
        method="search",
        optimal=True,
        seconds=time.perf_counter() - start,
        info={"walk": walk},
    )
