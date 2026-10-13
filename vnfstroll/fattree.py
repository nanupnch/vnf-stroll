"""k-ary fat-tree data-center topology (Al-Fares, Loukissas and Vahdat, SIGCOMM 2008)."""

from __future__ import annotations

import math

import numpy as np

from .graph import Graph

CORE, AGG, EDGE, HOST = 0, 1, 2, 3
LAYER_NAMES = {CORE: "core", AGG: "agg", EDGE: "edge", HOST: "host"}
LAYER_LETTERS = {CORE: "c", AGG: "a", EDGE: "e", HOST: "h"}
PAIR_TYPES = ("same-rack", "same-pod", "cross-pod")


class FatTree(Graph):
    """A k-ary fat-tree with k pods, k/2 edge and k/2 aggregation switches per pod,
    (k/2)^2 core switches and k/2 hosts per edge switch.

    Node ids: cores first, then aggregation, edge and host nodes. Core (g, i) is the i-th switch
    of core group g; aggregation switch j of every pod connects to all switches of core group j.
    Hosts are numbered pod by pod and edge by edge, as h1, h2, ... in the paper.
    The candidate locations for VNFs are the switches.
    """

    def __init__(self, k: int, weights=None):
        if k < 2 or k % 2:
            raise ValueError("k must be a positive even integer")
        half = k // 2
        self.k = k
        self.half = half
        self.num_core = half * half
        self.num_agg = k * half
        self.num_edge = k * half
        self.num_host = k * half * half
        self.num_switches = self.num_core + self.num_agg + self.num_edge
        num_nodes = self.num_switches + self.num_host

        layer = np.empty(num_nodes, dtype=np.int64)
        pod = np.full(num_nodes, -1, dtype=np.int64)
        index = np.empty(num_nodes, dtype=np.int64)
        rack = np.full(num_nodes, -1, dtype=np.int64)
        names = [""] * num_nodes

        for g in range(half):
            for i in range(half):
                v = self.core_id(g, i)
                layer[v], index[v], names[v] = CORE, g, f"c{g}.{i}"
        for p in range(k):
            for j in range(half):
                v = self.agg_id(p, j)
                layer[v], pod[v], index[v], names[v] = AGG, p, j, f"a{p}.{j}"
            for e in range(half):
                v = self.edge_id(p, e)
                layer[v], pod[v], index[v], names[v] = EDGE, p, e, f"e{p}.{e}"
                for x in range(half):
                    h = self.host_id(p, e, x)
                    layer[h], pod[h], index[h], rack[h] = HOST, p, e, v
                    names[h] = f"h{h - self.num_switches + 1}"

        links = []
        for p in range(k):
            for j in range(half):
                for i in range(half):
                    links.append((self.agg_id(p, j), self.core_id(j, i)))
            for e in range(half):
                for j in range(half):
                    links.append((self.edge_id(p, e), self.agg_id(p, j)))
                for x in range(half):
                    links.append((self.host_id(p, e, x), self.edge_id(p, e)))

        self.layer = layer
        self.pod = pod
        self.index = index
        self.rack = rack
        super().__init__(num_nodes, links, weights, candidates=np.arange(self.num_switches), names=names)

    @classmethod
    def num_links_for(cls, k: int) -> int:
        return 3 * k**3 // 4

    @classmethod
    def with_uniform_weights(cls, k: int, rng, mean: float = 1.5, var: float = 0.5) -> FatTree:
        """Fat-tree whose link weights are i.i.d. uniform with the given mean and variance.

        The paper's weighted experiments use link delays with mean 1.5 ms and variance 0.5 ms,
        i.e. U[1.5 - sqrt(1.5), 1.5 + sqrt(1.5)] = U[0.275, 2.725].
        """
        half_width = math.sqrt(3.0 * var)
        if mean - half_width <= 0:
            raise ValueError("weights must stay positive")
        weights = rng.uniform(mean - half_width, mean + half_width, size=cls.num_links_for(k))
        return cls(k, weights)

    # node ids
    def core_id(self, group: int, i: int) -> int:
        return group * self.half + i

    def agg_id(self, pod: int, j: int) -> int:
        return self.num_core + pod * self.half + j

    def edge_id(self, pod: int, e: int) -> int:
        return self.num_core + self.num_agg + pod * self.half + e

    def host_id(self, pod: int, e: int, x: int) -> int:
        return self.num_switches + (pod * self.half + e) * self.half + x

    def host(self, i: int) -> int:
        """Node id of host h_i (1-based, numbered as in the paper)."""
        if not 1 <= i <= self.num_host:
            raise ValueError(f"host index {i} out of range 1..{self.num_host}")
        return self.num_switches + i - 1

    @property
    def switches(self) -> np.ndarray:
        return np.arange(self.num_switches)

    @property
    def hosts(self) -> np.ndarray:
        return np.arange(self.num_switches, self.num_nodes)

    def switch_order(self, name: str = "core-agg-edge") -> np.ndarray:
        """Switch ids layer by layer, e.g. 'core-agg-edge' (the id order) or 'edge-agg-core'."""
        layers = {"core": CORE, "agg": AGG, "edge": EDGE}
        parts = name.split("-")
        if sorted(parts) != sorted(layers):
            raise ValueError("order must name the three layers, e.g. 'edge-agg-core'")
        sw = self.switches
        return np.concatenate([sw[self.layer[sw] == layers[p]] for p in parts])

    def pair_type(self, s: int, t: int) -> str:
        """Relation between two hosts: same-host, same-rack, same-pod or cross-pod."""
        if s == t:
            return "same-host"
        if self.rack[s] == self.rack[t]:
            return "same-rack"
        if self.pod[s] == self.pod[t]:
            return "same-pod"
        return "cross-pod"

    def representative_pairs(self) -> dict[str, tuple[int, int]]:
        """One ordered host pair of each type: h1 with h2, with the first host of the next rack,
        and with the first host of the next pod."""
        pairs = {}
        if self.half >= 2:
            pairs["same-rack"] = (self.host(1), self.host(2))
            pairs["same-pod"] = (self.host(1), self.host(self.half + 1))
        pairs["cross-pod"] = (self.host(1), self.host(self.half * self.half + 1))
        return pairs

    def describe(self, placement) -> str:
        """Layer pattern of a placement, e.g. 'c-a-e-a-c'."""
        return "-".join(LAYER_LETTERS[int(self.layer[v])] for v in placement)
