import math

import numpy as np
import pytest

from vnfstroll import FatTree
from vnfstroll.fattree import AGG, CORE, EDGE


@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_sizes_and_bipartite(k):
    ft = FatTree(k)
    assert ft.num_host == k**3 // 4
    assert ft.num_switches == 5 * k * k // 4
    assert ft.num_links == FatTree.num_links_for(k) == 3 * k**3 // 4
    assert ft.bipartite


@pytest.mark.parametrize("k", [4, 6, 8])
def test_host_to_host_distances(k):
    ft = FatTree(k)
    D = ft.distances(ft.hosts)[:, ft.hosts]
    expected = {"same-host": 0, "same-rack": 2, "same-pod": 4, "cross-pod": 6}
    for a, s in enumerate(ft.hosts):
        for b, t in enumerate(ft.hosts):
            assert D[a, b] == expected[ft.pair_type(s, t)]


@pytest.mark.parametrize("k", [4, 6, 8])
def test_host_to_switch_distances(k):
    ft = FatTree(k)
    D = ft.distances(ft.switches)
    for h in ft.hosts:
        for v in ft.switches:
            if ft.layer[v] == CORE:
                expected = 3  # every host is 3 hops from every core switch
            elif ft.layer[v] == AGG:
                expected = 2 if ft.pod[v] == ft.pod[h] else 4
            else:
                assert ft.layer[v] == EDGE
                expected = 1 if ft.rack[h] == v else (3 if ft.pod[v] == ft.pod[h] else 5)
            assert D[v, h] == expected


def test_paper_host_numbering():
    ft = FatTree(4)
    h4, h5 = ft.host(4), ft.host(5)
    assert (ft.pod[h4], ft.index[h4]) == (0, 1)  # pod 0, second edge switch
    assert (ft.pod[h5], ft.index[h5]) == (1, 0)  # pod 1, first edge switch
    assert ft.pair_type(h4, h5) == "cross-pod"
    assert ft.names[h4] == "h4"


def test_uniform_weights_follow_the_paper_delay_model():
    ft = FatTree.with_uniform_weights(16, np.random.default_rng(0))
    w = ft.weights
    assert not ft.unweighted
    assert w.min() >= 1.5 - math.sqrt(1.5) and w.max() <= 1.5 + math.sqrt(1.5)
    assert abs(w.mean() - 1.5) < 0.05
    assert abs(w.var() - 0.5) < 0.05
