"""The worked examples and Theorem 4 of Tran, Sun, Tang and Pan (IPDPS 2022)."""

import numpy as np
import pytest

from vnfstroll import Checker, FatTree, Flows, Graph, chain_instance, migration_instance, paper_rates, paper_workload
from vnfstroll.exact import solve_brute_force, solve_milp, top1_search


def fig4a_graph():
    """Fig. 4(a): hosts s and t, switches A, B, C, D."""
    s, a, b, c, d, t = range(6)
    links = [(s, a), (a, b), (b, t), (s, d), (d, t), (t, c)]
    weights = [3, 1, 3, 3, 2, 0.5]
    return Graph(6, links, weights, candidates=[a, b, c, d], names=["s", "A", "B", "C", "D", "t"])


def test_example2_optimal_2_stroll_costs_6():
    g = fig4a_graph()
    flows = Flows.single(0, 5)
    inst = chain_instance(g, flows, 2)
    assert solve_milp(inst).cost == pytest.approx(6)
    assert solve_brute_force(inst).cost == pytest.approx(6)
    checker = Checker(g)
    assert checker.cost(flows, (4, 3)) == pytest.approx(6)  # walk s, D, t, C, t
    assert checker.cost(flows, (1, 2)) == pytest.approx(7)  # path s, A, B, t


def test_example3_seven_vnfs_from_h4_to_h5_cost_8():
    ft = FatTree(4)
    s, t = ft.host(4), ft.host(5)
    flows = Flows.single(s, t)
    sol = top1_search(ft, s, t, 7)
    assert sol.cost == 8
    assert len(set(sol.placement)) == 7
    assert Checker(ft).cost(flows, sol.placement) == 8
    assert solve_milp(chain_instance(ft, flows, 7)).cost == pytest.approx(8)


def test_example1_migration_on_a_k2_fat_tree():
    ft = FatTree(2)  # the linear network h1 - e - a - c - a - e - h2
    h1, h2 = ft.host(1), ft.host(2)
    flows = Flows([h1, h2], [h1, h2], [100, 1])
    checker = Checker(ft)
    top = solve_milp(chain_instance(ft, flows, 2))
    assert top.cost == pytest.approx(100 * 4 + 1 * 10)
    later = flows.with_rates([1, 100])
    stay = checker.cost(later, top.placement)
    assert stay == pytest.approx(1004)
    tom = solve_milp(migration_instance(ft, later, top.placement, mu=1.0))
    assert tom.cost == pytest.approx(416)
    assert checker.cost(later, tom.placement, previous=top.placement, mu=1.0) == pytest.approx(416)
    assert checker.migration_distance(top.placement, tom.placement) == 6
    assert 1 - tom.cost / stay == pytest.approx(0.586, abs=0.001)


@pytest.mark.parametrize("n", [1, 3, 4, 6])
def test_theorem4_tom_with_zero_migration_cost_is_top(n):
    ft = FatTree(4)
    rng = np.random.default_rng(n)
    flows = paper_workload(ft, 40, rng)
    current = solve_milp(chain_instance(ft, flows, n)).placement
    later = flows.with_rates(paper_rates(rng, len(flows)))
    tom = solve_milp(migration_instance(ft, later, current, mu=0.0))
    top = solve_milp(chain_instance(ft, later, n))
    assert tom.cost == pytest.approx(top.cost)
