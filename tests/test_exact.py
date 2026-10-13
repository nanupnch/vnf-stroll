import itertools

import numpy as np
import pytest

from vnfstroll import Checker, FatTree, Flows, chain_instance, paper_workload
from vnfstroll.exact import (
    InnerTable,
    solve_brute_force,
    solve_milp,
    solve_top_decomposition,
    top1_closed_form,
    top1_lower_bound,
    top1_search,
    top_spread_cost,
)

# Cases where the TOP-1 optimum exceeds top1_lower_bound by 2 because the pods are small.
SMALL_K_EXCEPTIONS = {
    (4, "same-rack", 6): 10,
    (4, "same-pod", 5): 8,
    (4, "same-pod", 7): 10,
    (6, "same-pod", 7): 10,
}


@pytest.mark.parametrize("k", [4, 6, 8])
def test_top1_optimum_is_the_lower_bound_except_on_small_pods(k):
    ft = FatTree(k)
    for pair_type, (s, t) in ft.representative_pairs().items():
        for n in range(1, 14):
            expected = SMALL_K_EXCEPTIONS.get((k, pair_type, n), top1_lower_bound(pair_type, n))
            assert top1_search(ft, s, t, n).cost == expected, (k, pair_type, n)


@pytest.mark.parametrize("k", [8, 10, 12])
def test_top1_closed_form(k):
    ft = FatTree(k)
    for pair_type, (s, t) in ft.representative_pairs().items():
        for n in range(1, 14):
            assert top1_search(ft, s, t, n).cost == top1_closed_form(k, pair_type, n)


def test_top1_depends_only_on_pair_type_k4():
    ft = FatTree(4)
    for n in range(1, 9):
        seen = {}
        for s, t in itertools.permutations(ft.hosts.tolist(), 2):
            seen.setdefault(ft.pair_type(s, t), set()).add(top1_search(ft, s, t, n).cost)
        assert all(len(v) == 1 for v in seen.values()), (n, seen)


def test_search_milp_and_brute_force_agree_k4():
    ft = FatTree(4)
    checker = Checker(ft)
    for pair_type, (s, t) in ft.representative_pairs().items():
        flows = Flows.single(s, t)
        for n in range(1, 9):
            search = top1_search(ft, s, t, n)
            inst = chain_instance(ft, flows, n)
            milp = solve_milp(inst)
            assert milp.optimal
            assert milp.cost == pytest.approx(search.cost), (pair_type, n)
            assert checker.cost(flows, search.placement) == search.cost
            if n <= 4:
                assert solve_brute_force(inst).cost == pytest.approx(search.cost)


def test_milp_matches_brute_force_on_weighted_fat_tree():
    rng = np.random.default_rng(5)
    ft = FatTree.with_uniform_weights(4, rng)
    flows = paper_workload(ft, 30, rng)
    checker = Checker(ft)
    for n in (1, 2, 3, 4):
        inst = chain_instance(ft, flows, n)
        milp, brute = solve_milp(inst), solve_brute_force(inst)
        assert milp.cost == pytest.approx(brute.cost, rel=1e-9)
        assert checker.cost(flows, milp.placement) == pytest.approx(milp.cost, rel=1e-9)


def test_checker_matches_instance_cost():
    rng = np.random.default_rng(7)
    ft = FatTree.with_uniform_weights(4, rng)
    flows = paper_workload(ft, 25, rng)
    checker = Checker(ft)
    inst = chain_instance(ft, flows, 5)
    for _ in range(50):
        idx = rng.choice(inst.m, size=5, replace=False)
        assert checker.cost(flows, inst.candidates[idx]) == pytest.approx(inst.cost(idx), rel=1e-12)


@pytest.mark.parametrize("k", [4, 6, 8])
def test_inner_table_has_17_classes(k):
    table = InnerTable(FatTree(k), m_max=6, check="all" if k == 4 else "sample")
    assert table.num_classes == 17


def test_decomposition_matches_milp_k4():
    ft = FatTree(4)
    table = InnerTable(ft, m_max=6)
    for seed in range(3):
        flows = paper_workload(ft, 60, np.random.default_rng(seed))
        for n in range(1, 9):
            inst = chain_instance(ft, flows, n)
            assert solve_top_decomposition(ft, inst, table).cost == pytest.approx(solve_milp(inst).cost, rel=1e-9)


def test_decomposition_matches_milp_k8():
    ft = FatTree(8)
    table = InnerTable(ft, m_max=6)
    flows = paper_workload(ft, 200, np.random.default_rng(11))
    checker = Checker(ft)
    for n in (2, 3, 7, 8):
        inst = chain_instance(ft, flows, n)
        dec = solve_top_decomposition(ft, inst, table)
        assert dec.cost == pytest.approx(solve_milp(inst).cost, rel=1e-9)
        assert checker.cost(flows, dec.placement) == pytest.approx(dec.cost, rel=1e-9)


def test_top_spread_cost_k8():
    ft = FatTree(8)
    table = InnerTable(ft, m_max=11)
    flows = paper_workload(ft, 500, np.random.default_rng(2))
    for n in range(1, 14):
        inst = chain_instance(ft, flows, n)
        exact = solve_top_decomposition(ft, inst, table).cost / flows.total_rate
        assert exact == pytest.approx(top_spread_cost(ft, flows, n), rel=1e-12)


def test_top_spread_cost_on_concentrated_traffic():
    """Where the closed form claims to apply it is exact; it declines when one pod holds over half."""
    ft = FatTree(8)
    table = InnerTable(ft, m_max=7)
    rng = np.random.default_rng(4)
    predicted = 0
    for pods in ([0], [0, 1], [0, 1, 2], [0, 1, 2, 3]):
        for _ in range(4):
            flows = paper_workload(ft, 300, rng, source_pods=pods)
            for n in range(1, 10):
                claim = top_spread_cost(ft, flows, n)
                if len(pods) == 1:
                    assert claim is None
                if claim is not None:
                    predicted += 1
                    exact = solve_top_decomposition(ft, chain_instance(ft, flows, n), table).cost
                    assert exact / flows.total_rate == pytest.approx(claim, rel=1e-12)
    assert predicted > 0
