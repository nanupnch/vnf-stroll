import numpy as np
import pytest

from vnfstroll import FatTree, paper_workload
from vnfstroll.workload import diurnal_multiplier, eq9_tau, pod_shares


def test_paper_workload_mix():
    ft = FatTree(8)
    flows = paper_workload(ft, 20000, np.random.default_rng(0))
    assert (flows.src != flows.dst).all()
    assert (ft.layer[flows.src] == 3).all() and (ft.layer[flows.dst] == 3).all()
    same_rack = np.mean(ft.rack[flows.src] == ft.rack[flows.dst])
    assert same_rack == pytest.approx(0.8 + 0.2 * 3 / 127, abs=0.01)
    r = flows.rate
    assert np.mean(r < 3000) == pytest.approx(0.25, abs=0.01)
    assert np.mean((r >= 3000) & (r <= 7000)) == pytest.approx(0.70, abs=0.01)
    assert np.mean(r > 7000) == pytest.approx(0.05, abs=0.01)
    assert r.mean() == pytest.approx(4300, rel=0.02)


def test_source_pods_and_shares():
    ft = FatTree(8)
    flows = paper_workload(ft, 1000, np.random.default_rng(1), source_pods=[2])
    assert (ft.pod[flows.src] == 2).all()
    f_src, f_dst = pod_shares(ft, flows)
    assert f_src[2] == pytest.approx(1.0)
    assert f_dst.sum() == pytest.approx(1.0)


def test_eq9_shape():
    assert eq9_tau(0) == 0 and eq9_tau(12) == 0
    assert eq9_tau(6) == pytest.approx(0.8)
    assert eq9_tau(3) == pytest.approx(0.4)
    assert diurnal_multiplier(12, start_hour=6) == pytest.approx(1.8)  # east coast at noon
    assert diurnal_multiplier(12, start_hour=9) == pytest.approx(1.4)  # west coast at noon
    assert diurnal_multiplier(21, start_hour=9) == pytest.approx(1.0)
