"""VM flows and the traffic model of the IPDPS 2022 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fattree import FatTree


@dataclass(frozen=True)
class Flows:
    """Communicating VM pairs: flow i goes from host ``src[i]`` to host ``dst[i]`` at rate ``rate[i]``."""

    src: np.ndarray
    dst: np.ndarray
    rate: np.ndarray

    def __post_init__(self):
        src = np.atleast_1d(np.asarray(self.src, dtype=np.int64))
        dst = np.atleast_1d(np.asarray(self.dst, dtype=np.int64))
        rate = np.atleast_1d(np.asarray(self.rate, dtype=float))
        if not (src.shape == dst.shape == rate.shape) or src.ndim != 1:
            raise ValueError("src, dst and rate must be 1-D arrays of equal length")
        if (rate < 0).any():
            raise ValueError("rates must be non-negative")
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "dst", dst)
        object.__setattr__(self, "rate", rate)

    def __len__(self) -> int:
        return len(self.src)

    @property
    def total_rate(self) -> float:
        return float(self.rate.sum())

    @classmethod
    def single(cls, s: int, t: int, rate: float = 1.0) -> Flows:
        return cls([s], [t], [rate])

    def with_rates(self, rate) -> Flows:
        return Flows(self.src, self.dst, rate)


RATE_BANDS = ((0.25, 0.0, 3000.0), (0.70, 3000.0, 7000.0), (0.05, 7000.0, 10000.0))


def paper_rates(rng, size: int) -> np.ndarray:
    """Traffic rates of the paper's experiments: 25% light in [0, 3000), 70% medium in [3000, 7000]
    and 5% heavy in (7000, 10000]. The paper does not say how rates are distributed within a band;
    we draw them uniformly."""
    probs = [b[0] for b in RATE_BANDS]
    band = rng.choice(len(RATE_BANDS), size=size, p=probs)
    lo = np.array([b[1] for b in RATE_BANDS])[band]
    hi = np.array([b[2] for b in RATE_BANDS])[band]
    return rng.uniform(lo, hi)


def paper_workload(ft: FatTree, num_flows: int, rng, same_rack: float = 0.8, source_pods=None) -> Flows:
    """VM pairs as in the paper: with probability ``same_rack`` the two VMs are on different hosts under
    the same edge switch; otherwise the destination is a uniformly random other host.
    Sources are uniform over all hosts, or over the hosts of ``source_pods``."""
    half = ft.half
    if half < 2 and same_rack > 0:
        raise ValueError("same-rack pairs need at least two hosts per rack (k >= 4)")
    pods = np.arange(ft.k) if source_pods is None else np.asarray(source_pods, dtype=np.int64)
    hosts_per_pod = half * half

    pod = rng.choice(pods, size=num_flows)
    local = rng.integers(0, hosts_per_pod, size=num_flows)
    src = ft.num_switches + pod * hosts_per_pod + local

    dst = np.empty(num_flows, dtype=np.int64)
    in_rack = rng.random(num_flows) < same_rack
    if in_rack.any():
        slot = (src[in_rack] - ft.num_switches) % half
        shift = rng.integers(1, half, size=int(in_rack.sum()))
        dst[in_rack] = src[in_rack] - slot + (slot + shift) % half
    other = ~in_rack
    if other.any():
        pick = rng.integers(0, ft.num_host - 1, size=int(other.sum()))
        src_offset = src[other] - ft.num_switches
        pick = pick + (pick >= src_offset)
        dst[other] = ft.num_switches + pick
    return Flows(src, dst, paper_rates(rng, num_flows))


def pod_to_pod_workload(ft: FatTree, num_flows: int, rng, src_pod: int, dst_pod: int) -> Flows:
    """Directional traffic: every source host is in ``src_pod`` and every destination host in ``dst_pod``."""
    per_pod = ft.half * ft.half
    src = ft.num_switches + src_pod * per_pod + rng.integers(0, per_pod, size=num_flows)
    dst = ft.num_switches + dst_pod * per_pod + rng.integers(0, per_pod, size=num_flows)
    return Flows(src, dst, paper_rates(rng, num_flows))


def eq9_tau(h: float, period: int = 12, tau_min: float = 0.2) -> float:
    """Diurnal traffic factor of the paper's Eq. 9: 0 at the start and end of the period,
    rising linearly to (1 - tau_min) at its midpoint."""
    if h <= 0 or h >= period:
        return 0.0
    if h <= period / 2:
        return 2.0 * h / period * (1.0 - tau_min)
    return 2.0 * (period - h) / period * (1.0 - tau_min)


def diurnal_multiplier(hour: float, start_hour: float, period: int = 12, tau_min: float = 0.2) -> float:
    """Rate multiplier 1 + tau (the curve plotted in the paper's Fig. 8) for a coast whose day starts
    at ``start_hour``: 6 for the east coast and 9 for the west coast."""
    return 1.0 + eq9_tau(hour - start_hour, period, tau_min)


def pod_shares(ft: FatTree, flows: Flows) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of the total rate whose source (resp. destination) host is in each pod."""
    total = flows.total_rate
    f_src = np.bincount(ft.pod[flows.src], weights=flows.rate, minlength=ft.k) / total
    f_dst = np.bincount(ft.pod[flows.dst], weights=flows.rate, minlength=ft.k) / total
    return f_src, f_dst
