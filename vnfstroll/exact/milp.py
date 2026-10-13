"""Mixed-integer programs for TOP-1, TOP and TOM on any weighted graph, solved with HiGHS.

Variables: x[j, v] = 1 if VNF f_j is on candidate v (binary), and y[j, u, v] >= 0 for the transition
from f_j at u to f_{j+1} at v (u != v). Each VNF gets one candidate, each candidate holds at most one
VNF of the chain, and y carries one unit of flow from layer j to layer j + 1:

    sum_v y[j, u, v] = x[j, u]        sum_u y[j, u, v] = x[j + 1, v].

With x integral, y is forced to the transitions of the placement, so y can stay continuous.
Objective: sum_v A(v) x[1, v] + B(v) x[n, v] + Lambda sum c(u, v) y[j, u, v] + node costs.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..instance import ChainInstance, Solution


class _Model:
    """A sparse MILP assembled block by block."""

    def __init__(self):
        self.c, self.integer, self.ub = [], [], []
        self.rows, self.cols, self.vals = [], [], []
        self.lower, self.upper = [], []
        self.ncols = 0
        self.nrows = 0

    def add_columns(self, cost, integer: bool, ub: float) -> np.ndarray:
        cost = np.asarray(cost, dtype=float).ravel()
        idx = self.ncols + np.arange(len(cost))
        self.c.append(cost)
        self.integer.append(np.full(len(cost), 1.0 if integer else 0.0))
        self.ub.append(np.full(len(cost), ub))
        self.ncols += len(cost)
        return idx

    def add_rows(self, count: int, lower, upper) -> int:
        first = self.nrows
        self.lower.append(np.broadcast_to(np.asarray(lower, dtype=float), (count,)).copy())
        self.upper.append(np.broadcast_to(np.asarray(upper, dtype=float), (count,)).copy())
        self.nrows += count
        return first

    def add_entries(self, r, cols, value) -> None:
        cols = np.asarray(cols, dtype=np.int64)
        self.rows.append(np.broadcast_to(np.asarray(r, dtype=np.int64), cols.shape).ravel())
        self.cols.append(cols.ravel())
        self.vals.append(np.broadcast_to(np.asarray(value, dtype=float), cols.shape).ravel())

    def solve(self, time_limit, mip_rel_gap, verbose):
        matrix = coo_matrix(
            (np.concatenate(self.vals), (np.concatenate(self.rows), np.concatenate(self.cols))),
            shape=(self.nrows, self.ncols),
        ).tocsr()
        options = {"disp": verbose, "mip_rel_gap": mip_rel_gap}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        res = milp(
            np.concatenate(self.c),
            constraints=LinearConstraint(matrix, np.concatenate(self.lower), np.concatenate(self.upper)),
            integrality=np.concatenate(self.integer),
            bounds=Bounds(np.zeros(self.ncols), np.concatenate(self.ub)),
            options=options,
        )
        info = {
            "status": int(res.status),
            "message": res.message,
            "gap": getattr(res, "mip_gap", None),
            "nodes": getattr(res, "mip_node_count", None),
            "variables": self.ncols,
        }
        return res, info


def _add_chain(model: _Model, inst: ChainInstance) -> np.ndarray:
    """Add one chain's variables and constraints; returns the column index of x[j, v] as an (n, m) array."""
    m, n = inst.m, inst.n
    cx = np.zeros((n, m))
    cx[0] += inst.A
    cx[n - 1] += inst.B
    if inst.node_cost is not None:
        cx += inst.node_cost
    x = model.add_columns(cx, integer=True, ub=1.0).reshape(n, m)
    # each VNF on exactly one candidate
    first = model.add_rows(n, 1.0, 1.0)
    model.add_entries(first + np.repeat(np.arange(n), m), x.ravel(), 1.0)
    # each candidate holds at most one VNF of the chain
    first = model.add_rows(m, -np.inf, 1.0)
    model.add_entries(first + np.tile(np.arange(m), n), x.ravel(), 1.0)
    if n > 1:
        off_u, off_v = np.nonzero(~np.eye(m, dtype=bool))
        trans_cost = inst.total_rate * inst.D[off_u, off_v]
        for j in range(n - 1):
            y = model.add_columns(trans_cost, integer=False, ub=np.inf)
            out_row = model.add_rows(m, 0.0, 0.0)  # flow out of (j, u)
            model.add_entries(out_row + off_u, y, 1.0)
            model.add_entries(out_row + np.arange(m), x[j], -1.0)
            in_row = model.add_rows(m, 0.0, 0.0)  # flow into (j + 1, v)
            model.add_entries(in_row + off_v, y, 1.0)
            model.add_entries(in_row + np.arange(m), x[j + 1], -1.0)
    return x


def _placement(res, x: np.ndarray) -> np.ndarray:
    X = res.x[x]
    idx = X.argmax(axis=1)
    if not np.all(X[np.arange(len(idx)), idx] > 0.5):
        raise RuntimeError("MILP returned a fractional assignment")
    return idx


def solve_milp(
    inst: ChainInstance,
    time_limit: float | None = None,
    mip_rel_gap: float = 1e-9,
    verbose: bool = False,
) -> Solution:
    start = time.perf_counter()
    model = _Model()
    x = _add_chain(model, inst)
    res, info = model.solve(time_limit, mip_rel_gap, verbose)
    seconds = time.perf_counter() - start
    bound = getattr(res, "mip_dual_bound", None)
    if res.x is None:
        return Solution(np.inf, (), "milp", False, bound, seconds, info)
    idx = _placement(res, x)
    return Solution(
        cost=inst.cost(idx),
        placement=tuple(int(v) for v in inst.candidates[idx]),
        method="milp",
        optimal=res.status == 0,
        bound=bound,
        seconds=seconds,
        info=info,
    )
