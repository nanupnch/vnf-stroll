"""Exact solvers: n-stroll search, closed forms, MILP, TOP decomposition and brute force."""

from .brute_force import solve_brute_force
from .closed_form import parity_lower_bound, top1_closed_form, top1_lower_bound, top_spread_cost
from .decomposition import InnerTable, solve_top_decomposition
from .milp import solve_milp
from .search import SearchBudgetExceeded, first_distinct, stroll_search, top1_search

__all__ = [
    "solve_brute_force",
    "parity_lower_bound",
    "top1_closed_form",
    "top1_lower_bound",
    "top_spread_cost",
    "InnerTable",
    "solve_top_decomposition",
    "solve_milp",
    "SearchBudgetExceeded",
    "first_distinct",
    "stroll_search",
    "top1_search",
]
