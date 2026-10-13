"""Exact baselines for VNF placement and migration in fat-tree data centers."""

from .graph import Graph
from .fattree import FatTree
from .workload import Flows, paper_rates, paper_workload
from .instance import ChainInstance, Solution, chain_instance, migration_instance
from .checker import Checker

__all__ = [
    "Graph",
    "FatTree",
    "Flows",
    "paper_rates",
    "paper_workload",
    "ChainInstance",
    "Solution",
    "chain_instance",
    "migration_instance",
    "Checker",
]
