"""R2: exact TOP at the settings of the paper's Fig. 9 (unweighted k = 8 fat-tree, the paper's workload).

Fig. 9(a): n = 7 VNFs and l = 500..2000 VM pairs. Fig. 9(b): l = 1000 and n = 3..11. Each setting
averages 20 workloads, and n = 1..13 at l = 1000 is added to check the closed form. The optimum comes
from the ingress/egress decomposition; every placement is re-costed by the independent checker and
every instance is re-solved by the MILP.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import DATA, RESULTS, md_table, mean_ci, metadata, write_csv, write_report
from vnfstroll import Checker, FatTree, chain_instance, paper_workload
from vnfstroll.exact import InnerTable, solve_milp, solve_top_decomposition, top_spread_cost
from vnfstroll.fattree import LAYER_NAMES

FIG9A = [(l, 7) for l in (500, 1000, 1500, 2000)]
FIG9B = [(1000, n) for n in (3, 5, 7, 9, 11)]


def load_paper_values():
    """Plotted Optimal values keyed by (panel, l, n); both panels contain l = 1000, n = 7."""
    with (DATA / "ipdps2022_fig9.csv").open() as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    return {(r["panel"], int(r["l"]), int(r["n"])): float(r["optimal_1e6"]) * 1e6 for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--milp-seeds", type=int, default=20, help="re-solve the first S workloads of each setting by MILP")
    ap.add_argument("--n-max", type=int, default=13)
    args = ap.parse_args()
    started = time.perf_counter()

    ft = FatTree(args.k)
    dist = ft.distances(ft.candidates)
    table = InnerTable(ft, m_max=args.n_max - 2, check="sample", samples=10)
    checker = Checker(ft)
    configs = sorted(set(FIG9A) | set(FIG9B) | {(1000, n) for n in range(1, args.n_max + 1)})
    workloads = {
        (l, seed): paper_workload(ft, l, np.random.default_rng([seed, l]))
        for l in sorted({c[0] for c in configs})
        for seed in range(args.seeds)
    }

    rows = []
    for l, n in configs:
        for seed in range(args.seeds):
            flows = workloads[l, seed]
            inst = chain_instance(ft, flows, n, dist)
            dec = solve_top_decomposition(ft, inst, table)
            if not np.isclose(checker.cost(flows, dec.placement), dec.cost, rtol=1e-9):
                raise AssertionError("checker disagrees")
            predicted = top_spread_cost(ft, flows, n)
            if predicted is None:
                raise AssertionError("workload is not spread over the pods")
            row = {
                "l": l,
                "n": n,
                "seed": seed,
                "total_rate": round(flows.total_rate, 3),
                "optimal_cost": round(dec.cost, 3),
                "cost_per_unit_rate": round(dec.cost / flows.total_rate, 6),
                "closed_form": round(predicted, 6),
                "ingress": LAYER_NAMES[int(ft.layer[dec.placement[0]])],
                "egress": LAYER_NAMES[int(ft.layer[dec.placement[-1]])],
                "pattern": dec.info["pattern"],
                "decomposition_ms": round(1000 * dec.seconds, 3),
                "milp_cost": "",
                "milp_seconds": "",
            }
            if seed < args.milp_seeds:
                sol = solve_milp(inst)
                if not sol.optimal or not np.isclose(sol.cost, dec.cost, rtol=1e-9):
                    raise AssertionError(f"MILP disagrees at l={l} n={n} seed={seed}: {sol.cost} vs {dec.cost}")
                row["milp_cost"] = round(sol.cost, 3)
                row["milp_seconds"] = round(sol.seconds, 3)
            rows.append(row)
        print(f"l={l} n={n} done", flush=True)

    write_csv(RESULTS / "r2_top.csv", rows)
    paper = load_paper_values()
    plot(rows, paper)
    meta = metadata(**{
        "workloads": f"{args.seeds} per setting (seeds 0..{args.seeds - 1}), shared across n",
        "wall time": f"{time.perf_counter() - started:.0f} s",
    })
    write_report(RESULTS / "r2_top.md", "R2: exact TOP at the settings of Fig. 9 (k = 8, unweighted)", meta, report(args, rows, paper, table))
    print("wrote results/r2_top.csv, results/r2_top.md and results/r2_fig9.png")


def select(rows, l, n):
    return [r for r in rows if r["l"] == l and r["n"] == n]


def report(args, rows, paper, table) -> str:
    worst = max(abs(r["cost_per_unit_rate"] - r["closed_form"]) for r in rows)
    agreement = (
        "`top_spread_cost` predicts every instance below exactly."
        if worst < 1e-6
        else f"`top_spread_cost` differs from the exact optimum by up to {worst:.2e} per unit rate."
    )
    out = [
        "TOP places one chain of n VNFs, shared by l VM pairs, on n distinct switches (Eq. 1). The workload",
        "follows the paper: 80% of VM pairs under the same edge switch, rates 25% in [0, 3000), 70% in",
        "[3000, 7000] and 5% in (7000, 10000], drawn uniformly within each band. Every optimum is exact",
        "(decomposition over ingress/egress switches with an exact inner stroll), re-costed by the independent",
        "checker, and re-solved by the MILP for "
        + ("every instance." if args.milp_seeds >= args.seeds else f"the first {args.milp_seeds} of the {args.seeds} workloads of every setting."),
        "",
        "## Finding",
        "",
        "Every host is exactly 3 hops from every core switch. With the paper's workload, traffic is spread",
        "over all pods, so the optimal chain starts and ends at core switches and costs exactly Λ·(n + 5) for",
        "odd n, where Λ is the total rate, independent of where the VMs are. For even n, parity puts one end",
        "on an aggregation switch of the busiest pod: Λ·(n + 6 − 2·f_max), with f_max that pod's share of the",
        "traffic. " + agreement,
        "",
        "## Fig. 9(a): n = 7, varying l",
        "",
    ]
    base = np.mean([r["optimal_cost"] for r in select(rows, 1000, 7)])
    scale = []
    body = []
    for l, n in FIG9A:
        m, h = mean_ci([r["optimal_cost"] for r in select(rows, l, n)])
        p = paper["a", l, n]
        scale.append(m / p)
        body.append([l, f"{m / 1e6:.3f} ± {h / 1e6:.3f}", f"{m / base:.3f}", f"{p / 1e6:.3f}", f"{p / paper['a', 1000, 7]:.3f}", f"{m / p:.2f}"])
    out.append(md_table(["l", "exact optimum (×10⁶, mean ± 95% CI)", "relative to l = 1000", "paper (×10⁶)", "paper, relative to l = 1000", "exact ÷ paper"], body))
    out += ["", "## Fig. 9(b): l = 1000, varying n", ""]
    base = np.mean([r["optimal_cost"] for r in select(rows, 1000, 3)])
    body = []
    for l, n in FIG9B:
        sel = select(rows, l, n)
        m, h = mean_ci([r["optimal_cost"] for r in sel])
        per = [r["cost_per_unit_rate"] for r in sel]
        p = paper["b", l, n]
        scale.append(m / p)
        body.append([
            n, f"{m / 1e6:.3f} ± {h / 1e6:.3f}", f"{min(per):.4f}–{max(per):.4f}", f"{m / base:.4f}",
            f"{(n + 5) / 8:.4f}", f"{p / 1e6:.3f}", f"{p / paper['b', 1000, 3]:.4f}", f"{m / p:.2f}",
        ])
    out.append(md_table(["n", "exact optimum (×10⁶)", "optimum / Λ", "relative to n = 3", "(n + 5) / 8", "paper (×10⁶)", "paper, relative to n = 3", "exact ÷ paper"], body))
    out += [
        "",
        "The workloads are shared across n, so the exact ratios equal (n + 5)/8. With rates drawn as stated",
        f"(mean 4,300 per VM pair), the exact optimum is {min(scale):.2f}–{max(scale):.2f} times the plotted value",
        "at all nine points, i.e. a constant factor of about 10, as a different unit or scale of the rates would",
        "give. Ratios do not depend on that scale, so they are the quantities compared.",
        "",
        "## Closed form on every instance (l = 1000, n = 1..13)",
        "",
    ]
    body = []
    for n in range(1, args.n_max + 1):
        sel = select(rows, 1000, n)
        per = np.array([r["cost_per_unit_rate"] for r in sel])
        pred = np.array([r["closed_form"] for r in sel])
        ends = Counter(f"{r['ingress']} → {r['egress']}" for r in sel)
        dec_ms = np.mean([r["decomposition_ms"] for r in sel])
        milp_s = [r["milp_seconds"] for r in sel if r["milp_seconds"] != ""]
        body.append([
            n, f"{per.min():.4f}–{per.max():.4f}", f"{np.abs(per - pred).max():.1e}",
            ", ".join(f"{k} ({v})" for k, v in ends.most_common()),
            f"{dec_ms:.1f}", f"{np.mean(milp_s):.2f} / {np.max(milp_s):.2f}" if milp_s else "–",
        ])
    out.append(md_table(["n", "optimum / Λ (min–max)", "max |exact − closed form|", "ingress → egress (count)", "decomposition ms", "MILP s (mean / max)"], body))
    out += ["", "## Inner strolls H_m(u, v) used by the decomposition", "",
            "Fewest hops from switch u to switch v through m distinct other switches, per class of ordered switch pairs.", ""]
    body = [[label, count, *vals] for label, count, vals in table.rows()]
    out.append(md_table(["class", "pairs", *[f"m={m}" for m in range(table.m_max + 1)]], body))
    out += ["", f"{table.num_classes} classes; besides one representative, {table.checked_pairs} further pairs were solved and matched their class."]
    return "\n".join(out)


def plot(rows, paper):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ls = [l for l, _ in FIG9A]
    base = np.mean([r["optimal_cost"] for r in select(rows, 1000, 7)])
    exact = [np.mean([r["optimal_cost"] for r in select(rows, l, 7)]) / base for l in ls]
    axes[0].plot(ls, exact, "o-", label="exact optimum")
    axes[0].plot(ls, [paper["a", l, 7] / paper["a", 1000, 7] for l in ls], "s--", label="paper, Optimal")
    axes[0].set(xlabel="number of VM pairs l", ylabel="cost relative to l = 1000", title="Fig. 9(a) setting, n = 7")
    ns = [n for _, n in FIG9B]
    base = np.mean([r["optimal_cost"] for r in select(rows, 1000, 3)])
    exact = [np.mean([r["optimal_cost"] for r in select(rows, 1000, n)]) / base for n in ns]
    axes[1].plot(ns, exact, "o-", label="exact optimum")
    axes[1].plot(ns, [paper["b", 1000, n] / paper["b", 1000, 3] for n in ns], "s--", label="paper, Optimal")
    axes[1].plot(ns, [(n + 5) / 8 for n in ns], ":", color="gray", label="(n + 5) / 8")
    axes[1].set(xlabel="number of VNFs n", ylabel="cost relative to n = 3", title="Fig. 9(b) setting, l = 1000")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_fig9.png", dpi=150)


if __name__ == "__main__":
    main()
