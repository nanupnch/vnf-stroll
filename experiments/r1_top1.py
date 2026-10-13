"""R1: exact TOP-1 optimum on unweighted fat-trees.

Every ordered pair of distinct hosts for k = 4, 6, 8, and one pair of each type for k = 10..16,
n = 1..13. Each optimum comes from the exact search, its witness placement is re-costed by the
independent checker, and the representative pairs are re-solved by the MILP.
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict

from experiments.common import RESULTS, md_table, metadata, write_csv, write_report

from vnfstroll import Checker, FatTree, Flows, chain_instance
from vnfstroll.exact import solve_milp, top1_lower_bound, top1_search
from vnfstroll.exact.closed_form import PAIR_DISTANCE
from vnfstroll.fattree import PAIR_TYPES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-pairs", type=int, nargs="*", default=[4, 6, 8], help="k values solved for every host pair")
    ap.add_argument("--representatives", type=int, nargs="*", default=[10, 12, 14, 16], help="k values solved for one pair per type")
    ap.add_argument("--milp", type=int, nargs="*", default=[4, 8], help="k values whose representative pairs are re-solved by the MILP")
    ap.add_argument("--n-max", type=int, default=13)
    args = ap.parse_args()
    started = time.perf_counter()
    ns = range(1, args.n_max + 1)

    rows, tables, means, timing, milp_runs = [], {}, {}, {}, []
    for k in sorted(set(args.all_pairs) | set(args.representatives)):
        ft = FatTree(k)
        checker = Checker(ft)
        reps = ft.representative_pairs()
        every = k in args.all_pairs
        pairs = list(itertools.permutations(ft.hosts.tolist(), 2)) if every else list(reps.values())
        values = defaultdict(set)
        count = defaultdict(int)
        total = defaultdict(float)
        seconds, slowest, searches = 0.0, 0.0, 0
        t0 = time.perf_counter()
        for s, t in pairs:
            ptype = ft.pair_type(s, t)
            flows = Flows.single(s, t)
            for n in ns:
                sol = top1_search(ft, s, t, n)
                if checker.cost(flows, sol.placement) != sol.cost:
                    raise AssertionError(f"checker disagrees: k={k} {ptype} n={n}")
                values[ptype, n].add(int(sol.cost))
                count[ptype, n] += 1
                total[ptype, n] += sol.cost
                seconds += sol.seconds
                slowest = max(slowest, sol.seconds)
                searches += 1
        timing[k] = (len(pairs), searches, seconds, slowest, time.perf_counter() - t0)

        milp = {}
        if k in args.milp:
            for ptype, (s, t) in reps.items():
                for n in ns:
                    sol = solve_milp(chain_instance(ft, Flows.single(s, t), n))
                    milp[ptype, n] = sol
                    milp_runs.append((k, ptype, n, sol.cost, sol.seconds, sol.optimal))

        table = {}
        for ptype in PAIR_TYPES:
            for n in ns:
                vals = values[ptype, n]
                opt = min(vals)
                bound = top1_lower_bound(ptype, n)
                m = milp.get((ptype, n))
                if m is not None and (not m.optimal or abs(m.cost - opt) > 1e-6):
                    raise AssertionError(f"MILP disagrees: k={k} {ptype} n={n}: {m.cost} vs {opt}")
                table[ptype, n] = (opt, opt > bound)
                rows.append({
                    "k": k,
                    "pair_type": ptype,
                    "host_distance": PAIR_DISTANCE[ptype],
                    "n": n,
                    "pairs_solved": count[ptype, n],
                    "optimum_hops": opt,
                    "same_for_all_pairs": len(vals) == 1,
                    "lower_bound": bound,
                    "above_bound": opt - bound,
                    "milp_hops": "" if m is None else round(m.cost, 6),
                    "milp_seconds": "" if m is None else round(m.seconds, 3),
                })
        tables[k] = table
        if every:
            pairs_total = sum(count[p, 1] for p in PAIR_TYPES)
            means[k] = [sum(total[p, n] for p in PAIR_TYPES) / pairs_total for n in ns]
        print(f"k={k}: {len(pairs)} pairs, {searches} searches in {timing[k][4]:.1f} s", flush=True)

    write_csv(RESULTS / "r1_top1.csv", rows)
    meta = metadata(**{"wall time": f"{time.perf_counter() - started:.0f} s"})
    write_report(RESULTS / "r1_top1.md", "R1: exact TOP-1 optimum on unweighted fat-trees", meta, report(args, ns, tables, means, timing, milp_runs, rows))
    print("wrote results/r1_top1.csv and results/r1_top1.md")


def report(args, ns, tables, means, timing, milp_runs, rows) -> str:
    large = [r for r in rows if r["k"] >= 8]
    if any(r["above_bound"] for r in large):
        claim = "For k ≥ 8 it equals the counting lower bound `top1_lower_bound` except where marked * below:"
    else:
        claim = "For k ≥ 8 it always equals the counting lower bound `top1_lower_bound`:"
    out = [
        "TOP-1 places a chain of n VNFs on n distinct switches for one VM pair. With unit link weights its",
        "cost is the hop count of the walk s → p1 → … → pn → t. All values are exact: iterative-deepening",
        "search over walks (`vnfstroll/exact/search.py`), each witness placement re-costed by the independent",
        "checker (`vnfstroll/checker.py`), and the representative pairs of k = "
        + ", ".join(str(k) for k in args.milp)
        + " re-solved by the MILP (`vnfstroll/exact/milp.py`).",
        "",
        "Every ordered pair of distinct hosts is solved for k = "
        + ", ".join(f"{k} ({timing[k][0]:,} pairs)" for k in args.all_pairs)
        + "; one pair per type for k = "
        + ", ".join(str(k) for k in args.representatives)
        + ".",
        "",
        "## Finding",
        "",
        "The optimum depends only on n and on whether the hosts share a rack, a pod or neither.",
        claim,
        "",
        "- a walk through n distinct switches has at least n + 1 hops, and at least d(s, t) hops;",
        "- the fat-tree is bipartite, so every host-to-host walk has even length;",
        "- for two hosts on one edge switch, the walk starts and ends at that switch, so it needs n + 2 hops (n ≥ 2).",
        "",
        "For k = 4 and 6 a few values exceed the bound by 2, because their pods have too few edge and",
        "aggregation switches for the shorter walks (marked * below).",
        "",
    ]
    for k in args.all_pairs:
        table = tables[k]
        out.append(f"### k = {k} (all {timing[k][0]:,} host pairs)")
        out.append("")
        body = [[p] + [f"{table[p, n][0]}{'*' if table[p, n][1] else ''}" for n in ns] for p in PAIR_TYPES]
        body.append(["lower bound (rack / pod / cross)"] + [
            "/".join(str(top1_lower_bound(p, n)) for p in PAIR_TYPES) for n in ns
        ])
        out.append(md_table(["pair type \\ n", *ns], body))
        out.append("")
    tight = [k for k in args.representatives if not any(tables[k][key][1] for key in tables[k])]
    if tight:
        out.append(
            f"### k = {', '.join(str(k) for k in tight)} (one pair per type)\n\n"
            "Every value equals the lower bound, i.e. the same table as k = 8 (see `r1_top1.csv`)."
        )
        out.append("")

    exceptions = [r for r in rows if r["above_bound"]]
    out.append("Values above the bound: " + (
        "; ".join(f"k={r['k']} {r['pair_type']} n={r['n']}: {r['optimum_hops']} (bound {r['lower_bound']})" for r in exceptions)
        or "none"
    ) + ".")
    out.append("")
    uniform = all(r["same_for_all_pairs"] for r in rows if r["k"] in args.all_pairs)
    out.append(
        f"All pairs of the same type have the same optimum (k = {', '.join(str(k) for k in args.all_pairs)}): "
        f"{'yes' if uniform else 'NO'}."
    )
    out.append("")

    out.append("## Mean optimum over uniformly random distinct host pairs (hops)")
    out.append("")
    out.append(md_table(["k \\ n", *ns], [[k] + [f"{v:.3f}" for v in vals] for k, vals in means.items()]))
    out.append("")

    out.append("## Solve time")
    out.append("")
    trows = []
    for k, (pairs, searches, secs, slowest, wall) in timing.items():
        trows.append([k, f"{pairs:,}", f"{searches:,}", f"{1000 * secs / searches:.2f}", f"{1000 * slowest:.1f}"])
    out.append(md_table(["k", "host pairs", "searches", "mean ms per search", "max ms"], trows))
    out.append("")
    if milp_runs:
        worst = max(milp_runs, key=lambda r: r[4])
        out.append(
            f"MILP cross-check: {len(milp_runs)} representative cases, all proven optimal and equal to the search; "
            f"slowest {worst[4]:.2f} s (k={worst[0]}, {worst[1]}, n={worst[2]})."
        )
    return "\n".join(out)


if __name__ == "__main__":
    main()
