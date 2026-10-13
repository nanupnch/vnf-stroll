# R1: exact TOP-1 optimum on unweighted fat-trees

TOP-1 places a chain of n VNFs on n distinct switches for one VM pair. With unit link weights its
cost is the hop count of the walk s → p1 → … → pn → t. All values are exact: iterative-deepening
search over walks (`vnfstroll/exact/search.py`), each witness placement re-costed by the independent
checker (`vnfstroll/checker.py`), and the representative pairs of k = 4, 8 re-solved by the MILP (`vnfstroll/exact/milp.py`).

Every ordered pair of distinct hosts is solved for k = 4 (240 pairs), 6 (2,862 pairs), 8 (16,256 pairs); one pair per type for k = 10, 12, 14, 16.

## Finding

The optimum depends only on n and on whether the hosts share a rack, a pod or neither.
For k ≥ 8 it always equals the counting lower bound `top1_lower_bound`:

- a walk through n distinct switches has at least n + 1 hops, and at least d(s, t) hops;
- the fat-tree is bipartite, so every host-to-host walk has even length;
- for two hosts on one edge switch, the walk starts and ends at that switch, so it needs n + 2 hops (n ≥ 2).

For k = 4 and 6 a few values exceed the bound by 2, because their pods have too few edge and
aggregation switches for the shorter walks (marked * below).

### k = 4 (all 240 host pairs)

| pair type \ n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same-rack | 2 | 4 | 6 | 6 | 8 | 10* | 10 | 10 | 12 | 12 | 14 | 14 | 16 |
| same-pod | 4 | 4 | 4 | 6 | 8* | 8 | 10* | 10 | 10 | 12 | 12 | 14 | 14 |
| cross-pod | 6 | 6 | 6 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 | 14 | 14 |
| lower bound (rack / pod / cross) | 2/4/6 | 4/4/6 | 6/4/6 | 6/6/6 | 8/6/6 | 8/8/8 | 10/8/8 | 10/10/10 | 12/10/10 | 12/12/12 | 14/12/12 | 14/14/14 | 16/14/14 |

### k = 6 (all 2,862 host pairs)

| pair type \ n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same-rack | 2 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 | 14 | 14 | 16 |
| same-pod | 4 | 4 | 4 | 6 | 6 | 8 | 10* | 10 | 10 | 12 | 12 | 14 | 14 |
| cross-pod | 6 | 6 | 6 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 | 14 | 14 |
| lower bound (rack / pod / cross) | 2/4/6 | 4/4/6 | 6/4/6 | 6/6/6 | 8/6/6 | 8/8/8 | 10/8/8 | 10/10/10 | 12/10/10 | 12/12/12 | 14/12/12 | 14/14/14 | 16/14/14 |

### k = 8 (all 16,256 host pairs)

| pair type \ n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same-rack | 2 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 | 14 | 14 | 16 |
| same-pod | 4 | 4 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 | 14 | 14 |
| cross-pod | 6 | 6 | 6 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 | 14 | 14 |
| lower bound (rack / pod / cross) | 2/4/6 | 4/4/6 | 6/4/6 | 6/6/6 | 8/6/6 | 8/8/8 | 10/8/8 | 10/10/10 | 12/10/10 | 12/12/12 | 14/12/12 | 14/14/14 | 16/14/14 |

### k = 10, 12, 14, 16 (one pair per type)

Every value equals the lower bound, i.e. the same table as k = 8 (see `r1_top1.csv`).

Values above the bound: k=4 same-rack n=6: 10 (bound 8); k=4 same-pod n=5: 8 (bound 6); k=4 same-pod n=7: 10 (bound 8); k=6 same-pod n=7: 10 (bound 8).

All pairs of the same type have the same optimum (k = 4, 6, 8): yes.

## Mean optimum over uniformly random distinct host pairs (hops)

| k \ n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 5.467 | 5.600 | 5.733 | 6.000 | 6.400 | 8.133 | 8.400 | 10.000 | 10.133 | 12.000 | 12.133 | 14.000 | 14.133 |
| 6 | 5.623 | 5.698 | 5.774 | 6.000 | 6.075 | 8.000 | 8.302 | 10.000 | 10.075 | 12.000 | 12.075 | 14.000 | 14.075 |
| 8 | 5.717 | 5.764 | 5.811 | 6.000 | 6.047 | 8.000 | 8.047 | 10.000 | 10.047 | 12.000 | 12.047 | 14.000 | 14.047 |

## Solve time

| k | host pairs | searches | mean ms per search | max ms |
|---|---|---|---|---|
| 4 | 240 | 3,120 | 0.01 | 1.1 |
| 6 | 2,862 | 37,206 | 0.02 | 7.2 |
| 8 | 16,256 | 211,328 | 0.03 | 0.5 |
| 10 | 3 | 39 | 0.10 | 0.4 |
| 12 | 3 | 39 | 0.20 | 0.7 |
| 14 | 3 | 39 | 0.39 | 1.3 |
| 16 | 3 | 39 | 0.68 | 2.1 |

MILP cross-check: 78 representative cases, all proven optimal and equal to the search; slowest 12.22 s (k=8, same-rack, n=13).

## Run

- command: `python -m experiments.r1_top1`
- date (UTC): 2026-10-13 16:16
- code version: 3ca4afa
- python / numpy / scipy: 3.14.6 / 2.5.4 / 1.18.1
- machine: Darwin arm64, Apple M1
- wall time: 78 s
