# R2: exact TOP at the settings of Fig. 9 (k = 8, unweighted)

TOP places one chain of n VNFs, shared by l VM pairs, on n distinct switches (Eq. 1). The workload
follows the paper: 80% of VM pairs under the same edge switch, rates 25% in [0, 3000), 70% in
[3000, 7000] and 5% in (7000, 10000], drawn uniformly within each band. Every optimum is exact
(decomposition over ingress/egress switches with an exact inner stroll), re-costed by the independent
checker, and re-solved by the MILP for every instance.

## Finding

Every host is exactly 3 hops from every core switch. With the paper's workload, traffic is spread
over all pods, so the optimal chain starts and ends at core switches and costs exactly Λ·(n + 5) for
odd n, where Λ is the total rate, independent of where the VMs are. For even n, parity puts one end
on an aggregation switch of the busiest pod: Λ·(n + 6 − 2·f_max), with f_max that pod's share of the
traffic. `top_spread_cost` predicts every instance below exactly.

## Fig. 9(a): n = 7, varying l

| l | exact optimum (×10⁶, mean ± 95% CI) | relative to l = 1000 | paper (×10⁶) | paper, relative to l = 1000 | exact ÷ paper |
|---|---|---|---|---|---|
| 500 | 25.942 ± 0.251 | 0.502 | 2.554 | 0.497 | 10.16 |
| 1000 | 51.721 ± 0.326 | 1.000 | 5.142 | 1.000 | 10.06 |
| 1500 | 77.240 ± 0.418 | 1.493 | 7.700 | 1.497 | 10.03 |
| 2000 | 103.021 ± 0.484 | 1.992 | 10.303 | 2.004 | 10.00 |

## Fig. 9(b): l = 1000, varying n

| n | exact optimum (×10⁶) | optimum / Λ | relative to n = 3 | (n + 5) / 8 | paper (×10⁶) | paper, relative to n = 3 | exact ÷ paper |
|---|---|---|---|---|---|---|---|
| 3 | 34.481 ± 0.217 | 8.0000–8.0000 | 1.0000 | 1.0000 | 3.426 | 1.0000 | 10.06 |
| 5 | 43.101 ± 0.272 | 10.0000–10.0000 | 1.2500 | 1.2500 | 4.279 | 1.2490 | 10.07 |
| 7 | 51.721 ± 0.326 | 12.0000–12.0000 | 1.5000 | 1.5000 | 5.131 | 1.4977 | 10.08 |
| 9 | 60.341 ± 0.380 | 14.0000–14.0000 | 1.7500 | 1.7500 | 5.984 | 1.7466 | 10.08 |
| 11 | 68.961 ± 0.435 | 16.0000–16.0000 | 2.0000 | 2.0000 | 6.866 | 2.0041 | 10.04 |

The workloads are shared across n, so the exact ratios equal (n + 5)/8. With rates drawn as stated
(mean 4,300 per VM pair), the exact optimum is 10.00–10.16 times the plotted value
at all nine points, i.e. a constant factor of about 10, as a different unit or scale of the rates would
give. Ratios do not depend on that scale, so they are the quantities compared.

## Closed form on every instance (l = 1000, n = 1..13)

| n | optimum / Λ (min–max) | max |exact − closed form| | ingress → egress (count) | decomposition ms | MILP s (mean / max) |
|---|---|---|---|---|---|
| 1 | 6.0000–6.0000 | 0.0e+00 | core → core (20) | 0.0 | 0.00 / 0.00 |
| 2 | 7.6712–7.7314 | 0.0e+00 | agg → core (10), core → agg (10) | 0.1 | 0.12 / 0.13 |
| 3 | 8.0000–8.0000 | 0.0e+00 | core → core (20) | 0.1 | 0.38 / 0.41 |
| 4 | 9.6712–9.7314 | 0.0e+00 | agg → core (10), core → agg (10) | 0.1 | 0.42 / 0.54 |
| 5 | 10.0000–10.0000 | 0.0e+00 | core → core (20) | 0.1 | 1.31 / 1.79 |
| 6 | 11.6712–11.7314 | 0.0e+00 | agg → core (10), core → agg (10) | 0.1 | 0.98 / 1.67 |
| 7 | 12.0000–12.0000 | 0.0e+00 | core → core (20) | 0.1 | 0.65 / 0.67 |
| 8 | 13.6712–13.7314 | 0.0e+00 | agg → core (10), core → agg (10) | 0.1 | 1.11 / 2.59 |
| 9 | 14.0000–14.0000 | 0.0e+00 | core → core (20) | 1.0 | 1.78 / 2.53 |
| 10 | 15.6712–15.7314 | 0.0e+00 | agg → core (10), core → agg (10) | 0.6 | 1.62 / 2.91 |
| 11 | 16.0000–16.0000 | 0.0e+00 | core → core (20) | 0.4 | 1.87 / 3.14 |
| 12 | 17.6712–17.7314 | 0.0e+00 | agg → core (10), core → agg (10) | 0.3 | 2.03 / 3.34 |
| 13 | 18.0000–18.0000 | 0.0e+00 | core → core (20) | 0.2 | 2.14 / 3.47 |

## Inner strolls H_m(u, v) used by the decomposition

Fewest hops from switch u to switch v through m distinct other switches, per class of ordered switch pairs.

| class | pairs | m=0 | m=1 | m=2 | m=3 | m=4 | m=5 | m=6 | m=7 | m=8 | m=9 | m=10 | m=11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c-c d=2 | 48 | 2 | 2 | 4 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 |
| c-c d=4 | 192 | 4 | 4 | 4 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 |
| c-a d=1 | 128 | 1 | 3 | 3 | 5 | 5 | 7 | 7 | 9 | 9 | 11 | 11 | 13 |
| c-a d=3 | 384 | 3 | 3 | 3 | 5 | 5 | 7 | 7 | 9 | 9 | 11 | 11 | 13 |
| c-e d=2 | 512 | 2 | 2 | 4 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 |
| a-c d=1 | 128 | 1 | 3 | 3 | 5 | 5 | 7 | 7 | 9 | 9 | 11 | 11 | 13 |
| a-c d=3 | 384 | 3 | 3 | 3 | 5 | 5 | 7 | 7 | 9 | 9 | 11 | 11 | 13 |
| a-a d=2 | 224 | 2 | 2 | 4 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 |
| a-a d=2 same pod | 96 | 2 | 2 | 4 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 |
| a-a d=4 | 672 | 4 | 4 | 4 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 |
| a-e d=1 same pod | 128 | 1 | 3 | 3 | 5 | 5 | 7 | 7 | 9 | 9 | 11 | 11 | 13 |
| a-e d=3 | 896 | 3 | 3 | 3 | 5 | 5 | 7 | 7 | 9 | 9 | 11 | 11 | 13 |
| e-c d=2 | 512 | 2 | 2 | 4 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 |
| e-a d=1 same pod | 128 | 1 | 3 | 3 | 5 | 5 | 7 | 7 | 9 | 9 | 11 | 11 | 13 |
| e-a d=3 | 896 | 3 | 3 | 3 | 5 | 5 | 7 | 7 | 9 | 9 | 11 | 11 | 13 |
| e-e d=2 same pod | 96 | 2 | 2 | 4 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 |
| e-e d=4 | 896 | 4 | 4 | 4 | 4 | 6 | 6 | 8 | 8 | 10 | 10 | 12 | 12 |

17 classes; besides one representative, 170 further pairs were solved and matched their class.

## Run

- command: `python -m experiments.r2_top_fig9`
- date (UTC): 2026-10-13 03:40
- code version: no commit yet
- python / numpy / scipy: 3.14.6 / 2.5.4 / 1.18.1
- machine: Darwin arm64, Apple M1
- workloads: 20 per setting (seeds 0..19), shared across n
- wall time: 328 s
