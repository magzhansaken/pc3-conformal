# Changes in revision R1 (September 2026)

## Threshold (pc3.conformal_quantile)
* Now returns the m-th order statistic exactly, m = ceil((n+1)(1-alpha)), and +inf whenever fewer than m
  scores are finite (including m > n).  The submitted version used `np.quantile(..., m/n, method="higher")`,
  which returns the (m+1)-th order statistic for m < n (one rank more conservative) and the largest finite
  score for m > n.  The old behaviour is available with `legacy=True`.
* All tables and figures of the paper were regenerated with the corrected threshold (shifts of order 1/(n+1)).

## Finite-sample diagnostics (Theorem 3, Lemma 2, Corollary 4 of the revised paper)
* New functions: `clopper_pearson`, `fallback_probability`, `marginal_coverage_floor`,
  `exact_marginal_coverage` (closed form 1 - E[max(l, K~)]/(n+1)), `exact_marginal_coverage_sum`,
  `feasibility_diagnostics` (three-state verdict, coverage floors).
* `PC3.calibrate` exposes `K` and `diag`; `feasible` is kept as the alias for "Q finite" (the branch taken),
  not as a certificate that eta <= alpha.
* `ias.py`: records and the inference card report the confidence interval, the verdict, the branch,
  the fallback probability and the two coverage floors.

## Experiments
* `finite_sample_v2.py` (new): finite-sample study against the exact law of Theorem 3(ii)
  (Fig. N, Table A1, Fig. A5).  Supersedes the calibration-size part of `robust_concrete.py` and the
  old `finite_sample.py` protocol.
* `robust_cp_fast.py`: same numbers as `robust_cp.py` with the base fits shared across rho.
* `verification/`: numerical checks of Theorem 3, Lemma 2, Proposition 4 and Corollary 4.
* `tests/test_diagnostics.py`: 9 new tests (14 in total).
