"""Tests for the corrected conformal threshold and the finite-sample feasibility diagnostics (Theorem 3, Lemma 2, Corollary 4)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pc3 import (conformal_quantile, exact_marginal_coverage_sum, clopper_pearson, fallback_probability, marginal_coverage_floor,
                 exact_marginal_coverage, feasibility_diagnostics, PC3, make_synthetic_composite)
from scipy.stats import binom

def test_threshold_is_exact_mth_order_statistic():
    rng = np.random.default_rng(0)
    for n in [9, 19, 25, 50, 200, 1000]:
        for a in [0.05, 0.1, 0.2]:
            E = rng.random(n); m = int(np.ceil((n + 1) * (1 - a)))
            if m > n: assert np.isinf(conformal_quantile(E, a))
            else:     assert conformal_quantile(E, a) == np.sort(E)[m - 1]

def test_threshold_infinite_when_too_few_finite_scores():
    E = np.array([0.1, 0.2, np.inf, 0.3, np.inf, 0.05, 0.4, 0.15, 0.25, np.inf])   # n=10, m=10, 7 finite
    assert np.isinf(conformal_quantile(E, 0.1))
    assert np.isinf(conformal_quantile(np.random.rand(8), 0.1))                    # m=9 > n=8
    E2 = np.array([0.1, 0.2, 0.3, 0.05, 0.4, 0.15, 0.25, 0.35, 0.45, np.inf])       # 9 finite, m=10
    assert np.isinf(conformal_quantile(E2, 0.1))

def test_legacy_flag_reproduces_submitted_version():
    rng = np.random.default_rng(1); E = rng.random(50); n = 50; a = 0.1
    m = int(np.ceil((n + 1) * (1 - a)))
    assert conformal_quantile(E, a, legacy=True) == np.quantile(E, min(1, m / n), method="higher") == np.sort(E)[m]   # (m+1)-th

def test_clopper_pearson_edges_and_monotone():
    assert clopper_pearson(0, 100)[0] == 0.0 and clopper_pearson(100, 100)[1] == 1.0
    his = [clopper_pearson(k, 100)[1] for k in range(0, 101)]; los = [clopper_pearson(k, 100)[0] for k in range(0, 101)]
    assert all(np.diff(his) >= 0) and all(np.diff(los) >= 0)
    lo, hi = clopper_pearson(5, 100); assert lo < 0.05 < hi

def test_fallback_probability_matches_binomial_and_small_n():
    n, a, eta = 50, 0.1, 0.088; l = int(np.floor((n + 1) * a))
    assert abs(fallback_probability(n, a, eta) - sum(binom.pmf(k, n, eta) for k in range(l, n + 1))) < 1e-12
    assert fallback_probability(8, 0.1, 0.0) == 1.0                                # n < 1/alpha - 1

def test_marginal_floor_nonincreasing_in_eta_and_exact_above_floor():
    for n in [25, 50, 200]:
        etas = np.linspace(0, 0.5, 101); f = [marginal_coverage_floor(n, 0.1, e) for e in etas]
        assert all(np.diff(f) <= 1e-12)
        for e in etas[1:]:
            ex = exact_marginal_coverage(n, 0.1, e); assert marginal_coverage_floor(n, 0.1, e) - 1e-12 <= ex <= 1 - e + 1e-12

def test_three_state_feasibility():
    assert feasibility_diagnostics(0, 200, 0.1)["feasibility"] == "feasible"
    assert feasibility_diagnostics(60, 200, 0.1)["feasibility"] == "infeasible"
    assert feasibility_diagnostics(18, 200, 0.1)["feasibility"] == "indeterminate"
    d = feasibility_diagnostics(18, 200, 0.1); assert d["branch"] == "conformal"
    assert 0 < d["coverage_floor_closed_form"] <= d["coverage_floor_marginal"] < 0.9 and 0 < d["coverage_floor_realized"] < 0.9
    d2 = feasibility_diagnostics(25, 200, 0.1); assert d2["branch"] == "corridor" and abs(d2["coverage_floor_realized"] - (1 - d2["eta_hi"])) < 1e-12

def test_model_exposes_diagnostics():
    X, y, mono, bf, _, _ = make_synthetic_composite(n=700, seed=4)
    def bad(Xq):
        L, U = bf(Xq); return L, L + 0.6 * (U - L)
    m = PC3(mono, bad, robust=True).fit(X[:400], y[:400]).calibrate(X[400:600], y[400:600])
    assert m.diag["K"] == m.K == int(round(m.eta_hat * m.n_cal)) and m.diag["n_cal"] == 200
    assert m.diag["feasibility"] in {"feasible", "indeterminate", "infeasible"}
    assert (m.diag["branch"] == "conformal") == m.feasible

def test_closed_form_equals_summation_form():
    for n in [5, 8, 25, 50, 200, 1000]:
        for a in [0.05, 0.1, 0.2]:
            for e in [0.0, 0.03, 0.088, 0.1, 0.2, 0.5]:
                assert abs(exact_marginal_coverage(n, a, e) - exact_marginal_coverage_sum(n, a, e)) < 1e-12
