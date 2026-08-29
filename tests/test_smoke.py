"""Fast smoke tests for the PC3 method (no external data, no heavy baselines)."""
import numpy as np
import pc3


def _split(n_tr=200, n_cal=100, n=400, seed=0):
    X, y, mono, bf, *_ = pc3.make_synthetic_composite(n=n, seed=seed)
    return (X[:n_tr], y[:n_tr], X[n_tr:n_tr + n_cal], y[n_tr:n_tr + n_cal],
            X[n_tr + n_cal:], y[n_tr + n_cal:], mono, bf)


def test_coverage_at_target():
    """PC3 attains ~ the 1-alpha target coverage on synthetic data."""
    Xtr, ytr, Xcal, ycal, Xte, yte, mono, bf = _split()
    m = pc3.PC3(mono, bf, "cqr", True, True, True, alpha=0.1).fit(Xtr, ytr).calibrate(Xcal, ycal)
    _, lo, hi, _, _ = m.predict(Xte)
    cov = float(np.mean((yte >= lo) & (yte <= hi)))
    assert cov >= 0.80, f"coverage too low: {cov:.3f}"


def test_zero_physical_violations():
    """Projected intervals never leave the admissible corridor [L, U]."""
    Xtr, ytr, Xcal, ycal, Xte, yte, mono, bf = _split()
    m = pc3.PC3(mono, bf, "cqr", True, True, True, alpha=0.1).fit(Xtr, ytr).calibrate(Xcal, ycal)
    _, lo, hi, _, _ = m.predict(Xte)
    L, U = bf(Xte)
    assert np.all(lo >= L - 1e-9) and np.all(hi <= U + 1e-9), "interval left the corridor"


def test_robust_flag_matches_algorithm1():
    """robust=True: out-of-corridor calibration points get an infinite score, and the
    resulting interval contains the naive (calibrate-then-clip) interval for every x."""
    Xtr, ytr, Xcal, ycal, Xte, yte, mono, bf = _split()
    L, U = bf(Xte)
    tight = lambda Xq: (bf(Xq)[0], bf(Xq)[0] + 0.7 * (bf(Xq)[1] - bf(Xq)[0]))   # misspecified ceiling (eta ~ 1%)
    naive = pc3.PC3(mono, tight, "cqr", True, True, True, alpha=0.1).fit(Xtr, ytr).calibrate(Xcal, ycal)
    robust = pc3.PC3(mono, tight, "cqr", True, True, True, alpha=0.1, robust=True).fit(Xtr, ytr).calibrate(Xcal, ycal)
    assert robust.eta_hat > 0 and robust.Qg >= naive.Qg
    _, lo_n, hi_n, _, _ = naive.predict(Xte)
    _, lo_r, hi_r, Lt, Ut = robust.predict(Xte)
    assert np.all(lo_r <= lo_n + 1e-9) and np.all(hi_r >= hi_n - 1e-9), "robust interval must nest the naive one"
    assert np.all(lo_r >= Lt - 1e-9) and np.all(hi_r <= Ut + 1e-9)


def test_full_corridor_when_eta_exceeds_alpha():
    """When more than an alpha-fraction of calibration responses lie outside the corridor,
    Q = +inf and the projected interval is the full corridor (Theorem 3(ii))."""
    Xtr, ytr, Xcal, ycal, Xte, yte, mono, bf = _split()
    very_tight = lambda Xq: (bf(Xq)[0], bf(Xq)[0] + 0.5 * (bf(Xq)[1] - bf(Xq)[0]))   # eta ~ 60% > alpha
    m = pc3.PC3(mono, very_tight, "cqr", True, True, True, alpha=0.1, robust=True).fit(Xtr, ytr).calibrate(Xcal, ycal)
    assert m.eta_hat > 0.1 and not m.feasible and np.isinf(m.Qg)
    _, lo, hi, L, U = m.predict(Xte)
    assert np.allclose(lo, L) and np.allclose(hi, U)


def test_system_layer_record_fields():
    """The decision-support layer returns admissible intervals and informative flags."""
    import ias
    Xtr, ytr, Xcal, ycal, Xte, yte, mono, bf = _split()
    m = pc3.PC3(mono, bf, "cqr", True, True, True, alpha=0.1, robust=True).fit(Xtr, ytr).calibrate(Xcal, ycal)
    sysm = ias.PC3System(m, ["Vf", "Ef", "Em", "z1", "z2"], background=None, explain=False)
    rec = sysm.query(Xte[0])
    for k in ["y_hat", "lo", "hi", "L", "U", "point_clipped", "interval_clipped", "feasible", "eta_hat", "width_ratio"]:
        assert k in rec
    assert rec["L"] - 1e-9 <= rec["lo"] <= rec["y_hat"] <= rec["hi"] <= rec["U"] + 1e-9
    df = sysm.batch(Xte[:10])
    assert len(df) == 10 and df.feasible.all()
