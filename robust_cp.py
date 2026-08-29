#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust bound-constrained CP on FRP: recovery of 1-alpha under misspecified bounds.
The sweep tightens the upper bound; projection-aware calibration recovers the target
coverage while naive projection undercovers.

Run:  python robust_cp.py   (pc3.py and frp_experiment.py must be alongside)
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pc3
from frp_experiment import load_frp

ALPHA = 0.1; SEEDS = 5


class RobustPC3(pc3.PC3):
    """Projection-aware calibration (paper Algorithm 1 with robust flag r = 1).

    Kept as a thin alias for backward compatibility: it is ``pc3.PC3(..., robust=True)``,
    i.e. calibration points with Y outside [L, U] receive an infinite nonconformity score.
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robust", True)
        super().__init__(*args, **kwargs)


def make_bounds_mis(bf_true, rho):
    def bf(Xq):
        L, U = bf_true(Xq); return L, L + rho * (U - L)
    return bf


if __name__ == "__main__":
    X, y, mono, bf_true, feats, gf = load_frp()
    rhos = [1.0, 0.99, 0.98, 0.975, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90]
    rows = []
    for rho in rhos:
        bfm = make_bounds_mis(bf_true, rho)
        eta, c_naive, c_rob, c_cqr, v_rob = [], [], [], [], []
        w_naive, w_rob, w_cqr, w_cor = [], [], [], []                     # interval widths (efficiency)
        for s in range(SEEDS):
            Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=s)
            Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=s)
            Lt, Ut = bfm(Xte); eta.append(np.mean((yte < Lt) | (yte > Ut)))
            mn = pc3.PC3(mono, bfm, "cqr", True, True, True, alpha=ALPHA).fit(Xtr, ytr).calibrate(Xcal, ycal)
            _, lo, hi, L, U = mn.predict(Xte); c_naive.append(np.mean((yte >= lo) & (yte <= hi)))
            w_naive.append(np.mean(hi - lo)); w_cor.append(np.mean(U - L))
            mr = RobustPC3(mono, bfm, "cqr", True, True, True, alpha=ALPHA).fit(Xtr, ytr).calibrate(Xcal, ycal)
            _, lo2, hi2, L2, U2 = mr.predict(Xte); c_rob.append(np.mean((yte >= lo2) & (yte <= hi2)))
            v_rob.append(np.mean((lo2 < L2 - 1e-9) | (hi2 > U2 + 1e-9))); w_rob.append(np.mean(hi2 - lo2))
            mc = pc3.PC3(mono, bfm, "cqr", False, False, False, alpha=ALPHA).fit(Xtr, ytr).calibrate(Xcal, ycal)
            _, lo3, hi3, _, _ = mc.predict(Xte); c_cqr.append(np.mean((yte >= lo3) & (yte <= hi3)))
            w_cqr.append(np.mean(hi3 - lo3))
        rows.append((rho, np.mean(eta)*100, np.mean(c_naive)*100, np.mean(c_rob)*100,
                     np.mean(c_cqr)*100, np.mean(v_rob)*100,
                     np.mean(w_naive), np.mean(w_rob), np.mean(w_cqr), np.mean(w_cor)))

    print(f'\n{"="*78}\nCHECK Robust bound-constrained CP - FRP (5 seeds, alpha=0.1)\n{"="*78}')
    print(f'{"rho":>6}{"eta%":>8}{"naive%":>9}{"robust%":>9}{"CQR%":>8}{"rob.viol%":>11}'
          f'{"w_naive":>9}{"w_robust":>10}{"w_CQR":>8}{"w_[L,U]":>9}')
    print("-" * 87)
    for r in rows:
        print(f'{r[0]:>6.2f}{r[1]:>8.1f}{r[2]:>9.1f}{r[3]:>9.1f}{r[4]:>8.1f}{r[5]:>11.1f}'
              f'{r[6]:>9.1f}{r[7]:>10.1f}{r[8]:>8.1f}{r[9]:>9.1f}')
    print("(w_* = mean interval width in GPa; w_[L,U] = mean corridor width)")
    print("\nExpectation: for eta<10% robust recovers ~90% (naive is lower);")
    print("for eta>10% robust tracks 100-eta (the admissible frontier); CQR ~90% always but 0% admissibility.")

    eta = [r[1] for r in rows]; nai = [r[2] for r in rows]; rob = [r[3] for r in rows]; cqr = [r[4] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    xs = np.linspace(0, max(eta), 100); frontier = 100 - np.maximum(10, xs)
    ax.plot(xs, frontier, color="#888", ls="--", lw=1.3, label="admissible frontier 100 − max(α, η)")
    ax.plot(eta, nai, "s-", color="#C0392B", ms=6, label="naive projection (project-after-calibrate)")
    ax.plot(eta, rob, "o-", color="#2E5496", ms=6.5, label="PC³ (projection-aware), ours")
    ax.plot(eta, cqr, "^:", color="#3a8a4d", ms=5, label="CQR, no projection (0% admissible)")
    ax.axhline(90, color="#ccc", lw=0.8); ax.axvline(10, color="#ccc", lw=0.9, ls=":")
    ax.text(10.4, 40, "η = α\nrecovery\nboundary", fontsize=8, color="#666")
    ax.set_xlabel("Fraction of Y outside bounds  η = P(Y∉[L,U]), %")
    ax.set_ylabel("Test coverage, %"); ax.set_ylim(0, 100)
    ax.set_title("Robust bound-constrained CP: recovery of 1−α under misspecified bounds")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("out/figL_robust_cp.png", dpi=600); plt.close()
    print("\nFigure: out/figL_robust_cp.png")
