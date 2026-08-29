#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration order under misspecified bounds (paper Section 4.6 / Table 13 / Figure 12).

Three calibration rules are run on the SAME quantile models and the SAME corridor; they
differ only in when the projection onto [L, U] enters the calibration:
  cal-then-clip        split-conformal quantile of the unprojected CQR scores, interval
                       clipped afterwards. This is the post-processing order of
                       bound-constrained CP (Li et al., 2025, Eq. 15 applied after
                       calibration); their Theorem 1 shows it loses nothing when the bounds
                       are valid, which is exactly the assumption that fails here.
  clip-then-cal (n)    nested conformal prediction (Gupta et al., 2022) on the projected
                       family with the ceil(n(1-alpha))-th in-corridor score, i.e. without
                       the +1 finite-sample correction.
  PC3 (ours)           clip-then-calibrate with the ceil((n+1)(1-alpha))-th projection-aware
                       score (Eq. 6, +inf off the corridor); by Lemma 1 this is nested CP on
                       the projected family with the exact split-conformal guarantee.
NOTE: the OMLT mechanism of Li et al. (Optimal Minimal Length Threshold) addresses a
different failure mode - under-coverage in regions where the bounds are tight but VALID -
and is not reproduced here; earlier versions of this script mislabelled the second rule
as "OMLT-style".
Sweep of bound misspecification on FRP (U_mis = L + rho*(U-L)). Expected outcome:
cal-then-clip undercovers; the two clip-then-calibrate rules coincide up to the +1
correction and both track the admissible frontier 1 - max(alpha, eta).
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pc3
from frp_experiment import load_frp
from robust_cp import make_bounds_mis

ALPHA = 0.1; SEEDS = 8
def cov(y, lo, hi): return float(np.mean((y >= lo) & (y <= hi)))

def ncp_threshold_n(s_cal, in_corr, n, alpha):
    """clip-then-cal (n): smallest Q with >= ceil((1-alpha) n) in-corridor calibration scores <= Q (no +1 correction)."""
    s_in = np.sort(s_cal[in_corr])
    k = int(np.ceil((1 - alpha) * n))
    if k <= len(s_in):
        return s_in[k - 1]
    return np.inf   # unattainable -> full corridor

def main():
    X, y, mono, bf_true, feats, gf = load_frp()
    rhos = [1.0, 0.99, 0.98, 0.975, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90]
    acc = {r: {"eta": [], "CQR": [], "cal_then_clip": [], "clip_then_cal_n": [], "robust": []} for r in rhos}
    for s in range(SEEDS):
        Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=0.5, random_state=s)
        Xcal, Xte, ycal, yte = train_test_split(Xt, yt, test_size=0.5, random_state=s)
        ql = pc3.fit_quantile(Xtr, ytr, ALPHA / 2, mono, True)
        qh = pc3.fit_quantile(Xtr, ytr, 1 - ALPHA / 2, mono, True)
        qm = pc3.fit_quantile(Xtr, ytr, 0.5, mono, True)
        loC, hiC = ql.predict(Xcal), qh.predict(Xcal)
        loT, hiT = ql.predict(Xte), qh.predict(Xte)
        s_cal = np.maximum(loC - ycal, ycal - hiC)
        n = len(ycal)
        for r in rhos:
            bfm = make_bounds_mis(bf_true, r)
            Lc, Uc = bfm(Xcal); Lt, Ut = bfm(Xte)
            acc[r]["eta"].append(np.mean((yte < Lt) | (yte > Ut)))
            # CQR (no projection)
            Q = pc3.conformal_quantile(s_cal, ALPHA)
            acc[r]["CQR"].append(cov(yte, loT - Q, hiT + Q))
            # cal-then-clip: fixed quantile of unprojected scores, interval clipped afterwards
            lo, hi = np.maximum(loT - Q, Lt), np.minimum(hiT + Q, Ut)
            acc[r]["cal_then_clip"].append(cov(yte, lo, hi))
            # clip-then-cal (n): nested CP on the projected family without the +1 correction
            inc = (ycal >= Lc) & (ycal <= Uc)
            Qo = ncp_threshold_n(s_cal, inc, n, ALPHA)
            if np.isinf(Qo):
                loo, hio = Lt.copy(), Ut.copy()
            else:
                loo, hio = np.maximum(loT - Qo, Lt), np.minimum(hiT + Qo, Ut)
            acc[r]["clip_then_cal_n"].append(cov(yte, loo, hio))
            # robust (ours): +inf scores for out-of-corridor calibration points
            E = s_cal.copy(); E[~inc] = np.inf
            Qr = pc3.conformal_quantile(E, ALPHA)
            lor, hir = (np.maximum(loT - Qr, Lt), np.minimum(hiT + Qr, Ut)) if np.isfinite(Qr) else (Lt.copy(), Ut.copy())
            acc[r]["robust"].append(cov(yte, lor, hir))
    rows = [(r, np.mean(acc[r]["eta"])*100, np.mean(acc[r]["CQR"])*100, np.mean(acc[r]["cal_then_clip"])*100,
             np.mean(acc[r]["clip_then_cal_n"])*100, np.mean(acc[r]["robust"])*100) for r in rhos]
    print(f'\n{"="*78}\nFRP: calibration order under misspecified bounds ({SEEDS} seeds, alpha=0.1)\n{"="*78}')
    print(f'{"rho":>6}{"eta%":>8}{"CQR%":>8}{"cal>clip%":>11}{"clip>cal(n)%":>14}{"PC3(n+1)%":>11}')
    for r in rows: print(f'{r[0]:>6.3f}{r[1]:>8.1f}{r[2]:>8.1f}{r[3]:>11.1f}{r[4]:>14.1f}{r[5]:>11.1f}')
    d = max(abs(r[4] - r[5]) for r in rows)
    print(f"\nmax |clip-then-cal(n) - PC3(n+1)| = {d:.2f} pp  (the two differ only in the +1 order statistic)")

    eta = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    o = np.argsort(eta); eta = np.array(eta)[o]
    cpul = np.array([r[3] for r in rows])[o]; ncpn = np.array([r[4] for r in rows])[o]; rob = np.array([r[5] for r in rows])[o]
    xs = np.linspace(0, max(eta) + 0.5, 100)
    ax.plot(xs, 100 - np.maximum(10, xs), "--", color="#888", lw=1.2, label="admissible frontier 100−max(α,η)")
    ax.plot(eta, cpul, "s-", color="#C0392B", ms=6, label="calibrate, then clip (fixed quantile)")
    ax.plot(eta, ncpn, "^--", color="#E1A100", ms=7, label="clip, then calibrate: ⌈n(1−α)⌉-th score")
    ax.plot(eta, rob, "o-", color="#2E5496", ms=6, label="PC³: clip, then calibrate: ⌈(n+1)(1−α)⌉-th score", alpha=0.8)
    ax.axhline(90, color="#ccc", lw=0.8)
    ax.set_xlabel("Fraction of Y outside corridor, η, %"); ax.set_ylabel("Coverage, %"); ax.set_ylim(60, 95)
    ax.set_title("Calibration order under misspecified bounds:\ncalibrate-then-clip undercovers; clip-then-calibrate tracks the admissible frontier")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("out/figS_omlt_compare.png", dpi=600); plt.close()
    print("Figure: out/figS_omlt_compare.png")

if __name__ == "__main__":
    main()
