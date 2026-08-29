#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust bound-constrained CP on REAL concrete (UCI Concrete) with an empirical ceiling,
plus a finite-sample sensitivity study as the calibration size varies.

Run:  python robust_concrete.py   (pc3.py must be alongside)
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pc3
CQ = pc3.conformal_quantile; FQ = pc3.fit_quantile
ALPHA = 0.1

def fit_seed(Xtr, ytr, mono):
    return dict(
        lo=FQ(Xtr, ytr, ALPHA/2, mono, True), hi=FQ(Xtr, ytr, 1-ALPHA/2, mono, True),
        lon=FQ(Xtr, ytr, ALPHA/2, mono, False), hin=FQ(Xtr, ytr, 1-ALPHA/2, mono, False))

if __name__ == "__main__":
    X, y, mono, _, feats, gf = pc3.load_concrete()
    SEEDS = 6
    # precompute fits + predictions per seed
    cache = []
    for s in range(SEEDS):
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=s)
        Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=s)
        f = fit_seed(Xtr, ytr, mono)
        cache.append(dict(ytr=ytr, ycal=ycal, yte=yte,
            qlo_c=f["lo"].predict(Xcal), qhi_c=f["hi"].predict(Xcal),
            qlo_t=f["lo"].predict(Xte), qhi_t=f["hi"].predict(Xte),
            qlon_c=f["lon"].predict(Xcal), qhin_c=f["hin"].predict(Xcal),
            qlon_t=f["lon"].predict(Xte), qhin_t=f["hin"].predict(Xte)))

    def methods_at(c, U):
        yc, yt = c["ycal"], c["yte"]
        s_cal = np.maximum(c["qlo_c"] - yc, yc - c["qhi_c"])              # CQR score (flat band -> w=1)
        Qn = CQ(s_cal, ALPHA)
        lo = np.maximum(c["qlo_t"] - Qn, 0.0); hi = np.minimum(c["qhi_t"] + Qn, U)
        cov_n = np.mean((yt >= lo) & (yt <= hi))
        s_rob = s_cal.copy(); s_rob[(yc > U) | (yc < 0)] = np.inf
        Qr = CQ(s_rob, ALPHA)
        lo2 = np.maximum(c["qlo_t"] - Qr, 0.0); hi2 = np.minimum(c["qhi_t"] + Qr, U)
        cov_r = np.mean((yt >= lo2) & (yt <= hi2))
        s_np = np.maximum(c["qlon_c"] - yc, yc - c["qhin_c"])
        Qp = CQ(s_np, ALPHA)
        lo3 = c["qlon_t"] - Qp; hi3 = c["qhin_t"] + Qp
        cov_q = np.mean((yt >= lo3) & (yt <= hi3)); vio_q = np.mean((hi3 > U) | (lo3 < 0))
        return cov_n, cov_r, cov_q, vio_q, np.mean(hi - lo), np.mean(hi2 - lo2), np.mean(hi3 - lo3)

    pcts = [99, 97, 95, 93, 91, 90, 88, 86]; rows = []
    for pc_ in pcts:
        eta, cn, cr, cq, vq, wn, wr, wq = ([] for _ in range(8))
        for c in cache:
            U = float(np.percentile(c["ytr"], pc_)); eta.append(np.mean(c["yte"] > U))
            a, b, d, e, f1, f2, f3 = methods_at(c, U)
            cn.append(a); cr.append(b); cq.append(d); vq.append(e); wn.append(f1); wr.append(f2); wq.append(f3)
        rows.append((pc_, np.mean(eta)*100, np.mean(cn)*100, np.mean(cr)*100, np.mean(cq)*100, np.mean(vq)*100,
                     np.mean(wn), np.mean(wr), np.mean(wq)))
    print(f'\n(1) REAL eta-sweep on concrete (UCI Concrete, {SEEDS} seeds, alpha=0.1)')
    print(f'{"pct":>4}{"eta%":>7}{"naive%":>9}{"robust%":>9}{"CQRnp%":>8}{"CQRnp_viol%":>12}{"w_naive":>9}{"w_robust":>10}{"w_CQRnp":>9}')
    for r in rows:
        print(f'{r[0]:>4}{r[1]:>7.1f}{r[2]:>9.1f}{r[3]:>9.1f}{r[4]:>8.1f}{r[5]:>12.1f}{r[6]:>9.1f}{r[7]:>10.1f}{r[8]:>9.1f}')
    print("(w_* = mean interval width in MPa)")

    eta = [r[1] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(eta, [r[2] for r in rows], "s-", color="#C0392B", label="naive projection")
    ax[0].plot(eta, [r[3] for r in rows], "o-", color="#2E5496", ms=6.5, label="PC³ (ours)")
    ax[0].plot(eta, [r[4] for r in rows], "^:", color="#3a8a4d", label="CQR, no projection")
    xs = np.linspace(0, max(eta), 60); ax[0].plot(xs, 100-np.maximum(10, xs), "--", color="#999", label="frontier 100−max(α,η)")
    ax[0].axhline(90, color="#ccc", lw=.8); ax[0].axvline(10, color="#ccc", lw=.9, ls=":")
    ax[0].set_xlabel("Real fraction of Y>U on concrete, η, %"); ax[0].set_ylabel("Coverage, %")
    ax[0].set_title("Coverage (real concrete)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(eta, [0]*len(rows), "s-", color="#C0392B", label="naive projection (0%)")
    ax[1].plot(eta, [0]*len(rows), "o-", color="#2E5496", ms=6.5, label="robust-CP (0%)")
    ax[1].plot(eta, [r[5] for r in rows], "^:", color="#3a8a4d", label="CQR, no projection")
    ax[1].set_xlabel("Real fraction of Y>U on concrete, η, %"); ax[1].set_ylabel("Physics violations, %")
    ax[1].set_title("Admissibility (real concrete)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.suptitle("Main result on REAL data (concrete): PC³ — target coverage AND 0% violations", fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig("out/figM_concrete_sweep.png", dpi=600); plt.close()

    # (№3) finite-sample sensitivity
    pc_fix = 93; sizes = [40, 80, 150, 300, 600]; means, stds, etas = [], [], []
    sens = []
    for s in range(16):
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=s)
        Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=s)
        f = fit_seed(Xtr, ytr, mono)
        U = float(np.percentile(ytr, pc_fix))
        sens.append(dict(qlo_c=f["lo"].predict(Xcal), qhi_c=f["hi"].predict(Xcal),
                         qlo_t=f["lo"].predict(Xte), qhi_t=f["hi"].predict(Xte),
                         ycal=ycal, yte=yte, U=U, eta=np.mean(yte > U)))
    for m_cal in sizes:
        cov = []
        for s, c in enumerate(sens):
            n = len(c["ycal"]); idx = np.random.RandomState(s).choice(n, min(m_cal, n), replace=False)
            sc = np.maximum(c["qlo_c"][idx] - c["ycal"][idx], c["ycal"][idx] - c["qhi_c"][idx])
            sc[(c["ycal"][idx] > c["U"]) | (c["ycal"][idx] < 0)] = np.inf
            Q = CQ(sc, ALPHA)
            lo = np.maximum(c["qlo_t"] - Q, 0.0); hi = np.minimum(c["qhi_t"] + Q, c["U"])
            cov.append(np.mean((c["yte"] >= lo) & (c["yte"] <= hi)))
        means.append(np.mean(cov)*100); stds.append(np.std(cov)*100)
    eta_fix = np.mean([c["eta"] for c in sens]) * 100
    print(f'\n(3) Sensitivity to calibration size (concrete, pct={pc_fix}, eta~{eta_fix:.1f}%)')
    print(f'{"n_cal":>7}{"cov_mean%":>11}{"cov_std%":>10}')
    for ms, mm, sd in zip(sizes, means, stds): print(f'{ms:>7}{mm:>11.1f}{sd:>10.1f}')

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(sizes, means, yerr=stds, fmt="o-", color="#2E5496", capsize=4, label="PC³ coverage (±σ, 16 runs)")
    ax.axhline(90, color="#C0392B", ls="--", label="target 1−α = 90%")
    ax.set_xscale("log"); ax.set_xlabel("Calibration size n_cal (log)"); ax.set_ylabel("Coverage, %")
    ax.set_title(f"Finite-sample sensitivity (concrete, η≈{eta_fix:.0f}%):\nspread shrinks, mean approaches 90% as n_cal grows")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("out/figN_ncal_sensitivity.png", dpi=600); plt.close()
    print("\nFigures: out/figM_concrete_sweep.png, out/figN_ncal_sensitivity.png")
