#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(A) Honest 'base + clip' comparison on FRP E11.
(B) Sensitivity to misspecified bounds on FRP E11: coverage degrades but stays
    above (1-alpha) - P(Y outside [L,U]).

Run:  python revision_experiments.py   (pc3.py must be alongside)
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pc3
from frp_experiment import load_frp

ALPHA = 0.1; SEEDS = 5


# ============================================================
# (A) HONEST BASELINES: base + clip
# ============================================================
def clip_interval(point, lo, hi, L, U):
    lo = np.maximum(lo, L); hi = np.minimum(hi, U); point = np.clip(point, L, U)
    bad = lo > hi; lo[bad] = hi[bad] = point[bad]
    return point, lo, hi

def run_A(loader, name):
    print(f"\n{'='*92}\n(A) Honest 'base+clip' comparison - {name} (5 seeds, target 90%)\n{'='*92}")
    X, y, mono, bf, feats, gf = loader()
    # name -> (type, spec, clip?)
    cfg = [
        ("GP (native)",        "gp",  None, False),
        ("GP + clip",          "gp",  None, True),
        ("Split-Conformal",    "cf",  dict(score="residual", mono=False, proj=False, pa=False), False),
        ("Split-CP + clip",    "cf",  dict(score="residual", mono=False, proj=True,  pa=False), False),
        ("CQR",                "cf",  dict(score="cqr", mono=False, proj=False, pa=False), False),
        ("CQR + clip",         "cf",  dict(score="cqr", mono=False, proj=True,  pa=False), False),
        ("PC3 (full)",         "cf",  dict(score="cqr", mono=True,  proj=True,  pa=True),  False),
    ]
    keys = ["Coverage", "Width", "Violations", "WorstGrp", "Gap"]
    agg = {c[0]: {k: [] for k in keys} for c in cfg}
    for s in range(SEEDS):
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=s)
        Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=s)
        gte = gf(Xte); L, U = bf(Xte)
        for nm, kind, spec, clip in cfg:
            if kind == "gp":
                Xtra, ytra = np.vstack([Xtr, Xcal]), np.concatenate([ytr, ycal])
                p, lo, hi = pc3.gp_interval(Xtra, ytra, Xte, ALPHA)
                if clip: p, lo, hi = clip_interval(p, lo, hi, L, U)
            else:
                m = pc3.PC3(mono, bf, score=spec["score"], use_monotone=spec["mono"],
                            project=spec["proj"], physics_aware=spec["pa"], alpha=ALPHA).fit(Xtr, ytr).calibrate(Xcal, ycal)
                p, lo, hi, L, U = m.predict(Xte)
            r = pc3.evaluate(p, lo, hi, L, U, yte, groups=gte, target=1 - ALPHA)
            for k in keys: agg[nm][k].append(r[k])
    hdr = f'{"Method":<20}{"Cover":>8}{"Width":>9}{"Viol%":>8}{"WrstGrp":>9}{"Gap%":>7}'
    print(hdr); print("-" * len(hdr))
    for nm, *_ in cfg:
        a = {k: np.mean(v) for k, v in agg[nm].items()}
        print(f'{nm:<20}{a["Coverage"]*100:>7.1f}%{a["Width"]:>9.2f}{a["Violations"]*100:>7.1f}%'
              f'{a["WorstGrp"]*100:>8.1f}%{a["Gap"]*100:>6.1f}%')
    print("Takeaway: clipping zeros violations for ALL; the marginal guarantee comes from conformal calibration;")
    print("PC3 gain over 'CQR+clip' = only a moderate worst-group increase and a lower Gap.")
    return agg, cfg


# ============================================================
# (B) SENSITIVITY TO MISSPECIFIED BOUNDS
# ============================================================
def run_B(loader, name):
    print(f"\n{'='*92}\n(B) Sensitivity to misspecified bounds - {name}\n{'='*92}")
    X, y, mono, bf_true, feats, gf = loader()
    rhos = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]   # U_mis = L + rho*(U_true - L)
    cov, fout = [], []
    for rho in rhos:
        def bf_mis(Xq, rho=rho):
            L, U = bf_true(Xq); return L, L + rho * (U - L)
        cs, fs = [], []
        for s in range(SEEDS):
            Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=s)
            Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=s)
            m = pc3.PC3(mono, bf_mis, "cqr", True, True, True, alpha=ALPHA).fit(Xtr, ytr).calibrate(Xcal, ycal)
            p, lo, hi, L, U = m.predict(Xte)
            cs.append(np.mean((yte >= lo) & (yte <= hi)))
            Lm, Um = bf_mis(Xte); fs.append(np.mean((yte > Um) | (yte < Lm)))
        cov.append(np.mean(cs) * 100); fout.append(np.mean(fs) * 100)
    print(f'{"rho(U_mis)":>10}{"P(Y∉[L,U]),%":>14}{"Coverage,%":>12}{"floor=90−P,%":>14}')
    for rho, c, f in zip(rhos, cov, fout):
        print(f'{rho:>10.2f}{f:>14.1f}{c:>12.1f}{90-f:>14.1f}')
    print("Takeaway: under misspecified (too tight) bounds, coverage drops but not below (1-alpha)-P(Y outside [L,U]).")
    return rhos, cov, fout


if __name__ == "__main__":
    aggA, cfg = run_A(load_frp, "FRP E11")
    rhos, cov, fout = run_B(load_frp, "FRP E11")

    # --- Figure J: honest base+clip comparison (violations + worst-group) ---
    names = [c[0] for c in cfg]
    short = ["GP", "GP+clip", "Split-CP", "Split+clip", "CQR", "CQR+clip", "PC³"]
    vio = [np.mean(aggA[n]["Violations"]) * 100 for n in names]
    wg = [np.mean(aggA[n]["WorstGrp"]) * 100 for n in names]
    cols = ["#B0B0B0", "#7fb07f", "#B0B0B0", "#7fb07f", "#B0B0B0", "#7fb07f", "#2E5496"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].bar(short, vio, color=cols); ax[0].set_title("Physics violations, % (clipping → 0 for all)")
    ax[0].tick_params(axis="x", rotation=25)
    ax[1].bar(short, wg, color=cols); ax[1].axhline(90, ls="--", c="r", label="target")
    ax[1].set_title("Worst-group coverage, % (PC³ honest delta)"); ax[1].set_ylim(min(wg) - 4, 100)
    ax[1].tick_params(axis="x", rotation=25); ax[1].legend()
    plt.suptitle("Honest comparison (FRP): clipping zeros violations for all; PC³ adds over 'CQR+clip' "
                 "only a moderate gain in conditional coverage", fontweight="bold", fontsize=11)
    plt.tight_layout(); plt.savefig("out/figJ_clip_comparison.png", dpi=600); plt.close()

    # --- Figure K: bounds-misspecification sensitivity ---
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(fout, cov, "o-", color="#2E5496", label="PC³ empirical coverage")
    xs = np.linspace(0, max(fout), 50)
    ax.plot(xs, 90 - xs, "r--", label="theoretical lower bound: 90 − P(Y∉[L,U])")
    ax.axhline(90, color="#999", lw=0.8, ls=":")
    ax.set_xlabel("Fraction of Y outside bounds, P(Y∉[L,U]), %")
    ax.set_ylabel("Test coverage, %")
    ax.set_title("Sensitivity to misspecified bounds (FRP):\ncoverage stays above (1−α)−P(Y∉[L,U])")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("out/figK_bounds_sensitivity.png", dpi=600); plt.close()
    print("\nFigures: out/figJ_clip_comparison.png, out/figK_bounds_sensitivity.png")
