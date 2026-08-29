#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Second dataset: FRP - longitudinal modulus E11 of a unidirectional composite.
A micromechanics benchmark on real constituent ranges from the literature.
Here the Voigt-Reuss bounds are RIGOROUS for the effective modulus, and the upper
bound is genuinely binding (E11 is close to Voigt) -> a clean demonstration:
native methods violate physics, while PC3 has 0% violations by construction.

Run:  python frp_experiment.py   (pc3.py must be alongside)
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pc3


def load_frp(n=1500, seed=0):
    rs = np.random.RandomState(seed)
    Vf = rs.uniform(0.10, 0.48, n)          # fibre volume fraction (Tariq range)
    Ef = rs.uniform(69, 700, n)             # fibre modulus, GPa
    Em = rs.uniform(2, 10, n)               # matrix modulus, GPa
    radius = rs.uniform(3, 12, n)           # fibre radius, um (irrelevant feature)
    MMA = rs.uniform(0, 5, n)               # mean misalignment angle, deg (knockdown)
    voigt = Vf * Ef + (1 - Vf) * Em                       # upper bound (ROM, longitudinal)
    reuss = 1.0 / (Vf / Ef + (1 - Vf) / Em)               # lower bound
    s = np.clip(0.97 - 0.04 * MMA, 0.05, 0.999)           # position in corridor (decreases with misalignment)
    E11 = reuss + s * (voigt - reuss)                     # effective E11 (close to Voigt)
    y = np.clip(E11 + rs.normal(0, 0.04 * (voigt - reuss)), reuss, voigt)   # Y in [Reuss,Voigt] a.s.
    X = np.column_stack([Vf, Ef, Em, radius, MMA])
    feats = ["Vf", "Ef_GPa", "Em_GPa", "radius_um", "MMA_deg"]
    monotone = [+1, +1, +1, 0, -1]                        # E11 up with Vf,Ef,Em; down with misalignment
    def bounds_fn(Xq):
        vf, ef, em = Xq[:, 0], Xq[:, 1], Xq[:, 2]
        return 1.0 / (vf / ef + (1 - vf) / em), vf * ef + (1 - vf) * em
    def group_fn(Xq):
        vf = Xq[:, 0]; return np.where(vf <= 0.25, 0, np.where(vf <= 0.37, 1, 2))
    return X, y, monotone, bounds_fn, feats, group_fn


if __name__ == "__main__":
    res = pc3.run_experiment(load_frp, "FRP longitudinal modulus E11 (micromechanics, literature ranges)")
    abl = pc3.run_ablation(load_frp, "FRP longitudinal modulus E11")

    agg = res["agg"]; METH = list(pc3.METHODS.keys())
    short = {"GP (native)": "GP", "NGBoost (native)": "NGB", "Deep Ensemble (native)": "Ens",
             "MC-Dropout (native~)": "MCD", "Split-Conformal": "Split-CP", "CQR": "CQR",
             "PC3 (marginal)": "PC3", "PC3 + Mondrian": "PC3+Mon"}
    names = [short[m] for m in METH]
    cov = [np.mean(agg[m]["Coverage"]) * 100 for m in METH]
    wid = [np.mean(agg[m]["Width"]) for m in METH]
    vio = [np.mean(agg[m]["Violations"]) * 100 for m in METH]
    cols = ["#B0B0B0"] * 4 + ["#7FA8D0", "#7FA8D0", "#4C78A8", "#2E5496"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].bar(names, cov, color=cols); ax[0].axhline(90, ls="--", c="r", label="target")
    ax[0].set_title("Coverage, %"); ax[0].set_ylim(min(cov) - 4, 100); ax[0].legend()
    ax[1].bar(names, wid, color=cols); ax[1].set_title("Width, GPa (↓ better)")
    ax[2].bar(names, vio, color=cols); ax[2].set_title("Physics violations, % (↓ better)")
    for a in ax: a.tick_params(axis="x", rotation=30)
    plt.suptitle("FRP E11 (real constituent ranges, rigorous Voigt–Reuss): PC³ on target, 0% violations",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig("out/figH_frp_comparison.png", dpi=600); plt.close()

    X, y, mono, bf, feats, gf = load_frp()
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=0)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0)
    mdl = pc3.PC3(mono, bf, "cqr", True, True, True, alpha=0.1, mondrian=True, group_fn=gf).fit(Xtr, ytr).calibrate(Xcal, ycal)
    p, lo, hi, L, U = mdl.predict(Xte)
    idx = np.argsort(p)[:: max(1, len(p) // 120)]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.errorbar(yte[idx], p[idx], yerr=[np.maximum(p[idx] - lo[idx], 0), np.maximum(hi[idx] - p[idx], 0)],
                fmt="o", ms=3, color="#4C78A8", ecolor="#9ecae1", alpha=.85, label="PC³ ± interval")
    lim = [min(yte.min(), p.min()), max(yte.max(), p.max())]; ax.plot(lim, lim, "k--", label="ideal")
    ax.set_xlabel("True E11, GPa"); ax.set_ylabel("Prediction ± interval")
    ax.set_title("FRP E11: PC³ with intervals"); ax.legend()
    plt.tight_layout(); plt.savefig("out/figI_frp_parity.png", dpi=600); plt.close()
    print("\nFRP figures: out/figH_frp_comparison.png, out/figI_frp_parity.png")
