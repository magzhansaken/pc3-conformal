#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composition-grouped validation on cementitious families (paper Section 4.4, Table 11, Figure 9).

Splits by unique mixture composition (60/20/20 over groups, seeds 0–9) and repeats the
misspecification experiment on three public families:

  concrete  UCI Concrete (Yeh, 1998); auto-downloaded and cached to data/concrete.csv
  uhpc      UHPC compilation from github.com/Chen494820/code-and-dataset
            -> save "Raw material data.csv" as data/uhpc.csv
  mk        metakaolin geopolymer (223/83); aac  hybrid alkali-activated concrete (262/208)
  scc       SCC after elevated temperature from
            github.com/Quanchaochao/Explainable-prediction-model-for-high-temperature-
            compressive-strength-of-self-compacting-concrete
            -> save "real_data.csv" as data/scc_ht.csv

Bounds: empirical training-strength ceiling (percentile sweep) for concrete/uhpc;
the Eurocode 2 (EN 1992-1-2, siliceous) fire-decay envelope, scaled by the largest
ambient training strength, for scc — an intrinsically misspecified physical bound.

Usage:
  python grouped_families.py                 # all available families, 10 seeds
  python grouped_families.py --families concrete,scc --seeds 3
Outputs: out/grouped_results.csv, out/figT_grouped_families.png, printed Table-11 rows (submitted manuscript numbering).
"""
import argparse, os, sys, urllib.request, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pc3

ALPHA = 0.1
CONCRETE_URL = ("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
                "master/concrete.csv")

from robust_cp import RobustPC3   # = pc3.PC3(..., robust=True): projection-aware infinite score

def grouped_split(groups, seed, fr=(0.6, 0.2, 0.2)):
    u = np.unique(groups); rs = np.random.RandomState(seed); rs.shuffle(u)
    n1 = int(len(u) * fr[0]); n2 = int(len(u) * (fr[0] + fr[1]))
    S = [set(u[:n1]), set(u[n1:n2]), set(u[n2:])]
    return [np.array([g in s for g in groups]) for s in S]

def load(name):
    if name == "concrete":
        p = "data/concrete.csv"
        if not os.path.exists(p):
            os.makedirs("data", exist_ok=True)
            print("  downloading UCI Concrete ->", p)
            urllib.request.urlretrieve(CONCRETE_URL, p)
        d = pd.read_csv(p).drop_duplicates()
        X = d.drop(columns=["strength"]); y = d["strength"].values.astype(float)
        comp = [c for c in X.columns if c != "age"]; Tc = None
    elif name == "uhpc":
        d = pd.read_csv("data/uhpc.csv").drop_duplicates()
        X = d.drop(columns=["CS"]); y = d["CS"].values.astype(float)
        comp = [c for c in X.columns if c not in ("Age", "T")]; Tc = None
    elif name == "scc":
        d = pd.read_csv("data/scc_ht.csv").drop_duplicates()
        X = d.drop(columns=["Strength"]); y = d["Strength"].values.astype(float)
        comp = [c for c in X.columns if c not in ("Feature10", "Feature11")]
        Tc = list(X.columns).index("Feature10")
    elif name == "mk":
        d = pd.read_csv("data/mk_geopolymer.csv").drop_duplicates()
        d = d.drop(columns=[c for c in ["NO.", "Reference"] if c in d.columns])
        X = d.drop(columns=["CS."]); y = d["CS."].values.astype(float)
        comp = [c for c in X.columns if c not in ("Age", "CT")]; Tc = None
    elif name == "aac":
        d = pd.read_csv("data/hybrid_aac.csv").drop_duplicates()
        d = d.drop(columns=[c for c in ["Reference"] if c in d.columns])
        yc = "28-d Cubic compressive strength (MPa)"
        X = d.drop(columns=[yc]); y = d[yc].values.astype(float)
        comp = [c for c in X.columns if "curing" not in c.lower()]; Tc = None
    else:
        raise ValueError(name)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    g = X[comp].round(6).astype(str).agg("|".join, axis=1).values
    return X.values.astype(float), y, g, Tc

# EN 1992-1-2, siliceous aggregate
EC_T = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])
EC_K = np.array([1.00, 1.00, 0.95, 0.85, 0.75, 0.60, 0.45, 0.30, 0.15, 0.08, 0.04, 0.01])

def bf_ceiling(cap):
    return lambda Xq: (np.zeros(len(Xq)), np.full(len(Xq), float(cap)))

def bf_ec2(Tc, cap):
    return lambda Xq: (np.zeros(len(Xq)), np.interp(Xq[:, Tc], EC_T, EC_K) * float(cap))

def run_family(name, sweep, seeds):
    X, y, g, Tc = load(name)
    mono = [0] * X.shape[1]; rows = []
    for seed in seeds:
        mtr, mca, mte = grouped_split(g, seed)
        Xtr, ytr = X[mtr], y[mtr]; Xca, yca = X[mca], y[mca]; Xte, yte = X[mte], y[mte]
        base = pc3.PC3(mono, bf_ceiling(1e9), "cqr", False, True, True,
                       alpha=ALPHA).fit(Xtr, ytr)
        r2 = 1 - np.sum((yte - base.q_med.predict(Xte))**2) / np.sum((yte - yte.mean())**2)
        for tag, mk in sweep:
            bf = mk(ytr, Xtr, Tc)
            for M, cls, proj, pa in [("naive", pc3.PC3, True, True),
                                     ("robust", RobustPC3, True, True),
                                     ("cqr", pc3.PC3, False, False)]:
                m = cls(mono, bf, "cqr", False, proj, pa, alpha=ALPHA)
                m.q_lo, m.q_hi, m.q_med = base.q_lo, base.q_hi, base.q_med
                m.calibrate(Xca, yca)
                pt, lo, hi, L, U = m.predict(Xte)
                Lt, Ut = bf(Xte)
                rows.append(dict(family=name, seed=seed, tag=tag, method=M,
                                 eta=np.mean((yte < Lt) | (yte > Ut)),
                                 cov=np.mean((yte >= lo) & (yte <= hi)),
                                 width=np.mean(hi - lo), r2=r2))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="concrete,uhpc,scc,mk,aac")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    seeds = range(a.seeds)
    CEIL = [(f"p{p}", (lambda p: (lambda ytr, Xtr, Tc: bf_ceiling(np.percentile(ytr, p))))(p))
            for p in [99.9, 99, 97.5, 95, 92.5, 90, 85, 80]]
    SCC = [("EC2", lambda ytr, Xtr, Tc: bf_ec2(Tc, ytr[Xtr[:, Tc] <= 25].max()))]
    frames = []
    for fam in a.families.split(","):
        fam = fam.strip()
        need = {"concrete": None, "uhpc": "data/uhpc.csv", "scc": "data/scc_ht.csv",
                "mk": "data/mk_geopolymer.csv", "aac": "data/hybrid_aac.csv"}[fam]
        if need and not os.path.exists(need):
            print(f"[skip] {fam}: place the file at {need} (see data/README.md)"); continue
        print(f"\n== running {fam} ({a.seeds} seeds) ==")
        frames.append(run_family(fam, SCC if fam == "scc" else CEIL, seeds))
    if not frames:
        sys.exit("no families available — see data/README.md")
    df = pd.concat(frames); os.makedirs("out", exist_ok=True)
    df.to_csv("out/grouped_results.csv", index=False)
    piv = df.groupby(["family", "tag", "method"]).agg(
        eta=("eta", "mean"), cov=("cov", "mean"), sd=("cov", "std")).reset_index()
    print("\nfamily   tag     eta%   naive%        robust%       frontier%")
    for _, s in piv[piv.method == "robust"].iterrows():
        nai = piv[(piv.family == s.family) & (piv.tag == s.tag) &
                  (piv.method == "naive")].iloc[0]
        fr = 100 * (1 - max(ALPHA, s["eta"]))
        print(f"{s.family:8s} {s.tag:6s} {100*s['eta']:5.1f}  "
              f"{100*nai['cov']:5.1f}±{100*nai['sd']:4.1f}  "
              f"{100*s['cov']:5.1f}±{100*s['sd']:4.1f}   {fr:5.1f}")
    # figure
    fams = [f for f in ["concrete", "uhpc", "scc", "mk", "aac"] if f in set(df.family)]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.8), sharey=True, sharex=True)
    axes = axes.ravel()
    for ax, fam in zip(axes, fams):
        s = piv[piv.family == fam]
        xs = np.linspace(0, 23, 80)
        ax.plot(xs, 100 - np.maximum(10, xs), "--", c="#777", lw=1.2,
                label=r"frontier $1{-}\max(\alpha,\eta)$")
        for m, c, mk, lb in [("naive", "#C0392B", "s", "naive projection"),
                             ("robust", "#2E5496", "o", "projection-aware (robust)")]:
            ss = s[s.method == m].sort_values("eta")
            ax.errorbar(100 * ss["eta"], 100 * ss["cov"], yerr=100 * ss["sd"],
                        fmt=mk + "-", c=c, ms=6, capsize=3, lw=1.8, label=lb)
        ax.axhline(90, c="#ccc", lw=.8); ax.axvline(10, c="#ccc", lw=.8, ls=":")
        ax.set_title({"concrete":"Concrete","uhpc":"UHPC","scc":"SCC thermal (EC2)","mk":"MK geopolymer","aac":"Hybrid AAC"}.get(fam,fam), fontsize=11); ax.set_xlabel(r"$\eta$, %"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Test coverage, %"); axes[0].set_ylim(58, 101)
    if len(axes) > 3:
        axes[3].set_ylabel("Test coverage, %")
    for ax in axes[len(fams):]:                      # spare cell holds the shared legend
        ax.axis("off")
        ax.legend(handles=axes[0].get_legend_handles_labels()[0], loc="center",
                  fontsize=10.5, frameon=False)
    plt.tight_layout(); plt.savefig("out/figT_grouped_families.png", dpi=600)
    print("\nWrote out/grouped_results.csv and out/figT_grouped_families.png")

if __name__ == "__main__":
    main()
