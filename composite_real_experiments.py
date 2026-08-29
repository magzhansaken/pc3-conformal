#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust CP on REAL composites with LOOSE bounds (where Theorem 3 is exercised):
  SFRC    - steel-fibre-reinforced concrete, strength Fc; bound L=0 + empirical ceiling (percentile).
  Polymer - woven textile composite, warp tensile strength; physical rule-of-mixtures bound
            U = c * (fibre strength), c = warp reinforcement-efficiency factor (swept).
Quantile models do not depend on the bounds -> fit once per seed, reuse across the sweep.

Note: the textile-polymer dataset (Malashin et al., 2024) is downloaded automatically
from github.com/catauggie/TPCM on first run. It ships with Russian column headers; the
loader matches them verbatim, so those lookup strings are intentionally left in the
original language (an explicit error is raised if the schema ever changes).
"""
import os, urllib.request
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pc3
from robust_cp import RobustPC3

ALPHA = 0.1; DATA = "data"
TPCM_URL = "https://raw.githubusercontent.com/catauggie/TPCM/main/data.xlsx"
def cov(y, lo, hi): return float(np.mean((y >= lo) & (y <= hi)))

def load_polymer_df():
    """Load the textile-polymer dataset, downloading it from the public TPCM
    repository (Malashin et al., 2024) on first use and caching it under ./data/."""
    os.makedirs(DATA, exist_ok=True)
    local = f"{DATA}/Polymer_TPCM.xlsx"
    if not os.path.exists(local):
        print(f"downloading textile-polymer dataset from {TPCM_URL} ...")
        urllib.request.urlretrieve(TPCM_URL, local)
    return pd.read_excel(local)

def sweep_fit(Xtr, ytr, mono):
    dummy = lambda Xq: (np.zeros(len(Xq)), np.full(len(Xq), 1e12))
    mn = pc3.PC3(mono, dummy, "cqr", False, True, False, alpha=ALPHA).fit(Xtr, ytr)
    mr = RobustPC3(mono, dummy, "cqr", False, True, False, alpha=ALPHA)
    mr.q_lo, mr.q_hi, mr.q_med = mn.q_lo, mn.q_hi, mn.q_med
    mc = pc3.PC3(mono, dummy, "cqr", False, False, False, alpha=ALPHA).fit(Xtr, ytr)
    return mn, mr, mc

def plot(eta, nai, rob, title, path):
    o = np.argsort(eta); eta, nai, rob = np.array(eta)[o], np.array(nai)[o], np.array(rob)[o]
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    xs = np.linspace(0, max(eta) + 1, 100)
    ax.plot(xs, 100 - np.maximum(10, xs), "--", color="#888", lw=1.2, label="admissible frontier 100−max(α,η)")
    ax.plot(eta, nai, "s-", color="#C0392B", ms=6, label="naive projection")
    ax.plot(eta, rob, "o-", color="#2E5496", ms=6.5, label="PC³ (ours)")
    ax.axhline(90, color="#ccc", lw=0.8); ax.axvline(10, color="#ccc", lw=0.9, ls=":"); ax.text(10.3, 50, "η=α", fontsize=9, color="#666")
    ax.set_xlabel("Real fraction of Y outside the bound, η, %"); ax.set_ylabel("Coverage, %"); ax.set_ylim(40, 100)
    ax.set_title(title); ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=600); plt.close()

# ================= SFRC =================
def run_sfrc():
    df = pd.read_excel(f"{DATA}/SFRC_Data_v1.xlsx")
    y = df["Fc"].values.astype(float)
    feat = [c for c in df.columns if c not in ["Fc", "L/D", "RM"]]
    Xdf = df[feat].apply(pd.to_numeric, errors="coerce")
    X = Xdf.fillna(Xdf.median()).values.astype(float); mono = np.zeros(X.shape[1])
    print(f"\n{'='*70}\nSFRC (n={len(df)}) - empirical ceiling (10 seeds)\n{'='*70}")
    percs = [100, 99, 97, 95, 92, 90, 87, 84]
    acc = {p: {"eta": [], "naive": [], "robust": []} for p in percs}
    for s in range(10):
        Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=0.5, random_state=s)
        Xcal, Xte, ycal, yte = train_test_split(Xt, yt, test_size=0.5, random_state=s)
        mn, mr, mc = sweep_fit(Xtr, ytr, mono)
        for p in percs:
            U = np.percentile(np.concatenate([ytr, ycal]), p)
            bf = (lambda Xq, U=U: (np.zeros(len(Xq)), np.full(len(Xq), U)))
            acc[p]["eta"].append(np.mean(yte > bf(Xte)[1]))
            mn.bounds_fn = bf; mn.calibrate(Xcal, ycal); _, lo, hi, _, _ = mn.predict(Xte); acc[p]["naive"].append(cov(yte, lo, hi))
            mr.bounds_fn = bf; mr.calibrate(Xcal, ycal); _, l2, h2, _, _ = mr.predict(Xte); acc[p]["robust"].append(cov(yte, l2, h2))
    rows = [(p, np.mean(acc[p]["eta"])*100, np.mean(acc[p]["naive"])*100, np.mean(acc[p]["robust"])*100) for p in percs]
    print(f'{"perc":>6}{"eta%":>8}{"naive%":>9}{"robust%":>9}')
    for r in rows: print(f'{r[0]:>6}{r[1]:>8.1f}{r[2]:>9.1f}{r[3]:>9.1f}')
    plot([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
         "REAL SFRC: PC³ under an empirical ceiling", "out/figQ_sfrc_robust.png")

# ================= Polymer =================
def find(cols, *subs):
    for c in cols:
        if all(s in c for s in subs): return c
    return None

def run_polymer():
    df = load_polymer_df()
    C = df.columns
    tgt = find(C, "ПКМ", "Прочность на растяжение по основе")
    fib = find(C, "НИТИ", "Прочность при растяжении волокна")
    if tgt is None or fib is None:
        raise SystemExit(
            "Polymer dataset: expected target / fibre-strength columns were not found "
            "(the dataset schema may have changed). Cannot proceed.")
    feats = {
        "fabric_tensile": find(C, "ТКАНИ", "Прочность при растяжении по основе"),
        "fabric_mod": find(C, "ТКАНИ", "Модуль упругости при растяжении по основе"),
        "fabric_areal": find(C, "Поверхностная плотность"),
        "fibre_tensile": fib,
        "fibre_mod": find(C, "НИТИ", "Модуль упругости волокна"),
        "binder_tensile": find(C, "СВЯЗУЮЩЕГО", "Прочность на растяжение"),
        "binder_mod": find(C, "СВЯЗУЮЩЕГО", "Модуль упругости"),
    }
    feats = {k: v for k, v in feats.items() if v is not None}
    d = df[[tgt, fib] + [v for v in feats.values()]].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=[tgt, fib]).reset_index(drop=True)
    y = d[tgt].values.astype(float)
    Fdf = d[list(feats.values())].copy(); Fdf = Fdf.fillna(Fdf.median())
    fibre = d[fib].values.astype(float)
    X = np.column_stack([Fdf.values.astype(float), fibre]); ifib = X.shape[1] - 1   # fibre_tensile last
    mono = np.zeros(X.shape[1])
    print(f"\n{'='*70}\nPolymer (n={len(d)}) - rule of mixtures U=c*fibre_strength (10 seeds)\n{'='*70}")
    print("target %.0f–%.0f MPa; fibre_tensile %.0f–%.0f MPa" % (y.min(), y.max(), fibre.min(), fibre.max()))
    cs = [1.0, 0.8, 0.6, 0.5, 0.45, 0.40, 0.35, 0.30]
    acc = {c: {"eta": [], "naive": [], "robust": []} for c in cs}
    for s in range(10):
        Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=0.5, random_state=s)
        Xcal, Xte, ycal, yte = train_test_split(Xt, yt, test_size=0.5, random_state=s)
        mn, mr, mc = sweep_fit(Xtr, ytr, mono)
        for c in cs:
            bf = (lambda Xq, c=c: (np.zeros(len(Xq)), c * Xq[:, ifib]))
            acc[c]["eta"].append(np.mean(yte > bf(Xte)[1]))
            mn.bounds_fn = bf; mn.calibrate(Xcal, ycal); _, lo, hi, _, _ = mn.predict(Xte); acc[c]["naive"].append(cov(yte, lo, hi))
            mr.bounds_fn = bf; mr.calibrate(Xcal, ycal); _, l2, h2, _, _ = mr.predict(Xte); acc[c]["robust"].append(cov(yte, l2, h2))
    rows = [(c, np.mean(acc[c]["eta"])*100, np.mean(acc[c]["naive"])*100, np.mean(acc[c]["robust"])*100) for c in cs]
    print(f'{"c":>6}{"eta%":>8}{"naive%":>9}{"robust%":>9}')
    for r in rows: print(f'{r[0]:>6.2f}{r[1]:>8.1f}{r[2]:>9.1f}{r[3]:>9.1f}')
    plot([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
         "REAL polymer: PC³ under a rule-of-mixtures bound", "out/figR_polymer_robust.png")

def combine_panels(out="out/figQR_real_composites.png"):
    """Paper Figure 7: SFRC (a) and textile-polymer (b) panels side by side."""
    import matplotlib.image as mpimg
    a = mpimg.imread("out/figQ_sfrc_robust.png"); b = mpimg.imread("out/figR_polymer_robust.png")
    fig, ax = plt.subplots(1, 2, figsize=(14.4, 4.9))
    for axi, im, lab in zip(ax, [a, b], ["(a)", "(b)"]):
        axi.imshow(im); axi.axis("off")
        axi.text(0.01, 0.985, lab, transform=axi.transAxes, fontsize=15, fontweight="bold", va="top", ha="left")
    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.02)
    plt.savefig(out, dpi=600); plt.close()
    return out

if __name__ == "__main__":
    run_sfrc(); run_polymer(); combine_panels()
    print("\nFigures: out/figQ_sfrc_robust.png, out/figR_polymer_robust.png, out/figQR_real_composites.png (paper Figure 7)")
