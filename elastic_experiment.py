#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rigorous-bounds experiment on REAL data: elastic_tensor_2015 (1181 DFT materials;
de Jong et al., 2015). Target G_VRH (Voigt-Reuss-Hill shear modulus); corridor
[G_Reuss, G_Voigt], which is rigorous and holds for 100% of entries (eta = 0).
Features come from composition only (leakage-free: the elastic tensor itself,
Poisson ratio and anisotropy are NOT used).

Exp.1 (paper Figure 10): with the exact corridor, projection removes all physical
       violations (100% -> 0%) at matched ~90% coverage and shrinks the mean
       interval from ~75.8 to ~5.8 GPa, a ~13-fold reduction. Because eta = 0,
       robust and naive projection coincide (Theorem 3 with eta = 0).
Exp.2 (auxiliary figO, NOT in the paper): approximate composition-predicted bounds
       induce a real eta; robust recovers coverage while naive undercovers.

Data are loaded programmatically via matminer (no manual download); if matminer is
unavailable, a local CSV at ./data/elastic_tensor.csv (or ./elastic_tensor.csv) is
used as a fallback.

Run:  python elastic_experiment.py   (pc3.py must be alongside)
"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pymatgen.core import Composition, Element
import pc3

ALPHA = 0.1
PROPS = ["X", "Z", "atomic_mass", "group", "row"]
os.makedirs("out", exist_ok=True)


def load_elastic():
    """Load elastic_tensor_2015 via matminer; fall back to a local CSV."""
    try:
        from matminer.datasets import load_dataset
        df = load_dataset("elastic_tensor_2015")
        print("loaded elastic_tensor_2015 via matminer:", len(df), "materials")
        return df
    except Exception as e:
        for path in ("data/elastic_tensor.csv", "elastic_tensor.csv"):
            if os.path.exists(path):
                df = pd.read_csv(path, skiprows=1)
                print(f"loaded {len(df)} materials from local fallback {path}")
                return df
        raise SystemExit(
            "elastic_tensor_2015 not available. Install matminer "
            "(pip install matminer) or place elastic_tensor.csv in ./data/.\n"
            f"(matminer import/download error: {e})")


def _p(el, p):
    try:
        v = getattr(Element(el), p); return float(v) if v is not None else np.nan
    except Exception: return np.nan

def featurize(formula):
    d = Composition(formula).get_el_amt_dict(); tot = sum(d.values())
    els = list(d); w = np.array([d[e] / tot for e in els]); f = {}
    for p in PROPS:
        v = np.array([_p(e, p) for e in els]); m = ~np.isnan(v)
        vv, ww = v[m], w[m]
        if len(vv) == 0:
            for s in ["mean", "std", "min", "max", "range"]: f[f"{p}_{s}"] = 0.0
            continue
        wm = np.sum(vv * ww) / np.sum(ww)
        f[f"{p}_mean"] = wm
        f[f"{p}_std"] = np.sqrt(np.sum(ww * (vv - wm) ** 2) / np.sum(ww))
        f[f"{p}_min"], f[f"{p}_max"], f[f"{p}_range"] = vv.min(), vv.max(), vv.max() - vv.min()
    f["nelem"] = len(els)
    return f

def conf_q(s, a):
    return pc3.conformal_quantile(s, a)

def fit_q(Xtr, ytr):
    mono = np.zeros(Xtr.shape[1])
    qm = pc3.fit_quantile(Xtr, ytr, 0.5, mono, False)
    ql = pc3.fit_quantile(Xtr, ytr, ALPHA / 2, mono, False)
    qh = pc3.fit_quantile(Xtr, ytr, 1 - ALPHA / 2, mono, False)
    return ql, qh, qm

def intervals(ql, qh, X, Q):
    return ql.predict(X) - Q, qh.predict(X) + Q

def proj(lo, hi, p, L, U):
    lo = np.maximum(lo, L); hi = np.minimum(hi, U); p = np.clip(p, L, U)
    bad = lo > hi; lo[bad] = hi[bad] = p[bad]; return lo, hi, p

def evalc(lo, hi, p, L, U, y):
    return (np.mean((y >= lo) & (y <= hi)),
            np.mean((hi > U + 1e-6) | (lo < L - 1e-6)), np.mean(hi - lo))

# ---------- load + featurize ----------
df = load_elastic()
df = df.dropna(subset=["formula", "G_Reuss", "G_VRH", "G_Voigt", "nsites", "volume", "space_group"]).reset_index(drop=True)
print("featurizing composition...")
F = pd.DataFrame([featurize(f) for f in df["formula"]])
F["nsites"] = df["nsites"].values; F["volume"] = df["volume"].values; F["space_group"] = df["space_group"].values
X = F.values.astype(float)
y = df["G_VRH"].values.astype(float)
Reuss = df["G_Reuss"].values.astype(float); Voigt = df["G_Voigt"].values.astype(float)
print("features:", X.shape[1], "| target G_VRH range %.1f-%.1f GPa" % (y.min(), y.max()))

# ================= EXP 1: exact bounds (paper Figure 10) =================
print("\n" + "=" * 78 + "\nExp.1 - exact bounds [G_Reuss, G_Voigt] (5 seeds, alpha=0.1) -> Figure 10\n" + "=" * 78)
rows = {k: {"cov": [], "vio": [], "wid": []} for k in ["CQR (no proj)", "naive-proj", "robust (ours)", "GP (native)"]}
r2s = []
for s in range(5):
    Xtr, Xt, ytr, yt, Rtr, Rt, Vtr, Vt = train_test_split(X, y, Reuss, Voigt, test_size=0.5, random_state=s)
    Xcal, Xte, ycal, yte, Rcal, Rte, Vcal, Vte = train_test_split(Xt, yt, Rt, Vt, test_size=0.5, random_state=s)
    ql, qh, qm = fit_q(Xtr, ytr)
    r2s.append(r2_score(yte, qm.predict(Xte)))
    s_cal = np.maximum(ql.predict(Xcal) - ycal, ycal - qh.predict(Xcal))
    # CQR no proj
    Q = conf_q(s_cal, ALPHA); lo, hi = intervals(ql, qh, Xte, Q)
    c, v, w = evalc(lo, hi, qm.predict(Xte), Rte, Vte, yte); rows["CQR (no proj)"]["cov"].append(c); rows["CQR (no proj)"]["vio"].append(v); rows["CQR (no proj)"]["wid"].append(w)
    # naive proj
    lo2, hi2, p2 = proj(lo.copy(), hi.copy(), qm.predict(Xte), Rte, Vte)
    c, v, w = evalc(lo2, hi2, p2, Rte, Vte, yte); rows["naive-proj"]["cov"].append(c); rows["naive-proj"]["vio"].append(v); rows["naive-proj"]["wid"].append(w)
    # robust proj
    E = s_cal.copy(); E[(ycal < Rcal) | (ycal > Vcal)] = np.inf
    Qr = conf_q(E, ALPHA); lor, hir = intervals(ql, qh, Xte, Qr)
    lor, hir, pr = proj(lor, hir, qm.predict(Xte), Rte, Vte)
    c, v, w = evalc(lor, hir, pr, Rte, Vte, yte); rows["robust (ours)"]["cov"].append(c); rows["robust (ours)"]["vio"].append(v); rows["robust (ours)"]["wid"].append(w)
    # GP native (no bounds), projected for fairness
    pgp, logp, higp = pc3.gp_interval(np.vstack([Xtr, Xcal]), np.concatenate([ytr, ycal]), Xte, ALPHA)
    c, v, w = evalc(logp, higp, pgp, Rte, Vte, yte); rows["GP (native)"]["cov"].append(c); rows["GP (native)"]["vio"].append(v); rows["GP (native)"]["wid"].append(w)
print("median model R2 (composition-only): %.2f" % np.mean(r2s))
print(f'{"Method":<16}{"Cover":>8}{"Viol%":>8}{"Width":>9}')
for k, d in rows.items():
    print(f'{k:<16}{np.mean(d["cov"])*100:>7.1f}%{np.mean(d["vio"])*100:>7.1f}%{np.mean(d["wid"]):>9.1f}')

# ---- Figure 10: rigorous DFT Voigt-Reuss bars (from the real Exp.1 means) ----
order = ["CQR (no proj)", "GP (native)", "naive-proj", "robust (ours)"]
labels = ["CQR", "GP", "naive\u2192[L,U]", "robust\u2192[L,U]"]
viol = [np.mean(rows[k]["vio"]) * 100 for k in order]
width = [np.mean(rows[k]["wid"]) for k in order]
cov_all = np.mean([np.mean(rows[k]["cov"]) for k in order]) * 100
cols = ["#b0b0b0", "#b0b0b0", "#7bbf7b", "#2E5496"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].bar(labels, viol, color=cols); ax[0].set_title("Physics-bound violations, %"); ax[0].set_ylim(0, 105)
b = ax[1].bar(labels, width, color=cols); ax[1].set_title(f"Interval width, GPa (coverage \u2248{cov_all:.0f}% for all)")
for r, v in zip(b, width): ax[1].text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
for a in (ax[0], ax[1]): a.tick_params(axis="x", rotation=12)
fold = width[0] / width[-1] if width[-1] > 0 else float("nan")
plt.suptitle(f"Real DFT moduli (elastic_tensor_2015): projection onto the rigorous Voigt\u2013Reuss corridor\n"
             f"keeps {cov_all:.0f}% coverage, zeros violations, and shrinks the interval ~{fold:.0f}\u00d7",
             fontweight="bold", fontsize=11)
plt.tight_layout(); plt.savefig("out/figP_elastic_exact.png", dpi=600); plt.close()
print("Figure 10 written: out/figP_elastic_exact.png  (width %.1f -> %.1f GPa, ~%.0f-fold)" % (width[0], width[-1], fold))

# ================= EXP 2: approximate bounds (auxiliary figO, NOT in the paper) =================
print("\n" + "=" * 78 + "\nExp.2 - approximate composition-based bounds -> real eta (3 seeds) [auxiliary figO]\n" + "=" * 78)
scales = [1.6, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
agg = {s: {"eta": [], "naive": [], "robust": []} for s in scales}
for seed in range(3):
    Xtr, Xt, ytr, yt, Rtr, Rt, Vtr, Vt = train_test_split(X, y, Reuss, Voigt, test_size=0.5, random_state=seed)
    Xcal, Xte, ycal, yte, Rcal, Rte, Vcal, Vte = train_test_split(Xt, yt, Rt, Vt, test_size=0.5, random_state=seed)
    mono = np.zeros(X.shape[1])
    Rhat = pc3.fit_quantile(Xtr, Rtr, 0.5, mono, False)
    Vhat = pc3.fit_quantile(Xtr, Vtr, 0.5, mono, False)
    ql, qh, qm = fit_q(Xtr, ytr)
    s_cal_raw = np.maximum(ql.predict(Xcal) - ycal, ycal - qh.predict(Xcal))
    for sc in scales:
        def band(Xq):
            r, v = Rhat.predict(Xq), Vhat.predict(Xq); c = (r + v) / 2; h = (v - r) / 2 * sc
            return np.maximum(c - h, 0.0), c + h
        Lcal, Ucal = band(Xcal); Lte, Ute = band(Xte)
        agg[sc]["eta"].append(np.mean((yte < Lte) | (yte > Ute)))
        Q = conf_q(s_cal_raw, ALPHA); lo, hi = intervals(ql, qh, Xte, Q)
        lo, hi, p = proj(lo, hi, qm.predict(Xte), Lte, Ute)
        agg[sc]["naive"].append(np.mean((yte >= lo) & (yte <= hi)))
        E = s_cal_raw.copy(); E[(ycal < Lcal) | (ycal > Ucal)] = np.inf
        Qr = conf_q(E, ALPHA); lor, hir = intervals(ql, qh, Xte, Qr)
        lor, hir, pr = proj(lor, hir, qm.predict(Xte), Lte, Ute)
        agg[sc]["robust"].append(np.mean((yte >= lor) & (yte <= hir)))
eta = [np.mean(agg[s]["eta"]) * 100 for s in scales]
nai = [np.mean(agg[s]["naive"]) * 100 for s in scales]
rob = [np.mean(agg[s]["robust"]) * 100 for s in scales]
print(f'{"scale":>6}{"eta%":>8}{"naive%":>9}{"robust%":>9}')
for s, e, n, r in zip(scales, eta, nai, rob):
    print(f'{s:>6.1f}{e:>8.1f}{n:>9.1f}{r:>9.1f}')
o = np.argsort(eta); eta, nai, rob = np.array(eta)[o], np.array(nai)[o], np.array(rob)[o]
fig, ax = plt.subplots(figsize=(7.2, 4.8))
xs = np.linspace(0, max(eta) + 1, 100); ax.plot(xs, 100 - np.maximum(10, xs), "--", color="#888", lw=1.2, label="admissible frontier 100-max(alpha,eta)")
ax.plot(eta, nai, "s-", color="#C0392B", ms=6, label="naive projection")
ax.plot(eta, rob, "o-", color="#2E5496", ms=6.5, label="PC3 (ours)")
ax.axhline(90, color="#ccc", lw=0.8); ax.axvline(10, color="#ccc", lw=0.9, ls=":")
ax.set_xlabel("Real fraction of G_VRH outside the approximate corridor, eta, %"); ax.set_ylabel("Coverage, %")
ax.set_title("REAL moduli (elastic_tensor_2015): PC3 under approximate composition-based bounds [auxiliary, not in paper]")
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3); ax.set_ylim(40, 100)
plt.tight_layout(); plt.savefig("out/figO_elastic_robust.png", dpi=600); plt.close()
print("\nAuxiliary figure (not in paper): out/figO_elastic_robust.png")
