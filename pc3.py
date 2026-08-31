#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC3 - Physics-Constrained Conformal prediction for Composites (v2)
==================================================================
Reproducible reference implementation for the paper (submitted to Information, MDPI).

v2 adds:
  * Native UQ baselines: Gaussian Process, NGBoost, Deep Ensemble, MC-Dropout.
  * Mondrian conformal prediction - approximate CONDITIONAL coverage by family/regime.
  * Conditional-coverage metrics: WrstGrp (worst group) and Gap (max deviation from target).

PC3 core: monotone quantile models -> physics-aware conformal calibration
          -> physical projection into the corridor [L(x),U(x)] -> SHAP -> deployment inference.

Run:  python pc3.py
Dependencies: numpy pandas scikit-learn ngboost scipy shap matplotlib
(LightGBM/torch are NOT required: monotone quantiles use HistGradientBoostingRegressor;
 the neural baselines use sklearn MLP, and MC-Dropout is applied at inference.)
"""
import warnings, os, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as CK, RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
except Exception:                     # ngboost optional: only the native-UQ baseline needs it
    NGBRegressor = Normal = None

RNG = 0
SEEDS = 5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# ============================================================================
# 1. DATA + PHYSICS (monotonicity, bounds, groups for Mondrian)
# ============================================================================
def load_concrete(path="concrete.csv"):
    if not os.path.exists(path):
        import urllib.request
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path)
    target = "strength" if "strength" in df.columns else df.columns[-1]
    y = df[target].values.astype(float)
    feats = [c for c in df.columns if c != target]
    X = df[feats].values.astype(float)
    sign = {"cement": +1, "water": -1, "age": +1}             # physically motivated monotonicity
    monotone = [sign.get(f, 0) for f in feats]
    U_cap = float(np.percentile(y, 99) * 1.15)
    def bounds_fn(Xq):                                        # strength >= 0 (negatives are non-physical)
        n = len(Xq); return np.zeros(n), np.full(n, U_cap)
    ai = feats.index("age")
    def group_fn(Xq):                                         # regimes by age: <=7 / 8-28 / >28 days
        a = Xq[:, ai]; return np.where(a <= 7, 0, np.where(a <= 28, 1, 2))
    return X, y, monotone, bounds_fn, feats, group_fn


def make_synthetic_composite(n=1100, seed=RNG):
    rs = np.random.RandomState(seed)
    Vf = rs.uniform(0.10, 0.70, n); Ef = rs.uniform(200, 400, n); Em = rs.uniform(2.0, 5.0, n)
    z1, z2 = rs.normal(size=n), rs.normal(size=n)
    voigt = Vf * Ef + (1 - Vf) * Em
    reuss = 1.0 / (Vf / Ef + (1 - Vf) / Em)
    s = 0.35 + 0.40 * Vf
    E_true = reuss + s * (voigt - reuss)
    y = np.clip(E_true + rs.normal(0, 0.06 * (voigt - reuss)), reuss, voigt)   # Y in [Reuss,Voigt] a.s.
    X = np.column_stack([Vf, Ef, Em, z1, z2]); feats = ["Vf", "Ef", "Em", "noise1", "noise2"]
    monotone = [+1, +1, +1, 0, 0]
    def bounds_fn(Xq):
        vf, ef, em = Xq[:, 0], Xq[:, 1], Xq[:, 2]
        return 1.0 / (vf / ef + (1 - vf) / em), vf * ef + (1 - vf) * em        # exact Reuss/Voigt
    def group_fn(Xq):                                                         # regimes by fibre fraction Vf
        vf = Xq[:, 0]; return np.where(vf <= 0.30, 0, np.where(vf <= 0.50, 1, 2))
    return X, y, monotone, bounds_fn, feats, group_fn


# ============================================================================
# 2. MONOTONE QUANTILE MODELS + conformal quantile
# ============================================================================
def fit_quantile(Xtr, ytr, alpha, monotone, use_monotone):
    mc = monotone if use_monotone else [0] * Xtr.shape[1]
    m = HistGradientBoostingRegressor(loss="quantile", quantile=alpha, monotonic_cst=mc,
                                      max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                                      min_samples_leaf=20, l2_regularization=1.0, random_state=RNG)
    return m.fit(Xtr, ytr)

def conformal_quantile(scores, alpha):
    n = len(scores); level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, level, method="higher")


# ============================================================================
# 3. PC3 (conformal methods + Mondrian; flags assemble the ablations)
# ============================================================================
class PC3:
    """Physics-constrained conformal prediction (paper Algorithm 1).

    Parameters mirror the algorithm: ``use_monotone`` (monotone quantile base models),
    ``physics_aware`` (band-normalised CQR score, Eq. 4), ``project`` (intersection with the
    corridor [L(x), U(x)]) and ``robust`` (the projection-aware score of Eq. 6: a calibration
    point whose response lies outside the corridor receives an infinite nonconformity score).

    ``robust=True`` is the recommended setting whenever the bounds may be misspecified. It is
    exactly nested conformal prediction (Gupta et al., 2022) applied to the *projected* family
    of intervals, so the split-conformal guarantee holds with no assumption on the bounds.
    After ``calibrate`` the object exposes ``eta_hat`` (fraction of calibration responses
    outside the corridor) and ``feasible`` (``Q < inf``); when the target 1-alpha is not
    attainable inside the corridor, ``Q = +inf`` and ``predict`` returns the full corridor.
    """
    def __init__(self, monotone, bounds_fn, score="cqr", use_monotone=True, project=True,
                 physics_aware=True, alpha=0.1, mondrian=False, group_fn=None, robust=False):
        self.monotone, self.bounds_fn, self.score = monotone, bounds_fn, score
        self.use_monotone, self.project, self.physics_aware = use_monotone, project, physics_aware
        self.alpha, self.mondrian, self.group_fn = alpha, mondrian, group_fn
        self.robust = robust
        self.eta_hat, self.feasible, self.n_cal = None, None, None

    def fit(self, Xtr, ytr):
        a = self.alpha
        self.q_med = fit_quantile(Xtr, ytr, 0.5, self.monotone, self.use_monotone)
        if self.score == "cqr":
            self.q_lo = fit_quantile(Xtr, ytr, a / 2, self.monotone, self.use_monotone)
            self.q_hi = fit_quantile(Xtr, ytr, 1 - a / 2, self.monotone, self.use_monotone)
        return self

    def _band(self, X):
        L, U = self.bounds_fn(X); return np.maximum(U - L, 1e-9)

    def _scores(self, Xcal, ycal):
        if self.score == "cqr":
            E = np.maximum(self.q_lo.predict(Xcal) - ycal, ycal - self.q_hi.predict(Xcal))
        else:
            E = np.abs(ycal - self.q_med.predict(Xcal))
        if self.physics_aware:
            E = E / (self._band(Xcal) / self.wbar)
        if self.robust:                                                  # projection-aware score, Eq. (6)
            E = E.astype(float)
            L, U = self.bounds_fn(Xcal)
            E[(ycal < L - 1e-12) | (ycal > U + 1e-12)] = np.inf
        return E

    def calibrate(self, Xcal, ycal):
        self.wbar = self._band(Xcal).mean() if self.physics_aware else 1.0
        E = self._scores(Xcal, ycal)
        L, U = self.bounds_fn(Xcal)
        self.n_cal = len(ycal)
        self.eta_hat = float(np.mean((ycal < L - 1e-12) | (ycal > U + 1e-12)))  # bound-violation rate on calibration
        self.Qg = conformal_quantile(E, self.alpha)                     # global quantile
        self.feasible = bool(np.isfinite(self.Qg))                      # Q = +inf  =>  full corridor
        self.Qgrp = {}
        if self.mondrian and self.group_fn is not None:                 # per-group quantile
            g = self.group_fn(Xcal)
            for gg in np.unique(g):
                idx = g == gg
                self.Qgrp[gg] = conformal_quantile(E[idx], self.alpha) if idx.sum() >= 20 else self.Qg
        return self

    def _Q(self, X):
        if self.mondrian and self.group_fn is not None:
            g = self.group_fn(X); return np.array([self.Qgrp.get(gg, self.Qg) for gg in g])
        return np.full(len(X), self.Qg)

    def predict_base(self, X):
        """Unprojected (point, lo, hi): the conformalised interval of Eq. (5) before projection."""
        w = (self._band(X) / self.wbar) if self.physics_aware else np.ones(len(X))
        Q = self._Q(X)
        if self.score == "cqr":
            lo = self.q_lo.predict(X) - Q * w; hi = self.q_hi.predict(X) + Q * w
        else:
            med = self.q_med.predict(X); lo = med - Q * w; hi = med + Q * w
        return self.q_med.predict(X), lo, hi

    def predict(self, X):
        L, U = self.bounds_fn(X)
        point, lo, hi = self.predict_base(X)
        if self.project:
            lo = np.maximum(lo, L); hi = np.minimum(hi, U); point = np.clip(point, L, U)
            bad = lo > hi; lo[bad] = hi[bad] = point[bad]
        return point, lo, hi, L, U


# ============================================================================
# 4. NATIVE UQ BASELINES (own intervals, no conformal calibration)
# ============================================================================
def _z(alpha): return norm.ppf(1 - alpha / 2)

def gp_interval(Xtr, ytr, Xte, alpha, cap=500):
    if len(Xtr) > cap:
        idx = np.random.RandomState(0).choice(len(Xtr), cap, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    sc = StandardScaler().fit(Xtr); Xs, Xts = sc.transform(Xtr), sc.transform(Xte)
    k = CK(1.0) * RBF(np.ones(Xtr.shape[1])) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0,
                                  random_state=0).fit(Xs, ytr)
    mu, sd = gp.predict(Xts, return_std=True); z = _z(alpha)
    return mu, mu - z * sd, mu + z * sd

def ngb_interval(Xtr, ytr, Xte, alpha):
    # NGBoost draws from the global NumPy stream during boosting, so random_state
    # alone does not make it reproducible; seed the global stream as well.
    np.random.seed(0)
    m = NGBRegressor(Dist=Normal, n_estimators=300, learning_rate=0.04, verbose=False,
                     random_state=0).fit(Xtr, ytr)
    d = m.pred_dist(Xte); z = _z(alpha)
    return d.loc, d.loc - z * d.scale, d.loc + z * d.scale

def ensemble_interval(Xtr, ytr, Xte, alpha, K=4):
    sc = StandardScaler().fit(Xtr); Xs, Xts = sc.transform(Xtr), sc.transform(Xte)
    P = np.array([MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=250, alpha=1e-3,
                               random_state=k).fit(Xs, ytr).predict(Xts) for k in range(K)])
    mu, sd, z = P.mean(0), P.std(0) + 1e-6, _z(alpha)
    return mu, mu - z * sd, mu + z * sd

def mcdropout_interval(Xtr, ytr, Xte, alpha, T=50, p=0.1):
    sc = StandardScaler().fit(Xtr); Xs, Xts = sc.transform(Xtr), sc.transform(Xte)
    m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, alpha=1e-3, random_state=0).fit(Xs, ytr)
    W, b, rng = m.coefs_, m.intercepts_, np.random.RandomState(0)
    def fwd(X):
        a = X
        for i in range(len(W) - 1):
            a = np.maximum(0, a @ W[i] + b[i])
            a = a * ((rng.random(a.shape) > p) / (1 - p))      # inverted dropout at inference
        return (a @ W[-1] + b[-1]).ravel()
    O = np.array([fwd(Xts) for _ in range(T)])
    mu, sd, z = O.mean(0), O.std(0) + 1e-6, _z(alpha)
    return mu, mu - z * sd, mu + z * sd


# ============================================================================
# 5. METHODS + EVALUATION
# ============================================================================
METHODS = {
    "GP (native)":            {"kind": "gp"},
    "NGBoost (native)":       {"kind": "ngb"},
    "Deep Ensemble (native)": {"kind": "ens"},
    "MC-Dropout (native~)":   {"kind": "mcd"},
    "Split-Conformal":        {"kind": "cf", "score": "residual", "mono": False, "proj": False, "pa": False},
    "CQR":                    {"kind": "cf", "score": "cqr", "mono": False, "proj": False, "pa": False},
    "PC3 (marginal)":         {"kind": "cf", "score": "cqr", "mono": True, "proj": True, "pa": True},
    "PC3 + Mondrian":         {"kind": "cf", "score": "cqr", "mono": True, "proj": True, "pa": True, "mondrian": True},
}

def run_method(spec, Xtr, ytr, Xcal, ycal, Xte, monotone, bounds_fn, group_fn, alpha):
    if spec["kind"] == "cf":
        mdl = PC3(monotone, bounds_fn, score=spec["score"], use_monotone=spec["mono"],
                  project=spec["proj"], physics_aware=spec["pa"], alpha=alpha,
                  mondrian=spec.get("mondrian", False), group_fn=group_fn)
        return mdl.fit(Xtr, ytr).calibrate(Xcal, ycal).predict(Xte)
    Xtra, ytra = np.vstack([Xtr, Xcal]), np.concatenate([ytr, ycal])   # native models train on train+cal
    L, U = bounds_fn(Xte)
    fn = {"gp": gp_interval, "ngb": ngb_interval, "ens": ensemble_interval, "mcd": mcdropout_interval}[spec["kind"]]
    point, lo, hi = fn(Xtra, ytra, Xte, alpha)
    return point, lo, hi, L, U

def evaluate(point, lo, hi, L, U, ytest, groups=None, target=0.9):
    inside = (ytest >= lo) & (ytest <= hi)
    viol = ((lo < L - 1e-9) | (hi > U + 1e-9) | (point < L - 1e-9) | (point > U + 1e-9)).mean()
    d = dict(R2=r2_score(ytest, point), MAE=mean_absolute_error(ytest, point),
             RMSE=np.sqrt(mean_squared_error(ytest, point)),
             Coverage=inside.mean(), Width=(hi - lo).mean(), Violations=viol)
    if groups is not None:
        covs = [inside[groups == g].mean() for g in np.unique(groups) if (groups == g).sum() > 0]
        d["WorstGrp"] = min(covs); d["Gap"] = max(abs(c - target) for c in covs)
    return d


def run_experiment(loader, name, alpha=0.1, seeds=SEEDS):
    print(f"\n{'='*104}\nDATASET: {name}   (target = {int((1-alpha)*100)}%, {seeds} seeds)\n{'='*104}")
    X, y, monotone, bounds_fn, feats, group_fn = loader()
    keys = ["R2", "Coverage", "Width", "Violations", "WorstGrp", "Gap"]
    agg = {m: {k: [] for k in keys} for m in METHODS}
    for s in range(seeds):
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=s)
        Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=s)
        gte = group_fn(Xte)
        for m, spec in METHODS.items():
            res = evaluate(*run_method(spec, Xtr, ytr, Xcal, ycal, Xte, monotone, bounds_fn, group_fn, alpha),
                           yte, groups=gte, target=1 - alpha)
            for k in keys: agg[m][k].append(res[k])
        print(f"  seed {s+1}/{seeds} done")
    hdr = f'{"Method":<24}{"R2":>7}{"Cover":>8}{"Width":>9}{"Viol%":>8}{"WrstGrp":>9}{"Gap%":>7}'
    print(hdr); print("-" * len(hdr))
    for m in METHODS:
        a = {k: np.mean(v) for k, v in agg[m].items()}
        print(f'{m:<24}{a["R2"]:>7.3f}{a["Coverage"]*100:>7.1f}%{a["Width"]:>9.2f}'
              f'{a["Violations"]*100:>7.1f}%{a["WorstGrp"]*100:>8.1f}%{a["Gap"]*100:>6.1f}%')
    print(f'(target coverage={int((1-alpha)*100)}%; WrstGrp=worst-group coverage; Gap=max per-group deviation)')
    return dict(X=X, y=y, monotone=monotone, bounds_fn=bounds_fn, feats=feats, group_fn=group_fn, agg=agg)


# ============================================================================
# 6. FIGURES
# ============================================================================
def make_figures(synth, concrete, alpha=0.1):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    short = {"GP (native)": "GP", "NGBoost (native)": "NGB", "Deep Ensemble (native)": "Ens",
             "MC-Dropout (native~)": "MCD", "Split-Conformal": "Split-CP", "CQR": "CQR",
             "PC3 (marginal)": "PC3", "PC3 + Mondrian": "PC3+Mon"}
    ms = list(METHODS.keys()); names = [short[m] for m in ms]
    cov = [np.mean(synth["agg"][m]["Coverage"]) * 100 for m in ms]
    wid = [np.mean(synth["agg"][m]["Width"]) for m in ms]
    vio = [np.mean(synth["agg"][m]["Violations"]) * 100 for m in ms]
    cols = ["#B0B0B0"]*4 + ["#7FA8D0", "#7FA8D0", "#4C78A8", "#2E5496"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].bar(names, cov, color=cols); ax[0].axhline((1-alpha)*100, ls="--", c="r", lw=1.2, label="target")
    ax[0].set_title("Coverage, %"); ax[0].set_ylim(min(cov)-4, 100); ax[0].legend()
    ax[1].bar(names, wid, color=cols); ax[1].set_title("Width (↓ better)")
    ax[2].bar(names, vio, color=cols); ax[2].set_title("Physics violations, % (↓ better)")
    for a in ax: a.tick_params(axis="x", rotation=30)
    plt.suptitle("Synthetic: native UQ (grey) miscovers/violates physics; PC³ on target, 0% violations",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/figA_method_comparison.png", dpi=600); plt.close()

    # Fig B: PC3 calibration (alpha sweep)
    X, y, mono, bf, gf = synth["X"], synth["y"], synth["monotone"], synth["bounds_fn"], synth["group_fn"]
    nominal = np.array([0.80, 0.85, 0.90, 0.95]); emp = []
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=0)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0)
    for ct in nominal:
        al = 1 - ct
        mdl = PC3(mono, bf, "cqr", True, True, True, alpha=al).fit(Xtr, ytr).calibrate(Xcal, ycal)
        p, lo, hi, L, U = mdl.predict(Xte); emp.append(np.mean((yte >= lo) & (yte <= hi)))
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0.78, 0.97], [0.78, 0.97], "k--", label="ideal"); ax.plot(nominal, emp, "o-", color="#4C78A8", label="PC³")
    ax.set_xlabel("Nominal 1−α"); ax.set_ylabel("Empirical coverage")
    ax.set_title("PC³ calibration (synthetic)"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/figB_calibration.png", dpi=600); plt.close()

    # Fig C: parity + intervals on real concrete
    Xc, yc, mc, bfc, gfc = concrete["X"], concrete["y"], concrete["monotone"], concrete["bounds_fn"], concrete["group_fn"]
    featc = concrete["feats"]
    Xtr, Xtmp, ytr, ytmp = train_test_split(Xc, yc, test_size=0.5, random_state=0)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0)
    mdl = PC3(mc, bfc, "cqr", True, True, True, alpha=alpha).fit(Xtr, ytr).calibrate(Xcal, ycal)
    p, lo, hi, L, U = mdl.predict(Xte)
    idx = np.argsort(p)[:: max(1, len(p)//120)]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.errorbar(yte[idx], p[idx], yerr=[p[idx]-lo[idx], hi[idx]-p[idx]], fmt="o", ms=3,
                color="#4C78A8", ecolor="#9ecae1", alpha=0.85, label="PC³ ± interval")
    lim = [min(yte.min(), p.min()), max(yte.max(), p.max())]; ax.plot(lim, lim, "k--", label="ideal")
    ax.set_xlabel("True strength, MPa"); ax.set_ylabel("Prediction ± interval")
    ax.set_title("UCI Concrete: PC³ with intervals"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/figC_concrete_parity.png", dpi=600); plt.close()

    # Fig D: SHAP (concrete)
    try:
        import shap
        bg = shap.utils.sample(Xtr, 80, random_state=0); sub = Xte[:200]
        sv = np.array(shap.Explainer(mdl.q_med.predict, bg)(sub).values)
        plt.figure(); shap.summary_plot(sv, sub, feature_names=featc, show=False)
        plt.title("SHAP (concrete, PC³ median)"); plt.tight_layout()
        plt.savefig(f"{OUT}/figD_shap_concrete.png", dpi=600, bbox_inches="tight"); plt.close()
        print("\nMonotonicity via SHAP (sign of corr should match the prior):")
        for j, f in enumerate(featc):
            if mc[j] != 0:
                c = np.corrcoef(sub[:, j], sv[:, j])[0, 1]
                print(f"   {f:<10} prior {mc[j]:+d}  corr={c:+.2f}  {'OK' if np.sign(c)==np.sign(mc[j]) else '??'}")
    except Exception as e:
        print("SHAP skipped:", e)

    # Fig E: Mondrian - conditional coverage by group (concrete, by age)
    gte = gfc(Xte); gids = np.unique(gte); labels = {0: "age≤7", 1: "8–28", 2: ">28"}
    res = {}
    for tag, mon in [("PC³ (marginal)", False), ("PC³ + Mondrian", True)]:
        mdl = PC3(mc, bfc, "cqr", True, True, True, alpha=alpha, mondrian=mon, group_fn=gfc)
        mdl.fit(Xtr, ytr).calibrate(Xcal, ycal)
        p, lo, hi, L, U = mdl.predict(Xte); ins = (yte >= lo) & (yte <= hi)
        res[tag] = [ins[gte == g].mean()*100 for g in gids]
    ngrp = [int((gte == g).sum()) for g in gids]
    x = np.arange(len(gids)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    b1 = ax.bar(x-w/2, res["PC³ (marginal)"], w, label="PC$^3$ (marginal)",
                color="#9ecae1", edgecolor="#5a7fa6", lw=.6)
    b2 = ax.bar(x+w/2, res["PC³ + Mondrian"], w, label="PC$^3$ + Mondrian",
                color="#2E5496", edgecolor="#1d3a6b", lw=.6)
    ax.axhline((1-alpha)*100, ls="--", c="#C0392B", lw=1.2,
               label=r"target $1-\alpha=%d\%%$" % round((1-alpha)*100))
    for bars in (b1, b2):                                  # value labels: the caption carries the message,
        for r in bars:                                     # so the figure needs no in-figure title
            ax.annotate(f"{r.get_height():.1f}", (r.get_x()+r.get_width()/2, r.get_height()),
                        xytext=(0, 2.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{labels[g]}\n($n$ = {n})" for g, n in zip(gids, ngrp)])
    ax.set_xlabel("Curing age, days"); ax.set_ylabel("Per-group coverage, %")
    lo_y = min(min(res.values(), key=min)) - 5
    ax.set_ylim(lo_y, 106)                                 # headroom: a 100% bar must not touch the frame
    ax.legend(loc="upper right", fontsize=9, framealpha=.95)
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figE_mondrian_conditional.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"\nFigures saved to: {OUT}/")


# ============================================================================
# 7. Deployment inference (single-point demo)
# ============================================================================
def ias_demo(loader, name, alpha=0.1):
    X, y, mono, bf, feats, gf = loader()
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=0)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0)
    mdl = PC3(mono, bf, "cqr", True, True, True, alpha=alpha, mondrian=True, group_fn=gf, robust=True)
    mdl.fit(Xtr, ytr).calibrate(Xcal, ycal)
    x0 = Xte[:1]; p, lo, hi, L, U = mdl.predict(x0)
    raw = mdl.q_med.predict(x0)                                   # unprojected point, to report clipping
    print(f"\n{'-'*62}\nDeployment inference (single point, {name}):")
    print("  Input:", {f: round(float(v), 2) for f, v in zip(feats, x0[0])})
    print(f"  Prediction: {p[0]:.2f}   {int((1-alpha)*100)}% interval: [{lo[0]:.2f}, {hi[0]:.2f}]")
    print(f"  Physical corridor: [{L[0]:.2f}, {U[0]:.2f}]   point clipped: {bool(abs(raw[0]-p[0])>1e-9)}")
    print(f"  Calibration: n={mdl.n_cal}, eta_hat={100*mdl.eta_hat:.1f}%, target attainable inside corridor: {mdl.feasible}")
    try:
        import shap
        bg = shap.utils.sample(Xtr, 80, random_state=0)
        sv = np.array(shap.Explainer(mdl.q_med.predict, bg)(x0).values)[0]
        top = np.argsort(-np.abs(sv))[:3]
        print("  Top drivers (SHAP):", [(feats[j], round(float(sv[j]), 2)) for j in top])
    except Exception:
        pass


# ============================================================================
# 8. ABLATION - contribution of each PC3 component in turn
# ============================================================================
ABLATION = {
    "CQR (base)":             dict(mono=False, proj=False, pa=False),
    "+ monotone":             dict(mono=True,  proj=False, pa=False),
    "+ projection":           dict(mono=True,  proj=True,  pa=False),
    "+ physics-aware (=PC3)": dict(mono=True,  proj=True,  pa=True),
}

def run_ablation(loader, name, alpha=0.1, seeds=SEEDS):
    print(f"\n{'='*92}\nABLATION: {name}  (target={int((1-alpha)*100)}%, {seeds} seeds)\n{'='*92}")
    X, y, mono, bf, feats, gf = loader()
    keys = ["R2", "Coverage", "Width", "Violations", "WorstGrp", "Gap"]
    agg = {v: {k: [] for k in keys} for v in ABLATION}
    for s in range(seeds):
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=s)
        Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=s)
        gte = gf(Xte)
        for v, fl in ABLATION.items():
            mdl = PC3(mono, bf, "cqr", fl["mono"], fl["proj"], fl["pa"], alpha=alpha).fit(Xtr, ytr).calibrate(Xcal, ycal)
            res = evaluate(*mdl.predict(Xte), yte, groups=gte, target=1 - alpha)
            for k in keys: agg[v][k].append(res[k])
    hdr = f'{"Variant":<26}{"R2":>7}{"Cover":>8}{"Width":>9}{"Viol%":>8}{"WrstGrp":>9}{"Gap%":>7}'
    print(hdr); print("-" * len(hdr))
    for v in ABLATION:
        a = {k: np.mean(val) for k, val in agg[v].items()}
        print(f'{v:<26}{a["R2"]:>7.3f}{a["Coverage"]*100:>7.1f}%{a["Width"]:>9.2f}'
              f'{a["Violations"]*100:>7.1f}%{a["WorstGrp"]*100:>8.1f}%{a["Gap"]*100:>6.1f}%')
    return dict(agg=agg)

def make_ablation_figure(synth_abl, alpha=0.1):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    variants = list(ABLATION.keys()); short = ["CQR", "+mono", "+proj", "+phys(PC³)"]
    cov = [np.mean(synth_abl["agg"][v]["Coverage"]) * 100 for v in variants]
    wid = [np.mean(synth_abl["agg"][v]["Width"]) for v in variants]
    vio = [np.mean(synth_abl["agg"][v]["Violations"]) * 100 for v in variants]
    cols = ["#cccccc", "#9ecae1", "#6baed6", "#2E5496"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].bar(short, cov, color=cols); ax[0].axhline((1-alpha)*100, ls="--", c="r", label="target")
    ax[0].set_title("Coverage, %"); ax[0].set_ylim(min(cov)-3, 100); ax[0].legend()
    ax[1].bar(short, wid, color=cols); ax[1].set_title("Width")
    ax[2].bar(short, vio, color=cols); ax[2].set_title("Physics violations, %")
    for a in ax: a.tick_params(axis="x", rotation=15)
    plt.suptitle("PC³ ablation (synthetic): projection zeros physics violations; physics-aware refines conditional coverage",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/figF_ablation.png", dpi=600); plt.close()
    print(f"\nAblation figure: {OUT}/figF_ablation.png")


if __name__ == "__main__":
    synth = run_experiment(make_synthetic_composite, "Synthetic composite (Voigt–Reuss)")
    concrete = run_experiment(load_concrete, "UCI Concrete (real)")
    make_figures(synth, concrete)
    s_abl = run_ablation(make_synthetic_composite, "Synthetic composite (Voigt–Reuss)")
    c_abl = run_ablation(load_concrete, "UCI Concrete (real)")
    make_ablation_figure(s_abl)
    ias_demo(load_concrete, "UCI Concrete")
