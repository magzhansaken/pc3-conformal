#!/usr/bin/env python3
"""
finite_sample_v2.py -- finite-sample study of projection-aware calibration (Section 4, Figure N, Table A1).

Tests the exact coverage statement of Theorem 3(ii) on two benchmarks with misspecified bounds:
  * UCI Concrete, empirical strength ceiling U = P93(y_train), eta ~ 7 %
  * FRP micromechanics (synthetic), upper bound L + 0.965 (U-L), eta ~ 8.8 %

Protocol (per outer replicate, i.e. per training split):
  1. fit the monotone quantile models on the training split and fix the corridor;
  2. treat the non-training pool P as the population: eta_P = out-of-corridor rate in P,
     G_P = empirical law of in-corridor scores in P;
  3. draw R calibration samples of size n i.i.d. from P (with replacement -- the sampling
     model of Theorem 3), compute K, the m-th order statistic Q, and the REALIZED coverage
     c_r = P_P(Y in C_hat) evaluated on the whole pool;
  4. compare (a) the empirical fallback frequency with pi_inf = P(Bin(n, eta_P) >= floor((n+1)a)),
     (b) the marginal coverage mean_r c_r with the closed form c_n(eta_P) of Eq. (closed),
     (c) the spread of c_r across calibration samples with the law (1-eta_P) Beta(m, n-K+1-m),
     (d) E[c_r | K=k] with (1-eta_P) m/(n-k+1).
The three threshold rules of the earlier Table A1 are kept for continuity.
Outputs: out/finite_sample_v2.json, out/tableA1_v2.csv, out/figN_finite_sample.png
"""
import sys, os, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pc3
from frp_experiment import load_frp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from scipy.stats import binom, beta as beta_dist

ALPHA = 0.1
R = 2000                    # calibration draws per (replicate, n)
REPS = 4                    # outer replicates (training splits)
NCALS = {"concrete": [25, 50, 100, 200, 400], "frp": [25, 50, 100, 200, 400]}
os.makedirs("out", exist_ok=True)

def fit_q(Xtr, ytr, mono, q):
    return HGB(loss="quantile", quantile=q, max_iter=200, monotonic_cst=mono, random_state=0).fit(Xtr, ytr)

def prepare(name, rep):
    """Return dict with pool scores, violation flags, eta_P and the population law G_P."""
    if name == "concrete":
        X, y, mono, _, feats, _ = pc3.load_concrete()
        Xtr, Xpool, ytr, ypool = train_test_split(X, y, test_size=0.5, random_state=rep)
        qlo, qhi = fit_q(Xtr, ytr, mono, ALPHA/2), fit_q(Xtr, ytr, mono, 1-ALPHA/2)
        L = np.zeros(len(ypool)); U = np.full(len(ypool), float(np.percentile(ytr, 93)))
    else:
        X, y, mono, bf, feats, _ = load_frp()
        rs = np.random.RandomState(1000+rep); idx = rs.permutation(len(y)); tr, po = idx[:700], idx[700:]
        Xtr, ytr, Xpool, ypool = X[tr], y[tr], X[po], y[po]
        qlo, qhi = fit_q(Xtr, ytr, mono, ALPHA/2), fit_q(Xtr, ytr, mono, 1-ALPHA/2)
        Ltrue, Utrue = bf(Xpool); L = Ltrue; U = Ltrue + 0.965*(Utrue - Ltrue)
    w = np.maximum(U - L, 1e-9); w = w / w.mean()
    s_raw = np.maximum(qlo.predict(Xpool) - ypool, ypool - qhi.predict(Xpool))              # unnormalized CQR score
    s = s_raw / w                                                                          # band-normalized
    V = (ypool < L - 1e-12) | (ypool > U + 1e-12)
    return dict(s=s, V=V, eta=float(V.mean()), N=len(ypool), s_in_sorted=np.sort(s[~V]))

def law_quantiles(n, eta, alpha, qs=(0.05, 0.5, 0.95), draws=200_000, seed=0):
    """Quantiles and sd of the theoretical law of realized coverage (mixture over K)."""
    rng = np.random.default_rng(seed); m = int(np.ceil((n+1)*(1-alpha))); l = n+1-m
    K = rng.binomial(n, eta, draws); c = np.full(draws, 1-eta)
    fin = K <= n-m
    if fin.any():
        c[fin] = (1-eta) * rng.beta(m, n-K[fin]+1-m)
    return dict(sd=float(c.std()), q=[float(np.quantile(c, q)) for q in qs], below=float(np.mean(c < 1-alpha)))

def run_dataset(name):
    out = []
    for rep in range(REPS):
        P = prepare(name, rep); eta = P["eta"]; N = P["N"]; Sin = P["s_in_sorted"]; s = P["s"]; V = P["V"]
        G = lambda q: np.searchsorted(Sin, q, side="right") / len(Sin)
        rng = np.random.default_rng(100 + rep)
        for n in NCALS[name]:
            m = int(np.ceil((n+1)*(1-ALPHA))); l = n+1-m
            cov = np.empty(R); Ks = np.empty(R, int); Qinf = np.zeros(R, bool)
            cov_ctc = np.empty(R); cov_ctcn = np.empty(R)                                   # the two other rules of Table A1
            for r in range(R):
                idx = rng.integers(0, N, n)                                                 # i.i.d. from the pool
                sc, vc = s[idx], V[idx]; K = int(vc.sum()); Ks[r] = K
                E = sc.copy(); E[vc] = np.inf
                Q = pc3.conformal_quantile(E, ALPHA)                                        # exact m-th, +inf if < m finite
                if np.isfinite(Q): cov[r] = (1-eta) * G(Q)
                else: Qinf[r] = True; cov[r] = 1-eta
                # calibrate-then-clip: ceil((1-a)n)-th of all unprojected scores, then clip (= (1-eta) G(Qc) here since clipping to corridor)
                k1 = int(np.ceil((1-ALPHA)*n)); Qc = np.sort(sc)[min(k1, n)-1]; cov_ctc[r] = (1-eta) * G(Qc)
                # clip-then-calibrate (n): ceil((1-a)n)-th among in-corridor scores
                sin = np.sort(sc[~vc]); k2 = int(np.ceil((1-ALPHA)*n))
                Qo = sin[k2-1] if k2 <= len(sin) else np.inf
                cov_ctcn[r] = (1-eta) * G(Qo) if np.isfinite(Qo) else 1-eta
            pi_th = float(binom.sf(l-1, n, eta)); c_closed = pc3.exact_marginal_coverage(n, ALPHA, eta)
            lower = 1 - ALPHA - eta*pi_th; law = law_quantiles(n, eta, ALPHA)
            perk = []
            for k in np.argsort(-np.bincount(Ks))[:4]:
                sel = (Ks == k) & (~Qinf)
                if sel.sum() >= 30:
                    perk.append(dict(k=int(k), n_draws=int(sel.sum()), emp=float(cov[sel].mean()), th=float((1-eta)*m/(n-k+1))))
            cond_fin_emp = float(cov[~Qinf].mean()) if (~Qinf).any() else float("nan")
            ks = np.arange(0, n-m+1); b = binom.pmf(ks, n, eta)
            cond_fin_th = float((1-eta)*(b*m/(n-ks+1)).sum()/(1-pi_th)) if pi_th < 1 else float("nan")
            row = dict(dataset=name, rep=rep, n=n, m=m, l=l, eta_pool=eta, N_pool=N,
                       pi_emp=float(Qinf.mean()), pi_th=pi_th,
                       marg_emp=float(cov.mean()), marg_closed=c_closed, marg_lower=lower, ceiling=1-eta,
                       marg_mc_se=float(cov.std()/np.sqrt(R)),
                       sd_emp=float(cov.std()), sd_th=law["sd"],
                       q_emp=[float(np.quantile(cov, q)) for q in (0.05, 0.5, 0.95)], q_th=law["q"],
                       below_emp=float(np.mean(cov < 1-ALPHA)), below_th=law["below"],
                       cond_fin_emp=cond_fin_emp, cond_fin_th=cond_fin_th, per_k=perk,
                       ctc_marg=float(cov_ctc.mean()), ctcn_marg=float(cov_ctcn.mean()))
            out.append(row)
            print(f"{name:>8} rep{rep} n={n:>3} eta_P={100*eta:.1f}% | P(Q=inf) {row['pi_emp']:.3f}/{pi_th:.3f} | "
                  f"marg {100*row['marg_emp']:.2f} vs closed {100*c_closed:.2f} (lower {100*lower:.2f}, ceil {100*(1-eta):.2f}) | "
                  f"sd {100*row['sd_emp']:.2f}/{100*law['sd']:.2f} | q05 {100*row['q_emp'][0]:.1f}/{100*law['q'][0]:.1f} | "
                  f"below90 {row['below_emp']:.2f}/{law['below']:.2f} | ctc {100*row['ctc_marg']:.1f} ctcn {100*row['ctcn_marg']:.1f}")
    return out

def pooled(rows):
    """Average over replicates for each (dataset, n) -- reported in the paper."""
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows: g[(r["dataset"], r["n"])].append(r)
    res = []
    for (d, n), rs in sorted(g.items()):
        f = lambda key: float(np.mean([r[key] for r in rs]))
        res.append(dict(dataset=d, n=n, m=rs[0]["m"], l=rs[0]["l"], eta_pool=f("eta_pool"),
                        pi_emp=f("pi_emp"), pi_th=f("pi_th"), marg_emp=f("marg_emp"), marg_closed=f("marg_closed"),
                        marg_lower=f("marg_lower"), ceiling=f("ceiling"), sd_emp=f("sd_emp"), sd_th=f("sd_th"),
                        q05_emp=float(np.mean([r["q_emp"][0] for r in rs])), q05_th=float(np.mean([r["q_th"][0] for r in rs])),
                        q95_emp=float(np.mean([r["q_emp"][2] for r in rs])), q95_th=float(np.mean([r["q_th"][2] for r in rs])),
                        below_emp=f("below_emp"), below_th=f("below_th"), cond_fin_emp=f("cond_fin_emp"), cond_fin_th=f("cond_fin_th"),
                        ctc_marg=f("ctc_marg"), ctcn_marg=f("ctcn_marg")))
    return res

def make_figure(pool, rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    for i, name in enumerate(["concrete", "frp"]):
        pr = [r for r in pool if r["dataset"] == name]; ns = [r["n"] for r in pr]; eta = pr[0]["eta_pool"]
        nn = np.arange(int(min(ns)*0.8), int(max(ns)*1.25))
        lab = "UCI Concrete" if name == "concrete" else "FRP"
        # (a) marginal coverage
        ax = axes[i, 0]
        ax.plot(nn, [100*pc3.exact_marginal_coverage(int(x), ALPHA, eta) for x in nn], "-", color="#2E5496", lw=0.9, alpha=.6, label="exact law at mean η (sawtooth: m jumps with n)")
        ax.plot(nn, [100*(1-ALPHA-eta*binom.sf(int(np.floor((x+1)*ALPHA))-1, int(x), eta)) for x in nn], ":", color="#2E5496", lw=1.2, alpha=.8, label="lower bound, Thm 3(i)")
        ax.axhline(100*(1-eta), color="#7f7f7f", ls="-.", lw=1.2, label=f"ceiling 1−η (mean) = {100*(1-eta):.1f}%")
        ax.axhline(90, color="#C0392B", ls="--", lw=1.2, label="target 1−α")
        ax.plot(ns, [100*r["marg_closed"] for r in pr], "x", color="#2E5496", ms=10, mew=2, label="exact law at each replicate's η, averaged")
        ax.plot(ns, [100*r["marg_emp"] for r in pr], "o", color="#1a1a1a", ms=6.5, label="empirical marginal (mean over 8000 calibrations)")
        ax.set_xscale("log"); ax.set_xlabel("calibration size n"); ax.set_ylabel("marginal coverage, %")
        ax.set_title(f"({'ab'[i]}) {lab}, η≈{100*eta:.1f}%: marginal coverage"); ax.grid(alpha=.3)
        if i == 0: ax.legend(fontsize=7, loc="lower right")
        # (b) variation across calibration samples
        ax = axes[i, 1]
        ax.fill_between(ns, [100*r["q05_th"] for r in pr], [100*r["q95_th"] for r in pr], color="#9CC3E0", alpha=.55, label="law: 5–95% band, (1−η)·Beta mixed over K")
        ax.plot(ns, [100*r["q05_emp"] for r in pr], "v-", color="#1a1a1a", ms=5, label="empirical 5% and 95% quantiles")
        ax.plot(ns, [100*r["q95_emp"] for r in pr], "^-", color="#1a1a1a", ms=5)
        ax.plot(ns, [100*r["marg_emp"] for r in pr], "o-", color="#2E5496", ms=5, label="empirical mean")
        ax.axhline(90, color="#C0392B", ls="--", lw=1.2); ax.axhline(100*(1-eta), color="#7f7f7f", ls="-.", lw=1.2)
        ax.set_xscale("log"); ax.set_xlabel("calibration size n"); ax.set_ylabel("realized coverage of one calibration, %")
        ax.set_title(f"({'cd'[i]}) variation across calibration samples"); ax.grid(alpha=.3)
        if i == 0: ax.legend(fontsize=7, loc="lower right")
        # (c) fallback frequency
        ax = axes[i, 2]
        ax.plot(nn, [binom.sf(int(np.floor((x+1)*ALPHA))-1, int(x), eta) for x in nn], "-", color="#2E5496", lw=0.9, alpha=.6, label="π∞ = P(Bin(n,η) ≥ ⌊(n+1)α⌋) at mean η")
        ax.plot(ns, [r["pi_th"] for r in pr], "x", color="#2E5496", ms=10, mew=2, label="π∞ at each replicate's η, averaged")
        ax.plot(ns, [r["pi_emp"] for r in pr], "o", color="#1a1a1a", ms=6.5, label="empirical fallback frequency")
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("calibration size n"); ax.set_ylabel("P(Q = +∞): corridor returned")
        ax.set_title(f"({'ef'[i]}) fallback to the corridor"); ax.grid(alpha=.3)
        if i == 0: ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig("out/figN_finite_sample.png", dpi=600); plt.close()
    # ---- scatter test: every (replicate, n) configuration, empirical vs exact law
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    specs = [("marg_emp", "marg_closed", "marginal coverage, %", 100), ("sd_emp", "sd_th", "sd across calibrations, pp", 100),
             ("pi_emp", "pi_th", "fallback frequency P(Q=+∞)", 1), ("below_emp", "below_th", "fraction of calibrations below 1−α", 1)]
    for ax, (e, t, ttl, sc) in zip(axes, specs):
        for name, mk, col in [("concrete", "o", "#2E5496"), ("frp", "s", "#C0392B")]:
            rr = [r for r in rows if r["dataset"] == name]
            ax.plot([sc*r[t] for r in rr], [sc*r[e] for r in rr], mk, color=col, ms=5, alpha=.85, label="UCI Concrete" if name=="concrete" else "FRP")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("exact law (Theorem 3)"); ax.set_ylabel("empirical"); ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("out/figN2_exact_law_scatter.png", dpi=600); plt.close()

if __name__ == "__main__":
    t0 = time.time(); rows = run_dataset("concrete") + run_dataset("frp"); pool = pooled(rows)
    json.dump(dict(rows=rows, pooled=pool, R=R, REPS=REPS, alpha=ALPHA), open("out/finite_sample_v2.json", "w"), indent=1)
    import csv
    with open("out/tableA1_v2.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(list(pool[0].keys()))
        for r in pool: wr.writerow([r[k] for k in pool[0].keys()])
    make_figure(pool, rows)
    print(f"\ndone in {time.time()-t0:.0f}s -> out/finite_sample_v2.json, out/tableA1_v2.csv, out/figN_finite_sample.png")
