import numpy as np, sys, time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pc3 import PC3, make_synthetic_composite, conformal_quantile
from scipy.stats import binom
alpha=0.1; n=50; m=int(np.ceil((n+1)*(1-alpha)))
rs=np.random.RandomState(7)
# истинный коридор [Reuss,Voigt] -> НЕВЕРНЫЙ, сжатый: [R+d(V-R), V-d(V-R)]
tr = make_synthetic_composite(n=800, seed=1)
Xtr, ytr, monotone, true_bf = tr[0], tr[1], tr[2], tr[3]
# неверная ВЕРХНЯЯ граница: U' = L + c(U-L), c подбираем так, чтобы eta ~ 7% (как на concrete)
_p = make_synthetic_composite(n=400_000, seed=5); _L,_U = true_bf(_p[0]); _u=(_p[1]-_L)/(_U-_L)
c = float(np.quantile(_u, 0.93))
def bad_bounds(Xq):
    L,U = true_bf(Xq); return L, L + c*(U-L)
mdl = PC3(monotone, bad_bounds, robust=True, alpha=alpha).fit(Xtr, ytr)
# большой пул: оценка eta + тест
pool = make_synthetic_composite(n=2_000_000, seed=2); Xp, yp = pool[0], pool[1]
Lp,Up = bad_bounds(Xp); Vp = (yp < Lp-1e-12)|(yp > Up+1e-12); eta = Vp.mean()
N=200_000; Xt, yt, Lt, Ut, Vt = Xp[:N], yp[:N], Lp[:N], Up[:N], Vp[:N]
qlo_t, qhi_t = mdl.q_lo.predict(Xt), mdl.q_hi.predict(Xt); band_t = np.maximum(Ut-Lt,1e-9)
Xc_pool, yc_pool = Xp[N:N+400_000], yp[N:N+400_000]
def cov_given(Q, wbar):
    if not np.isfinite(Q): return (~Vt).mean()
    w = band_t/wbar; lo=np.maximum(qlo_t-Q*w, Lt); hi=np.minimum(qhi_t+Q*w, Ut)
    return np.mean((yt>=lo)&(yt<=hi))
R=2000; rng=np.random.default_rng(3)
res={"code":[], "exact_m":[]}; Ks=[]
t0=time.time()
for r in range(R):
    idx = rng.choice(len(yc_pool), n, replace=False)
    mdl.calibrate(Xc_pool[idx], yc_pool[idx])            # реальный путь кода: wbar, скоры, Q
    E = mdl._scores(Xc_pool[idx], yc_pool[idx]); K = int(np.isinf(E).sum()); Ks.append(K)
    res["code"].append(cov_given(mdl.Qg, mdl.wbar))       # индекс кода (m+1)
    Qm = np.partition(E, m-1)[m-1]                         # точный m-й
    res["exact_m"].append(cov_given(Qm, mdl.wbar))
Ks=np.array(Ks)
print(f"eta (по 2 млн точкам) = {eta:.4f};  n={n}, m={m};  время {time.time()-t0:.0f}s")
for tag, mu in [("код (m+1)-й", m+1), ("точный m-й", m)]:
    c=np.array(res["code" if mu==m+1 else "exact_m"])
    pi_inf=binom.sf(n-mu, n, eta); ks=np.arange(0,n-mu+1); b=binom.pmf(ks,n,eta)
    exact=(1-eta)*((b*mu/(n-ks+1)).sum()+pi_inf); lower=mu/(n+1)-eta*pi_inf
    Qinf = Ks > n-mu
    print(f"\n[{tag}] P(Q=inf): эмп {Qinf.mean():.4f} / Bin {pi_inf:.4f} | маргинальное: эмп {c.mean():.4f} / формула {exact:.4f} / нижн.гр {lower:.4f} / потолок {1-eta:.4f}")
    print(f"   усл. на Q<inf: эмп {c[~Qinf].mean():.4f} / формула {(1-eta)*(b*mu/(n-ks+1)).sum()/(1-pi_inf):.4f};  sd между калибровками {c.std():.4f}")
    for k in range(0, min(n-mu,4)+1):
        sel=(Ks==k)
        if sel.sum()>=30: print(f"   K={k} ({sel.sum():4d}): E[cov] эмп {c[sel].mean():.4f} / теория {(1-eta)*mu/(n-k+1):.4f}")
