"""Задача 2: численная проверка Леммы 2 (случай Q = +inf)."""
import os
import numpy as np
from scipy.stats import norm, binom
alpha=0.1
# (a) тождество n+1-m = floor((n+1)alpha) и формула пи_inf, включая n < 1/alpha - 1
print("(a) l = floor((n+1)a) == n+1-m ?")
ok=all(int(np.floor((n+1)*a))==n+1-int(np.ceil((n+1)*(1-a))) for n in range(1,2001) for a in [0.01,0.05,0.1,0.2,0.25,0.333])
print("    для всех n<=2000 и 6 значений alpha:", ok)

def sim(eta, n, R=8000, N=300_000, seed=0):
    rng=np.random.default_rng(seed); m=int(np.ceil((n+1)*(1-alpha))); l=n+1-m
    c=norm.ppf(1-eta/2); qlo,qhi=norm.ppf(alpha/2),norm.ppf(1-alpha/2)
    sc=lambda y: np.maximum(qlo-y,y-qhi)
    yt=rng.standard_normal(N); Vt=np.abs(yt)>c; St=sc(yt); Sin=np.sort(St[~Vt]); pin=1-Vt.mean()
    G=lambda q: np.searchsorted(Sin,q,side="right")/len(Sin)
    cov_alg=np.empty(R); cov_lvl=np.empty(R); cov_alt=np.empty(R); Qinf=np.zeros(R,bool); adm_viol=np.zeros(R)
    for r in range(R):
        y=rng.standard_normal(n); V=np.abs(y)>c; K=V.sum()
        if K>=l:                       # Q = +inf  (эквивалентно: меньше m конечных скоров)
            Qinf[r]=True
            cov_alg[r]=pin             # Алгоритм 1: коридор
            cov_lvl[r]=1.0             # множество уровня {E<=inf} = R: покрывает всё
            cov_alt[r]=1.0; adm_viol[r]=Vt.mean()   # альтернативный fallback: вернуть R (не admissible)
        else:
            E=sc(y); E[V]=np.inf; Q=np.partition(E,m-1)[m-1]
            g=G(Q); cov_alg[r]=pin*g; cov_lvl[r]=pin*g; cov_alt[r]=pin*g
    pi=binom.sf(l-1,n,eta)
    return dict(n=n,eta=eta,m=m,l=l,pi_emp=Qinf.mean(),pi_th=pi,
        cov_on_fallback=cov_alg[Qinf].mean() if Qinf.any() else np.nan, sd_on_fallback=cov_alg[Qinf].std() if Qinf.sum()>1 else np.nan,
        lvl=cov_lvl.mean(), alg=cov_alg.mean(), gap=cov_lvl.mean()-cov_alg.mean(), gap_th=eta*pi,
        alt=cov_alt.mean(), alt_bound=m/(n+1), alt_viol=adm_viol.mean(), alt_viol_th=eta*pi)
print("\n(b),(c),(d): n, eta | P(Q=inf) эмп/теор | покрытие НА fallback (=1-eta?) sd | P(E<=Q) - P(Y in C_alg) эмп/теор=eta*pi | альт. fallback (R): покрытие >= m/(n+1)? ; доля неадмиссибельных эмп/теор")
for n,eta in [(50,0.05),(50,0.10),(50,0.20),(25,0.088),(200,0.088),(8,0.05)]:
    o=sim(eta,n)
    print(f"  n={o['n']:3d} eta={o['eta']:.3f} m={o['m']:3d} l={o['l']:2d} | {o['pi_emp']:.3f}/{o['pi_th']:.3f} | {o['cov_on_fallback']:.4f} (1-eta={1-eta:.4f}) sd={o['sd_on_fallback']:.4f} | {o['gap']:.4f}/{o['gap_th']:.4f} | {o['alt']:.4f} >= {o['alt_bound']:.4f}: {o['alt']>=o['alt_bound']-0.003} ; {o['alt_viol']:.4f}/{o['alt_viol_th']:.4f}")
