"""Задача 6: проверка диагностики feasibility — точно по биному (a,b,d) и Монте-Карло (c)."""
import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import numpy as np
from scipy.stats import binom, norm, beta as beta_dist
from pc3 import clopper_pearson, feasibility_diagnostics, exact_marginal_coverage, fallback_probability
alpha=0.1; delta=0.05
print("=== (a) Доверие границ Клоппера–Пирсона: P(eta<=eta_hi) и P(eta>=eta_lo), точно по биному ===")
worst_hi=1; worst_lo=1
for n in [25,50,100,200,500,1000]:
    for eta in np.arange(0.005,0.4,0.005):
        ks=np.arange(n+1); b=binom.pmf(ks,n,eta)
        cov_hi=sum(b[k] for k in ks if eta<=clopper_pearson(k,n,delta)[1]+1e-12)
        cov_lo=sum(b[k] for k in ks if eta>=clopper_pearson(k,n,delta)[0]-1e-12)
        worst_hi=min(worst_hi,cov_hi); worst_lo=min(worst_lo,cov_lo)
print(f"   минимум по сетке (n,eta): P(eta<=eta_hi)={worst_hi:.4f}  P(eta>=eta_lo)={worst_lo:.4f}   (требуется >= {1-delta})")

print("\n=== (b) Ошибки трёхстадийного правила (точно по биному) ===")
print("   n   | max P(feasible | eta>alpha) | max P(infeasible | eta<=alpha) | P(feasible|eta=0.05) P(indet|eta=0.088) P(infeasible|eta=0.20)")
for n in [25,50,100,200,500,1000]:
    def probs(eta):
        ks=np.arange(n+1); b=binom.pmf(ks,n,eta); st=[feasibility_diagnostics(k,n,alpha,delta)["feasibility"] for k in ks]
        return {s:sum(b[k] for k in ks if st[k]==s) for s in ("feasible","indeterminate","infeasible")}
    e1=max(probs(e)["feasible"] for e in np.arange(0.1005,0.5,0.0025))
    e2=max(probs(e)["infeasible"] for e in np.arange(0.005,0.1001,0.0025))
    print(f"   {n:>4} | {e1:>27.4f} | {e2:>30.4f} | {probs(0.05)['feasible']:.3f}  {probs(0.088)['indeterminate']:.3f}  {probs(0.20)['infeasible']:.3f}")

print("\n=== (d) Порог маргинального покрытия с доверием: P( exact(eta_hi(K)) <= истинное маргинальное ) >= 1-delta ===")
worst=1
for n in [25,50,100,200,500]:
    for eta in np.arange(0.01,0.3,0.01):
        true=exact_marginal_coverage(n,alpha,eta); ks=np.arange(n+1); b=binom.pmf(ks,n,eta)
        ok=sum(b[k] for k in ks if feasibility_diagnostics(k,n,alpha,delta)["coverage_floor_marginal"]<=true+1e-12)
        worst=min(worst,ok)
print(f"   минимум по сетке: {worst:.4f}  (>= {1-delta} требуется)")

print("\n=== (c) Порог реализованного покрытия (этой калибровки), Монте-Карло: P(realized >= floor) >= 1-2delta ===")
rng=np.random.default_rng(0)
for n,eta in [(50,0.05),(200,0.088),(200,0.05),(100,0.15)]:
    m=int(np.ceil((n+1)*(1-alpha))); c=norm.ppf(1-eta/2); qlo,qhi=norm.ppf(alpha/2),norm.ppf(1-alpha/2)
    sc=lambda y: np.maximum(qlo-y,y-qhi)
    yt=rng.standard_normal(400_000); Vt=np.abs(yt)>c; Sin=np.sort(sc(yt[~Vt])); pin=1-Vt.mean()
    hits=0; R=4000; floors=[]
    for r in range(R):
        y=rng.standard_normal(n); V=np.abs(y)>c; K=int(V.sum()); E=sc(y); E[V]=np.inf
        fin=np.sort(E[np.isfinite(E)]); Q=fin[m-1] if m<=len(fin) else np.inf
        real = pin*(np.searchsorted(Sin,Q,side="right")/len(Sin)) if np.isfinite(Q) else pin
        fl=feasibility_diagnostics(K,n,alpha,delta)["coverage_floor_realized"]; floors.append(fl)
        hits += real>=fl
    print(f"   n={n:>4} eta={eta:.3f}: P(realized>=floor)={hits/R:.4f} (>= {1-2*delta});  медиана порога {100*np.median(floors):.1f}%  (истинное маргинальное {100*exact_marginal_coverage(n,alpha,eta):.1f}%)")

print("\n=== (e) Информативность: типичная карточка ===")
for n,K in [(200,10),(200,18),(25,2),(500,20),(206,15)]:
    d=feasibility_diagnostics(K,n,alpha,delta)
    print(f"   n={n:>3} K={K:>2}: eta_hat={100*d['eta_hat']:.1f}% CI90=[{100*d['eta_lo']:.1f},{100*d['eta_hi']:.1f}]  {d['feasibility']:<13} branch={d['branch']:<9} pi_inf(eta_hat)={d['pi_inf_hat']:.2f}  floor_marg={100*d['coverage_floor_marginal']:.1f}%  floor_real={100*d['coverage_floor_realized']:.1f}%")
