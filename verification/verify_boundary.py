"""Задача 3: граница eta = alpha, скорости и пределы (точные биномиальные расчёты + Монте-Карло)."""
import os
import numpy as np
from scipy.stats import binom, norm
alpha=0.1
def m_of(n): return int(np.ceil((n+1)*(1-alpha)))
def pi_inf(n,eta): return binom.sf(n+1-m_of(n)-1, n, eta)            # P(K >= l), l=n+1-m
def exact_marg(n,eta):
    m=m_of(n); ks=np.arange(0,n-m+1); b=binom.pmf(ks,n,eta)
    return (1-eta)*((b*m/(n-ks+1)).sum()+pi_inf(n,eta))
print("=== (1) P(Q=inf) как функция n по обе стороны границы и на ней (точный бином) ===")
ns=[25,50,100,200,500,1000,2000,5000,20000]
print(f"{'eta':>7}"+"".join(f"{n:>9}" for n in ns))
for eta in [0.07,0.088,0.097,0.10,0.103,0.112,0.13]:
    print(f"{eta:>7.3f}"+"".join(f"{pi_inf(n,eta):>9.3f}" for n in ns))
print("\n=== (2) Точное маргинальное покрытие -> 1-max(alpha,eta)? ===")
print(f"{'eta':>7}"+"".join(f"{n:>9}" for n in ns)+"   предел 1-max(a,eta)")
for eta in [0.07,0.088,0.10,0.112,0.13]:
    print(f"{eta:>7.3f}"+"".join(f"{exact_marg(n,eta):>9.4f}" for n in ns)+f"   {1-max(alpha,eta):.4f}")
print("\n=== (3) Хёффдинг: eta<alpha, n>=2(1-a)/(a-eta): pi_inf <= exp(-n(a-eta)^2/2);  eta>alpha: 1-pi_inf <= exp(-2n(eta-a)^2) ===")
viol=0; checked=0
for eta in np.arange(0.01,0.30,0.005):
    for n in range(1,3001):
        d=alpha-eta
        if d>0 and n>=2*(1-alpha)/d:
            checked+=1; viol+= pi_inf(n,eta) > np.exp(-n*d*d/2)+1e-12
        elif d<0:
            checked+=1; viol+= (1-pi_inf(n,eta)) > np.exp(-2*n*d*d)+1e-12
print(f"   проверено {checked} пар (n,eta), нарушений границ: {viol}")
print("\n=== (4) На границе eta=alpha: pi_inf -> 1/2; нормальное приближение Phi(-z), z=sqrt(n)(a-eta)/sqrt(eta(1-eta)) ===")
for n in [50,200,1000,5000,20000,100000]:
    print(f"   n={n:>6}: pi_inf(eta=alpha)={pi_inf(n,alpha):.4f}")
print("   Phi(-z) как практическое правило (eta=0.088, a=0.1):")
for n in [25,50,100,200,500,1000,2000]:
    z=np.sqrt(n)*(alpha-0.088)/np.sqrt(0.088*0.912)
    print(f"   n={n:>5}: точно {pi_inf(n,0.088):.3f}   Phi(-z)={norm.cdf(-z):.3f}   z={z:.2f}")
print("\n=== (5) Монте-Карло на границе eta=alpha, n=200 и n=1000 (10000 калибровочных выборок) ===")
rng=np.random.default_rng(1)
for n in [200,1000]:
    m=m_of(n); l=n+1-m; c=norm.ppf(1-alpha/2)
    K=(np.abs(rng.standard_normal((10000,n)))>c).sum(1)
    print(f"   n={n}: доля Q=inf эмп {np.mean(K>=l):.4f} / бином {pi_inf(n,alpha):.4f}   (старая формулировка обещала 1.000)")
