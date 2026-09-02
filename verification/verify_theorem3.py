"""
import os
Численная проверка исправленной Теоремы 3 (общий индекс m).
Модель: Y = f(X) + sigma*Z, Z~N(0,1); коридор [f-c, f+c] с c = sigma*z_{1-eta/2}, так что
P(Y вне коридора) = eta ТОЧНО. Скоры — CQR на ИСТИННЫХ квантилях (фиксированная функция),
что допустимо: теорема верна для любой фиксированной функции скора.
"""
import numpy as np, json, sys
from scipy.stats import norm, binom, beta as beta_dist, kstest

def run(eta, n, alpha, R=6000, N=400_000, seed=0, index="m"):
    rng = np.random.default_rng(seed)
    m = int(np.ceil((n+1)*(1-alpha)))
    m_use = m if index=="m" else min(n, m+1)          # "m+1" воспроизводит np.quantile(...,'higher')
    sigma = 1.0; c = sigma*norm.ppf(1-eta/2)           # коридор с точной долей нарушений eta
    qlo, qhi = norm.ppf(alpha/2)*sigma, norm.ppf(1-alpha/2)*sigma   # истинные квантили (f=0 WLOG)
    def scores(y):                                     # CQR-скор, w=1 (коридор постоянной ширины)
        return np.maximum(qlo - y, y - qhi)
    # большой тест-набор
    yt = sigma*rng.standard_normal(N); Vt = (np.abs(yt) > c); St = scores(yt)
    S_in_sorted = np.sort(St[~Vt]); p_in = 1 - Vt.mean()  # эмпирическая (1-eta) на тесте
    G = lambda q: np.searchsorted(S_in_sorted, q, side="right")/len(S_in_sorted)
    covs = np.empty(R); Ks = np.empty(R, int); Qinf = np.zeros(R, bool)
    for r in range(R):
        y = sigma*rng.standard_normal(n); V = np.abs(y) > c; K = V.sum()
        E = scores(y); E[V] = np.inf
        Q = np.partition(E, m_use-1)[m_use-1]           # m_use-й порядковый статистик
        Ks[r] = K
        if np.isfinite(Q): covs[r] = p_in*G(Q)          # (1-eta)*G(Q)
        else: Qinf[r] = True; covs[r] = p_in            # откат: коридор => ровно 1-eta
    # --- теория ---
    pi_inf = binom.sf(n-m_use, n, eta)                  # P(K >= n-m_use+1)
    ks = np.arange(0, n-m_use+1); b = binom.pmf(ks, n, eta)
    exact_marg = (1-eta)*( (b*m_use/(n-ks+1)).sum() + pi_inf )
    lower = m/(n+1) - eta*pi_inf if index=="m" else m_use/(n+1) - eta*pi_inf
    cond_Qfin_theory = (1-eta)*(b*m_use/(n-ks+1)).sum()/(1-pi_inf) if pi_inf<1 else np.nan
    cond_Qfin_bound = 1 - (1-m_use/(n+1))/(1-pi_inf) if pi_inf<1 else np.nan
    out = dict(eta=eta, n=n, alpha=alpha, m=m, m_used=m_use, R=R,
               P_Qinf_emp=Qinf.mean(), P_Qinf_theory=pi_inf,
               marginal_emp=covs.mean(), marginal_exact=exact_marg, marginal_lower=lower, ceiling=1-eta,
               cond_Qfin_emp=covs[~Qinf].mean() if (~Qinf).any() else np.nan,
               cond_Qfin_theory=cond_Qfin_theory, cond_Qfin_bound=cond_Qfin_bound,
               sd_across_cal=covs.std(), sd_across_cal_given_Qfin=covs[~Qinf].std() if (~Qinf).sum()>1 else np.nan)
    # условно на K=k: среднее и закон Beta
    perk = []
    for k in range(0, min(n-m_use, 6)+1):
        sel = (Ks==k) & (~Qinf)
        if sel.sum() < 40: continue
        a_, b_ = m_use, n-k+1-m_use
        theory_mean = (1-eta)*m_use/(n-k+1)
        # KS-тест: cov/(1-eta) ~ Beta(m, n-k+1-m)
        ks_p = kstest(covs[sel]/p_in, beta_dist(a_, b_).cdf).pvalue
        perk.append(dict(k=k, n_draws=int(sel.sum()), emp_mean=covs[sel].mean(), theory_mean=theory_mean,
                         emp_sd=covs[sel].std(), theory_sd=(1-eta)*np.sqrt(a_*b_/((a_+b_)**2*(a_+b_+1))), KS_p=ks_p))
    out["per_k"] = perk
    return out

if __name__ == "__main__":
    alpha=0.1; n=50
    results=[]
    for eta in [0.05, 0.10, 0.20]:
        for index in ["m","m+1"]:
            results.append(run(eta, n, alpha, index=index))
    json.dump(results, open("verify_theorem3.json","w"), indent=1, default=float)
    for o in results:
        tag = "точный m-й" if o["m_used"]==o["m"] else "код: (m+1)-й"
        print(f"\n=== eta={o['eta']:.2f}  alpha={alpha}  n={n}  m={o['m']}  [{tag}]  R={o['R']} ===")
        print(f"  P(Q=inf):      эмп {o['P_Qinf_emp']:.4f}   теория Bin {o['P_Qinf_theory']:.4f}")
        print(f"  маргинальное:  эмп {o['marginal_emp']:.4f}   точная формула {o['marginal_exact']:.4f}   нижняя граница {o['marginal_lower']:.4f}   потолок 1-eta {o['ceiling']:.4f}")
        print(f"  усл. на Q<inf: эмп {o['cond_Qfin_emp']:.4f}   формула {o['cond_Qfin_theory']:.4f}   простая граница {o['cond_Qfin_bound']:.4f}   (1-alpha={1-alpha})")
        print(f"  разброс между калибровочными выборками: sd {o['sd_across_cal']:.4f} (при Q<inf: {o['sd_across_cal_given_Qfin']:.4f})")
        for p in o["per_k"]:
            print(f"    K={p['k']:2d} ({p['n_draws']:4d} выборок): E[cov] эмп {p['emp_mean']:.4f} / теория {p['theory_mean']:.4f};  sd эмп {p['emp_sd']:.4f} / теория {p['theory_sd']:.4f};  KS p={p['KS_p']:.2f}")
