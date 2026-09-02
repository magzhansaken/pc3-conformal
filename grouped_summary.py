"""Prints the numbers quoted in Section 4.4 (composition-grouped protocol) from out/grouped_results.csv.
Run after `python grouped_families.py` (all five families) and paste the values into the sentences listed."""
import pandas as pd, numpy as np
d=pd.read_csv("out/grouped_results.csv")
g=d.groupby(["family","tag","method"]).agg(eta=("eta","mean"),cov_m=("cov","mean"),cov_s=("cov","std"),w=("width","mean")).reset_index()
def row(fam,tag):
    sub=g[(g.family==fam)&(g.tag==tag)]; r={m:sub[sub.method==m].iloc[0] for m in ["naive","robust","cqr"]}
    eta=100*r["naive"].eta; return eta, r
print("=== Table 10 rows (eta | naive | robust | frontier | CQR) ===")
for fam,tags in [("concrete",["p97.5","p95","p92.5","p90","p85","p80"]),("uhpc",["p97.5","p90","p85","p80"]),("mk",["p95","p90","p85","p80"]),("aac",["p95","p92.5","p85","p80"]),("scc",["EC2"])]:
    for t in tags:
        try: eta,r=row(fam,t)
        except Exception: print(f"{fam} {t}: нет данных"); continue
        print(f"{fam:<9}{t:<6} {eta:5.1f} & {100*r['naive'].cov_m:.1f} $\\pm$ {100*r['naive'].cov_s:.1f} & {100*r['robust'].cov_m:.1f} $\\pm$ {100*r['robust'].cov_s:.1f} & {100-max(10,eta):.1f} & {100*r['cqr'].cov_m:.1f}")
lev=d.groupby(["family","tag"]).eta.mean()
p=d.pivot_table(index=["family","seed","tag"],columns="method",values="cov").reset_index()
p["lev_eta"]=p.apply(lambda r: lev[(r.family,r.tag)],axis=1); pp=p[p.lev_eta>0]
inf=pp[pp.lev_eta>0.1]; fr=100-100*inf.lev_eta
print("\n=== Section 4.4 summary sentences (levels with mean eta>0; pairs = levels x seeds) ===")
print(f"levels with eta>0: {pp.groupby(['family','tag']).ngroups}; seed-level pairs: {len(pp)}")
print(f"robust >= naive in {(pp.robust>=pp.naive-1e-12).mean()*100:.0f}% of pairs; mean gain {100*(pp.robust-pp.naive).mean():.1f} pp")
print(f"eta>alpha: {inf.groupby(['family','tag']).ngroups} levels, {len(inf)} pairs; robust {(fr-100*inf.robust).mean():.1f} pp below the frontier, naive {(fr-100*inf.naive).mean():.1f} pp")
for fam,t in [("mk","p90"),("aac","p92.5"),("aac","p85"),("aac","p80"),("mk","p85"),("mk","p80")]:
    try: eta,r=row(fam,t); print(f"{fam} {t}: eta {eta:.1f}%  robust {100*r['robust'].cov_m:.1f} ± {100*r['robust'].cov_s:.1f}  frontier {100-max(10,eta):.1f}  (gap {100-max(10,eta)-100*r['robust'].cov_m:.1f})")
    except Exception: pass
