#!/usr/bin/env python3
"""Effect of the +1 finite-sample correction under misspecified bounds (supplementary study).

All three rules clip to the corridor; they differ only in how the threshold is set:
  cal-then-clip      : ceil((1-alpha) n)-th smallest score of ALL (unprojected) scores, then clip
  clip-then-cal (n)  : ceil((1-alpha) n)-th smallest score among IN-corridor points (no +1)
  PC3 (n+1)          : ceil((n+1)(1-alpha))-th smallest projection-aware score (+inf off corridor)
The (n+1) order statistic is what Theorem 3(i) needs for the finite-sample guarantee; the
study quantifies what omitting it costs at small calibration sizes. It is a property of the
quantile index, not a comparison with the OMLT mechanism of Li et al. (2025), whose nested
calibration already carries the (1+|I_cal|) factor; earlier versions of this file labelled
the second rule "OMLT-style", which was inaccurate.
"""
import numpy as np, warnings
warnings.filterwarnings('ignore')
from frp_experiment import load_frp
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

ALPHA=0.1; RHO=0.965; SEEDS=40
NCALS=[25,50,100,200]

def fit_q(Xtr,ytr,mono,q):
    return HGB(loss="quantile",quantile=q,max_iter=200,monotonic_cst=mono,random_state=0).fit(Xtr,ytr)

def run():
    X,y,mono,bf,feats,gf=load_frp()
    rows=[]
    for ncal in NCALS:
        res={k:[] for k in ['cal_then_clip','clip_then_cal_n','PC3','eta']}
        for s in range(SEEDS):
            rs=np.random.RandomState(1000+s); idx=rs.permutation(len(y))
            tr=idx[:700]; ca=idx[700:700+ncal]; te=idx[700+ncal:700+ncal+500]
            qlo=fit_q(X[tr],y[tr],mono,ALPHA/2); qhi=fit_q(X[tr],y[tr],mono,1-ALPHA/2)
            L,U=bf(X); Umis=L+RHO*(U-L)                      # misspecified upper bound
            w=np.maximum(Umis-L,1e-9)
            sc=lambda i: np.maximum(qlo.predict(X[i])-y[i], y[i]-qhi.predict(X[i]))/w[i]
            s_cal=sc(ca); inc=(y[ca]>=L[ca])&(y[ca]<=Umis[ca])
            n=len(ca)
            # cal-then-clip: quantile of all (unprojected) scores
            k1=int(np.ceil((1-ALPHA)*n)); Qc=np.sort(s_cal)[min(k1,n)-1]
            # clip-then-cal (n): ceil((1-a)n)-th among in-corridor scores
            s_in=np.sort(s_cal[inc]); k2=int(np.ceil((1-ALPHA)*n))
            Qo=s_in[k2-1] if k2<=len(s_in) else np.inf
            # PC3: ceil((n+1)(1-a))-th of projection-aware scores
            E=np.where(inc,s_cal,np.inf); k3=int(np.ceil((n+1)*(1-ALPHA)))
            Qp=np.sort(E)[min(k3,n)-1]
            for name,Q in [('cal_then_clip',Qc),('clip_then_cal_n',Qo),('PC3',Qp)]:
                lo=np.maximum(qlo.predict(X[te])-Q*w[te],L[te]); hi=np.minimum(qhi.predict(X[te])+Q*w[te],Umis[te])
                res[name].append(np.mean((y[te]>=lo)&(y[te]<=hi)))
            res['eta'].append(np.mean((y[te]<L[te])|(y[te]>Umis[te])))
        import json
        json.dump({k:[float(x) for x in v] for k,v in res.items()}, open(f'perseed_{ncal}.json','w'))
        rows.append((ncal,100*np.mean(res['eta']),
                     100*np.mean(res['cal_then_clip']),100*np.mean(res['clip_then_cal_n']),100*np.mean(res['PC3']),
                     100*np.mean(np.array(res['clip_then_cal_n'])<0.90),100*np.mean(np.array(res['PC3'])<0.90)))
    print(f"{'n_cal':>6}{'eta%':>7}{'cal>clip%':>11}{'clip>cal(n)%':>14}{'PC3(n+1)%':>11}{'(n)<90%':>9}{'(n+1)<90%':>11}")
    for r in rows: print(f"{r[0]:>6}{r[1]:>7.1f}{r[2]:>11.1f}{r[3]:>14.1f}{r[4]:>11.1f}{r[5]:>9.0f}{r[6]:>11.0f}")
    import csv
    with open('finite_sample.csv','w',newline='') as f:
        wcsv=csv.writer(f); wcsv.writerow(['n_cal','eta_pct','cal_then_clip_cov','clip_then_cal_n_cov','PC3_n_plus_1_cov','clip_then_cal_n_below_nominal_pct','PC3_below_nominal_pct'])
        wcsv.writerows([[f'{x:.2f}' if isinstance(x,float) else x for x in r] for r in rows])
    print('\nwrote finite_sample.csv')

if __name__=='__main__': run()
