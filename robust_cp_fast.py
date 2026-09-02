"""Equivalent of robust_cp.py __main__ (Tables 6 and 8, FRP block) with the base fits shared across rho
(fit() does not use the bounds, so results are identical to the script's; 30 fits instead of 150)."""
import sys, numpy as np, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,".")
import pc3; from frp_experiment import load_frp; from robust_cp import make_bounds_mis, RobustPC3
from sklearn.model_selection import train_test_split
ALPHA=0.1; SEEDS=5; X,y,mono,bf_true,feats,gf=load_frp()
rhos=[1.0,0.99,0.98,0.975,0.97,0.96,0.95,0.94,0.92,0.90]
fits=[]
for s in range(SEEDS):
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.5,random_state=s); Xcal,Xte,ycal,yte=train_test_split(Xtmp,ytmp,test_size=0.5,random_state=s)
    mm=pc3.PC3(mono,bf_true,"cqr",True,True,True,alpha=ALPHA).fit(Xtr,ytr); mp=pc3.PC3(mono,bf_true,"cqr",False,False,False,alpha=ALPHA).fit(Xtr,ytr)
    fits.append(dict(Xcal=Xcal,ycal=ycal,Xte=Xte,yte=yte,mm=mm,mp=mp)); print("fit seed",s,flush=True)
def clone(proto, bfm, robust):
    m=(RobustPC3 if robust else pc3.PC3)(mono,bfm,"cqr",proto.use_monotone,proto.project,proto.physics_aware,alpha=ALPHA)
    m.q_lo,m.q_hi,m.q_med=proto.q_lo,proto.q_hi,proto.q_med; return m
rows=[]
for rho in rhos:
    bfm=make_bounds_mis(bf_true,rho); eta=[];cn=[];cr=[];cq=[];vr=[];wn=[];wr=[];wq=[];wc=[]
    for f in fits:
        Xcal,ycal,Xte,yte=f["Xcal"],f["ycal"],f["Xte"],f["yte"]; Lt,Ut=bfm(Xte); eta.append(np.mean((yte<Lt)|(yte>Ut)))
        mn=clone(f["mm"],bfm,False).calibrate(Xcal,ycal); _,lo,hi,L,U=mn.predict(Xte); cn.append(np.mean((yte>=lo)&(yte<=hi))); wn.append(np.mean(hi-lo)); wc.append(np.mean(U-L))
        mr=clone(f["mm"],bfm,True).calibrate(Xcal,ycal); _,lo2,hi2,L2,U2=mr.predict(Xte); cr.append(np.mean((yte>=lo2)&(yte<=hi2))); vr.append(np.mean((lo2<L2-1e-9)|(hi2>U2+1e-9))); wr.append(np.mean(hi2-lo2))
        mc=clone(f["mp"],bfm,False).calibrate(Xcal,ycal); _,lo3,hi3,_,_=mc.predict(Xte); cq.append(np.mean((yte>=lo3)&(yte<=hi3))); wq.append(np.mean(hi3-lo3))
    rows.append(dict(rho=rho,eta=100*np.mean(eta),naive=100*np.mean(cn),robust=100*np.mean(cr),cqr=100*np.mean(cq),viol=100*np.mean(vr),w_naive=float(np.mean(wn)),w_rob=float(np.mean(wr)),w_cqr=float(np.mean(wq)),w_cor=float(np.mean(wc))))
json.dump(rows,open("out/robust_cp_rows.json","w"),indent=1)
print(f"{'rho':>6}{'eta%':>7}{'naive':>7}{'robust':>8}{'CQR':>6}{'viol':>6}{'w_n':>7}{'w_r':>7}{'w_cor':>7}")
for r in rows: print(f"{r['rho']:>6.3f}{r['eta']:>7.1f}{r['naive']:>7.1f}{r['robust']:>8.1f}{r['cqr']:>6.1f}{r['viol']:>6.1f}{r['w_naive']:>7.1f}{r['w_rob']:>7.1f}{r['w_cor']:>7.1f}")
