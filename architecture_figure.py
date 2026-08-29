#!/usr/bin/env python3
"""Figure 1: architecture of the PC3 decision-support system (paper Section 3.4).

Pure-matplotlib schematic; no data or model fitting required.
Writes out/fig1_architecture.png at 600 dpi.
"""
import os
os.makedirs("out", exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig,ax=plt.subplots(figsize=(9.0,5.9)); ax.axis('off')
ax.set_xlim(-1.2,101.2); ax.set_ylim(-2.6,66.5)
C={'data':'#E8EEF7','model':'#DCE7F5','cal':'#CBD9EE','phys':'#EAF2E4','out':'#F6EFE0'}
E={'data':'#4B6C9E','model':'#2E5496','cal':'#1F3B6E','phys':'#5C7A46','out':'#9A7B3F'}
LC='#5A6A82'
def box(x,y,w,h,txt,kind,fs=10.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.55,rounding_size=1.4",
        fc=C[kind],ec=E[kind],lw=1.5,zorder=3))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color='#16233A',linespacing=1.45,zorder=4)
def arrow(x1,y1,x2,y2,ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,
        lw=1.5,color=LC,linestyle=ls,shrinkA=0,shrinkB=0,zorder=2))
# -- row 1: TRAINING --
ax.text(2,61.8,"Training",fontsize=11.5,fontweight='bold',color=E['model'])
box(2,49,27,11,"Training split $(X_i,Y_i)$\ncomposition, age, processing",'data')
box(35.5,49,29,11,"Monotone quantile models\n$\\hat q_{lo},\\hat q_{med},\\hat q_{hi}$",'model')
box(71,49,27,11,"Physical bound maps\n$L(X),\\ U(X)$",'phys')
arrow(29.6,54.5,34.9,54.5); arrow(65.1,54.5,70.4,54.5)
# -- row 2: CALIBRATION --
ax.text(2,45.8,"Calibration",fontsize=11.5,fontweight='bold',color=E['cal'])
box(2,33,27,11,"Calibration split\nheld out, exchangeable",'data')
box(35.5,33,29,11,"Projection-aware scores\n$E_i=+\\infty$ if $Y_i\\notin[L,U]$",'cal')
box(71,33,27,11,"Conformal quantile $Q$\n$\\lceil (n{+}1)(1{-}\\alpha)\\rceil$-th score",'cal')
arrow(29.6,38.5,34.9,38.5); arrow(65.1,38.5,70.4,38.5)
arrow(50,48.4,50,44.6,ls='--'); arrow(84.5,48.4,84.5,44.6,ls='--')
# -- row 3: INFERENCE --
ax.text(2,29.8,"Inference",fontsize=11.5,fontweight='bold',color=E['data'])
box(2,17,27,11,"Query $x$\nnew mixture or laminate",'data')
box(35.5,17,29,11,"Interval construction\nprojected onto $[L(x),U(x)]$",'cal')
arrow(29.6,22.5,34.9,22.5)
ax.plot([84.5,84.5],[32.4,29.0],c=LC,lw=1.5,zorder=2)
ax.plot([84.5,71.5],[29.0,29.0],c=LC,lw=1.5,zorder=2)
arrow(71.5,29.0,65.4,24.6)
# -- row 4: OUTPUTS (single bus with visible stems) --
ax.text(2,14.0,"Decision-support output",fontsize=11.5,fontweight='bold',color=E['out'])
centres=[2+i*24.5+11 for i in range(4)]
ax.plot([50,50],[16.4,11.8],c=LC,lw=1.5,zorder=2)                 # drop
ax.plot([centres[0],centres[-1]],[11.8,11.8],c=LC,lw=1.5,zorder=2) # bus
for i,txt in enumerate(["Point prediction $\\hat y$","Admissible interval\n$C(x)\\subseteq[L(x),U(x)]$",
                        "Diagnostics: $\\hat\\eta$, $Q<\\infty$,\nclipping indicators","SHAP explanation"]):
    box(2+i*24.5,-0.4,22,8.5,txt,'out',fs=9.8)
    arrow(centres[i],11.8,centres[i],8.9)                          # stem length 2.9 -> visible
plt.tight_layout()
plt.savefig('out/fig1_architecture.png',dpi=600,bbox_inches='tight',pad_inches=0.06,facecolor='white')
from PIL import Image; import numpy as np
im=Image.open('out/fig1_architecture.png')
a=np.array(im.convert('L')); rows=np.where((a<200).any(axis=1))[0]
print('size',im.size,'aspect %.2f'%(im.size[0]/im.size[1]),'| margins: top',rows.min(),'bottom',a.shape[0]-1-rows.max())
