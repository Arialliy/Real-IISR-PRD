"""Replot local contrast retention from the paired, clean spot experiment."""
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
with (OUT/"per_pair_stage.csv").open(newline="") as f:
    rows=list(csv.DictReader(f))
routes=("reference_pair","q_base_e_only","q_global_cac_fixed_h_minus","s1_deployment")
labels=("Input","Reference codes, base decoder","Reference codes, global CAC","S1 predicted codes")
colors=("#475569","#06756d","#de8b35","#993061")
fig,(left,right)=plt.subplots(1,2,figsize=(10.4,4.1),layout="constrained",
                              gridspec_kw={"width_ratios":[1.38,1]})
amplitudes=[1,2,4,6,8,12]
for i,(route,label,color) in enumerate(zip(routes,labels,colors)):
    selected=[r for r in rows if r["route"]==route]
    med=[median(float(r["contrast_retention_ratio"])*100 for r in selected
                if float(r["nominal_peak_dn"])==a) for a in amplitudes]
    left.plot(amplitudes,med,label=label,color=color,marker=("o","o","s","D")[i],
              ls=("-","-","--","-")[i],lw=1.8,ms=4.5)
    n=sum(r["width_reason"]=="measured" for r in selected)
    right.barh(i,n,color=color,height=.52)
    right.text(n+4,i,f"{n}/213",va="center",fontsize=9)
left.axhline(0,lw=.6,color="#777")
left.set(xticks=amplitudes,xlabel="Injected peak (8-bit gray levels)",
         ylabel="Median paired contrast retained (%)",ylim=(-8,105))
left.legend(loc="center right",fontsize=8,frameon=False)
left.grid(axis="y",alpha=.15)
right.set(yticks=range(4),yticklabels=("Input","Base decoder","Global CAC","S1 output"),xlim=(0,260),
          xlabel="Spots with measurable half-height width",xticks=[0,50,100,150,200])
right.invert_yaxis()
for panel in (left,right):
    panel.spines[["right","top"]].set_visible(False)
fig.savefig(OUT/"spot_transfer.png",dpi=180)
