"""Render the registered update trace and the two independently archived measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
d=json.loads((OUT/"summary.json").read_text())
colors={"T-ALL":"#163f65","T-CE":"#b25817","T-IMG":"#088463"}
fig,axes=plt.subplots(2,2,figsize=(10.4,6.4),gridspec_kw={"width_ratios":[2.3,1]},constrained_layout=True)
for row,(k,ylabel) in enumerate([("whole_mae_dn","Whole-image MAE (DN)"),("plan_background_mean_error_dn","Signed background error (DN)")]):
    a,b=axes[row]
    for arm,color in colors.items():
        states=d["panel_trajectory"][arm]
        x=list(map(int,states))
        a.plot(x,[states[str(s)]["noisy"][k] for s in x],marker="o",ms=3,color=color,lw=1.8,label=arm)
    a.set_xlim(-.3,16.3)
    a.set_xticks([0,1,4,8,12,16])
    for condition,color,marker in [("noisy","#264d76","o"),("clean","#8d4944","s")]:
        vals=[d["panel_trajectory"]["T-ALL"]["0"][condition][k]]+[d["archived_panel"][str(s)][condition][k] for s in (50,54)]
        b.scatter([0,1,2],vals,marker=marker,color=color,label=condition,s=34)
    b.set_xticks([0,1,2],["S1 / 0","S2 / 50","S2 / 54"])
    b.set_xlim(-.38,2.38)
    for ax in (a,b):
        ax.axhline(0,color="#bbbbbb",lw=.6)
        ax.grid(axis="y",alpha=.22)
        ax.spines[["right","top"]].set_visible(False)
        ax.set_ylabel(ylabel)
axes[0,0].set_title("Replayed updates · noisy input",loc="left")
axes[0,1].set_title("Archived weights",loc="left")
axes[0,0].legend(frameon=False,ncol=3,fontsize=8)
axes[0,1].legend(frameon=False,fontsize=8)
axes[1,0].set_xlabel("Accepted AdamW update")
fig.suptitle("NUAA-SIRST training panel · 8 fixed images",fontsize=14)
fig.savefig(OUT/"trajectory.png",dpi=160)
