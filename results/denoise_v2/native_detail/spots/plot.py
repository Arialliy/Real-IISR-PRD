"""Compare spot contrast transfer and width coverage on clean and fresh noisy inputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
summary = json.loads((OUT/"summary.json").read_text())
methods = ("input", "column", "s1", "full_blend_half", "detail_half", "detail_full")
labels = ("Unprocessed input", "Column correction", "S1", "50% input blend",
          "Vertical detail, 50%", "Vertical detail, 100%")
ys = np.arange(len(methods))

fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.6), layout="constrained", sharey=True)
for condition, label, offset, color, marker in (
        ("clean","No added noise",-.13,"#0b7285","o"),
        ("fresh","Fresh added noise",.13,"#a94f22","D")):
    chosen = [summary[condition][m] for m in methods]
    med = np.asarray([r["retention_ratio"]["median"]*100 for r in chosen])
    quartiles = np.array([[r["retention_ratio"][q]*100 for r in chosen] for q in ("q25","q75")])
    axes[0].errorbar(med,ys+offset,xerr=[med-quartiles[0],quartiles[1]-med],fmt=marker,
                     color=color,mec="white",mew=.5,ms=6.4,capsize=2.2,elinewidth=1.2,label=label)
    n = [r["width_px"]["n"] for r in chosen]
    axes[1].scatter(n,ys+offset,color=color,s=30,marker=marker,edgecolor="white",linewidth=.4)
for ax in axes:
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x",alpha=.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y",length=0)
axes[0].invert_yaxis()
axes[0].set(yticks=ys,yticklabels=labels,xlabel="Median contrast retained (%)")
axes[1].set(xlabel="Spots with measurable width (of 213)",xticks=[0,50,100,150,213],xlim=(-8,220))
fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncols=2,frameon=False)
fig.savefig(OUT/"spot_results.png",dpi=180)
