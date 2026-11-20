"""Descriptive paired comparison for fixed published TIDY versus fixed Retrain-S1."""
import csv
import json
from pathlib import Path
from statistics import mean

OUT=Path(__file__).resolve().parent
MAIN=OUT.parents[1]/"protocol_v2/per_image.csv"
def load(path,method):
    with path.open(newline="") as f:
        return {(r["sample_id"],r["corruption"]):r for r in csv.DictReader(f)
                if r["method"]==method and r["corruption"] in ("draw0","draw1")}

left=load(MAIN,"s1_retrained")
right=load(OUT/"nuaa_per_image.csv","TIDY_published")
assert set(left)==set(right) and len(left)==426
rows=[]
for cond in ("draw0","draw1"):
    pairs=[(v,right[k]) for k,v in left.items() if k[1]==cond]
    assert len(pairs)==213
    assert all(a["input_pixels_sha256"]==b["input_sha256"] for a,b in pairs)
    for col_limit,rmse_limit in ((0.5,None),(0.5,1.),(1.,2.)):
        keep=[(a,b) for a,b in pairs if
              abs(float(a["weighted_background_column_residual_dn"])-float(b["weighted_background_column_residual_dn"]))<=col_limit
              and (rmse_limit is None or abs(float(a["plan_background_rmse_dn"])-float(b["plan_background_rmse_dn"]))<=rmse_limit)]
        metric="native_median_ring_contrast_abs_error_dn"
        rows.append(dict(draw=cond,n_originals=213,column_caliper_dn=col_limit,rmse_caliper_dn=rmse_limit,
                         n_matched=len(keep),matched_ids=[a["sample_id"] for a,b in keep],
                         s1_mean_background_rmse_dn=mean(float(a["plan_background_rmse_dn"]) for a,b in keep) if keep else None,
                         tidy_mean_background_rmse_dn=mean(float(b["plan_background_rmse_dn"]) for a,b in keep) if keep else None,
                         s1_mean_contrast_error_dn=mean(float(a[metric]) for a,b in keep) if keep else None,
                         tidy_mean_contrast_error_dn=mean(float(b[metric]) for a,b in keep) if keep else None))
(OUT/"background_overlap.json").write_text(json.dumps(rows,indent=2)+"\n")
