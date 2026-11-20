"""Analytic checks for the half-maximum, local incremental-response diameter."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from diagnose_spot_transfer import patch_metrics
from evaluate_warmup import save_report

y, x = np.mgrid[-10:11, -10:11]
r2 = x*x+y*y
core, ring = r2<=25, (r2>25) & (r2<=64)
base = np.full(r2.shape,100/255, dtype=np.float64)
def compute(change_dn):
    return patch_metrics(base,base+change_dn/255,core,ring)
def gaussian(sigma):
    return 6*np.exp(-r2/(2*sigma*sigma))*core
narrow,wide = (compute(gaussian(s)) for s in (1.5,2.5))
flat=compute(np.zeros_like(r2))
reversed_spot=compute(-gaussian(1.5))
point=compute(np.where(r2==0,6.,0.))
off_core=gaussian(1.5);off_core[10,16]=10
displaced=compute(off_core)
assert narrow["half_max_pixels"] == 9 and wide["half_max_pixels"] == 25
assert narrow["width_px"] < wide["width_px"]
assert np.isclose(narrow["width_px"],2*np.sqrt(9/np.pi))
assert np.isclose(wide["width_px"],2*np.sqrt(25/np.pi))
for record, reason in ((flat,"identical_local_outputs"),(reversed_spot,"nonpositive_contrast_increment"),
                       (point,"single_pixel_unresolved"),(displaced,"local_peak_outside_injection_core")):
    assert record["width_px"] is None and record["width_reason"] == reason
save_report(HERE/"width_checks.json",dict(narrow_gaussian_sigma_px=1.5,wide_gaussian_sigma_px=2.5,
            narrow=narrow,wide=wide,flat=flat,reversed=reversed_spot,
            single_pixel=point,displaced=displaced))
print("Six analytic width and missing-response checks passed.")
