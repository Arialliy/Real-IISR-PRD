"""Can the local and original 8x8 order terms distinguish a deleted calibration spot?"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

from evaluate_warmup import file_hash, save_report
from prd_data import DenoisePairDataset
from train_warmup import valid_order_loss

ROOT=Path(__file__).resolve().parent
UPSTREAM=ROOT/"vendor/Real-IISR/models/src/loss.py"


def main():
    spec=importlib.util.spec_from_file_location("release_thermal_order",UPSTREAM)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    published=module.ThermalOrderConsistencyLoss(reduction="mean")
    data=DenoisePairDataset(ROOT,seed=0)
    rows=[]
    for i in range(len(data)):
        sample=data.sample(i,epoch=0,draw_id=0)
        if not sample["pair_valid"]:
            continue
        minus, plus, valid = [torch.from_numpy(sample[k])[None] for k in ("reference_minus","reference_plus","valid_mask")]
        signal=plus-minus
        assert signal.max()>0 and signal.min()>=0
        fields={}
        for name,pred in (("identity",plus),("half_response",minus+0.5*signal),("deleted",minus),
                          ("inversion_control",1.-plus)):
            fields[name]={"local":valid_order_loss(pred,plus,valid).item(),"upstream":published(plus,pred).item()}
        rows.append(dict(sample_id=sample["meta"]["sample_id"],realized_peak_dn=float(signal.max())*255,
                         max_change_8x8_mean_dn=255*float(torch.nn.functional.avg_pool2d(signal,8,8).max()),
                         losses=fields))
    summary={}
    for name in rows[0]["losses"]:
        summary[name]={}
        for implementation in ("local","upstream"):
            arr=np.array([r["losses"][name][implementation] for r in rows])
            summary[name][implementation]=dict(exact_zero=int((arr==0.).sum()),mean=float(arr.mean()),max=float(arr.max()))
    assert summary["identity"]["local"]["exact_zero"]==len(rows)
    assert summary["inversion_control"]["local"]["max"]>0
    save_report(ROOT/"results/denoise_v2/spot_transfer/order_sensitivity.json",
                dict(epoch=0,draw_id=0,train_candidates=len(data),valid_pairs=len(rows),summary=summary,
                     upstream_source_sha256=file_hash(UPSTREAM),working_range="[0,1] display gray on original padded grid",
                     local_order_source_sha256=file_hash(ROOT/"train_warmup.py"),rows=rows))


if __name__=="__main__":
    with torch.inference_mode():
        main()
