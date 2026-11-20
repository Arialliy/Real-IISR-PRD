"""Audit registered replay and matched loss interventions, retaining per-update diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np


OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
ARMS=("T-ALL","T-CE","T-IMG")
COLS=("whole_mae_dn","plan_background_rmse_dn","plan_background_mean_error_dn",
      "native_median_ring_contrast_abs_error_dn","native_median_ring_contrast_bias_dn")
PROBES=("head.bias","blocks.23.attn.proj.bias","cb_cond.4.bias","vae.quantize.U","vae.quantize.V","vae.quantize.lora_gate")


def load(relative):
    return json.loads((OUT/relative).read_text())


def histogram(values):
    vals,counts=np.unique(values,return_counts=True)
    return {str(k):int(v) for k,v in zip(vals,counts)}


def compact_panel(r):
    result={"relative_parameter_L2":r["relative_L2_vs_S1"]}
    for c in ("noisy","clean"):
        s=r["summary"][c]
        samples=[v for v in r["rows"] if v["input_mode"]==c]
        levels=[dict(mean_ce=float(np.mean([v["levels"][i]["ce"] for v in samples])),
                     mean_disagreement=float(np.mean([v["levels"][i]["disagreement"] for v in samples])))
                for i in range(10)]
        result[c]={**{k:s[k] for k in COLS}, "mean_signed_contrast_dn":s["mean_predicted_signed_contrast_dn"],
                   "first_level_predicted_ids":histogram(s["first_tokens"]),"first_level_reference_ids":histogram(s["first_reference"]),
                   "first_level_wrong_count":sum(a!=b for a,b in zip(s["first_tokens"],s["first_reference"])),
                   "second_level_predicted_ids":histogram([t for v in samples for t in v["levels"][1]["predicted"]]),
                   "levels":levels,"first_margin_mean":float(np.mean([v["levels"][0]["top1_top2_margin"][0] for v in samples])),
                   "mean_valid_rgb_saturated":float(np.mean([v["output_valid_rgb_frac_saturated"] for v in samples]))}
    return result


def sourcehash(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def analyze():
    config=ROOT/"configs/denoise_v2/s2_update_diagnosis.json"
    spec=json.loads(config.read_text())
    schedule=json.loads((ROOT/"results/s2_common/schedule.json").read_text())
    source=[json.loads(l) for l in (ROOT/"results/s2_common/updates.jsonl").read_text().splitlines()]
    complete={a:load(f"{a}/complete.json") for a in ARMS}
    assert len({v["code_sha256"] for v in complete.values()})==1
    surveys={}
    training={}
    for a in ARMS:
        rows=[json.loads(l) for l in (OUT/a/"updates.jsonl").read_text().splitlines()]
        assert complete[a]["accepted_updates"]==len(rows)
        assert [k for r in rows for k in r["ids"]]==schedule[:4*len(rows)]
        ps=[load(f"{a}/panel_{i:02d}.json") for i in range(len(rows)+1)]
        for p in ps:
            for c in ("clean","noisy"):
                assert [x["id"] for x in p["rows"] if x["input_mode"]==c]==spec["panel_ids"]
            assert p["registration_sha256"]==sourcehash(config)
        surveys[a]={p["step"]:compact_panel(p) for p in ps}
        initial=surveys[a][0]
        assert initial==surveys["T-ALL"][0]
        training[a]=[]
        for r in rows:
            u=dict(update=r["step"],gradient_before_clip=r["grad_norm"],clipping_coefficient=r["clip_coef"],
                   loss_scale=r["loss_scale"],retries=r["retries"],gradient_tensors=r["grad_tensors"])
            samples=r["training_images"]
            for k in ("ce_by_level","ce_contribution_to_global_by_level","loss_logit_gradient_norm_per_image"):
                u["mean_"+k]=np.mean([v[k] for v in samples],axis=0).tolist()
            assert np.isfinite(u["mean_loss_logit_gradient_norm_per_image"]).all()
            u["sum_level_ce_contribution_error"]=abs(sum(u["mean_ce_contribution_to_global_by_level"])-r["token_ce"])
            training[a].append(u)
    flags={a:load(f"{a}/step1_flags.json")["trigger"] for a in ARMS}
    assert len(training["T-ALL"])==16
    assert len(training["T-CE"])==len(training["T-IMG"])==(4 if any(flags.values()) else 1)
    replay=[json.loads(l) for l in (OUT/"T-ALL/updates.jsonl").read_text().splitlines()]
    old_diff={k:max(r["old_step_abs_difference"][k] for r in replay) for k in replay[0]["old_step_abs_difference"]}
    identical=[r["step"] for r in replay if all(v==0 for v in r["old_step_abs_difference"].values())]
    unique={}
    with np.load(OUT/"T-CE/first_gradient_vectors.npz") as ce, np.load(OUT/"T-IMG/first_gradient_vectors.npz") as im, np.load(OUT/"T-ALL/first_gradient_vectors.npz") as joint:
        for name in PROBES:
            c=ce[name].astype(np.float64).ravel() if name in ce.files else None
            i=im[name].astype(np.float64).ravel() if name in im.files else None
            j=joint[name].astype(np.float64).ravel()
            nc=float(np.linalg.norm(c)) if c is not None else 0.
            ni=float(np.linalg.norm(i)) if i is not None else 0.
            nj=float(np.linalg.norm(j))
            residual=j-(c if c is not None else 0)-(i if i is not None else 0)
            unique[name]=dict(ce_path=c is not None,img_path=i is not None,ce_norm=nc,img_norm=ni,all_norm=nj,
                              ce_img_cosine=float(c@i/(nc*ni)) if nc and ni else None,
                              all_minus_sum_over_all=float(np.linalg.norm(residual)/nj) if nj else None)
    archives=load("archived/complete.json")["snapshots"]
    assert [a["step"] for a in archives]==[50,54] and archives[0]["saved_optimizer_steps"]==[50]
    archives_ps={s["step"]:compact_panel(load(f"archived/panel_{s['step']:02d}.json")) for s in archives}
    agreement={}
    for name,arm,step in (("s1","T-ALL",0),("s2","archived",54)):
        p=load(f"{arm}/panel_{step:02d}.json")
        for c in ("clean","noisy"):
            endpoint=ROOT/f"results/sirst_endpoints/{name}_draw{'clean' if c=='clean' else '0'}.json"
            prior={s["sample_id"]:s["metrics"]["greedy_argmax"] for s in json.loads(endpoint.read_text())["samples"]}
            samples=[v for v in p["rows"] if v["input_mode"]==c]
            differences={k:max(abs(s["metrics"][k]-prior[s["id"]][k]) for s in samples if k in prior[s["id"]]) for k in COLS}
            agreement[f"{name}_{c}"]=differences
    flags_by_arm={}
    for arm,trace in surveys.items():
        initial=trace[0]
        changes={}
        for mode in ("clean","noisy"):
            b=initial[mode]
            changes[mode]={
                "first_wrong_coarse_code":next((s for s,p in trace.items() if p[mode]["first_level_wrong_count"]>b["first_level_wrong_count"]),None),
                "first_mae_increase_2_dn":next((s for s,p in trace.items() if p[mode]["whole_mae_dn"]-b["whole_mae_dn"]>=2),None),
                "first_background_bias_shift_2_dn":next((s for s,p in trace.items() if abs(p[mode]["plan_background_mean_error_dn"]-b["plan_background_mean_error_dn"])>=2),None)}
        flags_by_arm[arm]=changes
    result=dict(registration_sha256=sourcehash(config),training_code_sha256=complete["T-ALL"]["code_sha256"],
                source_files_sha256_at_audit={k:sourcehash(ROOT/k) for k in ("prd_rollout.py","train_s2_common.py","prd_data.py","evaluate_warmup.py","diagnose_s2_updates.py")},
                split=spec["population"],panel_ids=spec["panel_ids"],all_registered_arm_summaries=complete,
                extension_trigger=flags,first_drift=flags_by_arm,old_TALL_log_bitwise_equal_numeric_updates=identical,
                old_TALL_worst_abs_delta=old_diff,first_update_gradient_audit={k:v for k,v in load("T-ALL/first_gradient_replay.json").items() if k!="actual"},
                scope="Training-panel results. CE and image interventions are independent restarts; old logs validate training statistics, not archived intermediate weights.",
                representative_gradients_before_clip=unique,grad_convention="Unscaled gradient of each full objective on its first four-image group; None means disconnected. Loss terms are weighted; FP16 backprop can cause deviation from float vector sums.",
                archived_sources=archives,external_same_endpoint_max_abs_metric_delta=agreement,
                panel_trajectory=surveys,archived_panel=archives_ps,training_trajectory=training)
    path=OUT/"summary.json"
    repeat_file=OUT/"pure_first_gradient/repeatability.json"
    if repeat_file.is_file():
        repeat=json.loads(repeat_file.read_text())
        result["original_unchanged_loss_repeatability"]={k:v for k,v in repeat.items() if k!="replicates"}
        result["original_unchanged_loss_repeatability"]["replicates"]=[
            {k:v for k,v in run.items() if k!="postclip_grad_norms"} for run in repeat["replicates"]]
    path.with_suffix(".partial").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    path.with_suffix(".partial").replace(path)
    print("historical log agreement",identical,old_diff)
    print("parameter gradients",unique)
    print("first_drift",flags_by_arm)
    for a in ARMS:
        for s,p in surveys[a].items():
            if s in (0,1,2,3,4,16):
                f=p["noisy"]
                print(a,s,f["whole_mae_dn"],f["plan_background_mean_error_dn"],f["first_level_wrong_count"])
    print("endpoint agreement",agreement)


if __name__=="__main__":
    analyze()
