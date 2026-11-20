"""Trace fixed-budget Free-S2 updates on the original training sources."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from evaluate_warmup import prepare_metrics, sample_digest, save_report
from prd_data import DenoisePairDataset
from prd_original_training import PROJECT, digest
from prd_rollout import free_rollout
from probe_backbone import model_input, to_gray_01
from train_local_baseline import load_trained
from train_s2_common import LOCK_SHA, REG, s2_optimizer
from train_warmup import valid_order_loss
from verify_rollout import cached_greedy


OUT = PROJECT / "results/denoise_v2/s2_update_diagnosis"
PLAN = PROJECT / "configs/denoise_v2/s2_update_diagnosis.json"
PLAN_HASH = "8d8770f5f57a74f155a6c685dcb532be8b84cea42d9ddecb39ce979e2208e581"
PROBES = ("head.bias", "blocks.23.attn.proj.bias", "cb_cond.4.bias", "vae.quantize.U",
          "vae.quantize.V", "vae.quantize.lora_gate")
METRICS = ("whole_mae_dn", "plan_background_rmse_dn", "plan_background_mean_error_dn",
           "native_median_ring_contrast_bias_dn", "native_median_ring_contrast_abs_error_dn",
           "mean_predicted_signed_contrast_dn", "mean_reference_signed_contrast_dn")
ARMS = ("T-ALL", "T-CE", "T-IMG")


def image_terms(model, sample, arm, scale):
    device = model.head.weight.device
    ref, inp, valid = [torch.from_numpy(sample[k])[None].to(device) for k in
                       ("reference_minus","noisy_minus","valid_mask")]
    with torch.no_grad():
        _, labels = model.vae.img_to_idxBl(ref.mul(2).sub(1).repeat(1,3,1,1))
        coverage = torch.cat([F.adaptive_avg_pool2d(valid.float(),n).flatten(1)
                              for n in model.patch_nums],dim=1)
    inp = inp.mul(2).sub(1).repeat(1,3,1,1)
    with torch.autocast("cuda",dtype=torch.float16,cache_enabled=False):
        image,scores,_ = free_rollout(model,inp,cfg=1.,temperature=1.)
    # Mirror order, dtype and reduction of train_s2_common.image_loss.
    token_map = F.cross_entropy(torch.cat(scores,dim=1).float().transpose(1,2),
                                torch.cat(labels,dim=1),reduction="none")
    ce = (token_map*coverage).sum()/coverage.sum()
    prediction=image.float()
    mse=(prediction-ref).square().mean(1,keepdim=True)[valid].mean()
    order=valid_order_loss(prediction.mean(1,keepdim=True),ref,valid)
    if arm=="T-ALL":
        loss=ce+0.2*mse+0.8*order
    elif arm=="T-CE":
        loss=ce
    else:
        loss=0.2*mse+0.8*order
    lengths=[int(s.shape[1]) for s in scores]
    with torch.no_grad():
        weights=coverage.split(lengths,dim=1)
        contributions=[(a*b).sum() for a,b in zip(token_map.split(lengths,dim=1),weights)]
        per_level=torch.stack([x/w.sum() for x,w in zip(contributions,weights)]).tolist()
        ce_mass=torch.stack(contributions).div(coverage.sum()).tolist()
    grad_logs={}
    for idx,logits in enumerate(scores):
        def observe(grad,i=idx):
            grad_logs[i]=float(grad.detach().float().norm())/(scale/4.)
        logits.register_hook(observe)
    (loss*(scale/4.)).backward()
    return torch.stack((ce.detach(),mse.detach(),order.detach())), dict(
        id=sample["meta"]["sample_id"],input_digest=sample_digest(sample),ce_by_level=per_level,
        ce_contribution_to_global_by_level=ce_mass,loss_logit_gradient_norm_per_image=[grad_logs[i] for i in range(10)])


def contrast(reference,valid,native):
    labels,_=ndimage.label(native,structure=np.ones((3,3),bool))
    vals=[]
    for idx in range(1,int(labels.max())+1):
        core=labels==idx
        ring=ndimage.binary_dilation(core,structure=np.ones((5,5),bool))&valid&~native
        if ring.any():
            vals.append(reference[core].astype("float64").mean()-np.median(reference[ring].astype("float64")))
    return float(np.mean(vals)*255.) if vals else None


def prepare_panel(model,data,indices):
    panel=[]
    for i in indices:
        s=data.sample(i,epoch=0,draw_id=0)
        ref=s["reference_minus"][0]
        valid=torch.from_numpy(s["valid_mask"])[None].cuda()
        with torch.no_grad():
            _,labels=model.vae.img_to_idxBl(model_input(s["reference_minus"],valid.device))
            coverage=[F.adaptive_avg_pool2d(valid.float(),n).flatten(1) for n in model.patch_nums]
        measure,_=prepare_metrics(s,extended=True)
        ref_contrast=contrast(ref,s["valid_mask"][0],s["native_mask"][0])
        panel.append((s,valid,labels,coverage,measure,ref_contrast))
    return panel


def row_metrics(model,item,condition):
    sample,valid,labels,coverage,measure,ref_contrast=item
    img=sample["reference_minus" if condition=="clean" else "noisy_minus"]
    ranges={}
    def latent_hook(mod,args):
        x=args[0].detach()
        ranges["latent_minmax"]=[float(x.min()),float(x.max())]
    def decoder_hook(mod,args,x):
        z=x.detach()[valid.expand_as(x)]
        ranges["preclamp_valid_rgb_minmax"]=[float(z.min()),float(z.max())]
        ranges["preclamp_valid_rgb_frac_outside"]=float(((z < -1)|(z > 1)).float().mean())
    h1=model.vae.post_quant_conv.register_forward_pre_hook(latent_hook)
    h2=model.vae.decoder.register_forward_hook(decoder_hook)
    try:
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):
            image,scores,ids=cached_greedy(model,model_input(img,valid.device))
    finally:
        h1.remove()
        h2.remove()
    with torch.no_grad():
        z=image[valid.expand_as(image)]
        ranges["output_valid_rgb_minmax"]=[float(z.min()),float(z.max())]
        ranges["output_valid_rgb_frac_saturated"]=float(((z==0)|(z==1)).float().mean())
        levels=[]
        for i,(logit,pred,target,w) in enumerate(zip(scores,ids,labels,coverage)):
            m=F.cross_entropy(logit.float().transpose(1,2),target,reduction="none")
            wrong=(pred!=target).float()
            level=dict(tokens=int(pred.numel()),coverage=float(w.sum()),ce=float((m*w).sum()/w.sum()),
                       disagreement=float(wrong.mean()),coverage_weighted_disagreement=float((wrong*w).sum()/w.sum()))
            if i<=1:
                best=logit.float().topk(2,dim=-1).values
                level.update(predicted=pred.flatten().tolist(),reference=target.flatten().tolist(),
                             top1_top2_margin=(best[...,0]-best[...,1]).flatten().tolist())
            levels.append(level)
        values=measure(to_gray_01(image).numpy())
    bias=values["native_median_ring_contrast_bias_dn"]
    values.update(mean_reference_signed_contrast_dn=ref_contrast,
                  mean_predicted_signed_contrast_dn=(ref_contrast+bias if bias is not None else None))
    return dict(id=sample["meta"]["sample_id"],input_mode=condition,
                input_sha256=hashlib.sha256(img.tobytes()).hexdigest(),
                reference_sha256=hashlib.sha256(sample["reference_minus"].tobytes()).hexdigest(),
                dataset_pair_digest=sample_digest(sample),metrics={k:values[k] for k in METRICS},levels=levels,**ranges)


def evaluate_panel(model,panel,out,step,parent_weights):
    rows=[row_metrics(model,item,c) for item in panel for c in ("clean","noisy")]
    summary={}
    for c in ("clean","noisy"):
        selected=[r for r in rows if r["input_mode"]==c]
        summary[c]={k:float(np.mean([r["metrics"][k] for r in selected if r["metrics"][k] is not None]))
                    for k in METRICS}
        summary[c].update(first_tokens=[r["levels"][0]["predicted"][0] for r in selected],
                          first_reference=[r["levels"][0]["reference"][0] for r in selected])
    totals={}
    with torch.no_grad():
        for name,p in model.named_parameters():
            if not p.requires_grad: continue
            initial=parent_weights[name]
            group=("CAC" if name.startswith(("cb_cond.","vae.quantize.")) else
                   "blocks" if name.startswith("blocks.") else "other")
            delta=(p.detach().cpu()-initial).norm().item()
            norm=initial.norm().item()
            acc=totals.setdefault(group,[0.,0.])
            acc[0]+=delta*delta
            acc[1]+=norm*norm
    totals["all"]=[sum(v[0] for v in totals.values()),sum(v[1] for v in totals.values())]
    change={k:(v[0]/v[1])**.5 for k,v in totals.items()}
    report=dict(step=step,registration_sha256=PLAN_HASH,summary=summary,relative_L2_vs_S1=change,rows=rows)
    save_report(out/f"panel_{step:02d}.json",report)
    print(f"panel {step} noisy MAE {summary['noisy']['whole_mae_dn']:.4f} DN bg bias {summary['noisy']['plan_background_mean_error_dn']:.4f} DN",flush=True)
    return report


def threshold(start,after):
    flags=[]
    for c in ("clean","noisy"):
        if start[c]["first_tokens"]!=after[c]["first_tokens"]: flags.append(c+"/first_token")
        for k in ("whole_mae_dn","plan_background_mean_error_dn"):
            if abs(start[c][k]-after[c][k])>=2.: flags.append(c+"/"+k)
    return flags


def verify_config():
    root=REG.parent
    spec=json.loads(PLAN.read_text())
    old=json.loads(REG.read_text())
    for path,expected in ((PLAN,PLAN_HASH),(REG,LOCK_SHA),(root/"schedule.json",spec["history_schedule_sha256"]),
                          (root/"updates.jsonl",spec["historic_updates_sha256"]),(PROJECT/spec["parent"],spec["parent_sha256"]),
                          (PROJECT/"manifests/train.jsonl",spec["manifest_sha256"]),(PROJECT/old["split_file"],spec["split_sha256"])):
        if digest(path)!=expected: raise ValueError(f"Registered source changed: {path}")
    return spec,old


def main(arm):
    spec,old=verify_config()
    torch.manual_seed(20261120)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0)
    out=OUT/arm
    out.mkdir(exist_ok=True,parents=True)
    if (out/"updates.jsonl").exists() or (out/"complete.json").exists():
        raise FileExistsError(out)
    parent=torch.load(PROJECT/spec["parent"],weights_only=True,map_location="cpu",mmap=True)["trainer"]["var_wo_ddp"]
    model=load_trained(PROJECT/spec["parent"],torch.device("cuda:0"))
    data=DenoisePairDataset(PROJECT,seed=0)
    mapping={r["sample_id"]:i for i,r in enumerate(data.records)}
    panel=prepare_panel(model,data,[mapping[i] for i in spec["panel_ids"]])
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    old_scope=json.loads((PROJECT/"results/local_retrain/load_audit.json").read_text())["trainable"]
    assert set(params)==set(old_scope) and len(params)==388
    schedule=json.loads((REG.parent/"schedule.json").read_text())
    previous=[json.loads(l) for l in (REG.parent/"updates.jsonl").read_text().splitlines()]
    scale=old["optimizer"]["loss_scale"]
    opt=s2_optimizer(model,old["optimizer"])
    checkpoints={"T-ALL":16,"T-CE":4,"T-IMG":4}
    if arm=="archived":
        rows=[]
        for step,path in ((50,PROJECT/"checkpoints/sirst_s2_common_resume.pt"),(54,PROJECT/"checkpoints/sirst_s2_common.pt")):
            ck=torch.load(path,weights_only=True,map_location="cpu",mmap=True)
            if ck["updates"]!=step or ck["parent_sha256"]!=spec["parent_sha256"] or ck["registration_sha256"]!=LOCK_SHA:
                raise ValueError("Archived snapshot identity mismatch")
            model.load_state_dict(ck["trainer"]["var_wo_ddp"],strict=True)
            steps=sorted({int(s["step"]) for s in ck.get("optimizer",{}).get("state",{}).values()})
            rows.append(dict(step=step,path=str(path.relative_to(PROJECT)),sha256=digest(path),saved_optimizer_steps=steps,
                             manifest_sha256=ck["manifest_sha256"],images=ck["images"]))
            evaluate_panel(model,panel,out,step,parent)
            del ck
        save_report(out/"complete.json",dict(source="actual archived snapshots read directly",snapshots=rows))
        return
    result0=evaluate_panel(model,panel,out,0,parent)
    hist=[]
    start=time.monotonic()
    for step in range(1,checkpoints[arm]+1):
        ids=schedule[4*(step-1):4*step]
        batch=[data.sample(mapping[k],epoch=20,draw_id=2) for k in ids]
        for retry in range(12):
            opt.zero_grad(set_to_none=True)
            values,diagnostics=[],[]
            for s in batch:
                v,d=image_terms(model,s,arm,scale)
                values.append(v)
                diagnostics.append(d)
            grads=[p.grad for p in params.values() if p.grad is not None]
            torch._foreach_mul_(grads,1./scale)
            vec={n:(p.grad.detach().cpu().numpy().copy() if p.grad is not None else None)
                 for n,p in params.items() if n in PROBES} if step==1 else {}
            norm=torch.nn.utils.clip_grad_norm_(params.values(),2.)
            if torch.isfinite(norm): break
            scale*=.5
            print(f"retry {step} with loss scale {scale}",flush=True)
        else: raise FloatingPointError(step)
        if step==1:
            np.savez(out/"first_gradient_vectors.npz",**{n:v for n,v in vec.items() if v is not None})
            if arm=="T-ALL":
                actual={n:(float(p.grad.norm()) if p.grad is not None else None) for n,p in params.items()}
                registered=json.loads((REG.parent/"first_update_grad_norms.json").read_text())
                save_report(out/"first_gradient_replay.json",dict(equal=sum(actual[n]==x for n,x in registered.items()),
                            max_abs_difference=max(abs(actual[n]-x) for n,x in registered.items()),actual=actual))
        opt.step()
        ce,mse,order=torch.stack(values).mean(0).tolist()
        n_updated=int(opt.state[model.head.weight]["step"])
        assert step==n_updated and float(norm)>0.
        record=dict(step=step,ids=ids,token_ce=ce,rgb_mse=mse,valid_order=order,grad_norm=float(norm),
                    grad_tensors=len(grads),clip_coef=min(1.,2./(float(norm)+1e-6)),loss_scale=scale,retries=retry,
                    training_images=diagnostics,elapsed=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if arm=="T-ALL":
            ref=previous[step-1]
            assert ids==ref["sample_ids"]
            record["old_step_abs_difference"]={k:abs(record[k]-ref[k]) for k in
                ("token_ce","rgb_mse","valid_order","grad_norm","loss_scale","retries")}
        with (out/"updates.jsonl").open("a") as f:
            f.write(json.dumps(record,allow_nan=False)+"\n")
        hist.append(record)
        del values,diagnostics,grads
        opt.zero_grad(set_to_none=True)
        report=evaluate_panel(model,panel,out,step,parent)
        if step==1:
            flags=threshold(result0["summary"],report["summary"])
            save_report(out/"step1_flags.json",dict(step=step,trigger=flags))
            if arm!="T-ALL":
                if not flags:
                    for _ in range(900):
                        paths=[OUT/name/"step1_flags.json" for name in ARMS]
                        if any(json.loads(p.read_text())["trigger"] for p in paths if p.is_file()): break
                        if all(p.exists() for p in paths): break
                        time.sleep(1)
                    else: raise TimeoutError("Waiting for first-step decision evidence")
                paths=[OUT/name/"step1_flags.json" for name in ARMS]
                if not any(json.loads(p.read_text())["trigger"] for p in paths if p.is_file()): break
    save_report(out/"complete.json",dict(arm=arm,images=4*len(hist),accepted_updates=len(hist),registration_sha256=PLAN_HASH,
                parent_sha256=spec["parent_sha256"],manifest_sha256=spec["manifest_sha256"],tests_used=0,
                optimizer_steps=sorted({int(s["step"]) for s in opt.state.values()}),optimizer_states=len(opt.state),
                sum_retries=sum(r["retries"] for r in hist),elapsed=time.monotonic()-start,
                code_sha256=digest(Path(__file__)),peak_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=(*ARMS,"archived"))
    main(parser.parse_args().arm)
