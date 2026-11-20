"""Measure paired clean calibration spot transfer through S1's VQ, CAC and deployment routes."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy import ndimage

from evaluate_e_only import DEST as TOKEN_DIR, encode, lineage
from evaluate_protocol_v2 import native_window, pixel_sha
from evaluate_sirst_endpoints import MANIFEST, OUT as ENDPOINTS, ROOT, S1
from evaluate_warmup import file_hash, sample_digest, save_report
from prd_data import DenoisePairDataset
from probe_backbone import model_input, to_gray_01
from train_local_baseline import load_trained
from verify_rollout import cached_greedy

DEST = ROOT / "results/denoise_v2/spot_transfer"
ROUTES = ("reference_pair", "q_base_e_only", "q_global_cac_fixed_h_minus",
          "q_global_cac_own_h", "s1_deployment")
DOSES = (1,2,4,6,8,12)


def patch_metrics(minus, plus, core, ring):
    """Intensity units are DN. Width is a local half-maximum equivalent diameter in native pixels."""
    a,b = minus.astype(np.float64)*255, plus.astype(np.float64)*255
    delta=b-a
    def contrast(im):
        return float(im[core].mean()-np.median(im[ring]))
    ca,cb=contrast(a),contrast(b)
    bg=float(np.median(delta[ring]))
    signal=delta-bg
    peak=float(signal[core].max())
    mad=float(1.4826*np.median(np.abs(delta[ring]-bg)))
    noise_gate=max(0.1,3*mad)
    result=dict(minus_contrast_dn=ca,plus_contrast_dn=cb,incremental_contrast_dn=cb-ca,
                difference_image_contrast_dn=contrast(delta),background_median_change_dn=bg,
                core_peak_after_ring_subtraction_dn=peak,ring_robust_sd_dn=mad,
                width_noise_threshold_dn=noise_gate,width_px=None,width_reason=None,half_max_pixels=None)
    footprint=core|ring
    if np.max(np.abs(delta[footprint]))==0:
        reason="identical_local_outputs"
    elif cb-ca<=0:
        reason="nonpositive_contrast_increment"
    elif peak<=noise_gate:
        reason="peak_below_local_noise_gate"
    elif signal[ring].max()>peak:
        reason="local_peak_outside_injection_core"
    elif .5*peak<=noise_gate:
        reason="half_height_below_local_noise_gate"
    else:
        selected=(signal>=.5*peak)&footprint
        labels,_=ndimage.label(selected,structure=np.ones((3,3),bool))
        anchor=np.argwhere(core&(signal==peak))[0]
        component=labels==labels[tuple(anchor)]
        n=int(component.sum())
        result["half_max_pixels"]=n
        if n<=1:
            reason="single_pixel_unresolved"
        elif np.any(component & (footprint & ~ndimage.binary_erosion(footprint,np.ones((3,3),bool)))):
            reason="half_height_contour_touches_window_edge"
        else:
            result["width_px"]=float(2*np.sqrt(n/np.pi))
            reason="measured"
    result["width_reason"]=reason
    return result


def plan(data):
    rows=[]
    for i in range(len(data)):
        s=data.sample(i,epoch=0,draw_id=0)
        signal=s["signal"][0]
        core,ring=s["support_mask"][0],s["ring_mask"][0]
        rec=data.records[i]
        if s["pair_valid"]:
            if (not np.any(core) or not np.any(ring) or np.any(core&ring)
                    or np.any((core|ring)&~s["valid_mask"][0]) or np.any((core|ring)&s["native_mask"][0])
                    or np.any(signal[~core]) or not np.array_equal(s["reference_plus"]-s["reference_minus"],s["signal"])):
                raise ValueError(f"Invalid injected pair: {rec['sample_id']}")
        est=float(signal.max()*255)
        declared=s["meta"]["signal_parameters"]["peak_dn"]
        if s["pair_valid"] and abs(est-declared)>2e-5:
            raise ValueError("Realized injected peak differs from nominal dose")
        x,y=s["meta"]["center_xy_patch"]; left,top=s["meta"]["pad_left_top"]
        rows.append(dict(sample_id=rec["sample_id"],source_png_sha256=rec["image_sha256"],
                         pair_valid=bool(s["pair_valid"]),nominal_peak_dn=int(declared),actual_peak_dn=est,
                         center_native_xy=[x-left,y-top] if s["pair_valid"] else None,core_pixels=int(core.sum()),
                         ring_pixels=int(ring.sum()),signal_sha256=pixel_sha(signal),
                         minus_sha256=pixel_sha(s["reference_minus"][0]),plus_sha256=pixel_sha(s["reference_plus"][0]),
                         seed=s["meta"]["signal_seed"],template=s["meta"]["signal_parameters"]))
    spec=dict(source="NUAA-SIRST train, originals only; one existing epoch0/draw0 template per source. Clean paired reference branches only; no added stripe/pixel noise.",
              n_sources=len(rows),n_placeable=sum(r["pair_valid"] for r in rows),
              per_nominal_peak_counts=dict(Counter(r["nominal_peak_dn"] for r in rows if r["pair_valid"])),
              endpoint="S1 after 20 local epochs",source_manifest_sha256=file_hash(MANIFEST),input_seed=0,epoch=0,draw=0,
              sampler="greedy, FP16 both TF32 off, FP32 VQ encoding, h and final CAC on S1, no rounding/AdaIN",
              allocation="All eligible sources at all five stages; group labels from originally assigned peak before viewing any outputs.",
              contrast="C=core mean minus median over the identical local ring. g=[C(output+)−C(output−)]/[C(input+)−C(input−)], signed. Also report C(output+−output−) separately.",
              width="In the predefined core+ring footprint subtract median ring change from the native delta image. Require positive incremental contrast and core peak at least as large as any ring value. Threshold is max(0.1 DN,3*1.4826*MAD_ring). Half-peak must exceed threshold. Eight-connected half-maximum component containing highest core pixel: equal-area diameter=2*sqrt(area/pi) pixels. Single-pixel, edge-truncated, low-contrast or absent responses have null width and separately counted reasons.",
              transfer_floor="Descriptive retained if signed contrast ratio >=0.5; also give raw unthresholded ratio and sign.",
              plan_code_sha256=file_hash(Path(__file__)),pairs=rows)
    previous=DEST/"protocol.json"
    if previous.exists() and json.loads(previous.read_text())!=json.loads(json.dumps(spec)):
        raise ValueError("Spot generation or registered settings changed")
    save_report(previous,spec)
    return spec


def run(data,spec):
    controls=json.loads((ENDPOINTS/"s1_drawclean.json").read_text())
    historical=json.loads((TOKEN_DIR/"s1_clean_six_routes.json").read_text())
    meta=json.loads((TOKEN_DIR/"s1_clean_tokens.json").read_text())
    archive=TOKEN_DIR/meta["token_archive"]
    if (file_hash(archive)!=meta["tokens_and_h_sha256"] or file_hash(S1)!=meta["checkpoint_sha256"]
            or spec["source_manifest_sha256"]!=meta["manifest_sha256"]
            or controls["checkpoint_sha256"]!=meta["checkpoint_sha256"]):
        raise ValueError("S1 checkpoint, clean controls or tokens changed")
    with np.load(archive,allow_pickle=False) as f:
        cached={k:f[k] for k in ("reference","predicted","h","ids","input_sha256")}
    device=torch.device("cuda",0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=load_trained(S1,device)
    for key,sha in lineage(model).items():
        if meta[key]!=sha:
            raise ValueError(f"S1 component lineage differs: {key}")
    scales=[n*n for n in model.patch_nums]
    outdir=DEST/"patches"
    outdir.mkdir(exist_ok=True)
    diagnostics=[]
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
        for i, specrow in enumerate(spec["pairs"]):
            if not specrow["pair_valid"]:
                continue
            sample=data.sample(i,epoch=0,draw_id=0)
            a,b=sample["reference_minus"],sample["reference_plus"]
            sample["noisy_minus"]=a
            box=native_window(sample)
            identifier=specrow["sample_id"]
            old=controls["samples"][i]; hdr=historical["samples"][i]
            if (identifier!=old["sample_id"] or identifier!=cached["ids"][i] or identifier!=hdr["sample_id"]
                    or sample_digest(sample)!=cached["input_sha256"][i] or pixel_sha(b[0])!=specrow["plus_sha256"]):
                raise ValueError("Clean sample does not match the registered S1 routes")
            hm=torch.from_numpy(cached["h"][i:i+1]).to(device)
            zm=torch.from_numpy(cached["reference"][i].astype(np.int64)).to(device)[None].split(scales,dim=1)
            zp=encode(model,b,device)
            capture=[]
            orig=model.vae.idxBl_to_img_lora
            def observe(tokens,h,*args,**kwargs):
                capture.append(h.detach())
                return orig(tokens,h,*args,**kwargs)
            with patch.object(model.vae,"idxBl_to_img_lora",observe):
                yplus,_,arplus=cached_greedy(model,model_input(b,device))
            if len(capture)!=1:
                raise ValueError("Expected one true final global CAC condition")
            hp=capture[0]
            def decode(z,h=None):
                raw=(model.vae.idxBl_to_img(z,same_shape=True,last_one=True) if h is None
                     else orig(z,h,same_shape=True,last_one=True))
                return to_gray_01((raw+1)*.5).numpy()[box]
            minus_base=decode(zm); minus_cac=decode(zm,hm)
            plus_base=decode(zp); plus_fixed=decode(zp,hm); plus_own=decode(zp,hp)
            for label,arr in (("reference_base",minus_base),("reference_cac",minus_cac)):
                if pixel_sha(arr)!=hdr["native_pixel_sha256"][label]:
                    raise ValueError(f"Archived clean decoder pixels differ: {identifier}/{label}")
            arpath=ENDPOINTS/old["native_gray_npy"]
            if file_hash(arpath)!=old["native_gray_sha256"]:
                raise ValueError("Original S1 unspotted deployment raster differs")
            minus_ar=np.load(arpath,allow_pickle=False)
            images=((a[0][box],b[0][box]),(minus_base,plus_base),(minus_cac,plus_fixed),
                    (minus_cac,plus_own),(minus_ar,to_gray_01(yplus).numpy()[box]))
            core,ring=sample["support_mask"][0][box],sample["ring_mask"][0][box]
            yy,xx=np.nonzero(core|ring)
            window=np.s_[yy.min():yy.max()+1,xx.min():xx.max()+1]
            pack=dict(core=core[window],ring=ring[window])
            info=dict(sample_id=identifier,nominal_peak_dn=specrow["nominal_peak_dn"],
                      crop_x0=int(xx.min()),crop_y0=int(yy.min()),actual_peak_dn=specrow["actual_peak_dn"],
                      h_max_abs_change=float((hm-hp).abs().max()),
                      vq_changed_by_scale=[int((u!=v).sum().item()) for u,v in zip(zm,zp,strict=True)],
                      ar_changed_total=int(np.count_nonzero(torch.cat(arplus,dim=1)[0].cpu().numpy()!=cached["predicted"][i])),
                      full_frame={})
            for stage,(m,p) in zip(ROUTES,images,strict=True):
                pack[stage+"_minus"]=m[window];pack[stage+"_plus"]=p[window]
                change=255*(p.astype(np.float64)-m)
                info["full_frame"][stage]=dict(equal_pixels=int((change==0).sum()),pixels=change.size,
                                               max_abs_change_dn=float(np.abs(change).max()),
                                               rmse_dn=float(np.sqrt(np.square(change).mean())))
            file=outdir/(data.records[i]["source_image_id"]+".npz")
            with file.with_suffix(".pending").open("wb") as f:np.savez_compressed(f,**pack)
            file.with_suffix(".pending").replace(file)
            info.update(patch_file=str(file.relative_to(DEST)),patch_sha256=file_hash(file))
            diagnostics.append(info)
            if len(diagnostics)%40==0:print(f"Captured {len(diagnostics)}/{spec['n_placeable']} clean pairs",flush=True)
    save_report(DEST/"pairs.json",dict(n_pairs=len(diagnostics),checkpoint_sha256=meta["checkpoint_sha256"],
                tokenizer_sha256=meta["tokenizer_sha256"],decoder_sha256=meta["decoder_sha256"],
                clean_token_archive_sha256=meta["tokens_and_h_sha256"],tf32_matmul=False,tf32_cudnn=False,
                draws_of_clean_source=1,stage_names=ROUTES,samples=diagnostics))


def report(spec):
    capture=json.loads((DEST/"pairs.json").read_text()); rows=[]
    source_ids=[r["sample_id"] for r in spec["pairs"] if r["pair_valid"]]
    if source_ids!=[r["sample_id"] for r in capture["samples"]]:
        raise ValueError("Captured source list differs")
    for r in capture["samples"]:
        file=DEST/r["patch_file"]
        if file_hash(file)!=r["patch_sha256"]:
            raise ValueError("Spot patch changed")
        with np.load(file,allow_pickle=False) as f:
            core,ring=f["core"],f["ring"]
            ref=patch_metrics(f["reference_pair_minus"],f["reference_pair_plus"],core,ring)
            if ref["incremental_contrast_dn"]<=0:raise ValueError("Registered input has no positive contrast increment")
            for stage in ROUTES:
                m,p=f[stage+"_minus"],f[stage+"_plus"]
                measured=patch_metrics(m,p,core,ring)
                ratio=measured["incremental_contrast_dn"]/ref["incremental_contrast_dn"]
                width=measured["width_px"]; baseline=ref["width_px"]
                rows.append(dict(sample_id=r["sample_id"],nominal_peak_dn=r["nominal_peak_dn"],route=stage,
                    actual_input_peak_dn=r["actual_peak_dn"],input_incremental_contrast_dn=ref["incremental_contrast_dn"],
                    contrast_retention_ratio=ratio,retained_at_half_input=ratio>=.5,input_width_px=baseline,
                    width_delta_px=(width-baseline) if width is not None and baseline is not None else None,
                    peak_gain=measured["core_peak_after_ring_subtraction_dn"]/r["actual_peak_dn"],
                    vq_change_count=sum(r["vq_changed_by_scale"]),ar_change_count=r["ar_changed_total"],**measured))
    result={}
    def describe(vals):
        return {"n":len(vals),"mean":float(np.mean(vals)) if vals else None,"median":float(np.median(vals)) if vals else None,
                "q25":float(np.quantile(vals,.25)) if vals else None,"q75":float(np.quantile(vals,.75)) if vals else None}
    for dose in ("all",*DOSES):
        result[str(dose)]={}
        for stage in ROUTES:
            chosen=[r for r in rows if r["route"]==stage and (dose=="all" or r["nominal_peak_dn"]==dose)]
            result[str(dose)][stage]=dict(n_pairs=len(chosen),n_retained_half=sum(r["retained_at_half_input"] for r in chosen),
                n_negative_gain=sum(r["contrast_retention_ratio"]<0 for r in chosen),
                n_zero_gain=sum(r["contrast_retention_ratio"]==0 for r in chosen),
                width_status=dict(Counter(r["width_reason"] for r in chosen)),
                **{k:describe([r[k] for r in chosen if r[k] is not None]) for k in
                   ("input_incremental_contrast_dn","incremental_contrast_dn","contrast_retention_ratio","peak_gain","width_px","width_delta_px")})
    with (DEST/"per_pair_stage.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    save_report(DEST/"summary.json",dict(n_sources=len(spec["pairs"]),n_placeable=spec["n_placeable"],
        n_eligible_rows=len(rows),n_unchanged_vq_codes=sum(sum(r["vq_changed_by_scale"])==0 for r in capture["samples"]),
        n_unchanged_deployment_codes=sum(r["ar_changed_total"]==0 for r in capture["samples"]),
        n_identical_h=sum(r["h_max_abs_change"]==0 for r in capture["samples"]),
        protocol_sha256=file_hash(DEST/"protocol.json"),pair_archive_index_sha256=file_hash(DEST/"pairs.json"),
        rows_sha256=file_hash(DEST/"per_pair_stage.csv"),by_nominal_peak_dn=result))
    print(f"Verified {len(rows)} paired stage observations",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("plan","run","report"))
    op=parser.parse_args().operation
    DEST.mkdir(parents=True,exist_ok=True)
    data=DenoisePairDataset(ROOT,seed=0)
    spec=plan(data)
    print(f"{spec['n_placeable']}/{spec['n_sources']} placeable, counts {spec['per_nominal_peak_counts']}",flush=True)
    if op=="run":run(data,spec)
    if op in ("report","run"):report(spec)
