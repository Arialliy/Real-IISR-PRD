"""Unified native grayscale assessment for fixed local SIRST training originals."""
import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity

from check_column_baseline import correct_columns
from evaluate_warmup import (METRIC_DIRECTION, checked_error_dn, file_hash, fixed_background,
                             paired_changes, prepare_metrics, sample_digest, save_report)
from prd_data import DenoisePairDataset

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "results/sirst_endpoints"
METHODS = ("input", "column", "published", "s1_retrained", "s2_common")
CORRUPTIONS = ("clean", "draw0", "draw1")
CHECKPOINTS = {"published": ("published", "Real-IISR.pth"), "s1_retrained": ("s1", "sirst_retrained_baseline.pt"),
               "s2_common": ("s2", "sirst_s2_common.pt")}
VERSION = "native_v2"
GAIN_MIN_DN = 1.0
LEGACY = ("valid_mae_dn","background_rmse_dn","background_column_residual_dn","native_contrast_abs_error_dn")


def native_window(sample):
    valid = sample["valid_mask"][0]
    if "pad_left_top" in sample.get("meta", {}):
        x, y = sample["meta"]["pad_left_top"]
        _, _, w, h = sample["meta"]["crop_xywh"]
    else:
        yy, xx = np.nonzero(valid)
        if not len(yy):
            raise ValueError("Native frame has no valid pixels")
        y, x = yy.min(), xx.min()
        h, w = yy.max()-y+1, xx.max()-x+1
    window = np.s_[y:y+h,x:x+w]
    if not valid[window].all() or valid.sum() != h*w or not valid.any():
        raise ValueError("Expected rectangular valid native extent")
    return window


def target_measurements(ref, pred, mask):
    labels, n = ndimage.label(mask, structure=np.ones((3,3), bool))
    rows = []
    for k, small in enumerate(ndimage.find_objects(labels),1):
        y, x = small
        roi = tuple(slice(max(s.start-2,0), min(s.stop+2,size)) for s,size in zip(small,mask.shape))
        core = labels[roi] == k
        dilated = ndimage.binary_dilation(core,structure=np.ones((5,5),bool))
        ring = dilated & ~mask[roi]
        border = any(s.start < 2 or s.stop > size-2 for s,size in zip(small,mask.shape))
        row = dict(target_id=k, target_area_px=int(core.sum()), ring_pixels=int(ring.sum()),
                   x0=x.start, y0=y.start, x1_exclusive=x.stop, y1_exclusive=y.stop,
                   ring_eligible=bool(ring.any()), touches_frame=any(s.start==0 or s.stop==size for s,size in zip(small,mask.shape)),
                   ring_clipped_by_frame=border,
                   ring_excluded_other_target=bool(np.any(dilated & mask[roi] & ~core)))
        ref_c = out_c = ref_mean = out_mean = None
        if ring.any():
            values = []
            for image in (ref,pred):
                local = image[roi]
                core_mean = local[core].mean()
                values.extend([float(255*(core_mean - np.median(local[ring]))),
                               float(255*(core_mean - local[ring].mean()))])
            ref_c, ref_mean, out_c, out_mean = values
        gate = ref_c is not None and abs(ref_c)>=GAIN_MIN_DN
        bias = None if ref_c is None else out_c-ref_c
        legacy = None if ref_mean is None else out_mean-ref_mean
        strength = None if ref_c is None else abs(ref_c)
        stratum = (None if strength is None else "lt10" if strength<10 else "10_to_lt30" if strength<30 else "ge30")
        row.update(reference_contrast_dn=ref_c,prediction_contrast_dn=out_c, contrast_bias_dn=bias,
                   contrast_abs_error_dn=None if bias is None else abs(bias), gain_eligible=gate,
                   gain=(out_c/ref_c) if gate else None,reference_abs_contrast_dn=strength,
                   reference_contrast_bin_dn=stratum, mean_ring_reference_contrast_dn=ref_mean,
                   mean_ring_prediction_contrast_dn=out_mean,mean_ring_contrast_bias_dn=legacy,
                   mean_ring_contrast_abs_error_dn=None if legacy is None else abs(legacy))
        rows.append(row)
    if n != len(rows):
        raise ValueError("Lost native targets")
    return rows


def measure_native(sample, prediction, *, details=False):
    """Score padded 2D grayscale [0,1] input. Return metrics and optional native target rows."""
    measure,_ = prepare_metrics(sample, extended=True)
    metrics = measure(prediction)
    box = native_window(sample)
    pred = np.asarray(prediction,dtype=np.float64)[box]
    ref = np.asarray(sample["reference_minus"][0],dtype=np.float64)[box]
    metrics["whole_ssim"] = (float(structural_similarity(ref,pred,data_range=1,win_size=11,
        gaussian_weights=True,sigma=1.5,use_sample_covariance=False)) if min(ref.shape)>=11 else None)
    return metrics, target_measurements(ref,pred,sample["native_mask"][0][box]) if details else []


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def checked_caches(methods, corruptions, ids, manifest_hash):
    reports, provenance = {}, {}
    from prd_original_training import HASHES
    pinned = {"published":"fe1c2600ec6aeb18bfa674f7726a1dcd0aec45d9fbba25c7f1b9fe266b3ef9a9",
              "s1_retrained":"fb2eaf0426e1634b8cde12c7d8951ded9d6b4f04a0ae2ef77a1c13f794caacab",
              "s2_common":"0be3668c63d2224c89b7c011bcd8167f01802e80acaab54d8c70639fcbb47010"}
    for method in methods:
        if method not in CHECKPOINTS:
            provenance[method] = {"kind":"identity" if method=="input" else "horizontal median column correction"}
            continue
        prefix, name = CHECKPOINTS[method]
        checkpoint = ROOT / "checkpoints" / name
        sha = file_hash(checkpoint)
        if sha != pinned[method]:
            raise ValueError(f"Wrong weights for {method}")
        parent = (HASHES if method=="s1_retrained" else {"s1_retrained": pinned["s1_retrained"]}
                  if method=="s2_common" else {"origin":"author released Real-IISR checkpoint"})
        provenance[method] = dict(checkpoint=str(checkpoint.relative_to(ROOT)),sha256=sha,parents=parent,
                                  reports={})
        for corruption in corruptions:
            draw = "clean" if corruption=="clean" else corruption[-1]
            path = CACHE / f"{prefix}_draw{draw}.json"
            report = json.loads(path.read_text())
            protocol = report["protocol"]
            if (report["draw"] != draw or report["endpoint"] != prefix or report["split"] != "NUAA-SIRST/train"
                    or report["manifest_sha256"] != manifest_hash or report["checkpoint_sha256"] != sha
                    or report["epoch"] != 0 or report["data_seed"] != 0 or "greedy_argmax" not in protocol["samplers"]
                    or protocol["matmul_tf32"] or protocol["cudnn_tf32"]
                    or [r["sample_id"] for r in report["samples"]] != ids):
                raise ValueError(f"Cached predictions mismatch protocol: {path}")
            reports[method,corruption] = report["samples"]
            provenance[method]["reports"][corruption] = dict(path=str(path.relative_to(ROOT)),sha256=file_hash(path),
                    protocol=protocol)
    return reports,provenance


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_one(i, data, methods, corruptions, reports, legacy_column, output):
    from scipy.ndimage import median_filter
    record = data.records[i]
    key, identifier = record["source_image_id"], record["sample_id"]
    png = ROOT / record["reference_path"]
    if file_hash(png) != record["image_sha256"]:
        raise ValueError(f"Original PNG changed: {key}")
    with Image.open(png) as im:
        ref = np.asarray(im.convert("L"),dtype=np.float32)/255
    with Image.open(ROOT / record["mask_path"]) as im:
        mask = np.asarray(im.convert("L")) != 0
    ref_sha, mask_sha = pixel_sha(ref), pixel_sha(mask)
    rows, targets, interventions = [],[],[]
    fields = {}
    fixed_hash = None
    for c in corruptions:
        sample = data.sample(i,epoch=0,draw_id=1 if c=="draw1" else 0)
        if c=="clean":
            sample["noisy_minus"] = sample["reference_minus"]
        full = sample["noisy_minus"][0]
        box = native_window(sample)
        bg = fixed_background(sample)[box]
        bg_hash = pixel_sha(bg)
        if (not np.array_equal(ref, sample["reference_minus"][0][box]) or np.any(full[~sample["valid_mask"][0]])
                or not np.array_equal(mask,sample["native_mask"][0][box]) or fixed_hash not in (None,bg_hash)):
            raise ValueError(f"Native PNG, shape, annotation or fixed background changed: {key}/{c}")
        fixed_hash = bg_hash
        inp = full[box]
        inp_sha = pixel_sha(inp)
        if c=="clean" and inp_sha != ref_sha:
            raise ValueError("Clean input must equal unmodified original grayscale pixels")
        x,y = sample["meta"]["pad_left_top"]
        h,w = ref.shape
        mh = sample_digest(sample)
        for mode in methods:
            entry = None
            if mode in ("input","column"):
                pred = full if mode=="input" else correct_columns(full,x,y,w,h)
                raster = output/"float"/mode/c/f"{key}.npy"
                np.save(raster,pred[box],allow_pickle=False)
                sha = file_hash(raster)
                if mode=="column":
                    old = inp.copy()
                    local = median_filter(old,size=(1,7),mode="reflect")
                    bias = np.median(old-local,axis=0)
                    old[:] = (old-(bias-np.median(bias))).clip(0,1)
                    if not np.array_equal(old,pred[box]):
                        raise ValueError("Default column correction changed legacy pixels")
            else:
                entry = reports[mode,c][i]
                raster = CACHE / entry["native_gray_npy"]
                sha=file_hash(raster)
                arr=np.load(raster,allow_pickle=False)
                if (entry["data_sha256"] != mh or sha!=entry["native_gray_sha256"] or arr.shape!=ref.shape
                        or arr.dtype!=np.float32 or not np.isfinite(arr).all() or arr.min()<0 or arr.max()>1):
                    raise ValueError(f"Invalid cached native prediction: {raster}")
                pred=np.zeros_like(full)
                pred[box]=arr
            metrics, details = measure_native(sample,pred,details=True)
            if entry:
                old_metrics = entry["metrics"]["greedy_argmax"]
                if any(metrics[k]!=v for k,v in old_metrics.items()):
                    raise ValueError(f"Historical endpoint metric disagreement: {key}/{c}/{mode}")
            elif c!="clean":
                old = legacy_column[c[-1]]["samples"][i]
                route="input" if mode=="input" else "column_correction"
                if old["data_sha256"]!=mh or any(metrics[k]!=v for k,v in old["metrics"][route].items()):
                    raise ValueError("Historical column/input metric disagreement")
            eligible = [t for t in details if t["ring_eligible"]]
            for per_target,per_image in (("contrast_abs_error_dn","native_median_ring_contrast_abs_error_dn"),
                      ("contrast_bias_dn","native_median_ring_contrast_bias_dn"),
                      ("mean_ring_contrast_abs_error_dn","native_contrast_abs_error_dn")):
                if eligible and abs(np.mean([t[per_target] for t in eligible])-metrics[per_image])>1e-10:
                    raise ValueError("Target rows do not reproduce image-level target metric")
            noise = sample["meta"]["noise_parameters"]
            row = dict(sample_id=identifier,method=mode,corruption=c,metric_version=VERSION,
                    reference_png=record["reference_path"],reference_png_sha256=record["image_sha256"],
                    reference_pixels_sha256=ref_sha,input_pixels_sha256=inp_sha,sample_sha256=mh,
                    native_mask_sha256=mask_sha,fixed_background_sha256=bg_hash,valid_pixels=int(ref.size),
                    native_background_pixels=int(bg.sum()),native_target_count=len(details),
                    eligible_targets=len(eligible),noise_seed=(None if c=="clean" else sample["meta"]["noise_seed"]),
                    stripe_sigma_dn=0. if c=="clean" else noise["stripe_sigma_dn"],
                    pixel_sigma_dn=0. if c=="clean" else noise["pixel_sigma_dn"],
                    native_float_path=os.path.relpath(raster,ROOT),native_float_sha256=sha,
                    whole_psnr_status=("perfect_reconstruction" if metrics["whole_psnr_db"] is None and
                                       metrics["whole_mae_dn"]==0 else "defined" if metrics["whole_psnr_db"] is not None
                                       else "unavailable"),**metrics)
            rows.append(row)
            targets.extend(dict(sample_id=identifier,method=mode,corruption=c,metric_version=VERSION,**t) for t in details)
            fields[mode,c]=pred[box].copy()
    if "clean" in corruptions:
        for method in methods:
            zero=fields[method,"clean"].astype(np.float64)
            e0=255*(zero-ref)
            for c in (s for s in corruptions if s!="clean"):
                noisy=fields[method,c].astype(np.float64)
                dn=255*(noisy-zero)
                en=255*(noisy-ref)
                residual=float(np.abs(en-(e0+dn)).max())
                if residual>1e-12:
                    raise ValueError("Intervention pixel identity failed")
                fieldfile=output/"intervention_fields"/method/c/f"{key}.npz"
                # Preserve exact float64 DN fields in a reproducible, compact archive. e_n = e0 + d_n.
                np.savez_compressed(fieldfile,e0_dn=e0,d_n_dn=dn)
                values = dict(sample_id=identifier,method=method,corruption=c,n_pixels=e0.size,
                              fields_npz=os.path.relpath(fieldfile,ROOT),fields_sha256=file_hash(fieldfile),
                              closure_max_abs_dn=residual)
                for name,field in (("e0",e0),("d_n",dn),("e_n",en)):
                    values[name+"_mae_dn"]=float(np.abs(field).mean())
                    values[name+"_rmse_dn"]=float(np.sqrt(np.square(field).mean()))
                    values[name+"_mean_dn"]=float(field.mean())
                values["twice_mean_e0_times_dn_dn2"]=float(2*np.mean(e0*dn))
                interventions.append(values)
    return rows,targets,interventions


def aggregate(values, name, psnr_states=()):
    usable=[float(v) for v in values if v is not None]
    nperfect=sum(s=="perfect_reconstruction" for s in psnr_states) if name=="whole_psnr_db" else 0
    seq=np.array(usable+[np.inf]*nperfect)
    out=dict(n_images=len(seq),n_missing=len(values)-len(seq),mean=None,median=None)
    if len(seq):
        med=float(np.median(seq))
        out.update(mean=float(seq.mean()) if not nperfect else None,median=med if np.isfinite(med) else None)
    if name=="whole_psnr_db":
        out.update(n_perfect=nperfect,n_finite=len(usable),mean_positive_infinity=bool(nperfect),
                   median_positive_infinity=bool(len(seq) and np.isposinf(np.median(seq))))
    if METRIC_DIRECTION[name]=="signed" and len(seq):
        out.update(q05=float(np.quantile(seq,.05)),q95=float(np.quantile(seq,.95)),
                   negative_fraction=float((seq<0).mean()),positive_fraction=float((seq>0).mean()),
                   mean_absolute_value=float(np.abs(seq).mean()))
    return out


def group_summary(rows,targets,interventions,methods,corruptions):
    result={}
    for c in corruptions:
        result[c]={}
        by_method={}
        for method in methods:
            sub=[r for r in rows if r["method"]==method and r["corruption"]==c]
            by_method[method]=sub
            metrics={name:aggregate([r[name] for r in sub],name,[r["whole_psnr_status"] for r in sub])
                     for name in METRIC_DIRECTION}
            result[c][method]=dict(n_images=len(sub),targets=sum(r["native_target_count"] for r in sub),
                                  eligible_targets=sum(r["eligible_targets"] for r in sub),metrics=metrics)
            selected=[t for t in targets if t["method"]==method and t["corruption"]==c]
            groups={}
            for attr in ("reference_contrast_bin_dn","area_bin"):
                for group in ("lt10","10_to_lt30","ge30") if attr!="area_bin" else ("1_to_4","5_to_16","ge17"):
                    grp=[t for t in selected if t["ring_eligible"] and ((t[attr]==group) if attr!="area_bin"
                         else (("1_to_4" if t["target_area_px"]<=4 else "5_to_16" if t["target_area_px"]<=16 else "ge17")==group))]
                    groups[group]={"instances":len(grp),"instance_mean_abs_error_dn":
                                   float(np.mean([t["contrast_abs_error_dn"] for t in grp])) if grp else None}
            result[c][method]["target_groups"]=groups
            iv=[r for r in interventions if r["method"]==method and r["corruption"]==c]
            if iv:
                result[c][method]["intervention"]={k:float(np.mean([v[k] for v in iv])) for k in
                                                   ("e0_mae_dn","d_n_mae_dn","e_n_mae_dn","d_n_rmse_dn","d_n_mean_dn")}
        if "input" in methods:
            for method in methods:
                paired=[{"metrics":{"input":{k:a[k] for k in METRIC_DIRECTION},
                                     "method":{k:b[k] for k in METRIC_DIRECTION}}}
                        for a,b in zip(by_method["input"],by_method[method],strict=True)]
                result[c][method]["changes_vs_input"]=paired_changes(paired,"input","method")
    return result


def git_head(directory):
    db=directory/".git"
    if db.is_file():
        db=(directory/db.read_text().strip().removeprefix("gitdir: ")).resolve()
    head=(db/"HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref=head[5:]
    if (db/ref).is_file():
        return (db/ref).read_text().strip()
    for line in (db/"packed-refs").read_text().splitlines():
        if line.endswith(" "+ref):
            return line.split()[0]
    raise ValueError(f"Unresolved Git HEAD at {directory}")


def checked_options(text, options):
    names=text.split(",") if isinstance(text,str) else text
    if not names or len(names)!=len(set(names)) or set(names)-set(options):
        raise ValueError(f"Expected a unique subset of {options}")
    return list(names)


def acceptance():
    ref=np.linspace(0.09,0.87,256,dtype=np.float32).reshape(16,16)
    valid=np.ones_like(ref,bool)
    core=np.zeros_like(valid);core[4:6,6:8]=True
    sample=dict(reference_minus=ref[None],valid_mask=valid[None],native_mask=core[None],background_mask=(~core)[None])
    score,targets=measure_native(sample,ref,details=True)
    if score["whole_ssim"]!=1 or score["whole_mae_dn"]!=0 or score["whole_psnr_db"] is not None or targets[0]["contrast_bias_dn"]!=0:
        raise AssertionError("Identity convention failed")
    for bad in (ref[:1],ref[None],np.full_like(ref,np.nan)):
        try:
            measure_native(sample,bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid arrays were broadcast or accepted")
    mod=ref.copy();mod[core]+=.02
    m,t=measure_native(sample,mod,details=True)
    if not (5.09<m["native_median_ring_contrast_bias_dn"]<5.11 and 5.09<t[0]["contrast_bias_dn"]<5.11):
        raise AssertionError("Signed target direction incorrect")
    comparison=[{"metrics":{"a":{"whole_psnr_db":20.,"whole_mae_dn":2.,"plan_background_mean_error_dn":1.},
                             "b":{"whole_psnr_db":None,"whole_mae_dn":0.,"plan_background_mean_error_dn":0.}}}]
    delta=paired_changes(comparison,"a","b")
    if delta["whole_psnr_db"]["improved_image_fraction"]!=1 or "improved_image_fraction" in delta["plan_background_mean_error_dn"]:
        raise AssertionError("Metric directions not respected")
    if not np.array_equal(correct_columns(ref,0,0,16,16,strength=0),ref):
        raise AssertionError("Zero column correction must reproduce the input")
    return {"identity_mae":0,"identity_psnr_status":"perfect_reconstruction","identity_ssim":1,
            "reject_broadcast_and_nonfinite":True,"strict_signed_contrast":True,"metric_directions":True,
            "zero_strength_identity":True}


def verify_output(output,workers):
    from concurrent.futures import ThreadPoolExecutor
    signature=json.loads((output/"protocol_v2.json").read_text())
    with (output/"per_image.csv").open() as f:
        rows=list(csv.DictReader(f))
    with (output/"per_target.jsonl").open() as f:
        targets=[json.loads(line) for line in f]
    with (output/"interventions.csv").open() as f:
        iv=list(csv.DictReader(f))
    by_key={(r["sample_id"],r["method"],r["corruption"]):r for r in rows}
    ids=[json.loads(l)["sample_id"] for l in (ROOT/"manifests/train.jsonl").read_text().splitlines()]
    methods,corruptions=signature["methods"],signature["corruptions"]
    expected={(i,m,c) for i in ids for m in methods for c in corruptions}
    if len(rows)!=len(expected) or set(by_key)!=expected:
        raise ValueError("Incomplete or duplicate image outputs")
    for identifier in ids:
        group=[by_key[identifier,mode,c] for mode in methods for c in corruptions]
        if any(len({r[k] for r in group})!=1 for k in ("reference_pixels_sha256","native_mask_sha256","fixed_background_sha256")):
            raise ValueError("Fixed region or reference changed between paired conditions")
        for c in corruptions:
            if len({by_key[identifier,m,c]["input_pixels_sha256"] for m in methods})!=1:
                raise ValueError("Methods received different inputs")
    def check_image(r):
        path=ROOT/r["native_float_path"]
        if file_hash(path)!=r["native_float_sha256"]:
            raise ValueError(f"Float cache changed: {path}")
        with Image.open(ROOT/r["reference_png"]) as im:
            ref=np.asarray(im.convert("L"),np.float32)/255
        arr=np.load(path,allow_pickle=False)
        err=checked_error_dn(arr,ref)
        if r["whole_mae_dn"]!=str(float(np.abs(err).mean())) or r["reference_pixels_sha256"]!=pixel_sha(ref):
            raise ValueError("Archived native grid disagrees with the native PNG or MAE")
    def check_iv(r):
        path=ROOT/r["fields_npz"]
        if file_hash(path)!=r["fields_sha256"]:
            raise ValueError("Intervention archive changed")
        key,method,c=r["sample_id"],r["method"],r["corruption"]
        clean,corrupt=[np.load(ROOT/by_key[key,method,c1]["native_float_path"],allow_pickle=False).astype(np.float64)
                       for c1 in ("clean",c)]
        with Image.open(ROOT/by_key[key,method,c]["reference_png"]) as im:
            ref=np.asarray(im.convert("L"),np.float32)/255
        with np.load(path,allow_pickle=False) as f:
            if not (np.array_equal(f["e0_dn"],255*(clean-ref)) and np.array_equal(f["d_n_dn"],255*(corrupt-clean))
                    and np.allclose(f["e0_dn"]+f["d_n_dn"],255*(corrupt-ref),atol=1e-12,rtol=0)):
                raise ValueError("Intervention fields or closure failed")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(check_image,rows):
            pass
        for _ in pool.map(check_iv,iv):
            pass
    target_keys={(t["sample_id"],t["method"],t["corruption"],t["target_id"]) for t in targets}
    if len(targets)!=sum(int(r["native_target_count"]) for r in rows) or len(target_keys)!=len(targets):
        raise ValueError("Lost or duplicated individual targets")
    if len(iv)!=(len(ids)*len(methods)*sum(c!="clean" for c in corruptions) if "clean" in corruptions else 0):
        raise ValueError("Incomplete intervention fields")
    files=("protocol_v2.json","summary.json","per_image.csv","per_target.csv","per_target.jsonl","interventions.csv")
    return dict(verification="all native float archives reread, original PNG MAE recomputed, all intervention fields independently rebuilt",
                n_sources=len(ids), n_image_rows=len(rows), n_target_rows=len(targets),
                n_interventions=len(iv), legacy_image_metrics_exact=True, fixed_background_across_methods_draws=True,
                same_input_across_methods=True, default_column_pixels_equal_legacy=True, tests=acceptance(),
                artifact_sha256={f:file_hash(output/f) for f in files})


def main():
    import fcntl
    from concurrent.futures import ThreadPoolExecutor
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition",choices=["train"])
    parser.add_argument("--endpoints")
    parser.add_argument("--corruptions")
    parser.add_argument("--metric-version",choices=[VERSION])
    parser.add_argument("--config",type=Path, help="JSON with partition, endpoints, corruptions and metric_version")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results/protocol_v2")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text()) if args.config else {}
    if set(cfg)-{"partition","endpoints","corruptions","metric_version"}:
        raise ValueError("Unknown config fields")
    methods=checked_options(args.endpoints or cfg.get("endpoints",METHODS),METHODS)
    corruptions=checked_options(args.corruptions or cfg.get("corruptions",CORRUPTIONS),CORRUPTIONS)
    if ((args.partition or cfg.get("partition","train"))!="train" or
            (args.metric_version or cfg.get("metric_version",VERSION))!=VERSION or args.workers<1):
        raise ValueError("This study evaluates only the registered primary training set under native_v2")
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/".run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.verify_only:
            verified=verify_output(output,args.workers)
            old=json.loads((output/"verification.json").read_text())
            if verified!=old:
                raise ValueError("Recorded audit and independent replay differ")
            print("Verified",verified["n_image_rows"],"images,",verified["n_interventions"],"interventions.",flush=True)
            return
        data=DenoisePairDataset(ROOT,seed=0)
        ids=[r["sample_id"] for r in data.records]
        if len(ids)!=213 or len(set(ids))!=213:
            raise ValueError("Unexpected training population")
        mh=file_hash(ROOT/"manifests/train.jsonl")
        reports,provenance=checked_caches(methods,corruptions,ids,mh)
        legacy_column=json.loads((ROOT/"results/column_baseline_extended.json").read_text())["draws"]
        split=ROOT/"datasets/NUAA-SIRST/img_idx/train_NUAA-SIRST.txt"
        code_names=("evaluate_protocol_v2.py","evaluate_warmup.py","check_column_baseline.py","prd_data.py",
                    "evaluate_sirst_endpoints.py","evaluate_retrained.py","verify_rollout.py","prd_backbone.py")
        spec=dict(metric_version=VERSION,partition="NUAA-SIRST/train",partition_path=str(split.relative_to(ROOT)),
                  partition_sha256=file_hash(split),manifest_sha256=mh,methods=methods,corruptions=corruptions,
                  n_sources=213,code_head=git_head(ROOT),vendor_head=git_head(ROOT/"vendor/Real-IISR"),
                  source_files_sha256={n:file_hash(ROOT/n) for n in code_names},
                  methods_provenance=provenance,seed=0,epoch=0,orientations=METRIC_DIRECTION,
                  noise="clean = original PNG gray. draw0/1 = fixed per-ID realization; independent uniform stripe sigma [0,12] DN and pixel sigma [0,4] DN, Gaussian offsets and pixel noise, clipping [0,1]. No injected spot.",
                  input_hash="SHA256 of contiguous native HxW float32 [0,1] bytes. Also save PNG hash, reference float hash, cached sample digest and native output NPY hash.",
                  inference="Center zero-pad to 512, native grid, cached first-index greedy; cfg=1, FP16, CUDA matmul/cuDNN TF32 off; FP32 arithmetic RGB mean, pre-AdaIN and pre-rounding.",
                  scope="Training images only; physical sensor cleanliness has no ground truth here. No fit/dev/test selection.",
                  metrics="Image-weighted means. Eight-connected native components; median ring in 2px square dilation excludes all native targets/outside native frame. All eligible target instances within each image get equal weight. Legacy four metrics retain original masks, 8-pixel minimum column and mean-ring contrast.",
                  weak_contrast="Reference-only absolute median-ring contrast in fixed bins (<10, [10,30), >=30 DN). Descriptive bins, not a calibrated physical definition.",
                  gain_minimum_reference_abs_contrast_dn=GAIN_MIN_DN,
                  perfect_psnr="whole_psnr_db=null plus explicit perfect_reconstruction status; remains +infinity in means, distinct from missing",
                  ssim="Native [0,1] grayscale float64, Gaussian 11x11 sigma 1.5, population covariance, valid windows",
                  intervention="e0=255*(F(clean)-reference), d_n=255*(F(noisy)-F(clean)), e_n=e0+d_n. Float64 DN archives store e0 and d_n; e_n is reconstructed without loss. No subtraction of scalar MAEs.")
        if (output/"protocol_v2.json").exists() and json.loads((output/"protocol_v2.json").read_text())!=spec:
            raise ValueError("Output belongs to a different protocol; select a fresh directory")
        if (output/"verification.json").exists():
            if verify_output(output,args.workers)!=json.loads((output/"verification.json").read_text()):
                raise ValueError("Existing artifacts failed verification")
            print("Verified existing outputs.",flush=True)
            return
        save_report(output/"protocol_v2.json",spec)
        for method in methods:
            for c in corruptions:
                if method in ("input","column"):
                    (output/"float"/method/c).mkdir(parents=True,exist_ok=True)
                if "clean" in corruptions and c!="clean":
                    (output/"intervention_fields"/method/c).mkdir(parents=True,exist_ok=True)
        images=[];target_rows=[];iv=[]
        def run(i):
            return score_one(i,data,methods,corruptions,reports,legacy_column,output)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i,(im,t,delta) in enumerate(pool.map(run,range(len(data))),1):
                images.extend(im);target_rows.extend(t);iv.extend(delta)
                if i%25==0:
                    print(f"Scored {i}/{len(data)} native source images.",flush=True)
        write_csv(output/"per_image.csv",images)
        write_csv(output/"per_target.csv",target_rows)
        with (output/"per_target.jsonl").open("w") as f:
            for t in target_rows:
                f.write(json.dumps(t,allow_nan=False)+"\n")
        if iv:
            write_csv(output/"interventions.csv",iv)
        else:
            (output/"interventions.csv").write_text("sample_id,method,corruption\n")
        save_report(output/"summary.json",group_summary(images,target_rows,iv,methods,corruptions))
        print("Scoring complete; verifying all native floats and intervention fields.",flush=True)
        audit=verify_output(output,args.workers)
        save_report(output/"verification.json",audit)
        print(f"Verified {len(images)} image rows, {len(target_rows)} target rows, {len(iv)} intervention fields.",flush=True)


if __name__=="__main__":
    main()
