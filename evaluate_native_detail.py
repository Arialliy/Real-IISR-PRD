"""Fixed-weight native detail experiment on the bundled SIRST training images."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_noise_levels import FIELDS, dataset, get_model, get_sample, write_csv
from evaluate_protocol_v2 import measure_native, native_window, pixel_sha
from evaluate_warmup import file_hash, save_report
from native_detail import iter_methods, predict_s1_native

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results/denoise_v2/native_detail"
NATIVE = OUT/"native"
CONFIG = ROOT/"configs/denoise_v2/native_detail.json"
SPEC = json.loads(CONFIG.read_text())
SENSITIVITY = SPEC["additional_noise_conditions"]
CONDITIONS = [*SPEC["paired_conditions"], *SENSITIVITY]
METHODS = SPEC["controls"]
CONTRAST = "native_median_ring_contrast_abs_error_dn"
COLUMN = "weighted_background_column_residual_dn"
BACKGROUND = "plan_background_rmse_dn"


def legacy_controls(condition):
    if condition in ("fresh","combined"):
        level = "fresh" if condition=="fresh" else "medium"
        controls = {}
        for name in ("column","s1"):
            shard = ROOT/f"results/protocol_v2_levels/shards/{name}_{level}.json"
            for r in json.loads(shard.read_text())["images"]:
                if r["method"] in ("input","column_1","s1"):
                    method = "column" if r["method"]=="column_1" else r["method"]
                    controls[(r["sample_id"],method)] = r
        return controls
    if condition in ("clean","draw0","draw1"):
        with (ROOT/"results/protocol_v2/per_image.csv").open(newline="") as f:
            return {(r["sample_id"], "s1" if r["method"]=="s1_retrained" else r["method"]):r
                    for r in csv.DictReader(f) if r["corruption"]==condition
                    and r["method"] in ("input","column","s1_retrained")}
    return {}


def collect(condition):
    data = dataset()
    if file_hash(ROOT/"manifests/train.jsonl")!=SPEC["train_manifest_sha256"]:
        raise ValueError("Wrong training population")
    if condition in SENSITIVITY:
        model,device,checkpoint_hash = get_model("s1")
        if checkpoint_hash!=SPEC["s1_sha256"]:
            raise ValueError("S1 checkpoint differs from the registered baseline")
    elif condition=="fresh":
        cap = json.loads((OUT/"capture/fresh.json").read_text())
        if [r["sample_id"] for r in cap["samples"]]!=[r["sample_id"] for r in data.records]:
            raise ValueError("Incomplete fresh capture")
    else:
        suffix = condition.removeprefix("draw")
        endpoint = json.loads((ROOT/f"results/sirst_endpoints/s1_draw{suffix}.json").read_text())
        if endpoint["checkpoint_sha256"]!=SPEC["s1_sha256"]:
            raise ValueError("Incorrect S1 archive")
    historical = legacy_controls(condition)
    images,targets=[],[]
    verified=0
    for i,rec in enumerate(data.records):
        if condition in SENSITIVITY:
            sample = data.sample(i,epoch=0,draw_id=SPEC["additional_noise_draw"],noise_override=SENSITIVITY[condition])
        else:
            sample = get_sample(data,i,condition)
        box = native_window(sample)
        x = sample["noisy_minus"][0][box]
        identifier = rec["sample_id"]
        if condition in SENSITIVITY:
            s = predict_s1_native(model,x,device)
        elif condition=="fresh":
            with np.load(OUT/"capture"/condition/(rec["source_image_id"]+".npz"),allow_pickle=False) as archive:
                if not np.array_equal(x,archive["x_minus"]):
                    raise ValueError("Fresh paired inputs disagree")
                s = archive["s1_minus"]
        else:
            entry = endpoint["samples"][i]
            path = ROOT/"results/sirst_endpoints"/entry["native_gray_npy"]
            if entry["sample_id"]!=identifier or file_hash(path)!=entry["native_gray_sha256"]:
                raise ValueError("Native archive hash or ID changed")
            s = np.load(path,allow_pickle=False)
        if x.shape!=s.shape or s.dtype!=np.float32:
            raise ValueError("Wrong stored grid or output dtype")
        x_sha = pixel_sha(x)
        for method,output in iter_methods(x,s):
            padded = np.zeros_like(sample["noisy_minus"][0])
            padded[box] = output
            scores,details = measure_native(sample,padded,details=True)
            ident = dict(condition=condition,method=method,sample_id=identifier)
            row = dict(**ident,input_sha256=x_sha,prediction_sha256=pixel_sha(output),
                       **{field:scores[field] for field in FIELDS})
            if historical and method in ("input","column","s1"):
                previous = historical[(identifier,method)]
                if previous.get("input_pixels_sha256",previous.get("input_sha256"))!=x_sha:
                    raise ValueError("Native input disagrees with archived control")
                for field in FIELDS:
                    old=previous[field]
                    if old=="" or old is None:
                        if row[field] is not None:
                            raise ValueError(f"Changed missing value in {field}")
                    elif not np.isclose(float(old),row[field],rtol=0,atol=1e-10):
                        raise ValueError(f"Control score disagreement for {identifier} {method} {field}")
                if "prediction_sha256" in previous and previous["prediction_sha256"]!=row["prediction_sha256"]:
                    raise ValueError("Recomputed native pixels differ from previous outputs")
                verified+=1
            images.append(row)
            targets.extend(ident|r for r in details)
    if verified!=len(historical):
        raise ValueError("Not all archived controls were cross-checked")
    save_report(NATIVE/f"{condition}.json",dict(condition=condition,config_sha256=file_hash(CONFIG),
                operator_sha256=file_hash(ROOT/"native_detail.py"),legacy_control_rows_verified=verified,
                images=images,targets=targets))
    print(f"{condition}: {len(images)} image records, {len(targets)} component records, {verified} checked controls",flush=True)


def summarize():
    data=dataset()
    expected=[r["sample_id"] for r in data.records]
    images,targets,means,intervals,overlap=[],[],[],[],[]
    grouped=defaultdict(list)
    verified=0
    for c in CONDITIONS:
        shard=json.loads((NATIVE/f"{c}.json").read_text())
        if shard["config_sha256"]!=file_hash(CONFIG) or shard["operator_sha256"]!=file_hash(ROOT/"native_detail.py"):
            raise ValueError("Source or config changed during the experiment")
        verified+=shard["legacy_control_rows_verified"]
        images+=shard["images"]
        targets+=shard["targets"]
        for r in shard["images"]:
            grouped[c,r["method"]].append(r)
    for (c,method),rows in grouped.items():
        if [r["sample_id"] for r in rows]!=expected:
            raise ValueError("Missing paired train images")
        vals={}
        for k in FIELDS:
            ok=[r[k] for r in rows if r[k] is not None]
            perfect = k=="whole_psnr_db" and len(ok)!=len(rows)
            vals[k]=float(np.mean(ok)) if ok and not perfect else None
        means.append(dict(condition=c,method=method,images=len(rows),
                         perfect_psnr_images=sum(r["whole_psnr_db"] is None for r in rows),**vals))
    rng=np.random.default_rng(20261121)
    bootstrap=rng.integers(len(data),size=(5000,len(data)))
    for c in CONDITIONS:
        for before,after in (("s1","detail_full"),("column","detail_full"),("full_blend_half","detail_half")):
            pairs=list(zip(grouped[c,before],grouped[c,after],strict=True))
            assert all(a["input_sha256"]==b["input_sha256"] for a,b in pairs)
            for k in (CONTRAST,BACKGROUND,COLUMN,"whole_mae_dn","plan_background_mean_error_dn"):
                diff=np.array([b[k]-a[k] for a,b in pairs])
                lo,hi=np.percentile(diff[bootstrap].mean(axis=1),[2.5,97.5])
                intervals.append(dict(condition=c,baseline=before,candidate=after,metric=k,
                                      mean_delta=diff.mean(),ci025=lo,ci975=hi))
            matched=[(a,b) for a,b in pairs if abs(a[COLUMN]-b[COLUMN])<=0.5 and abs(a[BACKGROUND]-b[BACKGROUND])<=1.]
            overlap.append(dict(condition=c,baseline=before,candidate=after,n_matched=len(matched),
                                mean_contrast_delta=(float(np.mean([b[CONTRAST]-a[CONTRAST] for a,b in matched]))
                                                     if matched else None),
                                matched_ids=[a["sample_id"] for a,b in matched]))
    write_csv(NATIVE/"per_image.csv",images)
    write_csv(NATIVE/"per_target.csv",targets)
    write_csv(NATIVE/"means.csv",means)
    write_csv(NATIVE/"paired_intervals.csv",intervals)
    save_report(NATIVE/"background_overlap.json",overlap)
    save_report(NATIVE/"verification.json",dict(train_images=len(data),conditions=CONDITIONS,
                methods=METHODS,images=len(images),targets=len(targets),legacy_control_rows_verified=verified,
                config_sha256=file_hash(CONFIG),operator_sha256=file_hash(ROOT/"native_detail.py")))


def real_pairs():
    from evaluate_tidy_control import scan_frames, SCAN_HASH
    metrics=("mae_dn","rmse_dn","col_residual_dn","mean_signed_dn")
    previous={}
    for method in ("s1","input","column"):
        path=ROOT/f"results/denoise_v2/tidy_control/scan_{method}_frames.csv"
        with path.open(newline="") as f:
            previous.update({(r["frame"],method):r for r in csv.DictReader(f)})
    model,device,sha=get_model("s1")
    if sha!=SPEC["s1_sha256"]:
        raise ValueError("Unexpected S1 checkpoint")
    rows=[]
    for frame,x,y in scan_frames():
        s=predict_s1_native(model,x,device)
        for method,pred in iter_methods(x,s):
            d=255*(pred.astype(np.float64)-y)
            r=dict(frame=frame,method=method,input_sha256=pixel_sha(x),reference_sha256=pixel_sha(y),
                   output_sha256=pixel_sha(pred),mae_dn=np.abs(d).mean(),rmse_dn=np.sqrt(np.square(d).mean()),
                   col_residual_dn=d.mean(axis=0).std(),mean_signed_dn=d.mean())
            if (frame,method) in previous:
                old=previous[frame,method]
                assert all(r[k]==old[k] for k in ("input_sha256","reference_sha256","output_sha256"))
                assert all(abs(r[k]-float(old[k]))<1e-10 for k in metrics)
            rows.append(r)
    OUT.mkdir(exist_ok=True,parents=True)
    write_csv(OUT/"scan_per_frame.csv",rows)
    summary={m:{k:np.mean([r[k] for r in rows if r["method"]==m]) for k in metrics} for m in METHODS}
    save_report(OUT/"scan.json",dict(sequence="300_floor5, same 32 train-side frames and central native 512 crop",
                registered_nuaa_config_sha256=file_hash(CONFIG),checkpoint_sha256=sha,
                archive_sha256=SCAN_HASH,control_hashes_reproduced=len(previous),means=summary))


def noise_response():
    """Subtract paired clean output to measure what the extra noise changed."""
    from evaluate_warmup import fixed_background
    data=dataset()
    measures=[]
    for i,record in enumerate(data.records):
        sample=get_sample(data,i,"clean")
        bg=fixed_background(sample)[native_window(sample)]
        counts=bg.sum(axis=0)
        use=counts>0
        filename=record["source_image_id"]+".npz"
        with np.load(OUT/"capture/clean"/filename,allow_pickle=False) as archive:
            reference=dict(iter_methods(archive["x_minus"],archive["s1_minus"]))
        for c in ("draw0","draw1","fresh"):
            with np.load(OUT/"capture"/c/filename,allow_pickle=False) as archive:
                for method,pred in iter_methods(archive["x_minus"],archive["s1_minus"]):
                    delta=(pred.astype(np.float64)-reference[method])*255.
                    col=(delta*bg).sum(axis=0)[use]/counts[use]
                    centered=col-np.average(col,weights=counts[use])
                    measures.append(dict(sample_id=record["sample_id"],condition=c,method=method,
                        bg_change_rmse_dn=np.sqrt(np.mean(delta[bg]**2)),
                        bg_change_column_sd_dn=np.sqrt(np.average(centered**2,weights=counts[use]))))
    write_csv(NATIVE/"noise_response.csv",measures)
    means=[]
    for c in ("draw0","draw1","fresh"):
        for m in METHODS:
            group=[r for r in measures if r["condition"]==c and r["method"]==m]
            means.append(dict(condition=c,method=m,n=len(group),
                              **{k:np.mean([r[k] for r in group]) for k in ("bg_change_rmse_dn","bg_change_column_sd_dn")}))
    write_csv(NATIVE/"noise_response_means.csv",means)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=[*CONDITIONS,"summary","scan","noise_response"])
    args=parser.parse_args()
    if args.action=="summary":
        summarize()
    elif args.action=="scan":
        real_pairs()
    elif args.action=="noise_response":
        noise_response()
    else:
        collect(args.action)
