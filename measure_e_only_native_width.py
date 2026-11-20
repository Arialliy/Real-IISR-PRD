"""Measure resolvable native apparent spot widths on uncorrupted training originals and six clean decodes."""
import argparse
import csv
import json
import math
from collections import Counter
from statistics import mean, median

import numpy as np
import torch
from scipy import ndimage

from evaluate_e_only import DEST, S1, S2, samples
from evaluate_protocol_v2 import pixel_sha
from evaluate_warmup import file_hash, save_report
from prd_backbone import PROJECT
from prd_data import DenoisePairDataset
from probe_backbone import to_gray_01


ROUTES = ("predicted_base","predicted_cac","input_base","input_cac","reference_base","reference_cac")
AREAS = ("1-4","5-16","17+")
CONTRASTS = ("lt10","10_to_lt30","ge30")


def width(patch, labels, target, ring, polarity, touches_frame):
    """Diameter of the 8-connected half-peak area, using the native reference polarity."""
    core=labels==target
    bg=float(np.median(patch[ring]))
    noise=float(1.4826*np.median(np.abs(patch[ring]-bg)))
    response=polarity*(patch-bg)
    peak=float(response[core].max())
    result=dict(peak_dn=peak, noise_mad_dn=noise, diameter_px=None, halfmax_area_px=None)
    if polarity==0:
        return dict(**result,status="undefined_reference_polarity")
    if peak <= max(1.,3*noise):
        return dict(**result,status="unresolved_peak")
    coords=np.argwhere(core & (response==peak))
    center=np.argwhere(core).mean(axis=0)
    y,x=coords[np.argmin(np.square(coords-center).sum(axis=1))]
    connected,_=ndimage.label(response>=0.5*peak, structure=np.ones((3,3),bool))
    lobe=connected==connected[y,x]
    if np.any(lobe & (labels!=0) & ~core):
        return dict(**result,status="merged_with_other_annotation")
    if response[lobe].max()>peak:
        return dict(**result,status="stronger_connected_peak_outside_core")
    if lobe[0].any() or lobe[-1].any() or lobe[:,0].any() or lobe[:,-1].any():
        return dict(**result,status="unclosed_at_frame_or_window" if touches_frame else "unclosed_at_window")
    area=int(lobe.sum())
    result.update(diameter_px=2*math.sqrt(area/math.pi),halfmax_area_px=area)
    return dict(**result,status="ok")


def measure(endpoint):
    device=torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2)
    data=DenoisePairDataset(PROJECT,seed=0)
    context=DEST/f"{endpoint}_clean_six_routes.json"
    outputs=json.loads(context.read_text())
    assert outputs["native_targets"]==270 and len(outputs["samples"])==213
    token_file=DEST/f"{endpoint}_clean_tokens.npz"
    assert file_hash(token_file)==outputs["tokens_sha256"]
    with np.load(token_file,allow_pickle=False) as f:
        cap={k:f[k] for k in f.files}
    assert np.array_equal(cap["input"],cap["reference"])
    if endpoint=="s1":
        from train_local_baseline import load_trained
        model=load_trained(S1,device)
    else:
        from train_s2_common import load_s2
        model=load_s2(S2,device)
    shapes=[x*x for x in model.patch_nums]
    all_rows=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        for i,(sample,prior,box) in enumerate(samples(data,"clean",outputs["samples"])):
            assert cap["ids"][i]==prior["sample_id"]
            h=torch.from_numpy(cap["h"][i:i+1]).to(device)
            pixels={"image_input":sample["reference_minus"][0][box]}
            for source in ("predicted","reference"):
                tokens=torch.from_numpy(cap[source][i]).to(device,dtype=torch.long)[None].split(shapes,dim=1)
                base=model.vae.idxBl_to_img(tokens,same_shape=True,last_one=True)
                cac=model.vae.idxBl_to_img_lora(tokens,h,same_shape=True,last_one=True)
                pixels[source+"_base"]=to_gray_01((base+1)*0.5).numpy()[box]
                pixels[source+"_cac"]=to_gray_01((cac+1)*0.5).numpy()[box]
            for mode in ("base","cac"):
                pixels["input_"+mode]=pixels["reference_"+mode]
            for key,image in pixels.items():
                assert pixel_sha(image)==prior["native_pixel_sha256"][key],(prior["sample_id"],key)
            labels,n=ndimage.label(sample["native_mask"][0][box],structure=np.ones((3,3),bool))
            truth=prior["native_targets"]
            assert n==len(truth["image_input"])
            for j,bounds in enumerate(ndimage.find_objects(labels)):
                target=j+1
                info=truth["image_input"][j]
                assert target==info["target_id"] and info["ring_eligible"]
                area=info["target_area_px"]
                ref_c=info["reference_contrast_dn"]
                crop=tuple(slice(max(0,b.start-4),min(extent,b.stop+4)) for b,extent in zip(bounds,labels.shape))
                touches=any(s.start==0 or s.stop==extent for s,extent in zip(crop,labels.shape))
                lab=labels[crop]
                core=lab==target
                ring=ndimage.binary_dilation(core,structure=np.ones((5,5),bool)) & (lab==0)
                assert int(core.sum())==area and int(ring.sum())==info["ring_pixels"]
                row=dict(sample_id=prior["sample_id"],target_id=target,area_px=area,
                         area_bin=AREAS[0 if area<=4 else 1 if area<=16 else 2],
                         contrast_bin=info["reference_contrast_bin_dn"],reference_contrast_dn=ref_c,
                         native_annotation_bbox_xyxy=[info[k] for k in ("x0","y0","x1_exclusive","y1_exclusive")],
                         measures={})
                for key,image in pixels.items():
                    local=image[crop].astype(np.float64)*255
                    output=truth[key][j]
                    check=float(local[core].mean()-np.median(local[ring]))
                    assert abs(check-output["prediction_contrast_dn"])<1e-9
                    result=width(local,lab,target,ring,float(np.sign(ref_c)),touches)
                    result.update(contrast_dn=output["prediction_contrast_dn"],contrast_bias_dn=output["contrast_bias_dn"],
                                  contrast_abs_error_dn=output["contrast_abs_error_dn"])
                    row["measures"][key]=result
                all_rows.append(row)
            if (i+1)%70==0:
                print(f"{endpoint}: measured native contrast and resolvable width on {i+1} of 213 images",flush=True)
    assert len(all_rows)==270
    save_report(DEST/f"{endpoint}_clean_native_points.json",dict(endpoint=endpoint,
        native_v2_clean_sha256=file_hash(context), token_archive_sha256=file_hash(token_file),
        protocol="On unpadded native originals, retain reference polarity from 8-connected annotated core mean minus median of its 2px square background ring without any annotation. For each route take strongest same-polarity pixel in core, ring-relative; require peak > max(1 DN, 3*1.4826*MAD(ring)). Use the 8-connected half-height lobe in the annotation bounding box expanded by 4 native pixels per side. Require lobe closed inside window/frame, no other labeled component and no stronger connected peak outside core. Tied peaks use closest to the native core centroid. Report equivalent circular diameter 2*sqrt(lobe_pixel_area/pi) in native pixels. All failures are explicit missing values. Pair widths only if both input and output measurable. Contrast is always reported separately for every eligible annotation. Real scene measurements describe apparent width; there is no latent physical PSF truth.",
        n_images=213,n_annotations=len(all_rows),samples=all_rows))


def summarize():
    rows=[]
    inputs={}
    for endpoint in ("s1","s2"):
        items=json.loads((DEST/f"{endpoint}_clean_native_points.json").read_text())["samples"]
        for target in items:
            key=target["sample_id"],target["target_id"]
            if endpoint=="s1":
                inputs[key]=target["measures"]["image_input"]
            else:
                assert inputs[key]==target["measures"]["image_input"]
        for area in (*AREAS,"all"):
            for cbin in (*CONTRASTS,"all"):
                selected=[t for t in items if (area=="all" or t["area_bin"]==area) and (cbin=="all" or t["contrast_bin"]==cbin)]
                ref=[t["measures"]["image_input"] for t in selected]
                for name in ROUTES:
                    rec=[t["measures"][name] for t in selected]
                    pairs=[(a["diameter_px"],b["diameter_px"]) for a,b in zip(ref,rec,strict=True) if a["status"]==b["status"]=="ok"]
                    diffs=[b-a for a,b in pairs]
                    rows.append(dict(endpoint=endpoint,route=name,area_bin=area,reference_contrast_bin_dn=cbin,n_labels=len(selected),
                                     n_images=len({t["sample_id"] for t in selected}),n_reference_width_measurable=sum(a["status"]=="ok" for a in ref),
                                     n_output_width_measurable=sum(a["status"]=="ok" for a in rec),n_paired_widths=len(pairs),
                                     mean_reference_contrast_dn=mean(a["contrast_dn"] for a in ref) if ref else None,
                                     mean_output_contrast_dn=mean(a["contrast_dn"] for a in rec) if rec else None,
                                     mean_contrast_bias_dn=mean(a["contrast_bias_dn"] for a in rec) if rec else None,
                                     mean_contrast_abs_error_dn=mean(a["contrast_abs_error_dn"] for a in rec) if rec else None,
                                     paired_median_reference_width_px=median(a for a,b in pairs) if pairs else None,
                                     paired_median_output_width_px=median(b for a,b in pairs) if pairs else None,
                                     paired_median_width_change_px=median(diffs) if pairs else None,
                                     paired_mean_width_change_px=mean(diffs) if pairs else None,
                                     paired_mean_abs_width_change_px=mean(map(abs,diffs)) if pairs else None,
                                     reference_status_counts=json.dumps(dict(Counter(a["status"] for a in ref)),sort_keys=True),
                                     output_status_counts=json.dumps(dict(Counter(a["status"] for a in rec)),sort_keys=True)))
    with (DEST/"clean_native_point_strata.csv").open("w",newline="") as out:
        w=csv.DictWriter(out,list(rows[0])); w.writeheader(); w.writerows(rows)
    print("Verified clean point records and wrote target-weighted size × reference contrast strata.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint",choices=["s1","s2","summarize"])
    choice=parser.parse_args().endpoint
    if choice=="summarize":
        summarize()
    else:
        measure(choice)
