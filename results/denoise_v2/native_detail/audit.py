"""Audit the native S1 detail branch from raw frames and from historical benchmark rows."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

CONDITIONS = ("clean", "draw0", "draw1", "fresh")
METHODS = ("input", "s1", "column", "detail_half", "detail_full", "full_blend_half")
FIELDS = ("whole_mae_dn", "plan_background_mean_error_dn", "plan_background_rmse_dn",
          "weighted_background_column_residual_dn", "native_median_ring_contrast_abs_error_dn",
          "native_median_ring_contrast_bias_dn", "native_target_mae_dn", "whole_psnr_db")
PROBES = (0, 17, 53, 106, 159, 212)
CHECKPOINT_DIGEST = "fb2eaf0426e1634b8cde12c7d8951ded9d6b4f04a0ae2ef77a1c13f794caacab"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixels(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def smooth_vertical(residual):
    # Half-sample symmetry: ... b a | a b c d | d c ...
    padded = np.pad(residual.astype(np.float64), ((2, 2), (0, 0)), mode="symmetric")
    height = residual.shape[0]
    total = sum(w * padded[k:k+height] for k, w in enumerate((1,4,6,4,1)))
    return (total / 16).astype(residual.dtype)


def restore(x, base, alpha):
    residual = x - base
    return np.clip(base + alpha * (residual - smooth_vertical(residual)), 0, 1)


def column_correct(x):
    padded = np.pad(x, ((0,0), (3,3)), mode="symmetric")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 7, axis=1)
    bias = np.median(x - np.median(windows, axis=-1), axis=0)
    return np.clip(x - (bias - np.median(bias)), 0, 1)


def analytic_checks():
    from native_detail import restore_native_detail
    rng = np.random.default_rng(450921)
    shapes = ((1,1), (1,7), (7,1), (2,4), (3,2), (16,1), (19,31))
    max_abs_error = 0.
    for shape in shapes:
        x, base = (rng.random(shape, dtype=np.float32) for _ in range(2))
        for alpha in (0., 0.5, 1.):
            expected = restore(x, base, alpha)
            output = restore_native_detail(x, base, strength=alpha)
            assert output.shape == shape and output.dtype == np.float32
            assert np.all(np.isfinite(output)) and output.min() >= 0 and output.max() <= 1
            np.testing.assert_allclose(output, expected, rtol=0, atol=2e-7)
            np.testing.assert_array_equal(restore_native_detail(x, base, strength=alpha), output)
            if alpha == 0 or shape[0] == 1:
                np.testing.assert_allclose(output, base, rtol=0, atol=1e-7)
            max_abs_error = max(max_abs_error, float(np.max(np.abs(output-expected))))
    base = np.full((9, 11), .5, dtype=np.float32)
    columns = base + np.arange(11, dtype=np.float32)[None, :]/32 - .125
    np.testing.assert_array_equal(restore_native_detail(columns, base, strength=1.), base)
    np.testing.assert_array_equal(restore_native_detail(base, base, strength=1.), base)
    # A one-pixel row or point has the expected signed high-pass response.
    row = base.copy()
    row[4, :] += .125
    expected = np.zeros_like(row)
    expected[2:7] = np.array([-1,-4,10,-4,-1])[:,None] / 128
    np.testing.assert_array_equal(restore_native_detail(row, base, strength=1.)-base, expected)
    point = base.copy()
    point[4, 5] += .125
    expected[:, :5] = 0.
    expected[:, 6:] = 0.
    np.testing.assert_array_equal(restore_native_detail(point, base, strength=1.)-base, expected)
    residual = rng.uniform(-.05, .05, (29, 23)).astype(np.float32)
    base = np.full_like(residual, .5)
    output = restore_native_detail(base + residual, base, strength=1.)
    max_column_sum = float(np.abs((output-base).astype(float).sum(0)).max())
    assert max_column_sum <= 2e-6
    # Saturation is allowed to change a column mean. This is a concrete counterexample.
    base = np.array([[.01],[.95],[.01],[.95]], dtype=np.float32)
    x = 1.-base
    raw = base + ((x-base) - smooth_vertical(x-base))
    assert np.any((raw<0)|(raw>1))
    np.testing.assert_array_equal(restore_native_detail(x,base,strength=1.),raw.clip(0.,1.))
    # The clipping example is symmetric overall; a single low-end peak also
    # demonstrates how one-sided clipping changes the native column total.
    base = np.zeros((7,1),np.float32)
    x = base.copy()
    x[3] = .5
    assert abs(restore_native_detail(x,base,strength=1.).sum() - .3125) < 1e-7
    return dict(shape_cases=list(shapes), strength_cases=[0,.5,1],
                maximum_independent_pixel_difference_0to1=max_abs_error,
                constant_and_arbitrary_column_only_inputs_pass=True,
                interior_point_and_row_impulse_pass=True,
                post_clip_column_sum_counterexample=.3125,
                max_unclipped_column_sum_abs_unit=max_column_sum,
                repeated_outputs_bitwise_identical=True,
                conclusion="Zero column sum applies to the full native grid before clipping; masked background means can change.")


def row_index(path, keys):
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    index = {tuple(row[k] for k in keys):row for row in rows}
    if len(rows) != len(index):
        raise ValueError(f"Repeated keys in {path}")
    return index


def same_number(expected, saved, tol=1e-8):
    if expected is None:
        assert saved is None or saved == "", (expected,saved)
    else:
        np.testing.assert_allclose(float(saved), expected, rtol=0, atol=tol)


def original_reference_and_mask(record):
    with Image.open(ROOT/record["reference_path"]) as f:
        y=np.asarray(f.convert("L"),dtype=np.float32)/255
    with Image.open(ROOT/record["mask_path"]) as f:
        mask=np.asarray(f.convert("L"))!=0
    assert y.shape == mask.shape and list(y.shape[::-1]) == record["reference_size"]
    return y,mask


def component_regions(mask):
    cc, _ = ndimage.label(mask, structure=np.ones((3,3),bool))
    boxes=[]
    for number, box in enumerate(ndimage.find_objects(cc), 1):
        area = tuple(slice(max(0,s.start-2), min(n,s.stop+2)) for s,n in zip(box,mask.shape,strict=True))
        core=cc[area]==number
        ring=(ndimage.maximum_filter(core.astype(np.uint8),size=5, mode="constant")>0) & ~mask[area]
        if np.any(ring):
            boxes.append((area,core,ring))
    return boxes


def measured(y, pred, mask, regions):
    y,pred = (q.astype(np.float64)*255 for q in (y,pred))
    e=pred-y
    bg=ndimage.maximum_filter(mask.astype(np.uint8),size=5,mode="constant")==0
    counts=bg.sum(0)
    col_mean=(e*bg).sum(0)[counts>0]/counts[counts>0]
    stripe=np.sqrt(np.average((col_mean-np.average(col_mean,weights=counts[counts>0]))**2,weights=counts[counts>0]))
    delta=[]
    for area,core,ring in regions:
        local_a,local_b = y[area],pred[area]
        delta.append((local_b[core].mean()-np.median(local_b[ring])) -
                     (local_a[core].mean()-np.median(local_a[ring])))
    mae=np.mean(abs(e))
    return dict(whole_mae_dn=mae, plan_background_mean_error_dn=e[bg].mean(),
                plan_background_rmse_dn=np.sqrt(np.mean(e[bg]**2)),weighted_background_column_residual_dn=stripe,
                native_median_ring_contrast_abs_error_dn=np.mean(np.abs(delta)),
                native_median_ring_contrast_bias_dn=np.mean(delta),native_target_mae_dn=np.mean(abs(e[mask])),
                whole_psnr_db=10.*np.log10(255**2/np.mean(e**2)) if mae>0 else None)


def candidates(x,base):
    return {"input":x,"s1":base,"column":column_correct(x),"detail_half":restore(x,base,.5),
            "detail_full":restore(x,base,1.),"full_blend_half":.5*x+.5*base}


def fresh_noise(y, name, draw):
    tokens = json.dumps([0,name,0,draw,"noise"]).encode()
    seed = int.from_bytes(hashlib.sha256(tokens).digest()[:8], "big")>>1
    noise = np.random.default_rng(seed)
    stripe,pixel = float(noise.uniform(0,12)),float(noise.uniform(0,4))
    bias = noise.standard_normal((1,y.shape[1]),dtype=np.float32)*(stripe/255.)
    pixel_field=noise.standard_normal(y.shape,dtype=np.float32)*(pixel/255.)+bias
    return np.clip(y+pixel_field,0,1)


def spot_metrics(a,b,core,ring):
    sys.path.insert(0,str(OUT.parent/"spot_transfer"))
    from audit_capture import direct_contrast, direct_width
    yy,xx = np.where(core|ring)
    area = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
    core,ring=core[area],ring[area]
    ad,bd=(arr[area].astype(np.float64)*255 for arr in (a,b))
    c=direct_contrast(bd,core,ring)-direct_contrast(ad,core,ring)
    width,reason=direct_width(bd-ad,c,core,ring)
    return c,width,reason


def audit_tables():
    from native_detail import restore_native_detail
    manifest = ROOT/"manifests/train.jsonl"
    records=[json.loads(l) for l in manifest.read_text().splitlines()]
    ids=[r["sample_id"] for r in records]
    lists=ROOT/"datasets/NUAA-SIRST/img_idx"
    train=(lists/"train_NUAA-SIRST.txt").read_text(encoding="utf-8-sig").split()
    held=(lists/"test_NUAA-SIRST.txt").read_text(encoding="utf-8-sig").split()
    assert [r["source_image_id"] for r in records]==train and all(r["split"]=="train" for r in records)
    assert len(ids)==len(set(ids))==213 and len(held)==len(set(held))==214 and set(train).isdisjoint(held)
    native_path, spot_path, means_path = OUT/"native/per_image.csv",OUT/"spots/per_pair.csv",OUT/"native/means.csv"
    nall=row_index(native_path, ("condition","method","sample_id"))
    spots=row_index(spot_path, ("condition","method","sample_id"))
    means=row_index(means_path, ("condition","method"))
    spot_summary_path=OUT/"spots/summary.json"
    spot_summary=json.loads(spot_summary_path.read_text())
    hist_main=row_index(ROOT/"results/protocol_v2/per_image.csv",("corruption","method","sample_id"))
    hist_level=row_index(ROOT/"results/protocol_v2_levels/per_image.csv",("condition","method","sample_id"))
    old_spots=row_index(OUT.parent/"spot_transfer/per_pair_stage.csv",("sample_id","route"))
    keys={(c,m,i) for c in CONDITIONS for m in METHODS for i in ids}
    assert keys <= set(nall) and keys==set(spots)
    expected_keys={(m,i) for m in METHODS for i in ids}
    for c in CONDITIONS:
        assert {(m,i) for cond,m,i in nall if cond==c}==expected_keys
        for m in METHODS:
            actual=spot_summary[c][m]
            spot_group=[spots[c,m,i] for i in ids]
            gains=np.asarray([float(r["retention_ratio"]) for r in spot_group])
            measurable=sum(r["width_reason"]=="measured" for r in spot_group)
            assert actual["retained_at_least_half_n"]==int((gains>=.5).sum())
            assert actual["negative_gain_n"]==int((gains<0).sum())
            same_number(float(np.median(gains)), actual["retention_ratio"]["median"])
            assert actual["width_px"]["n"]==measurable
            assert all((r["width_reason"]=="measured")==bool(r["width_px"]) for r in spot_group)
            for field in (*FIELDS, "whole_ssim"):
                raw=[nall[c,m,i][field] for i in ids]
                if field=="whole_psnr_db" and "" in raw:
                    expected=None
                else:
                    finite=[float(v) for v in raw if v!=""]
                    expected=np.mean(finite) if finite else None
                same_number(expected,means[c,m][field])
    checkpoint=ROOT/"checkpoints/sirst_retrained_baseline.pt"
    assert sha(checkpoint)==CHECKPOINT_DIGEST
    rows_checked,fields_checked,spot_checked,history_checked=0,0,0,0
    max_pixels=0.
    for cond in CONDITIONS:
        cap=json.loads((OUT/f"capture/{cond}.json").read_text())
        assert [s["sample_id"] for s in cap["samples"]]==ids
        for index,(r,item) in enumerate(zip(records,cap["samples"],strict=True)):
            ident=r["sample_id"]
            # Compare all control rows against the old table, irrespective of re-reading selected frames.
            for new,old in (("input","input"),("s1","s1_retrained"),("column","column")):
                if cond=="fresh":
                    old_name={"s1":"s1","column":"column_1","input":"input"}[new]
                    other=hist_level[cond,old_name,ident]
                else:
                    other=hist_main[cond,old,ident]
                for field in FIELDS:
                    v=other[field]
                    same_number(float(v) if v!="" else None,nall[cond,new,ident][field])
                history_checked+=1
            data=OUT/"capture"/cond/(r["source_image_id"]+".npz")
            assert sha(data)==item["archive_sha256"]
            with np.load(data, allow_pickle=False) as z:
                x,base=z["x_minus"],z["s1_minus"]
                q=hist_level[cond,"s1",ident] if cond=="fresh" else hist_main[cond,"s1_retrained",ident]
                if cond=="fresh":
                    assert pixels(x)==q["input_sha256"] and pixels(base)==q["prediction_sha256"]
                else:
                    assert pixels(x)==q["input_pixels_sha256"]
                    archived=ROOT/q["native_float_path"]
                    assert sha(archived)==q["native_float_sha256"]
                    np.testing.assert_array_equal(base,np.load(archived,allow_pickle=False))
                if index not in PROBES:
                    continue
                y,mask=original_reference_and_mask(r)
                dose=z["x_plus"]
                out_plus=z["s1_plus"]
                core,ring=z["core"],z["ring"]
            regions=component_regions(mask)
            expected=y if cond=="clean" else fresh_noise(y,ident,{"draw0":0,"draw1":1,"fresh":20261121}[cond])
            np.testing.assert_array_equal(x,expected)
            outputs=candidates(x,base)
            plus_outputs=candidates(dose,out_plus)
            clean_spot=float(old_spots[ident,"reference_pair"]["incremental_contrast_dn"])
            source_strength=spot_metrics(x,dose,core,ring)[0]
            for method,pred in outputs.items():
                row=nall[cond,method,ident]
                if pixels(pred)!=row["prediction_sha256"]:
                    raise ValueError(f"Independent full frame differs for {cond}/{method}/{ident}")
                for metric,value in measured(y,pred,mask,regions).items():
                    same_number(value,row[metric],tol=1e-5)
                    fields_checked+=1
                if method.startswith("detail_"):
                    alpha=1. if method=="detail_full" else .5
                    actual=restore_native_detail(x,base,strength=alpha)
                    diff=float(abs(actual-pred).max())
                    assert diff<2e-7
                    max_pixels=max(max_pixels,diff)
                v,width,reason=spot_metrics(pred,plus_outputs[method],core,ring)
                spot=spots[cond,method,ident]
                same_number(v,spot["incremental_contrast_dn"],tol=5e-5)
                same_number(v/clean_spot,spot["retention_ratio"],tol=2e-4)
                if source_strength>0:
                    same_number(v/source_strength,spot["retention_vs_input_ratio"],tol=2e-4)
                same_number(width,spot["width_px"])
                assert reason==spot["width_reason"],(ident,cond,method,reason,spot["width_reason"])
                rows_checked+=1
                spot_checked+=1
    return dict(train_n=213,reserved_test_manifest_n=214,train_test_id_overlap=0,
                selected_original_indices=list(PROBES),selected_sample_ids=[ids[i] for i in PROBES],
                capture_file_hashes_checked=len(ids)*len(CONDITIONS),
                matched_s1_native_frames=len(ids)*len(CONDITIONS),
                independently_rebuilt_with_original_PNG_or_seeded_noise=len(PROBES)*len(CONDITIONS),
                historical_input_s1_column_rows_verified=history_checked,
                native_rows_recomputed=rows_checked,independent_full_frame_hashes=rows_checked,
                independent_native_metric_values=fields_checked,
                artificial_pair_rows_remeasured=spot_checked,full_group_native_means_checked=len(CONDITIONS)*len(METHODS),
                spot_group_medians_counts_and_coverage_checked=len(CONDITIONS)*len(METHODS),
                checked_metric_names=list(FIELDS),max_independent_real_pixel_difference_0to1=max_pixels,
                sha256={"checkpoint":CHECKPOINT_DIGEST,"native_table":sha(native_path),"spot_table":sha(spot_path),
                        "means":sha(means_path),"train_manifest":sha(manifest)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-only",action="store_true")
    args=parser.parse_args()
    result={"analytic":analytic_checks()}
    if not args.analytic_only:
        result["real_frames"]=audit_tables()
        tmp=OUT/"audit.pending"
        tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        tmp.replace(OUT/"audit.json")
    print(json.dumps(result))


if __name__=="__main__":
    main()
