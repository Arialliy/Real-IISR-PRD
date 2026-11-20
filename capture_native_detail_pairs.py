"""Capture native S1 input/output pairs using the previously registered clean spots."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_noise_levels import dataset, get_model
from evaluate_protocol_v2 import native_window, pixel_sha
from evaluate_retrained import hard_prediction
from evaluate_warmup import file_hash, save_report

ROOT = Path(__file__).resolve().parent
CAPTURE = ROOT / "results/denoise_v2/native_detail/capture"
SPOTS = ROOT / "results/denoise_v2/spot_transfer"
ENDPOINTS = ROOT / "results/sirst_endpoints"
DRAWS = {"clean": 0, "draw0": 0, "draw1": 1, "fresh": 20261121}
ARRAYS = ("x_minus", "x_plus", "s1_minus", "s1_plus", "core", "ring")


def load_pair(condition, row):
    path = CAPTURE / condition / f"{row['source_image_id']}.npz"
    if (str(path.relative_to(ROOT)) != row["path"] or file_hash(path) != row["archive_sha256"]):
        raise ValueError(f"Pair archive identity differs: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAYS):
            raise ValueError(f"Incomplete pair archive: {path}")
        return {key: archive[key] for key in ARRAYS}


def contrast_increment(minus, plus, core, ring):
    def contrast(a):
        a = a.astype(np.float64)*255.
        return float(a[core].mean() - np.median(a[ring]))
    return contrast(plus) - contrast(minus)


def get_controls(condition):
    if condition == "fresh":
        path = ROOT / "results/protocol_v2_levels/shards/s1_fresh.json"
        old = json.loads(path.read_text())
        return path, old["checkpoint_sha256"], old["images"], {}
    label = "clean" if condition == "clean" else str(DRAWS[condition])
    path = ENDPOINTS / f"s1_draw{label}.json"
    old = json.loads(path.read_text())
    with (ROOT / "results/protocol_v2/per_image.csv").open(newline="") as f:
        input_hashes = {r["sample_id"]: r["input_pixels_sha256"] for r in csv.DictReader(f)
                        if r["method"] == "input" and r["corruption"] == condition}
    return path, old["checkpoint_sha256"], old["samples"], input_hashes


def verify_clean_patch(spot_archive, input_pair, output_pair, core, ring):
    path = SPOTS / spot_archive["patch_file"]
    if file_hash(path) != spot_archive["patch_sha256"]:
        raise ValueError("Previous clean spot archive hash differs")
    with np.load(path, allow_pickle=False) as old:
        x, y = spot_archive["crop_x0"], spot_archive["crop_y0"]
        h, w = old["core"].shape
        crop = np.s_[y:y+h, x:x+w]
        for current, name in zip((*input_pair,*output_pair,core,ring),
                                 ("reference_pair_minus","reference_pair_plus","s1_deployment_minus",
                                  "s1_deployment_plus","core","ring"), strict=True):
            if not np.array_equal(current[crop], old[name]):
                raise ValueError(f"Previous clean spot patch differs: {path.name}/{name}")
    diff = (output_pair[1].astype(np.float64)-output_pair[0])*255
    stats = spot_archive["full_frame"]["s1_deployment"]
    if (np.count_nonzero(diff==0) != stats["equal_pixels"] or diff.size != stats["pixels"] or
            np.abs(diff).max() != stats["max_abs_change_dn"] or
            np.sqrt(np.square(diff).mean()) != stats["rmse_dn"]):
        raise ValueError("Previous full-frame clean deployment difference does not reproduce")


def run(condition):
    data = dataset()
    specs = json.loads((SPOTS / "protocol.json").read_text())
    old_spots = json.loads((SPOTS / "pairs.json").read_text())["samples"]
    historic_path, old_model, controls, control_hashes = get_controls(condition)
    manifest_hash = file_hash(ROOT/"manifests/train.jsonl")
    model, device, ckpt_hash = get_model("s1")
    if old_model != ckpt_hash or specs["source_manifest_sha256"] != manifest_hash:
        raise ValueError("S1 or original training manifest differs from previous controls")
    records = data.records
    ids = [r["sample_id"] for r in records]
    for dataset_rows in (specs["pairs"],old_spots,controls):
        if [r["sample_id"] for r in dataset_rows] != ids:
            raise ValueError("Previous records differ from the ordered 213-image train list")
    outdir = CAPTURE / condition
    outdir.mkdir(exist_ok=True,parents=True)
    rows = []
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
        for i, record in enumerate(records):
            fixed = data.sample(i,epoch=0,draw_id=0)
            template = specs["pairs"][i]
            if (not fixed["pair_valid"] or not template["pair_valid"] or
                    record["image_sha256"] != template["source_png_sha256"] or
                    pixel_sha(fixed["signal"][0]) != template["signal_sha256"] or
                    pixel_sha(fixed["reference_plus"][0]) != template["plus_sha256"]):
                raise ValueError(f"Fixed spot no longer matches {record['sample_id']}")
            spot = fixed["signal"]
            box = native_window(fixed)
            core,ring=(fixed[k][0][box] for k in ("support_mask","ring_mask"))
            sample = fixed if DRAWS[condition]==0 else data.sample(i,epoch=0,draw_id=DRAWS[condition])
            if sample["meta"]["crop_xywh"] != (0,0,*record["reference_size"]) or native_window(sample)!=box:
                raise ValueError("Native crop changed")
            clean = condition == "clean"
            input_minus = sample["reference_minus"] if clean else sample["noisy_minus"]
            raw_plus = input_minus + spot
            input_plus = np.clip(raw_plus,0,1)
            if clean and not np.array_equal(input_plus,fixed["reference_plus"]):
                raise ValueError("Clean injected image changed")
            x_minus, x_plus = (a[0][box] for a in (input_minus,input_plus))
            if (np.any(x_plus[~core] != x_minus[~core]) or int(core.sum())!=template["core_pixels"] or
                    int(ring.sum())!=template["ring_pixels"] or np.any(core&ring)):
                raise ValueError("Spot escaped its registered support")
            entry = controls[i]
            input_sha = pixel_sha(x_minus)
            fresh = condition == "fresh"
            expected_input = entry["input_sha256"] if fresh else control_hashes[record["sample_id"]]
            if input_sha != expected_input:
                raise ValueError("Unspotted background differs from previous evaluation")
            if fresh:
                out_minus = hard_prediction(model,input_minus,device,greedy=True)[box]
                if pixel_sha(out_minus) != entry["prediction_sha256"]:
                    raise ValueError("Fresh unspotted S1 prediction does not reproduce")
            else:
                prev = ENDPOINTS / entry["native_gray_npy"]
                if file_hash(prev) != entry["native_gray_sha256"]:
                    raise ValueError("Previous S1 unspotted file hash changed")
                out_minus = np.load(prev,allow_pickle=False)
            out_plus = hard_prediction(model,input_plus,device,greedy=True)[box]
            for arr in (x_minus, x_plus, out_minus, out_plus):
                if arr.shape!=core.shape or arr.dtype!=np.float32 or not np.isfinite(arr).all() or arr.min()<0 or arr.max()>1:
                    raise ValueError("Invalid native input/output grid")
            if clean:
                verify_clean_patch(old_spots[i],(x_minus,x_plus),(out_minus,out_plus),core,ring)
            path=outdir / (record["source_image_id"]+".npz")
            tmp=path.with_suffix(".pending")
            with tmp.open("wb") as f:
                np.savez_compressed(f,x_minus=x_minus,x_plus=x_plus,s1_minus=out_minus,s1_plus=out_plus,core=core,ring=ring)
            tmp.replace(path)
            observation = contrast_increment(x_minus,x_plus,core,ring)
            pixels = raw_plus[0][box]
            row=dict(sample_id=record["sample_id"], source_image_id=record["source_image_id"], condition=condition,
                     path=str(path.relative_to(ROOT)), archive_sha256=file_hash(path),
                     nominal_peak_dn=template["nominal_peak_dn"], center_native_xy=template["center_native_xy"],
                     signal_sha256=template["signal_sha256"], core_sha256=pixel_sha(core), ring_sha256=pixel_sha(ring),
                     input_minus_sha256=input_sha, input_plus_sha256=pixel_sha(x_plus),
                     s1_minus_sha256=pixel_sha(out_minus),s1_plus_sha256=pixel_sha(out_plus),
                     core_pixels=int(core.sum()), ring_pixels=int(ring.sum()),
                     core_already_saturated_low=int(np.count_nonzero((x_minus==0)&core)),
                     core_already_saturated_high=int(np.count_nonzero((x_minus==1)&core)),
                     n_injected_pixels_clipped=int(np.count_nonzero((pixels>1)|(pixels<0))),
                     input_incremental_contrast_dn=observation, ratio_eligible=observation>0,
                     noise_parameters={"stripe_sigma_dn":0.,"pixel_sigma_dn":0.} if clean else sample["meta"]["noise_parameters"])
            if condition=="draw0":
                delta = (x_plus.astype(np.float64)-fixed["noisy_plus"][0][box])*255
                row.update(legacy_plus_pixel_mismatch=int(np.count_nonzero(delta)),
                           legacy_plus_max_abs_delta_dn=float(np.abs(delta).max()))
            rows.append(row)
            if len(rows)%50==0:
                print(f"{condition} captured {len(rows)}/{len(data)}",flush=True)
    save_report(CAPTURE/f"{condition}.json",dict(
        condition=condition, draw_id=DRAWS[condition], epoch=0, seed=0, n_pairs=len(rows),
        n_positive_denominators=sum(r["ratio_eligible"] for r in rows),
        n_pairs_with_clipping=sum(r["n_injected_pixels_clipped"]>0 for r in rows),
        n_pixels_clipped=sum(r["n_injected_pixels_clipped"] for r in rows),
        split="NUAA-SIRST/train",manifest_sha256=manifest_hash, checkpoint_sha256=ckpt_hash,
        spot_protocol_sha256=file_hash(SPOTS/"protocol.json"),historical_unspotted_path=str(historic_path.relative_to(ROOT)),
        historical_unspotted_sha256=file_hash(historic_path),code_sha256=file_hash(Path(__file__)),
        data_rule="Float32 x_plus=clip(x_minus+signal_from_epoch0_draw0,0,1); x_minus is the archived preclipped noisy branch or unmodified reference for clean.",
        inference="FP16 single-image greedy hard S1; TF32 disabled; RGB arithmetic mean prior to rounding/AdaIN",
        samples=rows))
    print(f"{condition} complete",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition",choices=DRAWS)
    run(parser.parse_args().condition)
