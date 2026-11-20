"""Evaluate both hard samplers on a completed SIRST endpoint and fixed training draw."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_retrained import ROOT, hard_prediction
from evaluate_warmup import file_hash, prepare_metrics, sample_digest, save_report, summarize
from prd_data import DenoisePairDataset

OUT = ROOT / "results/sirst_endpoints"
S1 = ROOT / "checkpoints/sirst_retrained_baseline.pt"
S2 = ROOT / "checkpoints/sirst_s2_common.pt"
MANIFEST = ROOT / "manifests/train.jsonl"
DRAWS = ("0", "1", "clean")
SAMPLERS = ("greedy_argmax", "upstream_seeded_top1")


def summary(rows):
    result = summarize(rows)
    for route, fields in result.items():
        values = [r["metrics"][route]["whole_psnr_db"] for r in rows]
        perfect = values.count(None)
        median = float(np.median([np.inf if x is None else x for x in values]))
        fields["whole_psnr_db"].update(n_images=len(values), n_finite=len(values)-perfect,
                                       n_positive_infinity=perfect, mean_positive_infinity=bool(perfect),
                                       median_positive_infinity=bool(np.isinf(median)),
                                       median=None if np.isinf(median) else median)
        if perfect:
            fields["whole_psnr_db"]["mean"] = None
    return result


def evaluate(endpoint, draw):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = DenoisePairDataset(ROOT, seed=0)
    asset = S1 if endpoint == "s1" else S2
    parent_hash = file_hash(S1)
    manifest = file_hash(MANIFEST)
    device = torch.device("cuda:0")
    if endpoint == "s1":
        from train_local_baseline import load_trained
        model = load_trained(asset, device)
    else:
        from train_s2_common import load_s2
        model = load_s2(asset, device)
    rows = []
    archive = OUT / f"{endpoint}_draw{draw}_float"
    archive.mkdir(parents=True, exist_ok=True)
    draw_id = 0 if draw == "clean" else int(draw)
    historical = None
    if draw != "clean":
        path = ROOT / f"results/local_retrain/train_eval_draw{draw}.json"
        historical = json.loads(path.read_text())
        if historical["manifest_sha256"] != manifest or historical["checkpoint_sha256"] != parent_hash:
            raise ValueError("Historical comparison uses a different training list or S1 checkpoint")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i, row in enumerate(data.records):
            sample = data.sample(i, epoch=0, draw_id=draw_id)
            if draw == "clean":
                sample["noisy_minus"] = sample["reference_minus"]
            digest = sample_digest(sample)
            gray = sample["noisy_minus"]
            measure, coverage = prepare_metrics(sample, extended=True)
            w, h = row["reference_size"]
            if sample["meta"]["crop_xywh"] != (0, 0, w, h) or coverage["valid_pixels"] != w*h:
                raise ValueError("Metrics must cover the complete native image")
            metrics = {"input": measure(gray[0])}
            for sampler in SAMPLERS:
                pred = hard_prediction(model, gray, device, greedy=(sampler == "greedy_argmax"))
                metrics[sampler] = measure(pred)
                if sampler == "greedy_argmax":
                    left, top = sample["meta"]["pad_left_top"]
                    native = pred[top:top+h, left:left+w].copy()
                    saved = archive / (row["source_image_id"] + ".npy")
                    if native.shape != (h, w) or native.dtype != np.float32:
                        raise ValueError("Greedy output must preserve original grayscale float32 pixels")
                    pending = saved.with_suffix(".pending")
                    with pending.open("wb") as handle:
                        np.save(handle, native, allow_pickle=False)
                    pending.replace(saved)
            if historical is not None:
                old = historical["samples"][i]
                if (row["sample_id"] != old["sample_id"] or old["data_sha256"] != digest
                        or any(metrics["input"][k] != v for k, v in old["metrics"]["input"].items())):
                    raise ValueError("Cached historical sample or input metrics differ")
            rows.append(dict(sample_id=row["sample_id"], data_sha256=digest, coverage=coverage, metrics=metrics,
                             native_gray_npy=str(saved.relative_to(OUT)), native_gray_sha256=file_hash(saved)))
            if len(rows) % 50 == 0:
                print(f"{endpoint} {draw}: {len(rows)}/{len(data)} training images", flush=True)
    if len(rows) != 213 or len({r["sample_id"] for r in rows}) != 213:
        raise ValueError("Incomplete primary training population")
    report = dict(endpoint=endpoint, draw=draw, data_seed=0, epoch=0, manifest_sha256=manifest,
                  checkpoint_sha256=parent_hash if endpoint == "s1" else file_hash(asset),
                  s1_sha256=parent_hash, split="NUAA-SIRST/train", samples=rows, all_images=summary(rows),
                  protocol=dict(samplers=list(SAMPLERS), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                cudnn_tf32=torch.backends.cudnn.allow_tf32, torch_version=str(torch.__version__),
                                sampler_options="cfg=1, top_k=1, top_p=0.75, g_seed=0; greedy uses argmax in upstream cached autoregressive entry; FP16",
                                image_output="float RGB arithmetic mean [0,1], preceding PNG rounding and AdaIN",
                                input="reference_minus" if draw == "clean" else "noisy_minus",
                                reference="reference_minus", scale="whole unpadded native valid pixels, 255 DN full scale",
                                aggregation="image means weighted equally, target contrast errors first averaged within image",
                                weighted_background="all valid pixels except square 2-pixel dilation of all native targets",
                                ring="square 2-pixel dilation of each native 8-connected target, valid excluding all native targets",
                                psnr_null="positive infinity, perfect reconstruction"))
    path = OUT / f"{endpoint}_draw{draw}.json"
    save_report(path, report)
    print(f"Saved {path}", flush=True)


def paired(rows, before, after):
    result = {}
    for field in rows[0]["metrics"][before]:
        vals = [(r["metrics"][before][field], r["metrics"][after][field]) for r in rows]
        diff = np.asarray([b-a for a,b in vals if a is not None and b is not None])
        item = {"finite_pairs": int(diff.size), "finite_mean_change": float(diff.mean()) if diff.size else None}
        if field == "whole_psnr_db":
            # PSNR has no missing support here. Null marks an exact reference.
            item["improved_images"] = sum((np.inf if b is None else b) > (np.inf if a is None else a) for a,b in vals)
        elif field not in ("native_median_ring_contrast_bias_dn", "plan_background_mean_error_dn"):
            item["improved_images"] = int((diff < 0).sum())
        result[field] = item
    return result


def compare():
    out = {}
    sources = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    expected_ids = [s["sample_id"] for s in sources]
    expected_hashes = {"s1": file_hash(S1), "s2": file_hash(S2)}
    manifest = file_hash(MANIFEST)
    for source in sources:
        if file_hash(ROOT / source["reference_path"]) != source["image_sha256"]:
            raise ValueError("Native reference image differs from the original training manifest")
    for draw in DRAWS:
        a,b = (json.loads((OUT / f"{s}_draw{draw}.json").read_text()) for s in ("s1","s2"))
        for endpoint, report in (("s1", a), ("s2", b)):
            if (report["endpoint"] != endpoint or report["draw"] != draw
                    or report["checkpoint_sha256"] != expected_hashes[endpoint]
                    or report["s1_sha256"] != expected_hashes["s1"] or report["manifest_sha256"] != manifest
                    or report["protocol"]["matmul_tf32"] or report["protocol"]["cudnn_tf32"]
                    or [r["sample_id"] for r in report["samples"]] != expected_ids
                    or report["all_images"] != summary(report["samples"])):
                raise ValueError(f"Wrong checkpoint, training population or precision for {endpoint}, {draw}")
        for key in ("manifest_sha256", "s1_sha256", "draw", "data_seed", "epoch", "protocol", "split"):
            if a[key] != b[key]:
                raise ValueError(f"Endpoint protocol differs: {key}")
        rows = []
        for source, first, second in zip(sources, a["samples"], b["samples"], strict=True):
            if (any(first[k] != second[k] for k in ("sample_id", "data_sha256", "coverage"))
                    or first["metrics"]["input"] != second["metrics"]["input"]):
                raise ValueError("Sample identity, native coverage or input differs")
            with Image.open(ROOT / source["reference_path"]) as im:
                reference = np.asarray(im.convert("L")).astype(np.float32) / 255
            for sample in (first, second):
                raster = OUT / sample["native_gray_npy"]
                if file_hash(raster) != sample["native_gray_sha256"]:
                    raise ValueError("Archived native output does not match the evaluated tensor")
                rec = np.load(raster, allow_pickle=False)
                if rec.shape != reference.shape or rec.dtype != np.float32:
                    raise ValueError("Archived prediction does not match native extent or precision")
                error = 255 * (rec.astype(np.float64) - reference)
                mse = np.square(error).mean()
                psnr = float(10 * np.log10(255**2 / mse)) if mse else None
                measured = sample["metrics"]["greedy_argmax"]
                if np.abs(error).mean() != measured["whole_mae_dn"] or psnr != measured["whole_psnr_db"]:
                    raise ValueError("Archived prediction does not reproduce native MAE and PSNR")
            vals = {"input": first["metrics"]["input"]}
            for sampler in SAMPLERS:
                vals["s1_"+sampler] = first["metrics"][sampler]
                vals["s2_"+sampler] = second["metrics"][sampler]
            rows.append({"metrics": vals})
        changes = {}
        for sampler in SAMPLERS:
            for previous, new in (("input", "s1_"+sampler), ("input", "s2_"+sampler),
                                  ("s1_"+sampler, "s2_"+sampler)):
                changes[new+"_minus_"+previous] = paired(rows, previous, new)
        out[draw] = dict(n_images=len(rows), means=summary(rows), paired_changes=changes,
                         manifest_sha256=a["manifest_sha256"], s1_sha256=a["checkpoint_sha256"],
                         s2_sha256=b["checkpoint_sha256"],
                         verified_native_outputs=2*len(rows), verified_image_aggregates=2)
    save_report(OUT / "comparison.json", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", choices=("s1", "s2"))
    parser.add_argument("--draw", choices=DRAWS)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare()
    elif args.endpoint and args.draw:
        evaluate(args.endpoint, args.draw)
    else:
        parser.error("Select --endpoint and --draw, or --compare")
