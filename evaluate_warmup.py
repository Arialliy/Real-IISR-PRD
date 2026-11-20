"""Compare hard single-frame reconstructions on the prespecified local training list."""
import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from prd_backbone import OFFICIAL_SHA256, load_backbone, official_inference
from prd_data import DenoisePairDataset
from probe_backbone import ROOT, VENDOR_REVISION, model_input, to_gray_01


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", prefix=path.name, dir=path.parent,
                                     encoding="utf-8", delete=False) as temp:
        json.dump(report, temp, ensure_ascii=False, allow_nan=False, indent=2)
        temp.write("\n")
    os.replace(temp.name, path)


METRIC_DIRECTION = {
    "whole_mae_dn": "lower", "valid_mae_dn": "lower", "background_rmse_dn": "lower",
    "plan_background_rmse_dn": "lower", "weighted_background_column_residual_dn": "lower",
    "background_column_residual_dn": "lower", "native_contrast_abs_error_dn": "lower",
    "native_median_ring_contrast_abs_error_dn": "lower", "native_target_mae_dn": "lower",
    "whole_psnr_db": "higher", "whole_ssim": "higher",
    "native_median_ring_contrast_bias_dn": "signed", "plan_background_mean_error_dn": "signed",
}


def checked_error_dn(prediction, reference):
    pred, ref = np.asarray(prediction), np.asarray(reference)
    if pred.shape != ref.shape or pred.ndim != 2:
        raise ValueError(f"Expected identical 2D grids, got {pred.shape} and {ref.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("Nonfinite prediction or reference")
    return (pred.astype(np.float64) - ref.astype(np.float64)) * 255


def fixed_background(sample):
    valid = sample["valid_mask"][0]
    target = sample["native_mask"][0] & valid
    return valid & ~ndimage.binary_dilation(target, structure=np.ones((5, 5), bool))


def prepare_metrics(sample, *, extended=False):
    reference = sample["reference_minus"][0]
    valid = sample["valid_mask"][0]
    background = sample["background_mask"][0]
    native = sample["native_mask"][0] & valid
    components, count = ndimage.label(native, structure=np.ones((3, 3), dtype=bool))
    local_regions = []
    for label, region in enumerate(ndimage.find_objects(components), start=1):
        if region is None:
            continue
        region = tuple(slice(max(0, s.start - 2), min(n, s.stop + 2))
                       for s, n in zip(region, native.shape))
        core = components[region] == label
        ring = ndimage.binary_dilation(core, structure=np.ones((5, 5), dtype=bool))
        ring &= valid[region] & ~native[region]
        if ring.any():
            local_regions.append((region, core, ring))
    bg_counts = background.sum(axis=0)
    columns = bg_counts >= 8
    if extended:
        plan_bg = fixed_background(sample)
        weights = plan_bg.sum(axis=0)
        active = weights > 0
        ref64 = reference.astype(np.float64)
        ref_contrasts = [ref64[r][m].mean() - np.median(ref64[r][a]) for r, m, a in local_regions]

    def measure(image):
        error = checked_error_dn(image, reference)
        contrast_errors = [
            abs(error[region][core].mean() - error[region][ring].mean())
            for region, core, ring in local_regions
        ]
        stripe = None
        if columns.sum() >= 2:
            col_mean_err = (error * background).sum(axis=0)[columns] / bg_counts[columns]
            stripe = float(np.std(col_mean_err))
        metrics = dict(valid_mae_dn=float(np.abs(error[valid]).mean()),
                    background_rmse_dn=float(np.sqrt(np.square(error[background]).mean()))
                    if background.any() else None,
                    background_column_residual_dn=stripe,
                    native_contrast_abs_error_dn=float(np.mean(contrast_errors))
                    if contrast_errors else None)
        if extended:
            image = np.asarray(image, dtype=np.float64)
            contrasts = np.asarray([image[r][m].mean() - np.median(image[r][a])
                                    for r, m, a in local_regions])
            delta = 255 * (contrasts - ref_contrasts)
            mse = float(np.square(error[valid]).mean())
            weighted_sd = None
            if active.any():
                selected = weights[active]
                means = (error * plan_bg).sum(axis=0)[active] / selected
                center = np.average(means, weights=selected)
                weighted_sd = float(np.sqrt(np.average((means-center)**2, weights=selected)))
            metrics.update(
                whole_mae_dn=metrics["valid_mae_dn"],
                whole_psnr_db=float(10 * np.log10(255**2 / mse)) if mse else None,
                weighted_background_column_residual_dn=weighted_sd,
                plan_background_rmse_dn=float(np.sqrt(np.square(error[plan_bg]).mean())) if active.any() else None,
                plan_background_mean_error_dn=float(error[plan_bg].mean()) if active.any() else None,
                native_target_mae_dn=float(np.abs(error[native]).mean()) if native.any() else None,
                native_median_ring_contrast_bias_dn=float(delta.mean()) if delta.size else None,
                native_median_ring_contrast_abs_error_dn=float(np.abs(delta).mean()) if delta.size else None,
            )
        return metrics

    coverage = dict(valid_pixels=int(valid.sum()), background_pixels=int(background.sum()),
                    eligible_background_columns=int(columns.sum()), native_components=int(count),
                    eligible_native_components=len(local_regions))
    if extended:
        coverage.update(plan_background_pixels=int(plan_bg.sum()), plan_background_columns=int(active.sum()))
    return measure, coverage


def sample_digest(sample):
    h = hashlib.sha256()
    for key in ("noisy_minus", "reference_minus", "valid_mask", "native_mask", "background_mask"):
        h.update(sample[key].tobytes())
    return h.hexdigest()


def summarize(rows):
    report = {}
    for route, metrics in rows[0]["metrics"].items():
        report[route] = {}
        for metric in metrics:
            values = [r["metrics"][route][metric] for r in rows if r["metrics"][route][metric] is not None]
            report[route][metric] = dict(n_images=len(values),
                                         mean=float(np.mean(values)) if values else None,
                                         median=float(np.median(values)) if values else None)
    return report


def paired_changes(rows, ref, current):
    changes = {}
    for metric in rows[0]["metrics"][ref]:
        direction = METRIC_DIRECTION[metric]
        delta = []
        for row in rows:
            before, after = (row["metrics"][route][metric] for route in (ref, current))
            if metric == "whole_psnr_db":
                if before is None and row["metrics"][ref].get("whole_mae_dn") == 0:
                    before = np.inf
                if after is None and row["metrics"][current].get("whole_mae_dn") == 0:
                    after = np.inf
            if before is not None and after is not None:
                delta.append(0.0 if before == after else after - before)
        if delta:
            arr = np.asarray(delta)
            mean = float(arr.mean()) if np.isfinite(arr).all() else None
            item = dict(n_images=len(delta), mean_delta=mean)
            if direction == "signed":
                item.update(mean_absolute_delta=float(np.abs(arr).mean()), negative_fraction=float((arr<0).mean()),
                            positive_fraction=float((arr>0).mean()), q05=float(np.quantile(arr,.05)),
                            median=float(np.median(arr)), q95=float(np.quantile(arr,.95)))
            else:
                good = arr < 0 if direction == "lower" else arr > 0
                item["improved_image_fraction"] = float(good.mean())
            if metric == "whole_psnr_db":
                item.update(positive_infinite_deltas=int(np.isposinf(arr).sum()),
                            negative_infinite_deltas=int(np.isneginf(arr).sum()))
            changes[metric] = item
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=Path, help="S1 adapter file for the paired second pass")
    parser.add_argument("--pilot", type=Path, help="S2 pilot file on top of the S1 adapter")
    parser.add_argument("--argmax", action="store_true", help="Evaluate only S2 vs input with deterministic cached argmax")
    parser.add_argument("--draw-id", type=int, default=0, help="Draw of paired training noise, geometry and calibration signal")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pilot and not args.warmup:
        parser.error("--pilot requires its parent --warmup")
    if args.argmax and not args.pilot:
        parser.error("--argmax is an S2-only diagnostic; supply --warmup and --pilot")
    suffix = f"_draw{args.draw_id}" if args.draw_id else ""
    pretrained_report = ROOT / f"results/warmup_pretrained_eval{suffix}.json"
    warmup_report = ROOT / f"results/warmup_paired_eval{suffix}.json"
    args.baseline = args.baseline or (warmup_report if args.pilot else pretrained_report)
    if args.output is None:
        args.output = warmup_report
        if args.pilot:
            prefix = "s2_argmax_eval" if args.argmax else "s2_pilot_paired_eval"
            args.output = ROOT / f"results/{prefix}{suffix}.json"
    if args.pilot:
        earlier_reports = (ROOT / "results").glob("warmup_*eval*.json")
        reserved = {p.resolve() for p in earlier_reports}
        reserved.update(p.resolve() for p in (args.warmup, args.pilot, args.baseline))
        if args.output.resolve() in reserved:
            parser.error("S2 output must be distinct from original checkpoints and earlier reports")
    torch.set_num_threads(1)
    started = time.monotonic()
    dataset = DenoisePairDataset(ROOT, seed=0)
    manifest_sha = file_hash(ROOT / "manifests/train.jsonl")
    protocol = dict(data="NUAA-SIRST", split="train", manifest_sha256=manifest_sha,
                    official_weights_sha256=OFFICIAL_SHA256, vendor_revision=VENDOR_REVISION,
                    data_seed=0, epoch=0, draw_id=args.draw_id, decode_seed_per_call=0,
                    branch="noisy_minus against its pre-corruption reference", padding="0 DN",
                    decode="upstream top_k=1 top_p=0.75 cfg=1, fp16 CUDA autocast, float32 RGB mean; no color matching",
                    aggregation="per-image metric, then equal-weight mean over eligible images",
                    native_contrast="mean absolute error in per-component core-minus-ring contrast; "
                                    "8-connected native mask components, 5x5 square dilation clipped "
                                    "to valid pixels with all native components excluded from ring",
                    column_residual="population SD of mean background error for columns with >=8 "
                                    "background pixels; requires at least two columns",
                    units="0–255 grayscale DN", sample_count=len(dataset))
    baseline = None
    if args.warmup:
        baseline = json.loads(args.baseline.read_text())
        if baseline["protocol"] != protocol or len(baseline["samples"]) != len(dataset):
            raise ValueError("Baseline protocol or coverage differs")
        if args.output.resolve() == args.baseline.resolve():
            parser.error("The paired report needs its own output file")
    if args.argmax:
        from verify_rollout import cached_greedy
        # The official trace above verifies input identities and S1 exposure only.
        protocol["decode"] = "upstream cached forward with deterministic first-index argmax; cfg=1, FP16 CUDA autocast; float32 RGB mean, no color matching"
        protocol["decode_seed_per_call"] = None
    device = torch.device("cuda:0")
    model = load_backbone(ROOT / "checkpoints/Real-IISR.pth", device)
    report = dict(protocol=protocol, samples=[])
    current = "pretrained"
    if args.pilot:
        from prd_backbone import apply_s2_pilot
        if baseline["warmup"]["checkpoint_sha256"] != file_hash(args.warmup):
            raise ValueError("S1 baseline and parent checkpoint disagree")
        if baseline["warmup"]["baseline_report_sha256"] != file_hash(pretrained_report):
            raise ValueError("S1 report was generated from a different pretrained trace")
        meta = apply_s2_pilot(model, args.warmup, args.pilot)
        if meta["manifest_sha256"] != manifest_sha:
            raise ValueError("S2 pilot used a different training manifest")
        current = "s2_pilot"
        report["warmup"] = baseline["warmup"]
        report[current] = dict(checkpoint_sha256=file_hash(args.pilot), train_steps=meta["train_steps"],
                               distinct_update_images=len(set(meta["training_ids"])), epoch=meta["epoch"],
                               data_seed=meta.get("data_seed", meta.get("seed")), draw_id=meta["draw_id"],
                               parent_report_sha256=file_hash(args.baseline))
    elif args.warmup:
        from prd_backbone import apply_warmup
        meta = apply_warmup(model, args.warmup)
        current = "warmup"
        report[current] = dict(checkpoint_sha256=file_hash(args.warmup), train_steps=meta["train_steps"],
                               distinct_update_images=len(set(meta["training_ids"])),
                               baseline_report_sha256=file_hash(args.baseline))
    if args.warmup:
        trained_ids = set(meta["training_ids"])
        all_ids = {row["sample_id"] for row in dataset.records}
        if not trained_ids.issubset(all_ids):
            raise ValueError("Unrecognized training sample identities in checkpoint")
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for i in range(len(dataset)):
            sample = dataset.sample(i, epoch=0, draw_id=args.draw_id)
            image = sample["noisy_minus"]
            measure, coverage = prepare_metrics(sample)
            identity = dict(sample_id=sample["meta"]["sample_id"], data_sha256=sample_digest(sample),
                            noise_seed=sample["meta"]["noise_seed"], coverage=coverage)
            if baseline is None:
                metrics = {"input": measure(image[0])}
            else:
                old = baseline["samples"][i]
                if any(old[key] != value for key, value in identity.items()):
                    raise ValueError(f"Sample ID, data, noise or coverage mismatch at {i}")
                metrics = {"input": measure(image[0])} if args.argmax else old["metrics"].copy()
                identity["used_for_update"] = identity["sample_id"] in trained_ids
                if args.pilot:
                    identity["used_for_s1_update"] = old["used_for_update"]
                    identity["used_for_s2_update"] = identity["used_for_update"]
                    identity["used_same_s2_draw"] = (identity["used_for_update"]
                            and (meta["data_seed"], meta["epoch"], meta["draw_id"])
                            == (protocol["data_seed"], protocol["epoch"], protocol["draw_id"]))
            model_x = model_input(image, device)
            output = cached_greedy(model, model_x)[0] if args.argmax else official_inference(model, model_x)
            pred = to_gray_01(output).numpy()
            metrics[current] = measure(pred)
            report["samples"].append(dict(**identity, metrics=metrics))
            if (i+1) % 25 == 0 or i+1 == len(dataset):
                print(f"{i+1}/{len(dataset)} training images, {time.monotonic()-started:.1f}s", flush=True)
    rows = report["samples"]
    report["all_images"] = summarize(rows)
    if args.warmup:
        refs = [ref for ref in rows[0]["metrics"] if ref != current]
        report[f"paired_{current}_minus"] = {ref: paired_changes(rows, ref, current) for ref in refs}
        report["update_exposure"] = {}
        for used in (True, False):
            group = [r for r in rows if r["used_for_update"] == used]
            if group:
                counts = dict(n_images=len(group), metrics=summarize(group))
                counts[f"paired_{current}_minus_{refs[-1]}"] = paired_changes(group, refs[-1], current)
                report["update_exposure"]["updated" if used else "other_training_images"] = counts
    report["wall_seconds"] = time.monotonic() - started
    output_path = args.output if args.warmup else args.baseline
    save_report(output_path, report)
    print(output_path, json.dumps(report["all_images"]))


if __name__ == "__main__":
    main()
