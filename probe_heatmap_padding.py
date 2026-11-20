"""Isolate valid-pixel thermal statistics in a frozen-model, training-only padding probe."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from prd_backbone import OFFICIAL_SHA256, load_backbone, official_inference
from prd_data import DenoisePairDataset
from probe_backbone import ROOT, VENDOR_REVISION, model_input, to_gray_01


def valid_stats_heatmap(ir_image, valid, alpha=3.0, eps=1e-5):
    """Change the reduction domain only; apply the same sigmoid to every input pixel."""
    if valid.shape != ir_image.shape:
        raise ValueError("Mask shape must match the one-channel image")
    means, stds = [], []
    for plane, mask in zip(ir_image, valid):
        values = plane[mask]
        if values.numel() < 2:
            raise ValueError("Sample standard deviation needs at least two valid pixels")
        means.append(values.mean())
        stds.append(values.std(correction=1))
    mean = torch.stack(means).view(-1, 1, 1, 1)
    std = torch.stack(stds).view(-1, 1, 1, 1)
    return torch.sigmoid(alpha * ((ir_image - mean) / (std + eps)))


def probe_sample(model, module, sample, baseline_row, device):
    valid = torch.from_numpy(sample["valid_mask"][0])
    bg = torch.from_numpy(sample["background_mask"][0])
    ref = torch.from_numpy(sample["reference_minus"][0])
    noisy = sample["noisy_minus"]
    valid_np = sample["valid_mask"]
    mean = float(noisy[valid_np].mean())
    alt = np.where(valid_np, noisy, mean)
    if not np.array_equal(noisy[valid_np], alt[valid_np]) or np.all(valid_np):
        raise ValueError("This comparison requires identical valid data and a nonempty padding region")
    images = {"zero": model_input(noisy, device), "valid_mean": model_input(alt, device)}
    valid_gpu = valid[None, None].to(device)
    maps = {fill: x.mean(dim=1, keepdim=True) for fill, x in images.items()}
    original = module.compute_thermal_response_map
    full_difference = (valid_stats_heatmap(maps["zero"], torch.ones_like(valid_gpu)) - original(maps["zero"])).abs()
    full_max = full_difference.max().item()
    if full_max > 1e-6:
        raise AssertionError(f"All-valid heatmap equivalence failed: {full_max}")
    outputs = {}
    row = dict(sample_id=sample["meta"]["sample_id"], crop_xywh=sample["meta"]["crop_xywh"],
               noise_seed=sample["meta"]["noise_seed"], valid_fraction=float(valid_np.mean()),
               mean_fill_dn=mean * 255, all_valid_heatmap_max_abs_error=full_max, modes={})
    left, top = sample["meta"]["pad_left_top"]
    _, _, width, height = sample["meta"]["crop_xywh"]
    if min(width, height) <= 32:
        raise ValueError("The valid image is too small for the specified 16-pixel interior")
    interior = np.s_[top+16:top+height-16, left+16:left+width-16]
    for mode in ("original", "valid_pixels"):
        calls = []

        def heat_fn(x, alpha=3.0, eps=1e-5):
            calls.append(x.shape)
            if mode == "original":
                return original(x, alpha=alpha, eps=eps)
            return valid_stats_heatmap(x, valid_gpu, alpha=alpha, eps=eps)

        heat_diff = (heat_fn(maps["zero"]) - heat_fn(maps["valid_mean"])).abs()
        feature_mae = heat_diff[valid_gpu].mean().item()
        if mode == "valid_pixels" and heat_diff[valid_gpu].max().item() > 1e-6:
            raise AssertionError("Changing padding altered the valid-only normalized feature on valid pixels")
        calls.clear()
        with patch.object(module, "compute_thermal_response_map", heat_fn):
            outputs = {fill: to_gray_01(official_inference(model, x)) for fill, x in images.items()}
            repeated = to_gray_01(official_inference(model, images["zero"]))
        if len(calls) != 3:
            raise AssertionError(f"Expected one heatmap computation per inference, got {len(calls)} total")
        repeat_max_dn = (repeated - outputs["zero"]).abs().max().item() * 255
        if repeat_max_dn != 0:
            raise AssertionError(f"Identical-input hard output did not repeat: {repeat_max_dn} DN")
        delta = (outputs["zero"] - outputs["valid_mean"]).abs() * 255
        row["modes"][mode] = {
            "valid_heatmap_mae_between_fills": feature_mae,
            "valid_output_mae_between_fills_dn": delta[valid].mean().item(),
            "interior_output_mae_between_fills_dn": delta[interior].mean().item(),
            "repeat_zero_fill_output_max_abs_dn": repeat_max_dn,
            **{f"{fill}_fill_bg_rmse_dn": ((image[bg] - ref[bg]) * 255).square().mean().sqrt().item()
               for fill, image in outputs.items()},
        }
    metrics = row["modes"]["original"]
    baseline_padding = baseline_row["padding_zero_to_valid_mean"]
    baseline_bg = baseline_row["routes"]["pred_tokens_noisy_h"]["background_minus_rmse_dn"]
    replay_deltas = (
        metrics["valid_output_mae_between_fills_dn"] - baseline_padding["valid_output_mae_dn"],
        metrics["interior_output_mae_between_fills_dn"] - baseline_padding["valid_interior_16px_output_mae_dn"],
        metrics["zero_fill_bg_rmse_dn"] - baseline_bg,
    )
    row["original_trace_max_metric_difference_dn"] = max(abs(v) for v in replay_deltas)
    if row["original_trace_max_metric_difference_dn"] > 1e-4:
        raise AssertionError(f"Original decode differs from fixed-seed representation probe: {replay_deltas}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/Real-IISR.pth")
    parser.add_argument("--trace", type=Path, default=ROOT / "results/representation_probe.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/heatmap_padding_probe.json")
    args = parser.parse_args()
    trace_bytes = args.trace.read_bytes()
    trace = json.loads(trace_bytes)
    if trace["split"] != "train" or trace["checkpoint_sha256"] != OFFICIAL_SHA256 or trace["decode_seed_per_call"] != 0:
        raise ValueError("Expected the training-only, reproducible official checkpoint trace")
    seed = trace["data_seed"]
    dataset = DenoisePairDataset(ROOT, seed=seed)
    manifest_hash = hashlib.sha256((ROOT / "manifests/train.jsonl").read_bytes()).hexdigest()
    if manifest_hash != trace["training_manifest_sha256"] or trace["source_revision"] != VENDOR_REVISION:
        raise ValueError("Dataset or model source differs from the reference trace")
    device = torch.device("cuda:0")
    model = load_backbone(args.checkpoint, device)
    from models import var as var_module

    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for index, baseline_row in enumerate(trace["samples"]):
            sample = dataset[index]
            if sample["meta"]["sample_id"] != baseline_row["meta"]["sample_id"]:
                raise ValueError("The two probes must use the same training images in the same order")
            rows.append(probe_sample(model, var_module, sample, baseline_row, device))
            print(f"Processed {sample['meta']['sample_id']}", flush=True)
    summary = {}
    for mode in ("original", "valid_pixels"):
        summary[mode] = {}
        for key in rows[0]["modes"][mode]:
            values = [row["modes"][mode][key] for row in rows]
            summary[mode][key] = {"n": len(values), "mean": float(np.mean(values)),
                                  "median": float(np.median(values)), "min": min(values), "max": max(values)}
    report = {
        "dataset": "NUAA-SIRST", "split": "train", "data_seed": seed, "decode_seed_per_call": 0,
        "scope": "frozen original SR weights; same synthetic noisy-minus branch; only thermal map mean and sample standard deviation reduction domain changes",
        "unchanged": "same sigmoid (alpha=3, eps=1e-5) at all pixels including pad; original input convolution, edges, AFF, attention, CAC and decoder; fixed float RGB mean output",
        "source_revision": VENDOR_REVISION, "checkpoint_sha256": OFFICIAL_SHA256,
        "training_manifest_sha256": manifest_hash, "representation_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "samples": rows, "descriptive_by_image": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
