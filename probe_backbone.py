"""Run an offline, training-split-only representation probe with original Real-IISR weights."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from prd_data import DenoisePairDataset


ROOT = Path(__file__).resolve().parent
VENDOR_REVISION = "70355f7e9d6b849272d429d12e854f5890b89ae2"


def model_input(gray, device):
    image = torch.from_numpy(gray).to(device=device, dtype=torch.float32)
    return (image.unsqueeze(0) * 2 - 1).repeat(1, 3, 1, 1)


def to_gray_01(rgb):
    if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Expected a finite, clipped [0, 1] model image")
    return rgb[0].float().mean(dim=0).cpu()


def run_branch(model, reference, noisy, device, official_inference):
    """Observe the original final decode once, retaining its actual hard tokens and h."""
    captured = []
    decode = model.vae.idxBl_to_img_lora

    def capture_decode(tokens, condition, *args, **kwargs):
        captured.append((tuple(tokens), condition))
        return decode(tokens, condition, *args, **kwargs)

    with patch.object(model.vae, "idxBl_to_img_lora", capture_decode):
        prediction = official_inference(model, model_input(noisy, device))
    if len(captured) != 1:
        raise RuntimeError(f"Expected one final CAC decode, got {len(captured)}")
    predicted_tokens, noisy_h = captured[0]
    # The upstream trainer computes reference tokens outside its AMP region.
    # In particular, the in-place nearest-code addmm expects FP32 embeddings.
    with torch.autocast(device_type="cuda", enabled=False):
        _, ref_tokens = model.vae.img_to_idxBl(model_input(reference, device))
    expected_lengths = [size * size for size in model.patch_nums]
    for tokens in (predicted_tokens, ref_tokens):
        if [t.shape[1] for t in tokens] != expected_lengths:
            raise RuntimeError("Token scales disagree with the loaded model")

    # Both alternatives use reference-image tokens; only the second uses the
    # exact h extracted from this branch's original noisy-image inference.
    base = model.vae.idxBl_to_img(ref_tokens, same_shape=True, last_one=True)
    conditional = decode(ref_tokens, noisy_h, same_shape=True, last_one=True)
    gray = {
        "noisy_input": torch.from_numpy(noisy[0]),
        "ref_tokens_base_vq": to_gray_01((base + 1) * 0.5),
        "ref_tokens_noisy_h": to_gray_01((conditional + 1) * 0.5),
        "pred_tokens_noisy_h": to_gray_01(prediction),
    }
    return gray, ref_tokens, predicted_tokens, noisy_h


def measure(sample, displays):
    ref = torch.from_numpy(sample["reference_minus"][0])
    signal = torch.from_numpy(sample["signal"][0])
    valid = torch.from_numpy(sample["valid_mask"][0])
    background = torch.from_numpy(sample["background_mask"][0])
    support = torch.from_numpy(sample["support_mask"][0])
    native = torch.from_numpy(sample["native_mask"][0]) & valid
    expanded = torch.nn.functional.max_pool2d(native[None, None].float(), 5, stride=1, padding=2)
    native_ring = expanded[0, 0].bool() & ~native & valid
    rows = {}
    for route in displays[0]:
        minus, plus = displays[0][route], displays[1][route]
        err_minus = (minus - ref) * 255
        metrics = dict(valid_minus_mae_dn=err_minus[valid].abs().mean().item())
        if background.any():
            metrics["background_minus_rmse_dn"] = err_minus[background].square().mean().sqrt().item()
        if native.any() and native_ring.any():
            metrics["native_mask_minus_mae_dn"] = err_minus[native].abs().mean().item()
            metrics["native_minus_signed_contrast_error_dn"] = (
                err_minus[native].mean() - err_minus[native_ring].mean()
            ).item()
        if sample["pair_valid"] and support.any():
            delta = plus[support] - minus[support]
            ideal = signal[support]
            metrics["injection_support_mae_dn"] = (delta - ideal).abs().mean().item() * 255
            metrics["injection_gain"] = (torch.dot(delta, ideal) / torch.dot(ideal, ideal)).item()
        rows[route] = metrics
    return rows


def probe_pair(model, sample, device, official_inference, swap_condition):
    displays, ref_codes, pred_codes, conditions = [], [], [], []
    for branch in ("minus", "plus"):
        gray, ref, pred, h = run_branch(
            model, sample[f"reference_{branch}"], sample[f"noisy_{branch}"], device, official_inference
        )
        displays.append(gray)
        ref_codes.append(ref)
        pred_codes.append(pred)
        conditions.append(h)
    row = {
        "meta": sample["meta"],
        "pair_valid": bool(sample["pair_valid"]),
        "routes": measure(sample, displays),
        "changed_token_counts_by_scale": {
            name: [(minus != plus).sum().item() for minus, plus in zip(branches[0], branches[1])]
            for name, branches in (("reference", ref_codes), ("predicted", pred_codes))
        },
    }
    if swap_condition:
        swapped = model.vae.idxBl_to_img_lora(ref_codes[1], conditions[0], same_shape=True, last_one=True)
        difference = (to_gray_01((swapped + 1) * 0.5) - displays[1]["ref_tokens_noisy_h"]).abs()
        valid = torch.from_numpy(sample["valid_mask"][0])
        row["fixed_plus_ref_tokens_swap_plus_h_to_minus_h_valid_mae_dn"] = difference[valid].mean().item() * 255
    valid = sample["valid_mask"]
    x = sample["noisy_minus"]
    if not np.all(valid):
        mean = float(x[valid].mean())
        alt = np.where(valid, x, mean)
        assert np.array_equal(x[valid], alt[valid])
        output = official_inference(model, model_input(alt, device))
        diff = (to_gray_01(output) - displays[0]["pred_tokens_noisy_h"]).abs() * 255
        left, top = sample["meta"]["pad_left_top"]
        _, _, width, height = sample["meta"]["crop_xywh"]
        padding = {"valid_fraction": float(valid.mean()), "gray_fill_dn": mean * 255,
                   "valid_output_mae_dn": diff[torch.from_numpy(valid[0])].mean().item()}
        if min(width, height) > 32:
            inside = diff[top+16:top+height-16, left+16:left+width-16]
            padding["valid_interior_16px_output_mae_dn"] = inside.mean().item()
        row["padding_zero_to_valid_mean"] = padding
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/Real-IISR.pth")
    parser.add_argument("--warmup", type=Path, help="Apply the common S1 checkpoint before probing")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap-condition", action="store_true")
    parser.add_argument("--official-example", action="store_true", help="Also check the first bundled example before color postprocessing")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from prd_backbone import OFFICIAL_SHA256, apply_warmup, load_backbone, official_inference

    device = torch.device("cuda:0")
    dataset = DenoisePairDataset(ROOT, seed=args.seed)
    if not 1 <= args.samples <= len(dataset):
        parser.error(f"--samples must be between 1 and {len(dataset)}")
    model = load_backbone(args.checkpoint, device)
    if args.warmup:
        metadata = apply_warmup(model, args.warmup)
    model.eval()
    torch.manual_seed(args.seed)
    if model.patch_nums[-1] * model.vae.downsample != dataset.patch_size:
        raise ValueError("Training patch and loaded model grid disagree")
    results = {
        "source_revision": VENDOR_REVISION,
        "training_manifest_sha256": hashlib.sha256((ROOT / "manifests/train.jsonl").read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint.name, "checkpoint_sha256": OFFICIAL_SHA256,
        "dataset": "NUAA-SIRST", "split": "train", "seed": args.seed, "data_seed": args.seed,
        "decode_seed_per_call": 0,
        "hard_token_choice": "upstream top_k=1, top_p=0.75, multinomial with generator reset to seed 0 on each call; equal maxima can tie",
        "selection": "first requested original IDs in bundled training-list order, epoch/draw 0",
        "model": "original pretrained SR model without denoising adaptation",
        "precision": "FP32 reference tokenization; FP16 CUDA autocast for original prediction and all final decodes; FP32 metrics",
        "output_domain": "fixed RGB-channel mean of clipped [0,1], no color matching; 1 DN = 1/255",
        "native_contrast": "per-image union of available native mask pixels minus a two-pixel rectangular-dilation ring, valid pixels only",
        "patch_nums": list(model.patch_nums), "samples": [],
    }
    if args.warmup:
        with args.warmup.open("rb") as source:
            results["warmup_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
        results["model"] = "S1 common same-size restoration warmup on the released backbone"
        results["train_steps"] = metadata["train_steps"]
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        if args.official_example:
            path = sorted((ROOT / "vendor/Real-IISR/testset/LR").glob("*.png"))[0]
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                if rgb.height != 512:
                    rgb = rgb.resize((512, 512))
                arr = np.asarray(rgb, dtype=np.float32).copy() / 255
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            output = official_inference(model, x)
            to_gray_01(output)
            results["official_example_smoke"] = {
                "input": str(path.relative_to(ROOT)), "output_shape": list(output.shape),
                "rgb_min_01": output.min().item(), "rgb_max_01": output.max().item(),
            }
        for index in range(args.samples):
            sample = dataset[index]
            results["samples"].append(probe_pair(model, sample, device, official_inference, args.swap_condition))
            print(f"Processed {sample['meta']['sample_id']}", flush=True)
    summaries = defaultdict(list)
    for sample in results["samples"]:
        for route, metrics in sample["routes"].items():
            for metric, value in metrics.items():
                summaries[f"{route}/{metric}"].append(value)
        for metric, value in sample.get("padding_zero_to_valid_mean", {}).items():
            summaries[f"padding_zero_to_valid_mean/{metric}"].append(value)
    results["descriptive_by_image"] = {
        key: {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
              "min": min(values), "max": max(values)} for key, values in summaries.items()
    }
    filename = "s1_representation_probe.json" if args.warmup else "representation_probe.json"
    path = args.output or ROOT / "results" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    report = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.with_suffix(".json.tmp").open("w", encoding="utf-8") as output_file:
        output_file.write(report)
    path.with_suffix(".json.tmp").replace(path)
    print(path)


if __name__ == "__main__":
    main()
