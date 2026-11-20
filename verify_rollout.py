"""Verify a stateless full-prefix rollout against cached greedy inference on training images."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from prd_backbone import OFFICIAL_SHA256, PROJECT, apply_warmup, load_backbone, official_inference
from prd_data import DenoisePairDataset
from prd_rollout import free_rollout
from probe_backbone import model_input


def cached_greedy(model, image):
    from models import var
    codes, scores = [], []

    def greedy(logits, **kwargs):
        codes.append(logits.argmax(-1))
        scores.append(logits.detach())
        return codes[-1].unsqueeze(-1)

    # The release sampler calls multinomial even at top_k=1 when logits tie.
    # Override only that decision rule to match the S2 deployment argmax.
    with patch.object(var, "sample_with_top_k_top_p_", greedy):
        output = official_inference(model, image)
    return output, scores, codes


def measure(model, x, reference, valid, precision, gradient):
    enabled = precision == "fp16"
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", enabled=enabled, dtype=torch.float16):
        original, cached_scores, cached_ids = cached_greedy(model, x)
        test, scores, ids = free_rollout(model, x)
    lengths = [code.numel() for code in ids]
    if lengths != [s * s for s in model.patch_nums] or len(ids) != len(cached_ids):
        raise ValueError("Token count differs from the pre-trained model scales")
    error = (original.float() - test.float()).abs() * 255
    result = dict(
        token_counts=lengths,
        mismatch_counts=[(a != b).sum().item() for a, b in zip(ids, cached_ids)],
        valid_rgb_mae_dn=error[valid.expand_as(error)].mean().item(),
        max_rgb_abs_dn=error.max().item(),
        max_logit_difference=[(a - b).abs().max().item() for a, b in zip(scores, cached_scores)],
    )
    del original, cached_scores, cached_ids, scores, ids, error
    if gradient:
        # Restrict leaves only for a bounded probe. Every block and every
        # feedback embedding still participates in differentiate-through-input.
        model.requires_grad_(False)
        names = {"head.bias", f"blocks.{model.depth - 1}.attn.proj.bias", "cb_cond.4.bias"}
        tracked = {name: p for name, p in model.named_parameters() if name in names}
        if set(tracked) != names:
            raise RuntimeError("Unexpected upstream parameter names for gradient probe")
        for weight in tracked.values():
            weight.requires_grad_(True)
        with torch.autocast("cuda", enabled=enabled, dtype=torch.float16):
            pred, grad_logits, _ = free_rollout(model, x)
            for logits in grad_logits:
                logits.retain_grad()
            mask = valid.expand_as(pred)
            loss = F.l1_loss(pred.float()[mask], reference.expand_as(pred)[mask])
        loss_scale = 1024. if enabled else 1.
        (loss * loss_scale).backward()
        norm_by_leaf = {name: (p.grad.float() / loss_scale).norm().item()
                        for name, p in tracked.items()}
        norm_by_scale = [(l.grad.float() / loss_scale).norm().item() for l in grad_logits]
        if not all(n > 0 and math.isfinite(n) for n in [*norm_by_leaf.values(), *norm_by_scale]):
            raise ValueError(f"Broken gradient: {norm_by_leaf}, {norm_by_scale}")
        result.update(
            loss_scale=loss_scale, image_only_l1=loss.item(), param_grad_norms=norm_by_leaf,
            logit_grad_norms=norm_by_scale,
            st_minus_hard_max_abs_dn=(pred.detach().float() - test.float()).abs().max().item() * 255,
        )
    result["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--backward", action="store_true", help="Verify gradients on the first sample")
    parser.add_argument("--warmup", type=Path, help="Apply the project's S1 checkpoint before verifying")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    dataset = DenoisePairDataset(PROJECT, seed=0)
    if not 1 <= args.samples <= len(dataset):
        parser.error("samples must fit within the SIRST training split")
    model = load_backbone(PROJECT / "checkpoints/Real-IISR.pth", device)
    report = dict(split="train", checkpoint_sha256=OFFICIAL_SHA256, sampler="deterministic argmax",
                  precision=args.precision, image_branch="noisy_minus", data_seed=0, samples=[])
    if args.warmup:
        apply_warmup(model, args.warmup)
        with args.warmup.open("rb") as checkpoint:
            report["s1_checkpoint_sha256"] = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    for i in range(args.samples):
        sample = dataset[i]
        if sample["meta"]["split"] != "train":
            raise ValueError("This diagnostic accepts only local training images")
        x = model_input(sample["noisy_minus"], device)
        ref = torch.from_numpy(sample["reference_minus"])[None].to(device)
        valid = torch.from_numpy(sample["valid_mask"])[None].to(device)
        row = measure(model, x, ref, valid, args.precision, args.backward and i == 0)
        row["sample_id"] = sample["meta"]["sample_id"]
        report["samples"].append(row)
        print(row["sample_id"], "mismatched tokens", sum(row["mismatch_counts"]),
              "max RGB difference in DN", row["max_rgb_abs_dn"], flush=True)
    report["exact_hard_parity"] = all(
        sum(row["mismatch_counts"]) == 0 and row["max_rgb_abs_dn"] == 0
        and row.get("st_minus_hard_max_abs_dn", 0) == 0 for row in report["samples"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    print("Saved", args.output, flush=True)
    if not report["exact_hard_parity"]:
        raise RuntimeError("Readiness check failed; see saved per-scale mismatches")


if __name__ == "__main__":
    main()
