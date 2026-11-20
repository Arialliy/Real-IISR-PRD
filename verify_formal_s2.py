"""Test the formal S1 endpoint for image-only free-prefix S2 training."""

import json
import math
from unittest.mock import patch

import torch

from check_s1_hard_sampler import published
from evaluate_warmup import file_hash, sample_digest, save_report
from prd_backbone import PROJECT
from prd_data import DenoisePairDataset
from prd_rollout import free_rollout
from probe_backbone import model_input
from train_local_baseline import FINAL, load_trained
from verify_rollout import cached_greedy


def measure(model, sample):
    x = model_input(sample["noisy_minus"], "cuda")
    valid = torch.from_numpy(sample["valid_mask"])[None].cuda()
    reference = torch.from_numpy(sample["reference_minus"])[None].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        cached, cached_scores, cached_ids = cached_greedy(model, x)
        released, published_ids, tie_counts = published(model, x)
    model.zero_grad(set_to_none=True)
    with patch.object(model.vae, "img_to_idxBl", side_effect=AssertionError("Unexpected tokenization during free rollout")):
        with torch.autocast("cuda", dtype=torch.float16):
            pred, scores, ids = free_rollout(model, x)
            for score in scores:
                score.retain_grad()
            gray = pred.float().mean(1, keepdim=True)
            loss = (gray[valid] - reference[valid]).abs().mean()
        scale = 1024.0
        (scale * loss).backward()
    errors_dn = (pred.detach().float() - cached.float()).abs() * 255.0
    gray_delta = (gray.detach() - cached.float().mean(1, keepdim=True)).abs() * 255.0
    official_gray_delta = (gray.detach() - released.float().mean(1, keepdim=True)).abs() * 255.0
    norms = {name: (p.grad.detach().float()/scale).norm().item() if p.grad is not None else None
             for name, p in model.named_parameters() if p.requires_grad}
    score_norms = [(v.grad.float()/scale).norm().item() if v.grad is not None else None for v in scores]
    return {
        "id": sample["meta"]["sample_id"], "data_sha256": sample_digest(sample),
        "scale_tokens": [v.numel() for v in ids],
        "greedy_differing_tokens": [int((a != b).sum()) for a,b in zip(ids, cached_ids)],
        "greedy_max_abs_logit_diff_by_scale": [(a.detach()-b).abs().max().item() for a,b in zip(scores, cached_scores)],
        "greedy_max_abs_rgb_dn": errors_dn.max().item(), "greedy_valid_mean_abs_gray_dn": gray_delta[valid].mean().item(),
        "greedy_valid_max_abs_gray_dn": gray_delta[valid].max().item(),
        "published_differing_tokens": [int((a != b).sum()) for a,b in zip(ids, published_ids)],
        "published_nonfirst_maximum_counts": tie_counts, "published_valid_mean_abs_gray_dn": official_gray_delta[valid].mean().item(),
        "published_valid_max_abs_gray_dn": official_gray_delta[valid].max().item(),
        "image_only_gray_l1_01": loss.item(), "unscaled_param_grad_norms": norms,
        "unscaled_logit_grad_norms": score_norms,
        "finite_positive_grads": all(v is not None and math.isfinite(v) and v > 0 for v in [*norms.values(), *score_norms]),
    }


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_trained(FINAL, torch.device("cuda:0")).requires_grad_(False)
    names = {"head.bias", f"blocks.{model.depth-1}.attn.proj.bias", "cb_cond.4.bias"}
    for name, p in model.named_parameters():
        if name in names or id(p) in (id(model.vq.U), id(model.vq.V), id(model.vq.lora_gate)):
            p.requires_grad_(True)
    if sum(1 for p in model.parameters() if p.requires_grad) != 6:
        raise ValueError("Formal S1 checkpoint has an unexpected parameter layout")
    dataset = DenoisePairDataset(PROJECT, seed=0)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for i in range(2):
        sample = dataset.sample(i, epoch=0, draw_id=0)
        if sample["meta"]["split"] != "train":
            raise ValueError("Expected local training split")
        rows.append(measure(model, sample))
        print(rows[-1]["id"], rows[-1]["greedy_differing_tokens"], rows[-1]["unscaled_param_grad_norms"], flush=True)
    passed = all(r["greedy_max_abs_rgb_dn"] == 0 and sum(r["greedy_differing_tokens"]) == 0
                 and max(r["greedy_max_abs_logit_diff_by_scale"]) == 0 and r["finite_positive_grads"] for r in rows)
    report = {
        "checkpoint": str(FINAL.relative_to(PROJECT)), "checkpoint_sha256": file_hash(FINAL),
        "device": torch.cuda.get_device_name(), "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "data": "first two SIRST bundled train IDs; seed=0, epoch=0, draw=0, noisy_minus, 0 DN padding",
        "protocol": "FP16 autocast, CFG=1. Differentiable hard argmax vs upstream cached first-index argmax; separate untouched published top_k=1, top_p=.75, g_seed=0 comparison.",
        "backward": "Masked grayscale L1 between predicted image and reference; no CE, token targets or pair loss. Loss scaled by 1024, reported gradient norms unscaled.",
        "reference_access": "Only x passed to free_rollout; reference used only after generation for pixel loss. img_to_idxBl blocked by a raising guard throughout free forward and backward.",
        "rows": rows, "exact_greedy_and_gradients_pass": passed,
    }
    path = PROJECT/"results/local_retrain/s2_preflight.json"
    save_report(path, report)
    print(path, json.dumps({"passed":passed, "peak_allocated_gib":report["peak_allocated_gib"]}),flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
