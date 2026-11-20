"""Compare cached greedy and published hard sampling on the first eight local S1 training pairs."""

from unittest.mock import patch

import numpy as np
import torch

from evaluate_warmup import file_hash, prepare_metrics, sample_digest, save_report
from prd_backbone import OFFICIAL_SHA256, PROJECT, apply_warmup, load_backbone, official_inference
from prd_data import DenoisePairDataset
from probe_backbone import model_input, to_gray_01
from verify_rollout import cached_greedy
import json


def published(model, x):
    from models import var
    sampler = var.sample_with_top_k_top_p_
    tokens, nonfirst_maxima = [], []

    def capture(logits, **kwargs):
        # The release truncates logits in-place before multinomial sampling.
        first = logits.argmax(dim=-1)
        selected = sampler(logits, **kwargs)
        code = selected.squeeze(-1)
        tokens.append(code)
        nonfirst_maxima.append((first != code).sum().item())
        return selected

    with patch.object(var, "sample_with_top_k_top_p_", capture):
        out = official_inference(model, x)
    return out, tokens, nonfirst_maxima


def main():
    baseline_path = PROJECT / "results/warmup_paired_eval.json"
    baseline = json.loads(baseline_path.read_text())
    checkpoint = PROJECT / "checkpoints/s1_warmup.pt"
    digest = file_hash(checkpoint)
    if baseline["warmup"]["checkpoint_sha256"] != digest:
        raise ValueError("S1 checkpoint differs from main evaluation")
    datas = DenoisePairDataset(PROJECT, seed=0)
    device = torch.device("cuda:0")
    model = load_backbone(PROJECT / "checkpoints/Real-IISR.pth", device)
    apply_warmup(model, checkpoint)
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(8):
            s = datas.sample(i, epoch=0, draw_id=0)
            prior = baseline["samples"][i]
            if prior["sample_id"] != s["meta"]["sample_id"] or prior["data_sha256"] != sample_digest(s):
                raise ValueError("Input or reference differs from the main S1 trace")
            x = model_input(s["noisy_minus"], device)
            g, _, g_tokens = cached_greedy(model, x)
            f, f_tokens, tie_choices = published(model, x)
            valid = s["valid_mask"][0]
            m, _ = prepare_metrics(s)
            g, f = to_gray_01(g).numpy(), to_gray_01(f).numpy()
            g_stats, f_stats = m(g), m(f)
            for field in ("valid_mae_dn", "background_rmse_dn"):
                if abs(f_stats[field] - prior["metrics"]["warmup"][field]) > 1.e-5:
                    raise ValueError(f"Published sampler failed to replay {prior['sample_id']} {field}")
            rows.append(dict(
                sample_id=prior["sample_id"], data_sha256=prior["data_sha256"],
                valid_between_sampler_mae_dn=float(np.abs((g[valid]-f[valid]).astype(np.float64)).mean()*255.),
                greedy_valid_mae_to_reference_dn=g_stats["valid_mae_dn"],
                published_valid_mae_to_reference_dn=f_stats["valid_mae_dn"],
                greedy_background_rmse_dn=g_stats["background_rmse_dn"],
                published_background_rmse_dn=f_stats["background_rmse_dn"],
                differences_by_scale=[(a!=b).sum().item() for a,b in zip(g_tokens, f_tokens)],
                nonfirst_max_choices_by_scale=tie_choices))
    keys = [k for k in rows[0] if k.endswith("_dn")]
    summary = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    summary["images_with_any_different_token"] = sum(any(row["differences_by_scale"]) for row in rows)
    summary["different_token_positions"] = sum(sum(row["differences_by_scale"]) for row in rows)
    summary["total_token_positions"] = len(rows)*sum(size**2 for size in model.patch_nums)
    report = dict(base_checkpoint_sha256=OFFICIAL_SHA256, s1_checkpoint_sha256=digest,
                  reference_report_sha256=file_hash(baseline_path), split="train", data_seed=0,
                  epoch=0, draw_id=0, branch="noisy_minus", padding_dn=0, dtype="fp16 CUDA autocast",
                  samplers="upstream cached deterministic argmax and upstream cached top_k=1/top_p=0.75/g_seed=0; cfg=1 in both",
                  population="first eight training IDs in local bundled order, equal weight per image",
                  summary=summary, rows=rows)
    save_report(PROJECT / "results/s1_hard_sampler_check.json", report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
