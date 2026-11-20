"""Check the local training pairs, P/C losses, and the upstream thermal input feature."""

import argparse
import hashlib
import json
from runpy import run_path
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from prd_data import DenoisePairDataset
from prd_losses import paired_loss


def check_loss_gradients():
    support = torch.zeros(3, 1, 2, 5, dtype=torch.bool)
    support[:, :, 0, 1] = True
    valid = torch.ones_like(support)
    valid[..., -1] = False
    ring = torch.zeros_like(support)
    ring[:, :, 1, 1] = True
    batch = dict(signal=support.float() / 255, support_mask=support, ring_mask=ring,
                 valid_mask=valid, pair_valid=torch.tensor([True, False, True]))
    result = {}
    for variant, expected in (("P", 1 / (255 * 8)), ("C", 1.0)):
        plus = torch.zeros_like(batch["signal"], requires_grad=True)
        minus = torch.zeros_like(plus, requires_grad=True)
        loss = paired_loss(minus, plus, batch, variant)
        torch.testing.assert_close(loss, torch.tensor(expected))
        loss.backward()
        assert plus.grad.abs().sum() > 0 and torch.equal(plus.grad, -minus.grad)
        assert not plus.grad[1].any() and not plus.grad[..., -1].any()
        empty = dict(batch, pair_valid=torch.zeros(3, dtype=torch.bool))
        assert paired_loss(minus, plus, empty, variant).item() == 0
        result[variant] = {"analytic_loss": expected, "measured_loss": loss.item()}
    return result


def check_pairs(root, seed):
    # This diagnostic uses only the upstream transform, without initializing the full model.
    heatmap = run_path(str(root / "vendor/Real-IISR/models/src/Heatmap.py"))
    compute_thermal_response_map = heatmap["compute_thermal_response_map"]

    dataset = DenoisePairDataset(root, seed=seed)
    stats = defaultdict(list)
    skipped = []
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = {key: torch.from_numpy(value).unsqueeze(0)
                 for key, value in sample.items() if isinstance(value, np.ndarray)}
        batch["pair_valid"] = torch.tensor([sample["pair_valid"]])
        image, reference = batch["noisy_minus"], batch["reference_minus"]
        valid, background = batch["valid_mask"], batch["background_mask"]
        spot, ring = batch["support_mask"], batch["ring_mask"]
        signal = batch["signal"]
        assert not ((spot | ring) & (~valid | batch["native_mask"])).any()
        assert not (spot & ring).any() and not (background & ~valid).any()
        torch.testing.assert_close(batch["reference_plus"] - reference, signal, atol=1e-7, rtol=0)
        torch.testing.assert_close(batch["noisy_plus"] - image, signal, atol=1.2e-7, rtol=0)
        delta_error = batch["noisy_plus"] - image - signal
        stats["noisy_pair_max_abs_error_01"].append(delta_error.abs().amax().item())
        if background.any():
            bg_error = (image - reference)[background]
            stats["identity_background_rmse_dn"].append(bg_error.square().mean().sqrt().item() * 255)

        if (~valid).any():
            # Change only padding values, using the upstream feature on the [-1, 1] input.
            x = image * 2 - 1
            alt = torch.where(valid, x, x[valid].mean())
            diff = (compute_thermal_response_map(x) - compute_thermal_response_map(alt)).abs()
            stats["thermal_map_padding_mae_on_valid_pixels"].append(diff[valid].mean().item())

        if not sample["pair_valid"]:
            skipped.append(sample["meta"]["group_id"])
            continue
        for variant in ("P", "C"):
            identity = paired_loss(image, batch["noisy_plus"], batch, variant)
            zero = paired_loss(reference, batch["reference_plus"], batch, variant)
            erased = paired_loss(reference, reference, batch, variant)
            half = paired_loss(reference, reference + 0.5 * signal, batch, variant)
            assert zero.item() == 0 and erased.item() > 0
            torch.testing.assert_close(half, 0.5 * erased, rtol=1e-3, atol=1e-9)
            limit = 2e-7 if variant == "P" else 4e-5
            assert identity.item() < limit, (sample["meta"]["group_id"], variant, identity.item())
            stats[f"identity_{variant}"].append(identity.item())
            stats[f"erased_{variant}"].append(erased.item())

    if not stats["erased_P"]:
        raise RuntimeError("No valid calibration pair to check")
    summaries = {key: dict(n=len(values), mean=float(np.mean(values)), max=float(np.max(values)))
                 for key, values in stats.items()}
    return dict(dataset="NUAA-SIRST", split="train", seed=seed,
                manifest_sha256=hashlib.sha256((root / "manifests/train.jsonl").read_bytes()).hexdigest(),
                image_count=len(dataset), pair_valid_count=len(stats["erased_P"]),
                invalid_pair_ids=skipped, controls=summaries,
                padding_comparison="zero DN versus per-image mean DN; thermal feature only",
                scale="DN values on a 0–255 scale; P, C and thermal features dimensionless")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        result = check_pairs(Path(__file__).resolve().parent, args.seed)
    result["analytic_loss_and_output_gradient_checks"] = check_loss_gradients()
    report = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
