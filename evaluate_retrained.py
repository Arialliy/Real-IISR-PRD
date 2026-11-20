"""Measure the locally retrained Real-IISR on a fixed training draw."""
import argparse
from pathlib import Path

import torch

from evaluate_warmup import file_hash, paired_changes, prepare_metrics, sample_digest, save_report, summarize
from prd_backbone import official_inference
from prd_data import DenoisePairDataset
from probe_backbone import ROOT, model_input, to_gray_01


def hard_prediction(model, gray, device, *, greedy=False):
    """Reconstruct one frame with the common FP16 hard-inference entry."""
    image = model_input(gray, device)
    if greedy:
        from verify_rollout import cached_greedy
        output = cached_greedy(model, image)[0]
    else:
        output = official_inference(model, image)
    return to_gray_01(output).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sirst_retrained_baseline.pt")
    parser.add_argument("--draw-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from train_local_baseline import load_trained

    device = torch.device("cuda:0")
    model = load_trained(args.checkpoint, device)
    data = DenoisePairDataset(ROOT, seed=0)
    rows = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for i in range(len(data)):
            sample = data.sample(i, epoch=0, draw_id=args.draw_id)
            noisy = sample["noisy_minus"]
            predicted = hard_prediction(model, noisy, device)
            measure, coverage = prepare_metrics(sample)
            rows.append({
                "sample_id": sample["meta"]["sample_id"], "data_sha256": sample_digest(sample),
                "coverage": coverage, "metrics": {"input": measure(noisy[0]), "retrained": measure(predicted)},
            })
    report = {
        "split": "NUAA-SIRST/train", "manifest_sha256": file_hash(ROOT / "manifests/train.jsonl"),
        "checkpoint_sha256": file_hash(args.checkpoint), "data_seed": 0, "epoch": 0, "draw_id": args.draw_id,
        "branch": "noisy_minus against reference_minus", "units": "0–255 grayscale DN",
        "hard_inference": "upstream top_k=1 top_p=0.75 cfg=1 g_seed=0 FP16, no color matching",
        "aggregation": "per-image metric then equal-weight mean over eligible images",
        "samples": rows, "all_images": summarize(rows),
        "change_from_input": paired_changes(rows, "input", "retrained"),
    }
    output = args.output or ROOT / f"results/local_retrain/train_eval_draw{args.draw_id}.json"
    save_report(output, report)
    print(f"Evaluated {len(rows)} training images; saved {output}", flush=True)


if __name__ == "__main__":
    main()
