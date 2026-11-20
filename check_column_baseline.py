"""Measure a fixed column-bias correction on the project's two SIRST training draws."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from evaluate_warmup import paired_changes, prepare_metrics, sample_digest, save_report, summarize
from prd_data import DenoisePairDataset


ROOT = Path(__file__).resolve().parent


def correct_columns(noisy, x, y, width, height):
    result = noisy.copy()
    field = result[y:y + height, x:x + width]
    local = median_filter(field, size=(1, 7), mode="reflect")
    bias = np.median(field - local, axis=0)
    field[:] = (field - (bias - np.median(bias))).clip(0, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true",
                        help="Also evaluate weighted column noise, PSNR and median-ring contrast.")
    args = parser.parse_args()
    data = DenoisePairDataset(ROOT, seed=0)
    report = {
        "method": "Per-row 7-pixel horizontal median with reflect boundaries on the unpadded input. "
                  "Subtract each column's median residual, centered across columns. Clip to [0,1].",
        "split": "NUAA-SIRST/train",
        "epoch": 0,
        "data_seed": 0,
        "branch": "noisy_minus",
        "draws": {},
    }
    for draw_id in (0, 1):
        suffix = "_draw1" if draw_id == 1 else ""
        trace = json.loads((ROOT / f"results/warmup_paired_eval{suffix}.json").read_text())
        if len(trace["samples"]) != len(data):
            raise ValueError("Training-list length changed")
        rows = []
        for i, saved in enumerate(trace["samples"]):
            sample = data.sample(i, epoch=0, draw_id=draw_id)
            identifier = sample["meta"]["sample_id"]
            digest = sample_digest(sample)
            if (identifier, digest) != (saved["sample_id"], saved["data_sha256"]):
                raise ValueError(f"Input or region changed for {identifier}")
            measure, _ = prepare_metrics(sample, extended=args.extended)
            left, top = sample["meta"]["pad_left_top"]
            _, _, width, height = sample["meta"]["crop_xywh"]
            noisy = sample["noisy_minus"][0]
            metrics = {"input": measure(noisy),
                       "column_correction": measure(correct_columns(noisy, left, top, width, height))}
            if any(metrics["input"][key] != value for key, value in saved["metrics"]["input"].items()):
                raise ValueError(f"Input metrics changed for {identifier}")
            rows.append({"sample_id": identifier, "data_sha256": digest, "metrics": metrics})
        report["draws"][draw_id] = {
            "all_images": summarize(rows),
            "paired_change_from_input": paired_changes(rows, "input", "column_correction"),
            "samples": rows,
        }
    output = "column_baseline_extended.json" if args.extended else "column_baseline.json"
    save_report(ROOT / "results" / output, report)


if __name__ == "__main__":
    main()
