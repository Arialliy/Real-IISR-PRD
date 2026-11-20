"""Verify original PNGs, compare native floating predictions and score SSIM in one protocol."""
import json
from pathlib import Path

import numpy as np
import skimage
from PIL import Image
from skimage.metrics import structural_similarity

from evaluate_sirst_endpoints import MANIFEST, OUT, ROOT, S1, file_hash, prepare_metrics, sample_digest, save_report
from prd_backbone import OFFICIAL_SHA256
from prd_data import DenoisePairDataset

FIELDS = ("whole_mae_dn", "whole_psnr_db", "whole_ssim", "plan_background_rmse_dn",
          "weighted_background_column_residual_dn", "native_median_ring_contrast_bias_dn",
          "native_median_ring_contrast_abs_error_dn", "native_target_mae_dn")


def main():
    data = DenoisePairDataset(ROOT, seed=0)
    hashes = {"published": OFFICIAL_SHA256, "s1": file_hash(S1)}
    membership = file_hash(MANIFEST)
    paths = {name: OUT / f"{name}_drawclean.json" for name in hashes}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for name, r in reports.items():
        if (r["endpoint"] != name or r["draw"] != "clean" or r["checkpoint_sha256"] != hashes[name]
                or r["manifest_sha256"] != membership or r["split"] != "NUAA-SIRST/train"
                or len(r["samples"]) != len(data) or r["protocol"]["cudnn_tf32"]
                or r["protocol"]["matmul_tf32"] or "greedy_argmax" not in r["protocol"]["samplers"]):
            raise ValueError(f"Invalid evaluation provenance: {name}")
    for key in reports["s1"]["protocol"]:
        if key != "samplers" and reports["s1"]["protocol"][key] != reports["published"]["protocol"][key]:
            raise ValueError(f"Endpoints differ in {key}")
    rows = []
    for i, record in enumerate(data.records):
        img_file = ROOT / record["reference_path"]
        if file_hash(img_file) != record["image_sha256"]:
            raise ValueError(f"Original image digest changed: {img_file}")
        with Image.open(img_file) as im:
            src = np.asarray(im.convert("L")).astype(np.float32) / 255
        sample = data.sample(i, epoch=0, draw_id=0)
        sample["noisy_minus"] = sample["reference_minus"]
        w, h = record["reference_size"]
        x0, y0 = sample["meta"]["pad_left_top"]
        valid = sample["valid_mask"][0]
        clean = sample["noisy_minus"][0]
        window = np.s_[y0:y0+h, x0:x0+w]
        if (sample["meta"]["crop_xywh"] != (0,0,w,h) or not valid[window].all() or valid.sum() != w*h
                or not np.array_equal(clean[window], src) or np.any(clean[~valid])):
            raise ValueError(f"Actual clean input differs from native PNG: {img_file}")
        identity = sample_digest(sample)
        measure, coverage = prepare_metrics(sample, extended=True)
        scores = {}
        for name in hashes:
            entry = reports[name]["samples"][i]
            if (entry["sample_id"] != record["sample_id"] or entry["data_sha256"] != identity
                    or entry["coverage"] != coverage or entry["metrics"]["input"]["whole_mae_dn"] != 0):
                raise ValueError(f"Sample input or support differs: {name}, {img_file}")
            raster = OUT / entry["native_gray_npy"]
            if file_hash(raster) != entry["native_gray_sha256"]:
                raise ValueError(f"Stored grayscale array digest differs: {raster}")
            rec = np.load(raster, allow_pickle=False)
            if (rec.shape != src.shape or rec.dtype != np.float32 or not np.isfinite(rec).all()
                    or rec.min() < 0 or rec.max() > 1):
                raise ValueError(f"Invalid native prediction: {raster}")
            reconstructed = np.zeros_like(clean)
            reconstructed[window] = rec
            m = measure(reconstructed)
            if m != entry["metrics"]["greedy_argmax"]:
                raise ValueError(f"Recomputed per-image metrics disagree: {raster}")
            m["whole_ssim"] = float(structural_similarity(
                src.astype(np.float64), rec.astype(np.float64), data_range=1, win_size=11,
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
            scores[name] = {key: m[key] for key in FIELDS}
        rows.append({"id": record["sample_id"], "source_sha256": record["image_sha256"], "width": w, "height": h,
                     "input_sha256": identity, "native_components": coverage["eligible_native_components"],
                     "metrics": scores})
    if len(rows) != 213 or len({s["id"] for s in rows}) != 213:
        raise ValueError("Incomplete native image set")
    means = {name: {} for name in hashes}
    changes = {}
    for key in FIELDS:
        for name in hashes:
            values = [s["metrics"][name][key] for s in rows]
            if any(v is None for v in values):
                raise ValueError(f"Missing metric or infinite PSNR requires special reporting: {name}, {key}")
            means[name][key] = float(np.mean(values))
        diff = np.asarray([s["metrics"]["s1"][key]-s["metrics"]["published"][key] for s in rows])
        changes[key] = {"mean_s1_minus_published": float(diff.mean()), "n_equal": int((diff == 0).sum())}
        if key != "native_median_ring_contrast_bias_dn":
            better = (diff > 0) if key in ("whole_psnr_db", "whole_ssim") else (diff < 0)
            changes[key]["n_s1_better"] = int(better.sum())
    result = {"population":"NUAA-SIRST train, 213 original PNGs, zero extra noise, 270 native components",
              "verified_clean_inputs":len(rows), "verified_float_arrays":2*len(rows),
              "manifest_sha256":membership, "checkpoint_sha256":hashes, "report_sha256":{n:file_hash(p) for n,p in paths.items()},
              "processing":"Native grid, zero-padding centered to 512 square, deterministic cached greedy, cfg=1, FP16, both TF32 off; arithmetic mean RGB as float32 [0,1], no PNG rounding or AdaIN; remove all padding before SSIM, MAE and PSNR.",
              "ssim":"Native grayscale float64, data_range=1, Gaussian window 11x11 sigma=1.5, population covariance, valid windows, per-image arithmetic mean.",
              "skimage":skimage.__version__, "aggregation":"Every image equally weighted; native contrast uses component core mean minus median in two-pixel square ring, absolute error averaged over eligible instances within each image.",
              "native_components":sum(row["native_components"] for row in rows),
              "means":means,"paired_changes":changes,"per_image":rows}
    save_report(OUT / "published_vs_s1_clean.json", result)
    print(json.dumps({"images":len(rows),"components":result["native_components"],"means":means,"paired_changes":changes}),flush=True)


if __name__ == "__main__":
    main()
