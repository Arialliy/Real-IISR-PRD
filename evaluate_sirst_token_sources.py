"""Compare three token sources with the same noisy-image global CAC on the formal SIRST endpoints."""
import argparse
import json

import numpy as np
import torch

from evaluate_sirst_endpoints import MANIFEST, OUT, S1, S2, paired, summary
from evaluate_warmup import file_hash, prepare_metrics, sample_digest, save_report
from prd_backbone import PROJECT
from prd_data import DenoisePairDataset
from probe_backbone import model_input, run_branch, to_gray_01
from verify_rollout import cached_greedy


FIELDS = (
    "weighted_background_column_residual_dn", "native_median_ring_contrast_abs_error_dn",
    "native_median_ring_contrast_bias_dn", "whole_mae_dn", "whole_psnr_db",
)
PREDICTED = "pred_tokens_noisy_h"
NOISY = "noisy_tokens_noisy_h"
REFERENCE = "ref_tokens_noisy_h"


def greedy_output(model, image):
    return cached_greedy(model, image)[0]


def evaluate(endpoint):
    path = OUT / f"{endpoint}_draw0.json"
    exported = json.loads(path.read_text())
    data = DenoisePairDataset(PROJECT, seed=0)
    asset = S1 if endpoint == "s1" else S2
    manifest_sha, checkpoint_sha = file_hash(MANIFEST), file_hash(asset)
    records = exported["samples"]
    ids = [r["sample_id"] for r in data.records]
    if (len(ids) != 213 or len(set(ids)) != 213 or [r["sample_id"] for r in records] != ids
            or exported["split"] != "NUAA-SIRST/train" or exported["endpoint"] != endpoint
            or (exported["data_seed"], exported["epoch"], exported["draw"]) != (0, 0, "0")
            or exported["manifest_sha256"] != manifest_sha or exported["checkpoint_sha256"] != checkpoint_sha):
        raise ValueError("The endpoint evaluation must cover the identical formal checkpoint and 213 training samples")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    if endpoint == "s1":
        from train_local_baseline import load_trained
        model = load_trained(asset, device)
    else:
        from train_s2_common import load_s2
        model = load_s2(asset, device)
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i, (row, saved) in enumerate(zip(data.records, records, strict=True)):
            sample = data.sample(i, epoch=0, draw_id=0)
            noisy = sample["noisy_minus"]
            measure, coverage = prepare_metrics(sample, extended=True)
            digest = sample_digest(sample)
            w, h = row["reference_size"]
            if (saved["data_sha256"] != digest or saved["coverage"] != coverage
                    or sample["meta"]["crop_xywh"] != (0, 0, w, h) or coverage["valid_pixels"] != w*h):
                raise ValueError(f"Input identity or native support differs on {row['sample_id']}")
            images, ref_codes, pred_codes, noisy_h = run_branch(
                model, sample["reference_minus"], noisy, device, greedy_output)
            left, top = sample["meta"]["pad_left_top"]
            native_pred = images[PREDICTED].numpy()[top:top+h, left:left+w]
            archived = OUT / saved["native_gray_npy"]
            archive_array = np.load(archived, allow_pickle=False)
            if (file_hash(archived) != saved["native_gray_sha256"] or archive_array.dtype != np.float32
                    or not np.array_equal(native_pred, archive_array)):
                raise ValueError(f"Greedy native float raster differs on {row['sample_id']}")
            with torch.autocast("cuda", enabled=False):
                _, noisy_codes = model.vae.img_to_idxBl(model_input(noisy, device))
            if [c.shape for c in noisy_codes] != [c.shape for c in ref_codes]:
                raise ValueError("Noisy and reference VQ layouts differ")
            decoded = model.vae.idxBl_to_img_lora(noisy_codes, noisy_h, same_shape=True, last_one=True)
            images[NOISY] = to_gray_01((decoded + 1) * 0.5)
            metrics = {}
            for route in ("noisy_input", PREDICTED, NOISY, REFERENCE):
                full = measure(images[route].numpy())
                if route in ("noisy_input", PREDICTED):
                    control = "input" if route == "noisy_input" else "greedy_argmax"
                    if full != saved["metrics"][control]:
                        raise ValueError(f"Endpoint metrics differ on {row['sample_id']}/{control}")
                metrics[route] = {key: full[key] for key in FIELDS}
            rows.append(dict(sample_id=row["sample_id"], data_sha256=digest, native_size_wh=[w,h],
                             native_components=coverage["eligible_native_components"], metrics=metrics,
                             different_from_ref_tokens_by_scale={
                                 name: [int((a != b).sum()) for a,b in zip(tokens, ref_codes, strict=True)]
                                 for name,tokens in ((PREDICTED, pred_codes), (NOISY, noisy_codes))}))
            if len(rows) % 50 == 0:
                print(f"{endpoint}: verified {len(rows)} of {len(data)} training images", flush=True)
    report = dict(endpoint=endpoint, draw=0, epoch=0, data_seed=0, split=exported["split"],
                  manifest_sha256=manifest_sha, checkpoint_sha256=checkpoint_sha,
                  endpoint_json=path.name, endpoint_sha256=file_hash(path), verified_native_rasters=len(rows),
                  verified_raster_check="Exact float32 equality between predicted output and archived native greedy image; all endpoint input/greedy metrics equal.",
                  token_sources=[PREDICTED, NOISY, REFERENCE], token_counts_by_scale=[p*p for p in model.patch_nums],
                  protocol="Deterministic upstream cached argmax CFG=1, CUDA FP16 autocast prediction/decode, TF32 disabled. In accord with the official trainer, VQ encoding uses its FP32 nearest-code search. All three decodes share the global CAC, frozen VQ decoder, and the same h produced from the noisy input. Reference encoding is diagnostic and gives no theoretical upper bound.",
                  scoring="Native extent; same reference, mask, weighted columns and per-image median-ring contrast as the endpoint draw 0. Equal weight per image. PSNR on unpadded original pixels, peak 255 DN.",
                  all_images=summary(rows), contrasts_vs_prediction={r:paired(rows, PREDICTED, r) for r in (NOISY, REFERENCE)},
                  samples=rows)
    save_report(OUT / f"{endpoint}_draw0_token_sources.json", report)
    print(f"Saved {endpoint}, {len(rows)} exact native rasters", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, choices=("s1", "s2"))
    evaluate(parser.parse_args().endpoint)
