"""Score paired final decodes using the shared native_v2 image and target definitions."""
import hashlib
import inspect
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import evaluate_protocol_v2 as native
from evaluate_e_only import DEST, OUT, SOURCES, samples
from evaluate_sirst_endpoints import summary
from evaluate_warmup import file_hash, save_report
from probe_backbone import to_gray_01


DECODERS = ("base", "cac")
HISTORICAL = {"predicted": "pred_tokens_noisy_h", "input": "noisy_tokens_noisy_h", "reference": "ref_tokens_noisy_h"}


def score_image(sample, prior, box, pixels, older, draw, offsets, codes):
    expected = np.load(OUT/prior["native_gray_npy"], allow_pickle=False)
    if not np.array_equal(pixels["predicted_cac"][box], expected):
        raise ValueError(f"Stored tokens/h did not reproduce native greedy float32 output: {prior['sample_id']}")
    if draw == "clean":
        for final in DECODERS:
            if not np.array_equal(pixels["input_"+final], pixels["reference_"+final]):
                raise ValueError("Two independent encodings of the same clean image decoded differently")
    comparison = {}
    for source in SOURCES:
        diff = 255*(pixels[source+"_cac"][box].astype(np.float64)-pixels[source+"_base"][box])
        comparison[source] = {"mean_abs_dn":float(np.abs(diff).mean()), "max_abs_dn":float(np.abs(diff).max()),
                              "identical_pixels":int((diff==0).sum()), "pixels":int(diff.size)}
    pixels["image_input"] = sample["noisy_minus"][0]
    metrics, targets, hashes = {}, {}, {}
    for route, image in pixels.items():
        hashes[route] = native.pixel_sha(image[box])
        metrics[route], targets[route] = native.measure_native(sample, image, details=True)
        if route == "predicted_cac" or route == "image_input":
            historical = prior["metrics"]["greedy_argmax" if route=="predicted_cac" else "input"]
            if any(metrics[route][k] != v for k,v in historical.items()):
                raise ValueError(f"Stored endpoint metrics differ on {prior['sample_id']} / {route}")
    if older is not None:
        assert older["sample_id"] == prior["sample_id"] and older["data_sha256"] == prior["data_sha256"]
        for s,old_name in HISTORICAL.items():
            if any(metrics[s+"_cac"][k] != v for k,v in older["metrics"][old_name].items()):
                raise ValueError("Existing draw0 global CAC branch metrics failed to replay")
    token_diff = {}
    for source in ("predicted", "input"):
        token_diff[source] = [int(np.count_nonzero(codes[source][i:j] != codes["reference"][i:j]))
                              for i,j in zip(offsets[:-1], offsets[1:], strict=True)]
    return dict(sample_id=prior["sample_id"], data_sha256=prior["data_sha256"],
                native_cac_prediction_sha256=prior["native_gray_sha256"], native_pixel_sha256=hashes,
                metrics=metrics, native_targets=targets, paired_decode_pixels=comparison,
                differences_from_reference_codes_by_scale=token_diff)


def score(model, device, data, controls, endpoint, draw, identity):
    stem = DEST / f"{endpoint}_{draw}_tokens"
    record = json.loads(stem.with_suffix(".json").read_text())
    archive = stem.with_suffix(".npz")
    if any(record[k] != v for k,v in identity.items()) or record["tokens_and_h_sha256"] != file_hash(archive):
        raise ValueError("Token archive, model or training population hash differs")
    historical = (json.loads((OUT / f"{endpoint}_draw0_token_sources.json").read_text())["samples"]
                  if draw == "0" else None)
    with np.load(archive, allow_pickle=False) as f:
        captured = {k:f[k] for k in f.files}
    shapes = [p*p for p in model.patch_nums]
    offsets = np.r_[0,np.cumsum(shapes)]
    assert np.array_equal(offsets,captured["offsets"])
    rows, pending = [], deque()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16), ThreadPoolExecutor(max_workers=3) as workers:
        for i, (sample, prior, box) in enumerate(samples(data, draw, controls)):
            assert captured["ids"][i] == prior["sample_id"] and captured["input_sha256"][i] == prior["data_sha256"]
            h = torch.from_numpy(captured["h"][i:i+1]).to(device)
            pixels, codes = {}, {}
            for s in SOURCES:
                codes[s] = captured[s][i]
                tokens = torch.from_numpy(codes[s]).to(device, dtype=torch.long)[None].split(shapes, dim=1)
                base = model.vae.idxBl_to_img(tokens, same_shape=True, last_one=True)
                cac = model.vae.idxBl_to_img_lora(tokens, h, same_shape=True, last_one=True)
                pixels[s+"_base"] = to_gray_01((base+1)*0.5).numpy()
                pixels[s+"_cac"] = to_gray_01((cac+1)*0.5).numpy()
            pending.append(workers.submit(score_image, sample, prior, box, pixels,
                                          historical[i] if historical else None, draw, offsets, codes))
            if len(pending) >= 6:
                rows.append(pending.popleft().result())
            if (i+1)%65 == 0:
                print(f"{endpoint} draw {draw}: decoded {i+1}/213", flush=True)
        rows.extend(future.result() for future in pending)
    assert len(rows) == 213
    assert all(sum(len(r["native_targets"][k]) for r in rows)==270 for k in rows[0]["native_targets"])
    metrics_source = [native.measure_native,native.native_window,native.target_measurements,
                      native.prepare_metrics,native.checked_error_dn,native.fixed_background,native.pixel_sha]
    metric_code_hash = hashlib.sha256("".join(inspect.getsource(fn) for fn in metrics_source).encode()).hexdigest()
    report = dict(**identity, endpoint=endpoint, split="NUAA-SIRST/train", draw=draw, data_seed=0, epoch=0,
                  metrics_version=native.VERSION, native_v2_functions_sha256=metric_code_hash,
                  code_sha256={"score":file_hash(Path(__file__)), "capture":file_hash(Path(__file__).with_name("evaluate_e_only.py"))},
                  tokens_sha256=record["tokens_and_h_sha256"], global_h_dtype=record["global_h_dtype"],
                  precision="FP32 native VQ encoding and nearest-code search; CUDA FP16 hard cached AR and final decodes; first-index argmax, CFG=1, TF32 disabled",
                  comparison="Each source uses identical saved multi-scale token IDs in both decodes. The global CAC uses the identical input-image h for all three sources. E-only switches only the final code lookup to its base table; the AR feedback and the frozen image decoder are unchanged.",
                  original_native_rasters_verified=len(rows), existing_global_routes_verified=3 if historical else 1,
                  native_targets=270, token_counts_per_scale=shapes, per_image_summary=summary(rows), samples=rows)
    save_report(DEST/f"{endpoint}_{draw}_six_routes.json",report)
    print(f"{endpoint} draw {draw}: native_v2 metrics, targets and archived outputs verified", flush=True)
