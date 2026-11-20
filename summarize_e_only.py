"""Verify the six SIRST endpoint/draw reports and summarize paired E-only versus global CAC."""
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np


OUT = Path(__file__).resolve().parent / "results/denoise_v2/token_sources"
SOURCES = ("predicted", "input", "reference")
METRICS = ("whole_mae_dn", "whole_psnr_db", "whole_ssim", "plan_background_mean_error_dn",
           "plan_background_rmse_dn", "weighted_background_column_residual_dn", "native_target_mae_dn",
           "native_median_ring_contrast_abs_error_dn", "native_median_ring_contrast_bias_dn")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def collect(endpoint, draw):
    stem = f"{endpoint}_{draw}"
    path = OUT/(stem+"_six_routes.json")
    item = json.loads(path.read_text())
    capture = json.loads((OUT/(stem+"_tokens.json")).read_text())
    archive = OUT/capture["token_archive"]
    assert digest(archive) == capture["tokens_and_h_sha256"] == item["tokens_sha256"]
    assert item["metrics_version"]=="native_v2" and item["original_native_rasters_verified"]==213
    assert (item["endpoint"],item["draw"]) == (endpoint,draw)
    assert len(item["samples"])==213 and item["native_targets"]==270
    assert item["existing_global_routes_verified"]==(3 if draw=="0" else 1)
    assert capture["clean_independent_encodings_verified"]==(213 if draw=="clean" else 0)
    with np.load(archive, allow_pickle=False) as archived:
        assert archived["h"].shape[0]==213 and np.isfinite(archived["h"]).all()
        codes={s:archived[s] for s in SOURCES}
        assert all(a.shape==(213,2240) for a in codes.values())
        assert list(archived["ids"])==[r["sample_id"] for r in item["samples"]]
        assert list(archived["input_sha256"])==[r["data_sha256"] for r in item["samples"]]
        if draw=="clean":
            assert np.array_equal(codes["input"],codes["reference"])
    return item, codes, {path.name:digest(path), archive.name:digest(archive)}


def main():
    results, witnesses, hashes = {}, {}, set()
    report_digests = {}
    for draw in ("0","clean","1"):
        first = {}
        for endpoint in ("s1","s2"):
            report,codes,files = collect(endpoint,draw)
            report_digests.update(files)
            hashes.add((report["tokenizer_sha256"],report["decoder_sha256"],report["manifest_sha256"],report["native_v2_functions_sha256"]))
            rows=report["samples"]
            if endpoint=="s1":
                first=dict(codes=codes, rows=rows)
            else:
                for s in ("reference","input"):
                    assert np.array_equal(first["codes"][s],codes[s])
                    for old,new in zip(first["rows"],rows,strict=True):
                        assert (old["sample_id"],old["data_sha256"])==(new["sample_id"],new["data_sha256"])
                        assert old["native_pixel_sha256"][s+"_base"]==new["native_pixel_sha256"][s+"_base"]
            reference_hashes=[r["native_pixel_sha256"]["reference_base"] for r in rows]
            if "reference_base" not in witnesses:
                witnesses["reference_base"]=reference_hashes
            assert witnesses["reference_base"]==reference_hashes
            data={}
            for source in SOURCES:
                group={}
                for decoder in ("base","cac"):
                    route=source+"_"+decoder
                    keys={k:mean(r["metrics"][route][k] for r in rows) for k in METRICS}
                    assert all(np.isclose(keys[k],report["per_image_summary"][route][k]["mean"],rtol=0,atol=1e-12) for k in METRICS)
                    group[decoder]=keys
                diff=[r["paired_decode_pixels"][source] for r in rows]
                group["between_same_token_decodes"]={
                    "mean_image_mae_dn": mean(r["mean_abs_dn"] for r in diff),
                    "largest_pixel_difference_dn": max(r["max_abs_dn"] for r in diff),
                    "exact_native_images":sum(r["identical_pixels"]==r["pixels"] for r in diff),
                    "pixel_fraction_identical":sum(r["identical_pixels"] for r in diff)/sum(r["pixels"] for r in diff)}
                data[source]=group
            results[endpoint+"_"+draw]=data
    assert len(hashes)==1
    token_h,decoder_h,manifest_h,metric_h=hashes.pop()
    summary = dict(cases=results, training_manifest_sha256=manifest_h, tokenizer_sha256=token_h,
                   decoder_sha256=decoder_h, native_v2_functions_sha256=metric_h, artifact_sha256=report_digests,
                   checked="For every draw, two endpoints yield identical reference/input VQ indices and corresponding E-only native rasters. Reference E-only native rasters agree across both endpoints and all draws. Both clean encodings and both clean reconstructions agree. Every predicted CAC raster reproduces the archived greedy output; draw0 additionally replays all three earlier global CAC metric rows.",
                   eligible_images_per_condition=213,native_components_per_condition=270,greedy_control_rasters_verified=1278)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print("Verified 1,278 predictor replays and 6 six-route tables.", flush=True)


if __name__=="__main__":
    main()
