"""Score official final PNGs against the actual 512-square model input on 213 training originals."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ORIGINAL = PROJECT / "results/official_sirst_train"
AUDIT = ORIGINAL / "float_audit"
S1 = HERE / "s1"
STAGES = ("published", "s1")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(name, rows):
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def final_png(path):
    with Image.open(path) as im:
        if (im.mode, im.size) != ("RGB", (512, 512)):
            raise ValueError(f"Unexpected final PNG: {path}")
        return np.asarray(im).astype(np.float32).mean(axis=2)


def whole_metrics(ref, rec):
    a, b = ref.astype(np.float64), rec.astype(np.float64)
    error = b-a
    mse = np.square(error).mean()
    return dict(mae_dn=float(np.abs(error).mean()),
                psnr_db=float(10*np.log10(255**2/mse)) if mse else None,
                ssim=float(structural_similarity(a, b, win_size=11, data_range=255, gaussian_weights=True,
                                                 sigma=1.5, use_sample_covariance=False)))


def project(native):
    return np.asarray(Image.fromarray(native).resize((512, 512), Image.Resampling.NEAREST))


def components(mask_path):
    with Image.open(mask_path) as im:
        native = np.asarray(im.convert("L")) != 0
    labels, count = ndimage.label(native, structure=np.ones((3, 3), bool))
    projected = project(labels.astype(np.int32))
    for i in range(1, count+1):
        core = projected == i
        native_ring = ndimage.binary_dilation(labels == i, structure=np.ones((5, 5), bool)) & ~native
        ring = project(native_ring.astype(np.uint8)).astype(bool)
        if not core.any() or not ring.any():
            raise ValueError(f"Empty projected support on {mask_path}, component {i}")
        yield i, core, ring


def main():
    complete = S1 / "complete.json"
    if not complete.is_file():
        raise FileNotFoundError("The complete formal S1 replay is required")
    inputs = ORIGINAL / "input_manifest.json"
    train = PROJECT / "manifests/train.jsonl"
    bundled = PROJECT / "datasets/NUAA-SIRST/img_idx/train_NUAA-SIRST.txt"
    ids = bundled.read_text(encoding="utf-8-sig").split()
    completion = json.loads(complete.read_text())
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    initial = json.loads((AUDIT/"summary.json").read_text())
    formal = json.loads((PROJECT/"results/local_retrain/training_summary.json").read_text())
    assert completion["status"] == "complete" and completion["test_images"] == 0
    assert completion["count"] == completion["exact_input_tensor_matches"] == completion["exact_repostprocessed_pngs"] == 213
    assert completion["split_sha256"] == protocol["split_sha256"] == digest(bundled)
    assert completion["weights_sha256"] == formal["checkpoint"]["sha256"] == protocol["s1_weights_sha256"]
    assert completion["official_weights_sha256"] == initial["checkpoint_sha256"] == protocol["published_weights_sha256"]
    assert completion["source_entry_sha256"] == initial["source_entry_sha256"] == protocol["source_sha256"]
    rows = json.loads(inputs.read_text())
    train_rows = [json.loads(x) for x in train.read_text().splitlines()]
    replay_rows = csv_rows(AUDIT / "per_image.csv")
    replay_targets = {(r["id"], int(r["component"])):r for r in csv_rows(AUDIT / "per_target.csv")}
    assert len(ids) == len(set(ids)) == 213
    assert ids == [r["id"] for r in rows] == [r["source_image_id"] for r in train_rows] == [r["id"] for r in replay_rows]
    assert all(r["dataset"] == "NUAA-SIRST" and r["split"] == "train" for r in train_rows)
    official_names = [r["official_filename"] for r in rows]
    assert [r["filename"] for r in completion["completed_rows"]] == official_names
    outdirs = {s:d/"testset/VARPrediction" for s,d in zip(STAGES, (AUDIT, S1), strict=True)}
    for d in outdirs.values():
        assert sorted(p.name for p in d.glob("*.png")) == official_names
    image_scores, target_scores = [], []
    for r,t,a,done in zip(rows, train_rows, replay_rows, completion["completed_rows"], strict=True):
        image_id, filename = r["id"], r["official_filename"]
        src = PROJECT / t["reference_path"]
        mask = PROJECT / t["mask_path"]
        prev_arr = AUDIT / a["arrays"]
        new_arr = S1 / "arrays" / (Path(filename).stem+".npz")
        assert src == PROJECT / r["input_png"]
        assert digest(src) == a["source_sha256"] == r["image_sha256"] == t["image_sha256"]
        assert digest(prev_arr) == a["array_sha256"]
        assert digest(new_arr) == done["s1_npz_sha256"]
        with np.load(prev_arr, allow_pickle=False) as rec:
            observed = rec["input_chw"]
        with np.load(new_arr, allow_pickle=False) as rec:
            assert np.array_equal(rec["input_chw"], observed)
        with Image.open(src) as im:
            prepared = np.asarray(im.convert("RGB").resize((512, 512))).astype(np.float32)/255*2-1
        x = observed.transpose(1,2,0)
        assert x.dtype == np.float32 and np.array_equal(x, prepared)
        reference = ((x+1)*127.5).mean(axis=2)
        paths = {s:d/filename for s,d in outdirs.items()}
        outputs = {s:final_png(p) for s,p in paths.items()}
        assert digest(paths["published"]) == a["replay_png_sha256"]
        assert digest(paths["s1"]) == done["s1_png_sha256"]
        result = dict(id=image_id, filename=filename, source_sha256=t["image_sha256"],
                      native_mask_sha256=digest(mask), input_chw_sha256=hashlib.sha256(observed.tobytes()).hexdigest())
        for s in STAGES:
            result[s+"_final_png_sha256"] = digest(paths[s])
            result.update({s+"_"+k:v for k,v in whole_metrics(reference, outputs[s]).items()})
        local = []
        for i,core,ring in components(mask):
            def contrast(image):
                return float(np.mean(image[core], dtype=np.float64) - np.median(image[ring]))
            measures = dict(id=image_id, component=i, projected_core_pixels=int(core.sum()),
                            projected_ring_pixels=int(ring.sum()), input_contrast_dn=contrast(reference))
            for s in STAGES:
                v = contrast(outputs[s])
                delta = v-measures["input_contrast_dn"]
                measures.update({s+"_contrast_dn":v, s+"_contrast_bias_dn":delta, s+"_contrast_abs_error_dn":abs(delta)})
            previous = replay_targets[(image_id, i)]
            assert measures["projected_core_pixels"] == int(previous["projected_core_pixels"])
            assert measures["projected_ring_pixels"] == int(previous["projected_ring_pixels"])
            assert abs(measures["published_contrast_dn"] - float(previous["final_contrast_dn"])) < 1e-6
            assert abs(measures["input_contrast_dn"] - float(previous["input_contrast_dn"])) < 1e-6
            local.append(measures)
        result["components"] = len(local)
        for k in ("input_contrast_dn", *(s+"_"+v for s in STAGES for v in ("contrast_dn","contrast_bias_dn","contrast_abs_error_dn"))):
            result[k] = float(np.mean([q[k] for q in local]))
        target_scores.extend(local)
        image_scores.append(result)
        if len(image_scores) % 50 == 0:
            print(f"Scored {len(image_scores)} of {len(rows)}", flush=True)
    assert len(target_scores) == len(replay_targets) == 270
    per_image, per_target = "image_quality_per_image.csv", "image_quality_per_target.csv"
    write_rows(per_image, image_scores)
    write_rows(per_target, target_scores)
    means = {}
    directions = {"mae_dn":-1,"ssim":1,"psnr_db":1,"contrast_abs_error_dn":-1}
    for s in STAGES:
        keys = ("mae_dn","ssim","psnr_db","contrast_dn","contrast_bias_dn","contrast_abs_error_dn")
        means[s] = {k:float(np.mean([r[s+"_"+k] for r in image_scores])) for k in keys}
    paired = {}
    for k,direction in directions.items():
        changes = np.array([r["s1_"+k]-r["published_"+k] for r in image_scores])
        paired[k] = {"mean_s1_minus_published": float(changes.mean()), "images_improved":int((changes*direction>0).sum())}
    sources = (inputs, bundled, train, protocol_path, Path(__file__).resolve(), AUDIT/"summary.json", AUDIT/"per_image.csv", AUDIT/"per_target.csv",
               PROJECT/"vendor/Real-IISR/test.py", PROJECT/"vendor/Real-IISR/dataloader/testdataset.py", complete)
    result = dict(split="NUAA-SIRST/train", n_images=len(rows), n_components=len(target_scores),
                  shared_inputs_exact=213, published_targets_replayed=270,
                  input_mean_contrast_dn=float(np.mean([r["input_contrast_dn"] for r in image_scores])),
                  image_weighted_means=means, paired=paired,
                  protocol={"reference":"Actual input_chw from both captures, bitwise equal and rechecked versus original image through official RGB resize/ToTensor. RGB arithmetic mean in 0–255 DN. Same raw source assigned to HR and LR.",
                            "outputs":"Final 512x512 RGB uint8 PNG after rounding and AdaIN at both checkpoints. Published PNGs use the same replay as the captured input (float_audit), not the first archived execution.",
                            "whole":"Grayscale full-frame MAE and PSNR peak=255; Gaussian SSIM 11x11 sigma=1.5 with valid windows, population covariance, K1=.01/K2=.03. Per-image metrics, equally weighted across images.",
                            "local":"Label native mask using 8-connectivity; square dilation by 2 native pixels excluding all targets. Nearest-neighbor projection of original core labels/ring masks to 512x512. Contrast is core mean minus ring median in DN. Bias is output minus input. Absolute errors averaged by component within each image, then equally over images.",
                            "scope":"Identity preservation on unmodified local training images; reference is the resized input. No independent clean observation, synthetic corruption or test images."},
                  checkpoint_sha256={"published":initial["checkpoint_sha256"], "s1":formal["checkpoint"]["sha256"]},
                  provenance_sha256={str(p.relative_to(PROJECT)):digest(p) for p in sources},
                  score_files_sha256={s:digest(HERE/s) for s in (per_image,per_target)})
    (HERE/"image_quality_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(means), flush=True)


if __name__ == "__main__":
    main()
