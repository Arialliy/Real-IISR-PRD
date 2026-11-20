"""Evaluate predefined SIRST train noise levels and fixed column-filter strengths."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from check_column_baseline import correct_columns
from evaluate_protocol_v2 import measure_native, native_window, pixel_sha
from evaluate_warmup import file_hash, sample_digest, save_report
from prd_data import DenoisePairDataset

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs/denoise_v2/noise_levels.json"
OUT = ROOT / "results/protocol_v2_levels"
SPEC = json.loads(CONFIG.read_text())
CONDITIONS = SPEC["conditions"] | SPEC["column_curve_extra_conditions"]
FIELDS = ("whole_mae_dn", "whole_psnr_db", "whole_ssim", "plan_background_rmse_dn",
          "plan_background_mean_error_dn", "weighted_background_column_residual_dn",
          "native_target_mae_dn", "native_median_ring_contrast_abs_error_dn",
          "native_median_ring_contrast_bias_dn")


def dataset():
    data = DenoisePairDataset(ROOT, seed=SPEC["seed"])
    if len(data) != 213 or len({r["sample_id"] for r in data.records}) != len(data):
        raise ValueError("Incomplete or duplicated train population")
    return data


def get_sample(data, index, condition):
    return data.sample(index, epoch=SPEC["epoch"], **CONDITIONS[condition])


def check_data():
    data = dataset()
    archives = [json.loads((ROOT / f"results/sirst_endpoints/s1_draw{draw}.json").read_text())["samples"]
                for draw in (0, 1)]
    max_error = 0.
    for i, record in enumerate(data.records):
        for draw, archive in enumerate(archives):
            s = data.sample(i, epoch=0, draw_id=draw)
            assert record["sample_id"] == archive[i]["sample_id"]
            assert sample_digest(s) == archive[i]["data_sha256"]
        clean = get_sample(data, i, "clean")
        assert np.array_equal(clean["reference_minus"], clean["noisy_minus"])
        assert np.array_equal(clean["reference_plus"], clean["noisy_plus"])
        assert np.array_equal(clean["noisy_plus"] - clean["noisy_minus"], clean["signal"])
        low, medium, high = (get_sample(data, i, k) for k in ("low", "medium", "high"))
        x, a, b, c = (s[k][0].astype(np.float64) for s,k in (
            (low,"reference_minus"),(low,"noisy_minus"),(medium,"noisy_minus"),(high,"noisy_minus")))
        expected = np.clip(x + 2*(a-x), 0, 1)
        expected_high = np.clip(x + 4*(a-x), 0, 1)
        max_error = max(max_error, float(np.abs(expected-b).max()), float(np.abs(expected_high-c).max()))
    assert max_error < 5e-7
    for bad in ({}, {"stripe_sigma_dn": -1, "pixel_sigma_dn": 2},
                {"stripe_sigma_dn": 1, "pixel_sigma_dn": float("nan")}):
        try:
            data.sample(0, noise_override=bad)
        except ValueError:
            continue
        raise AssertionError("Invalid level accepted")
    hashes = {p.name: file_hash(p) for p in sorted((ROOT/"manifests").glob("*.jsonl"))}
    save_report(OUT/"data_checks.json", dict(original_train_images=len(data), original_draw_hashes_checked=426,
                zero_noise_exact_images=len(data), invalid_cases_rejected=3, manifest_sha256=hashes,
                shared_field_max_abs_error_0to1=max_error, config_sha256=file_hash(CONFIG)))


def get_model(name):
    import torch
    from prd_backbone import load_backbone
    from train_local_baseline import load_trained
    from train_s2_common import load_s2

    ckpt = ROOT / "checkpoints" / SPEC["models"][name]
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = {"published": load_backbone, "s1": load_trained, "s2": load_s2}[name](ckpt, device)
    return model, device, file_hash(ckpt)


def run(name, conditions):
    data = dataset()
    checkpoint_hash = None
    if name != "column":
        import torch
        from evaluate_retrained import hard_prediction
        model, device, checkpoint_hash = get_model(name)
        if name == "published":
            test = data.sample(0, epoch=0, draw_id=0)
            raw = ROOT / "results/sirst_endpoints/published_draw0_float" / (data.records[0]["source_image_id"] + ".npy")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                passes = [hard_prediction(model,test["noisy_minus"],device,greedy=True)[native_window(test)]
                          for _ in range(2)]
            assert np.array_equal(passes[0],passes[1])
            assert np.array_equal(passes[0], np.load(raw, allow_pickle=False))
            save_report(OUT/"published_repeat.json", dict(sample_id=data.records[0]["sample_id"],
                        matches_two_runs_and_previous_archive=True, native_prediction_sha256=pixel_sha(passes[0]),
                        archive_file_sha256=file_hash(raw), checkpoint_sha256=checkpoint_hash))
    for condition in conditions:
        if name != "column" and condition not in SPEC["conditions"]:
            raise ValueError("Clean and historical model outputs already exist under protocol_v2")
        images, targets = [], []
        for i, record in enumerate(data.records):
            sample = get_sample(data, i, condition)
            noisy = sample["noisy_minus"]
            box = native_window(sample)
            _, _, w, h = sample["meta"]["crop_xywh"]
            if sample["meta"]["crop_xywh"] != (0,0,*record["reference_size"]):
                raise ValueError("Expected entire original image")
            digest = pixel_sha(noisy[0][box])
            if name == "column":
                x, y = sample["meta"]["pad_left_top"]
                predictions = [(f"column_{s:g}" if s else "input",
                                correct_columns(noisy[0], x,y,w,h,window=SPEC["column_window"],strength=s)
                                if s else noisy[0]) for s in SPEC["column_strengths"]]
            else:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    predictions = [(name, hard_prediction(model, noisy, device, greedy=True))]
            for method, pred in predictions:
                metrics, rows = measure_native(sample, pred, details=True)
                ident = dict(condition=condition, method=method, sample_id=record["sample_id"])
                images.append(dict(**ident, input_sha256=digest, prediction_sha256=pixel_sha(pred[box]),
                                   **sample["meta"]["noise_parameters"],
                                   **{k:metrics[k] for k in FIELDS}))
                targets.extend(ident | row for row in rows)
        target = OUT / "shards" / f"{name}_{condition}.json"
        save_report(target, dict(condition=condition, runner=name, config_sha256=file_hash(CONFIG),
                                 train_sha256=file_hash(ROOT/"manifests/train.jsonl"),
                                 checkpoint_sha256=checkpoint_hash, images=images, targets=targets))
        print(f"{target.name}: {len(images)} image records, {len(targets)} component records", flush=True)


def write_csv(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def check_main_table(grouped):
    """Reconcile independently generated conventional inputs and alpha=1 scores."""
    with (ROOT/"results/protocol_v2/per_image.csv").open(newline="") as f:
        main = {(r["corruption"],r["method"],r["sample_id"]):r for r in csv.DictReader(f)}
    checks = 0
    for condition in SPEC["column_curve_extra_conditions"]:
        for method, main_name in (("input","input"), ("column_1","column")):
            for r in grouped[condition,method]:
                other = main[condition, main_name, r["sample_id"]]
                assert other["input_pixels_sha256"] == r["input_sha256"]
                for k in FIELDS:
                    x, y = r[k], other[k]
                    if x is None:
                        assert y == ""
                    elif not np.isclose(x,float(y),rtol=0,atol=1e-10):
                        raise ValueError(f"Main table disagrees on {condition} {method} {k}: {x}, {y}")
                checks += 1
    return checks


def paired_intervals(grouped):
    """Paired percentile bootstrap across originals on the fresh noisy condition."""
    rng = np.random.default_rng(20261121)
    n = len(grouped["fresh","s1"])
    indices = rng.integers(n, size=(5000,n))
    output = []
    for first, second in (("s1","input"),("s1","column_1"),("s1","published"),("s2","s1")):
        for key in ("whole_mae_dn","plan_background_mean_error_dn","native_median_ring_contrast_abs_error_dn",
                    "weighted_background_column_residual_dn","plan_background_rmse_dn"):
            a,b = (grouped["fresh",m] for m in (first,second))
            if any(u["sample_id"] != v["sample_id"] for u,v in zip(a,b,strict=True)):
                raise ValueError("Bootstrap pairing lost")
            delta = np.array([u[key]-v[key] for u,v in zip(a,b,strict=True)])
            lo,hi = np.percentile(delta[indices].mean(axis=1), [2.5,97.5])
            output.append(dict(condition="fresh", first=first, second=second, metric=key,
                               n_originals=n, mean_first_minus_second=delta.mean(), ci025=lo, ci975=hi))
    write_csv(OUT/"fresh_paired_intervals.csv", output)


def summarize():
    from collections import defaultdict
    data = dataset()
    assets = {"config_sha256":file_hash(CONFIG), "train_sha256":file_hash(ROOT/"manifests/train.jsonl")}
    ckpts = {k:file_hash(ROOT/"checkpoints"/v) for k,v in SPEC["models"].items()}
    ids = [r["sample_id"] for r in data.records]
    grouped, shared = defaultdict(list), {}
    image_rows, target_rows, summary = [], [], []
    for name in ["column", *SPEC["models"]]:
        for condition in CONDITIONS if name == "column" else SPEC["conditions"]:
            obj = json.loads((OUT/"shards"/f"{name}_{condition}.json").read_text())
            if any(obj[k] != v for k,v in assets.items()) or obj["checkpoint_sha256"] != ckpts.get(name):
                raise ValueError("Shard provenance differs from config/checkpoint")
            image_rows.extend(obj["images"])
            target_rows.extend(obj["targets"])
            for row in obj["images"]:
                key = row["condition"],row["sample_id"]
                if shared.setdefault(key,row["input_sha256"]) != row["input_sha256"]:
                    raise ValueError("Comparators received different native inputs")
                grouped[condition,row["method"]].append(row)
    for (condition, method), rows in grouped.items():
        if [i["sample_id"] for i in rows] != ids:
            raise ValueError("Records do not match complete train population")
        means = {}
        for k in FIELDS:
            v = [r[k] for r in rows if r[k] is not None]
            perfect = k=="whole_psnr_db" and len(v)!=len(rows)
            means[k] = float(np.mean(v)) if v and not perfect else None
        summary.append(dict(condition=condition, method=method, images=len(rows),
                            perfect_psnr_images=sum(r["whole_psnr_db"] is None for r in rows), **means))
    write_csv(OUT/"per_image.csv", image_rows)
    write_csv(OUT/"per_target.csv", target_rows)
    write_csv(OUT/"means.csv", summary)
    paired_intervals(grouped)
    checked = check_main_table(grouped)
    save_report(OUT/"verification.json", dict(**assets, checkpoint_sha256=ckpts, paired_native_inputs=len(shared),
                 matching_conventional_rows_from_main_table=checked, conditions=len(CONDITIONS),
                 groups=len(grouped), image_rows=len(image_rows), target_rows=len(target_rows)))
    plot(summary)


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2,figsize=(10,4.5),layout="constrained")
    for ax, x, label in zip(axs, ("weighted_background_column_residual_dn","plan_background_rmse_dn"),
                            ("Background column residual (DN)", "Background RMSE (DN)")):
        for condition in ("low","medium","high","fresh","clean"):
            lookup = {r["method"]:r for r in summary if r["condition"]==condition}
            curve = [lookup["input"]] + [lookup[f"column_{a:g}"] for a in SPEC["column_strengths"] if a]
            y = "native_median_ring_contrast_abs_error_dn"
            line, = ax.plot([r[x] for r in curve],[r[y] for r in curve],marker="o",label=condition)
            ax.plot(lookup["column_1"][x],lookup["column_1"][y],marker="s",mfc="white",color=line.get_color())
        ax.set(xlabel=label,ylabel="Contrast absolute error (DN)")
        ax.grid(alpha=.25)
    axs[0].legend(title="Noise condition",fontsize=8)
    fig.suptitle("Fixed column correction strengths on 213 training images (squares mark 1.0)")
    fig.savefig(OUT/"column_strengths.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["checks","summary","column",*SPEC["models"]])
    parser.add_argument("conditions", nargs="*", choices=None)
    args = parser.parse_args()
    if args.action=="checks":
        check_data()
    elif args.action=="summary":
        summarize()
    else:
        conditions = args.conditions or list(CONDITIONS if args.action=="column" else SPEC["conditions"])
        run(args.action, conditions)
