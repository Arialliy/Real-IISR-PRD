"""Measure paired calibration-spot transfer through a fixed native detail path on NUAA train."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from capture_native_detail_pairs import load_pair
from diagnose_spot_transfer import DOSES, patch_metrics
from evaluate_noise_levels import write_csv
from evaluate_warmup import file_hash, save_report
from native_detail import iter_methods

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "results/denoise_v2/native_detail"
CAPTURE = EXPERIMENT / "capture"
OUT = EXPERIMENT / "spots"
HISTORICAL = ROOT / "results/denoise_v2/spot_transfer"
CONDITIONS = ("clean", "draw0", "draw1", "fresh")
METHODS = ("input", "column", "s1", "detail_half", "detail_full", "full_blend_half")
SEED = 20261121
BOOT_REPS = 5000


def read_json(path):
    return json.loads(path.read_text())


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(n=0, mean=None, q25=None, median=None, q75=None)
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return dict(n=len(values), mean=float(np.mean(values)), q25=float(q25),
                median=float(median), q75=float(q75))


def load_history():
    with (HISTORICAL / "per_pair_stage.csv").open() as stream:
        history = {(r["sample_id"], r["route"]): r for r in csv.DictReader(stream)
                   if r["route"] in ("reference_pair", "s1_deployment")}
    return history


def verify_clean_history(row, measured, previous):
    checks = dict(measured)
    checks.update(contrast_retention_ratio=row["retention_ratio"],
                  input_incremental_contrast_dn=row["reference_increment_dn"],
                  input_width_px=row["reference_width_px"], width_delta_px=row["width_error_px"])
    if row["nominal_peak_dn"] != int(previous["nominal_peak_dn"]):
        raise ValueError("Historical spot dose changed")
    for name, actual in checks.items():
        expected = previous[name]
        if actual is None:
            match = expected == ""
        elif isinstance(actual, str):
            match = expected == actual
        else:
            match = abs(float(expected) - actual) <= 1e-10
        if not match:
            raise ValueError(f"Historical disagreement {row['sample_id']}, {row['method']}, {name}")


def measure_source(condition, pair, clean, spot, history):
    sample_id = spot["sample_id"]
    core, ring = pair["core"], pair["ring"]
    if not np.array_equal(clean["core"], core) or not np.array_equal(clean["ring"], ring):
        raise ValueError(f"Clean and noisy frames use different spots on {sample_id}")
    y, x = np.nonzero(core | ring)
    box = np.s_[y.min():y.max()+1, x.min():x.max()+1]
    core, ring = core[box], ring[box]

    def measure(a, b):
        return patch_metrics(a[box], b[box], core, ring)

    original = measure(clean["x_minus"], clean["x_plus"])
    reference_delta = original["incremental_contrast_dn"]
    if reference_delta <= 0:
        raise ValueError(f"No positive reference injection on {sample_id}")
    x_minus, x_plus = pair["x_minus"], pair["x_plus"]
    input_delta = measure(x_minus, x_plus)["incremental_contrast_dn"]
    injected = clean["x_plus"] - clean["x_minus"]
    unclipped = x_minus + injected
    clip_pixels = int((unclipped > 1).sum())
    if not np.array_equal(np.clip(unclipped, 0, 1), x_plus):
        raise ValueError(f"Input pair differs from the common-injection specification on {sample_id}")
    reference_width = original["width_px"]
    rows = []
    minus_routes = iter_methods(x_minus, pair["s1_minus"])
    plus_routes = iter_methods(x_plus, pair["s1_plus"])
    for (method, minus), (plus_name, plus) in zip(minus_routes, plus_routes, strict=True):
        if method != plus_name:
            raise ValueError("Misaligned reconstruction methods")
        measured = measure(minus, plus)
        output_delta = measured["incremental_contrast_dn"]
        gain = output_delta / reference_delta
        width = measured["width_px"]
        width_error = None if width is None or reference_width is None else width-reference_width
        row = dict(condition=condition, method=method, sample_id=sample_id,
                   nominal_peak_dn=spot["nominal_peak_dn"], reference_increment_dn=reference_delta,
                   input_increment_dn=input_delta, injection_clipped_pixels=clip_pixels,
                   retention_ratio=gain, retention_vs_input_ratio=output_delta/input_delta if input_delta>0 else None,
                   retained_at_least_half=gain >= .5, within_half_of_reference=abs(gain-1) <= .5,
                   abs_contrast_increment_error_dn=abs(output_delta-reference_delta),
                   reference_width_px=reference_width, width_error_px=width_error,
                   abs_width_error_px=abs(width_error) if width_error is not None else None,
                   **measured)
        if condition == "clean" and method in ("input", "s1"):
            historical_route = "reference_pair" if method == "input" else "s1_deployment"
            verify_clean_history(row, measured, history[sample_id, historical_route])
        rows.append(row)
    if len(rows) != len(METHODS) or {r["method"] for r in rows} != set(METHODS):
        raise ValueError("A fixed comparator was lost or duplicated")
    return rows


def group_summary(rows):
    gains = [r["retention_ratio"] for r in rows]
    return dict(
        n_images=len(rows),
        retained_at_least_half_n=sum(r["retained_at_least_half"] for r in rows),
        retention_0p5_to_1p5_n=sum(r["within_half_of_reference"] for r in rows),
        negative_gain_n=sum(g < 0 for g in gains),
        input_increment_nonpositive_n=sum(r["input_increment_dn"] <= 0 for r in rows),
        injection_clipped_pairs_n=sum(r["injection_clipped_pixels"] > 0 for r in rows),
        injection_clipped_pixels=sum(r["injection_clipped_pixels"] for r in rows),
        width_status=dict(Counter(r["width_reason"] for r in rows)),
        **{k: describe([r[k] for r in rows]) for k in (
            "reference_increment_dn", "input_increment_dn", "incremental_contrast_dn",
            "retention_ratio", "retention_vs_input_ratio", "abs_contrast_increment_error_dn",
            "reference_width_px", "width_px", "abs_width_error_px", "width_error_px")})


def paired_intervals(groups):
    output = []
    rng = np.random.default_rng(SEED)
    for condition in CONDITIONS:
        candidate = groups[condition, "detail_full"]
        indices = rng.integers(0, len(candidate), size=(BOOT_REPS, len(candidate)))
        for comparator in ("s1", "column", "full_blend_half"):
            control = groups[condition, comparator]
            if [r["sample_id"] for r in candidate] != [r["sample_id"] for r in control]:
                raise ValueError("Bootstrap rows must be paired per source")
            for field, statistic in (("retention_ratio", np.median), ("retained_at_least_half", np.mean),
                                     ("abs_contrast_increment_error_dn", np.mean)):
                a, b = (np.asarray([r[field] for r in rows]) for rows in (candidate, control))
                boot = statistic(a[indices], axis=1) - statistic(b[indices], axis=1)
                low, high = np.percentile(boot, [2.5, 97.5])
                output.append(dict(condition=condition, comparator=comparator, metric=field,
                                   statistic=statistic.__name__, full_minus_comparator=float(statistic(a)-statistic(b)),
                                   percentile_95_low=float(low), percentile_95_high=float(high)))
    return output


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    prior = read_json(HISTORICAL / "protocol.json")
    spots = prior["pairs"]
    ids = [r["sample_id"] for r in spots]
    manifest = ROOT / "manifests/train.jsonl"
    train = [json.loads(line) for line in manifest.read_text().splitlines()]
    if (len(set(ids)) != 213 or [r["sample_id"] for r in train] != ids
            or not all(r["split"] == "train" for r in train)
            or not all(r["pair_valid"] for r in spots)):
        raise ValueError("Expected the same full registered set of 213 NUAA training pairs")
    historical = load_history()
    if len(historical) != 2 * len(ids):
        raise ValueError("Incomplete historical clean baseline")
    indices = {c: read_json(CAPTURE / (c + ".json"))["samples"] for c in CONDITIONS}
    for c, entries in indices.items():
        if [entry["sample_id"] for entry in entries] != ids:
            raise ValueError(f"Unexpected capture manifest for {c}")
    rows = []
    for i, spot in enumerate(spots):
        clean = load_pair("clean", indices["clean"][i])
        for condition in CONDITIONS:
            pair = clean if condition == "clean" else load_pair(condition, indices[condition][i])
            rows.extend(measure_source(condition, pair, clean, spot, historical))
        if (i+1) % 50 == 0:
            print(f"Measured {i+1}/213 origins", flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"], row["method"]].append(row)
    summaries = {}
    strata = []
    for (condition, method), group in groups.items():
        if [r["sample_id"] for r in group] != ids:
            raise ValueError("Different origin order or missing sample")
        summaries.setdefault(condition, {})[method] = group_summary(group)
        for dose in DOSES:
            selected = [r for r in group if r["nominal_peak_dn"] == dose]
            d = group_summary(selected)
            strata.append(dict(condition=condition, method=method, nominal_peak_dn=dose,
                               n_pairs=len(selected), retention_median=d["retention_ratio"]["median"],
                               retention_q25=d["retention_ratio"]["q25"], retention_q75=d["retention_ratio"]["q75"],
                               n_retained_at_least_half=d["retained_at_least_half_n"],
                               n_negative_gain=d["negative_gain_n"], n_width_measurable=d["width_px"]["n"],
                               n_paired_width=d["width_error_px"]["n"], mean_abs_width_error_px=d["abs_width_error_px"]["mean"],
                               mean_abs_contrast_error_dn=d["abs_contrast_increment_error_dn"]["mean"]))
    write_csv(OUT/"per_pair.csv", rows)
    write_csv(OUT/"by_dose.csv", strata)
    write_csv(OUT/"paired_intervals.csv", paired_intervals(groups))
    save_report(OUT/"summary.json", summaries)
    save_report(OUT/"verification.json", dict(
        train_origins=len(ids), draw_ids=CONDITIONS, method_ids=METHODS, total_rows=len(rows), held_out_test_used=0,
        checked_clean_input_history=213, checked_clean_s1_history=213, historical_absolute_tolerance=1e-10,
        capture_indices_sha256={c:file_hash(CAPTURE/(c+".json")) for c in CONDITIONS},
        hashes={p.name:file_hash(p) for p in (manifest, HISTORICAL/"protocol.json", HISTORICAL/"per_pair_stage.csv",
                Path(__file__), ROOT/"native_detail.py", ROOT/"diagnose_spot_transfer.py", OUT/"per_pair.csv",
                OUT/"by_dose.csv", OUT/"paired_intervals.csv", OUT/"summary.json")},
        gain_denominator="positive reference C(clean+) minus C(clean-); input-relative denominator reported separately",
        bootstrap=dict(seed=SEED, draws=BOOT_REPS, unit="source image separately within each condition",
                        quantiles=[.025,.975], formula="statistic(full) minus statistic(comparator) after paired resampling")))
    print(f"Wrote {len(rows)} paired method observations", flush=True)


if __name__ == "__main__":
    main()
