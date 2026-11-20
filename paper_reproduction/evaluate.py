"""Native 4x evaluation on the fixed local FLIR holdout using the original test.py."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import PIL
import skimage

from check_examples import measure


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
UPSTREAM = PROJECT / "vendor/Real-IISR"
TILE = 512
SCALE = 4
WIDTH, HEIGHT = 1024, 768
POSITIONS = [(x, y) for y in (0, 256) for x in (0, 256, 512)]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_png(path, image):
    tmp = path.with_suffix(".pending")
    image.save(tmp, format="PNG", compress_level=2)
    tmp.replace(path)


def load_split(config):
    parts = {}
    for side, count in (("train", 1192), ("test", 265)):
        path = PROJECT / config[side + "_ids"]
        lines = [s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        if any(not re.fullmatch(r"[0-9]{6}(?:\.bmp)?", s) for s in lines):
            raise ValueError(f"Invalid released basename in {path}")
        ids = [Path(s).stem for s in lines]
        if len(ids) != len(set(ids)) or len(ids) != count:
            raise ValueError(f"Expected {count} unique {side} IDs")
        parts[side] = ids
    all_ids = set(parts["test"]) | set(parts["train"])
    if len(all_ids) != 1457:
        raise ValueError("Training and held-out IDs overlap")
    root = PROJECT / config["dataset_root"]
    for sub in ("HR", "LR", "LR_4x"):
        if {p.name for p in (root / sub).iterdir() if p.is_file()} != {f"{s}.bmp" for s in all_ids}:
            raise ValueError(f"Release inventory differs for {sub}")
    signature = {s + "_sha256": digest(PROJECT / config[s + "_ids"]) for s in parts}
    return root, sorted(parts["test"]), signature


def scene_fingerprints(root, ids):
    return {s: {sub: digest(root / sub / (s + ".bmp")) for sub in ("HR", "LR_4x")} for s in ids}


def names_for(scene):
    return [f"{scene}_y{y:03d}_x{x:03d}.png" for x, y in POSITIONS]


def check_folder(folder, expected):
    if not folder.is_dir() or {p.name for p in folder.iterdir() if p.is_file()} != expected:
        raise ValueError(f"File membership differs in {folder}")


def prepared(out, root, ids, split, create):
    layout = out / "testset"
    stage = out / "inputs.json"
    scenes = scene_fingerprints(root, ids)
    identity = dict(split=split, scene_sha256=scenes, scale=SCALE, tile=TILE, coordinates=POSITIONS,
                    preparation_sha256=digest(Path(__file__)), pillow=PIL.__version__)
    # Convert serialized coordinate tuples to their JSON representation before comparison.
    identity["coordinates"] = [list(p) for p in POSITIONS]
    expected = {name for s in ids for name in names_for(s)}
    if stage.exists():
        saved = json.loads(stage.read_text())
        if saved["identity"] != identity:
            raise ValueError("Dataset, split or evaluation code changed, use a fresh output directory")
        for sub in ("HR", "LR"):
            check_folder(layout / sub, expected)
            for name in expected:
                if digest(layout / sub / name) != saved["tile_sha256"][sub][name]:
                    raise ValueError(f"Prepared tile bytes changed: {sub}/{name}")
        return expected
    if not create:
        raise FileNotFoundError("Run the prepare stage first")
    hashes = {"HR": {}, "LR": {}}
    for sub in hashes:
        (layout / sub).mkdir(parents=True, exist_ok=True)
    for s in ids:
        with Image.open(root / "HR" / (s + ".bmp")) as hr, Image.open(root / "LR_4x" / (s + ".bmp")) as lr:
            if hr.size != (WIDTH, HEIGHT) or lr.size != (WIDTH // SCALE, HEIGHT // SCALE):
                raise ValueError(f"Unexpected original 4x pair dimensions: {s} {hr.size} {lr.size}")
            for name, (x, y) in zip(names_for(s), POSITIONS):
                for sub, im, divisor in (("HR", hr, 1), ("LR", lr, SCALE)):
                    box = tuple(q // divisor for q in (x, y, x+TILE, y+TILE))
                    file = layout / sub / name
                    save_png(file, im.crop(box))
                    hashes[sub][name] = digest(file)
    for sub in hashes:
        check_folder(layout / sub, expected)
    write_json(stage, {"identity": identity, "tile_sha256": hashes})
    return expected


def checkpoint_identity(path, config, split):
    import torch
    checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if (checkpoint.get("format") != "flir_original_super_resolution_v1"
            or checkpoint.get("updates") != config["accepted_optimizer_updates"]
            or checkpoint["updates"] != 10000
            or any(checkpoint["split"].get(k) != v for k, v in split.items())
            or checkpoint["config_sha256"] != digest(HERE / "config.json")
            or not checkpoint["trainer"].get("var_wo_ddp")):
        raise ValueError("Checkpoint must contain the completed local 10,000-update run on these exact lists")
    return {"checkpoint_sha256": digest(path), "updates": checkpoint["updates"],
            "config_sha256": checkpoint["config_sha256"], "split": split}


def inference_identity(out, checkpoint, config, split):
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["upstream_commit"]:
        raise ValueError(f"Unexpected upstream code revision: {commit}")
    subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--quiet", "HEAD"], check=True)
    return {**checkpoint_identity(checkpoint, config, split), "input_manifest_sha256": digest(out / "inputs.json"),
            "upstream_commit": commit, "test_entry_sha256": digest(UPSTREAM / "test.py"),
            "seed": 0, "batch_size": 4, "label_B": 0}


def verify_predictions(out, expected, identity):
    record = json.loads((out / "inference.json").read_text())
    if record["identity"] != identity or set(record["prediction_sha256"]) != expected:
        raise ValueError("Completed prediction manifest belongs to a different protocol")
    folder = out / "testset/VARPrediction"
    check_folder(folder, expected)
    for name, expected_hash in record["prediction_sha256"].items():
        if digest(folder / name) != expected_hash:
            raise ValueError(f"Prediction digest mismatch: {name}")


def infer(out, expected, identity, checkpoint):
    folder = out / "testset/VARPrediction"
    if (out / "inference.json").exists():
        verify_predictions(out, expected, identity)
        print("Verified and reused all original predictions.", flush=True)
        return
    if folder.exists() and {p.name for p in folder.iterdir()} - expected:
        raise ValueError("Unexpected files in the inference output folder")
    cmd = [sys.executable, str(UPSTREAM / "test.py"), f"--var_test_path={checkpoint}", "--seed=0"]
    with (out / "official_test.log").open("w") as log:
        subprocess.run(cmd, cwd=out, check=True, stdout=log, stderr=subprocess.STDOUT)
    check_folder(folder, expected)
    hashes = {}
    for name in sorted(expected):
        file = folder / name
        with Image.open(file) as im:
            if im.size != (TILE, TILE) or im.mode != "RGB":
                raise ValueError(f"Unexpected official output: {file}")
            im.verify()
        hashes[name] = digest(file)
    import torch
    write_json(out / "inference.json", {"identity": identity, "prediction_sha256": hashes,
               "command": cmd, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
               "cuda": torch.version.cuda, "pillow": PIL.__version__})


def blend_weights(x, y):
    # Linear 256-pixel crossfades in original output coordinates.
    half = TILE // 2
    ramp = (np.arange(half, dtype=np.float64) + .5) / half
    def axis(origin, length):
        a = np.ones(TILE, dtype=np.float64)
        if origin > 0:
            a[:half] = ramp
        if origin + TILE < length:
            a[half:] = ramp[::-1]
        return a
    return axis(y, HEIGHT)[:, None] * axis(x, WIDTH)[None, :]


def mosaic(tiles):
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.float64)
    denominator = np.zeros((HEIGHT, WIDTH), np.float64)
    count = 0
    for (x, y), rgb in zip(POSITIONS, tiles, strict=True):
        if rgb.shape != (TILE, TILE, 3) or rgb.dtype != np.uint8:
            raise ValueError("Expected decoded 8-bit RGB tile")
        weight = blend_weights(x, y)
        canvas[y:y+TILE, x:x+TILE] += rgb * weight[:, :, None]
        denominator[y:y+TILE, x:x+TILE] += weight
        count += 1
    if count != 6 or not np.allclose(denominator, 1, atol=1e-12, rtol=0):
        raise ValueError("Native frame is not fully and consistently covered")
    return Image.fromarray(np.rint(canvas / denominator[:, :, None]).clip(0, 255).astype(np.uint8))


def read_tiles(folder, scene):
    for name in names_for(scene):
        with Image.open(folder / name) as im:
            yield np.asarray(im.convert("RGB"))


def score(out, root, ids, identity):
    from statistics import mean
    rows = []
    for sub in ("SR", "bicubic"):
        (out / sub).mkdir(exist_ok=True)
    for i, scene in enumerate(ids, 1):
        hr_path = root / "HR" / (scene + ".bmp")
        lr_path = root / "LR_4x" / (scene + ".bmp")
        with Image.open(hr_path) as hr, Image.open(lr_path) as lr:
            rgb = np.asarray(hr.convert("RGB"))
            bicubic = lr.convert("RGB").resize(hr.size, Image.Resampling.BICUBIC)
        sr = mosaic(read_tiles(out / "testset/VARPrediction", scene))
        row = {"id": scene}
        for sub, im in (("bicubic", bicubic), ("SR", sr)):
            dst = out / sub / (scene + ".png")
            save_png(dst, im)
            row[sub] = {**measure(im, rgb), "png_sha256": digest(dst)}
        rows.append(row)
        if i % 25 == 0:
            print(f"Scored {i}/{len(ids)} native pairs.", flush=True)
    avg = {}
    for sub in ("bicubic", "SR"):
        p = [r[sub]["psnr_db"] for r in rows]
        avg[sub] = {"psnr_db": None if None in p else mean(p),
                    "ssim": mean(r[sub]["ssim"] for r in rows)}
    write_json(out / "per_image.json", rows)
    write_json(out / "metrics.json", {"population": "independent FLIR-IISR holdout", "n_pairs": len(rows),
               "inference": identity, "tile_count": len(rows) * len(POSITIONS), "averages": avg,
               "image_size_wh": [WIDTH, HEIGHT], "scale": SCALE, "boundary_crop": 0,
               "metrics": "uint8 RGB PSNR from full-frame RGB MSE, per-channel Gaussian SSIM 11x11 sigma 1.5 population covariance, valid windows; equal image weights. Null PSNR denotes positive infinity.",
               "metric_source_sha256": digest(HERE / "check_examples.py"),
               "skimage": skimage.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
               "per_image_sha256": digest(out / "per_image.json")})


def check():
    rgb = np.random.default_rng(42).integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    rec = mosaic(rgb[y:y+TILE, x:x+TILE] for x, y in POSITIONS)
    if not np.array_equal(np.asarray(rec), rgb) or measure(rec, rgb) != {"psnr_db": None, "ssim": 1.0}:
        raise AssertionError("Identity stitching or metric test failed")
    print("Native RGB reconstruction exact for all pixels; identity SSIM=1, PSNR=+infinity.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "prepare", "infer", "score"])
    parser.add_argument("--checkpoint", type=Path, default=HERE / "flir_10000/final.pt")
    parser.add_argument("--output-dir", type=Path, default=HERE / "evaluate_holdout")
    args = parser.parse_args()
    if args.stage == "check":
        check()
        return
    config = json.loads((HERE / "config.json").read_text())
    root, ids, split = load_split(config)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        expected = prepared(out, root, ids, split, create=args.stage == "prepare")
        if args.stage == "prepare":
            print(f"Verified {len(ids)} held-out 4x pairs and {len(expected)} paired tiles.", flush=True)
            return
        ck = args.checkpoint.resolve()
        identity = inference_identity(out, ck, config, split)
        if args.stage == "infer":
            infer(out, expected, identity, ck)
        else:
            verify_predictions(out, expected, identity)
            score(out, root, ids, identity)


if __name__ == "__main__":
    main()
