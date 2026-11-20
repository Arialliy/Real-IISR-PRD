"""Observe S1 inside the original test.py and verify every input against the published run."""
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[1]
OLD = PROJECT / "results/official_sirst_train"
S1 = RUN / "s1"
WEIGHTS = PROJECT / "checkpoints/sirst_retrained_baseline.pt"
ENTRY = PROJECT / "vendor/Real-IISR/test.py"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    manifest = json.loads((RUN / "input_manifest.json").read_text())
    info = json.loads((OLD / "run_summary.json").read_text())
    old_audit = json.loads((RUN / "published/summary.json").read_text())
    s1_summary = json.loads((PROJECT / "results/local_retrain/training_summary.json").read_text())
    expected = "fb2eaf0426e1634b8cde12c7d8951ded9d6b4f04a0ae2ef77a1c13f794caacab"
    if sha(WEIGHTS) != expected or sha(ENTRY) != old_audit["source_entry_sha256"] or old_audit["exact_inputs"] != 213:
        raise ValueError("Model, entry or earlier audited inputs changed")
    ids = (PROJECT / info["list_file"]).read_text().split()
    if [r["id"] for r in manifest] != ids or len(set(ids)) != 213:
        raise ValueError("Input manifest and bundled NUAA-SIRST training IDs differ")
    if (S1 / "complete.json").exists():
        raise FileExistsError("This inference run already has verified final outputs")
    for r in manifest:
        if sha(PROJECT / r["input_png"]) != r["image_sha256"]:
            raise ValueError(f"Input file changed: {r['id']}")
    layout = S1 / "testset"
    layout.mkdir(exist_ok=True)
    for side in ("HR", "LR"):
        link = layout / side
        source = OLD / "testset" / side
        if not link.exists():
            link.symlink_to(os.path.relpath(source, link.parent), target_is_directory=True)
        if link.resolve() != source.resolve() or sorted(p.name for p in link.glob("*.png")) != [r["official_filename"] for r in manifest]:
            raise ValueError(f"Wrong official {side} inputs")
    arrays = S1 / "arrays"
    arrays.mkdir(exist_ok=True)
    os.chdir(S1)
    sys.path.insert(0, str(ENTRY.parent))
    from models import Real_IISR
    from myutils.wavelet_color_fix import adain_color_fix

    official_iter = DataLoader.__iter__
    official_infer = Real_IISR.autoregressive_infer_cfg
    names, batch_sizes, outputs = [], [], []

    def observe_loader(loader):
        for batch in official_iter(loader):
            names[:] = [Path(p).name for p in batch["path"]]
            yield batch

    def observe_infer(model, *args, **kwargs):
        inp = kwargs["lr_inp"]
        prediction = official_infer(model, *args, **kwargs)
        x = inp.detach().cpu().numpy()
        y = prediction.detach().cpu().numpy()
        if len(names) != len(x) or y.shape != x.shape or y.shape[1:] != (3,512,512):
            raise ValueError("Unexpected candidate input/output tensor shapes")
        if y.dtype != np.float16 or not np.isfinite(y).all() or y.min() < 0 or y.max() > 1:
            raise ValueError("Unexpected model output precision or range")
        batch_sizes.append(len(names))
        for i, name in enumerate(names):
            row = manifest[len(outputs)]
            if row["official_filename"] != name:
                raise ValueError("Unexpected official evaluation order")
            npz = Path(name).with_suffix(".npz")
            with np.load(RUN / "published/arrays" / npz) as other:
                if x[i].dtype != other["input_chw"].dtype or not np.array_equal(x[i], other["input_chw"]):
                    raise ValueError(f"Model inputs differ across the two runs: {row['id']}")
            dst = arrays / npz
            tmp = dst.with_suffix(".partial")
            with tmp.open("wb") as f:
                np.savez(f, input_chw=x[i], prediction_chw=y[i])
            tmp.replace(dst)
            outputs.append(dict(sample_id=row["id"], filename=name))
        if len(outputs) % 32 == 0 or len(outputs) == 213:
            print(f"Observed {len(outputs)}/213 S1 predictions", flush=True)
        return prediction

    sys.argv = [str(ENTRY), f"--var_test_path={WEIGHTS}", "--seed=0"]
    with patch.object(DataLoader, "__iter__", observe_loader), patch.object(Real_IISR, "autoregressive_infer_cfg", observe_infer):
        runpy.run_path(str(ENTRY), run_name="__main__")
    if batch_sizes != [4] * 53 + [1]:
        raise ValueError(f"Unexpected batch sizes: {batch_sizes}")
    for row in outputs:
        name = row["filename"]
        p = layout / "VARPrediction" / name
        npz = arrays / Path(name).with_suffix(".npz")
        with np.load(npz) as data:
            raw = data["prediction_chw"].astype(np.float32).transpose(1,2,0)
        rounded = Image.fromarray((raw * 255).round().astype(np.uint8), "RGB")
        with Image.open(layout / "LR" / name) as original:
            expected_image = adain_color_fix(rounded, original.convert("RGB").resize((512,512)))
        with Image.open(p) as actual:
            if actual.mode != "RGB" or actual.size != (512,512) or not np.array_equal(np.asarray(expected_image), np.asarray(actual)):
                raise ValueError(f"Official final PNG differs from postprocessing its actual raw output: {name}")
        row.update(s1_npz_sha256=sha(npz), s1_png_sha256=sha(p))
    summary = dict(status="complete", count=len(outputs), batch_sizes=batch_sizes, split=info["list_file"],
                   split_sha256=sha(PROJECT / info["list_file"]), source_entry=info["upstream_entry"],
                   source_entry_sha256=sha(ENTRY), source_commit=info["upstream_commit"],
                   weights=str(WEIGHTS.relative_to(PROJECT)), weights_sha256=expected, s1_training_updates=s1_summary["accepted_optimizer_updates"],
                   official_weights_sha256=old_audit["checkpoint_sha256"], published_reference="results/official_sirst_train/float_audit",
                   device=torch.cuda.get_device_name(0), peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                   torch_version=str(torch.__version__), command="bash results/original_input_official_protocol/run_s1.sh 1",
                   settings=dict(seed_cli=0, per_call_g_seed=None, cfg=1, top_k=1, top_p=0.75, more_smooth=False, mixed_precision="fp16", final="8-bit round then original AdaIN"),
                   exact_input_tensor_matches=213, exact_repostprocessed_pngs=213, test_images=0, noise_added=False,
                   completed_rows=outputs)
    dest = S1 / "complete.json"
    tmp = dest.with_suffix(".partial")
    tmp.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    tmp.replace(dest)
    print(f"Verified {len(outputs)} same-input predictions and AdaIN PNGs",flush=True)


if __name__ == "__main__":
    main()
