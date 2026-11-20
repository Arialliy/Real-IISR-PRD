"""Replay the unmodified official entry and preserve its actual inputs and predictions."""
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader


AUDIT = Path(__file__).resolve().parent
PREVIOUS = AUDIT.parent
PROJECT = PREVIOUS.parents[1]
UPSTREAM = PROJECT / "vendor/Real-IISR"


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    manifest = json.loads((PREVIOUS / "input_manifest.json").read_text())
    previous = json.loads((PREVIOUS / "run_summary.json").read_text())
    ids = (PROJECT / previous["list_file"]).read_text().split()
    if ([row["id"] for row in manifest] != ids or len(ids) != 213
            or sha256(UPSTREAM / "test.py") != previous["test_py_sha256"]
            or sha256(PROJECT / previous["checkpoint"]) != previous["checkpoint_sha256"]):
        raise RuntimeError("Training list, original source, or checkpoint changed")
    out = AUDIT / "arrays"
    out.mkdir(exist_ok=True)
    layout = AUDIT / "testset"
    layout.mkdir(exist_ok=True)
    for side in ("HR", "LR"):
        link = layout / side
        source = PREVIOUS / "testset" / side
        if not link.exists():
            link.symlink_to(os.path.relpath(source, link.parent), target_is_directory=True)
        if link.resolve() != source or sorted(p.name for p in link.glob("*.png")) != [r["official_filename"] for r in manifest]:
            raise RuntimeError("The audit must use the original ordered training inputs")
    if (layout / "VARPrediction").resolve() == (PREVIOUS / "testset/VARPrediction").resolve():
        raise RuntimeError("The audit must have its own output directory")

    os.chdir(AUDIT)
    sys.path.insert(0, str(UPSTREAM))
    from models import Real_IISR

    original_iter = DataLoader.__iter__
    original_infer = Real_IISR.autoregressive_infer_cfg
    batch_files = []
    completed = 0

    def observe_batches(loader):
        nonlocal batch_files
        for batch in original_iter(loader):
            batch_files = [Path(p).name for p in batch["path"]]
            yield batch

    def observe_inference(model, *args, **kwargs):
        nonlocal completed
        inp = kwargs["lr_inp"]
        output = original_infer(model, *args, **kwargs)
        if len(inp) != len(batch_files) or len(inp) != len(output):
            raise RuntimeError("Mismatch between official batch paths, inputs and outputs")
        input_cpu = inp.detach().cpu().numpy()
        output_cpu = output.detach().cpu().numpy()
        for i, name in enumerate(batch_files):
            row = manifest[completed]
            if name != row["official_filename"]:
                raise RuntimeError(f"Unexpected input order at {completed}: {name}")
            x, y = input_cpu[i], output_cpu[i]
            if (x.shape != (3, 512, 512) or y.shape != x.shape or not np.isfinite(x).all()
                    or not np.isfinite(y).all() or np.min(y) < 0 or np.max(y) > 1):
                raise ValueError(f"Unexpected values or tensor shape: {name}")
            dest = out / (Path(name).stem + ".npz")
            temp = dest.with_suffix(".pending")
            with temp.open("wb") as f:
                np.savez(f, input_chw=x, prediction_chw=y)
            temp.replace(dest)
            completed += 1
        if completed % 32 == 0 or completed == len(manifest):
            print(f"Captured {completed} of {len(manifest)}", flush=True)
        return output

    entry = UPSTREAM / "test.py"
    sys.argv = [str(entry), f"--var_test_path={PROJECT / previous['checkpoint']}", "--seed=0"]
    with patch.object(DataLoader, "__iter__", observe_batches), patch.object(Real_IISR, "autoregressive_infer_cfg", observe_inference):
        runpy.run_path(str(entry), run_name="__main__")
    if completed != len(manifest):
        raise RuntimeError(f"Incomplete archive: {completed}")
    print(f"Complete; peak allocated GPU memory {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
