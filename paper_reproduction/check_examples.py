"""Verify released Real-IISR on the paired examples included by the authors."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


PROJECT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT / "paper_reproduction" / "upstream_examples"


def measure(image, reference):
    diff = np.asarray(image, dtype=np.float64) - reference
    mse = np.square(diff).mean()
    ssim = structural_similarity(reference, np.asarray(image), channel_axis=2, data_range=255,
                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return {"psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None, "ssim": float(ssim)}


def main():
    folder = EXAMPLES / "testset"
    inputs = sorted((folder / "HR").glob("*.png"))
    outputs = sorted((folder / "VARPrediction").glob("*.png"))
    if not inputs or [p.name for p in inputs] != [p.name for p in outputs]:
        raise ValueError("Official test output does not cover all paired examples")
    rows = []
    for pair, output in zip(inputs, outputs):
        with Image.open(pair) as hr, Image.open(folder / "LR" / pair.name) as lr, Image.open(output) as rec:
            if lr.size != (128, 128) or hr.size != (512, 512) or rec.size != hr.size:
                raise ValueError(f"Unexpected dimensions for {pair.name}")
            reference = np.asarray(hr.convert("RGB"))
            bicubic = lr.convert("RGB").resize(hr.size, Image.Resampling.BICUBIC)
            rows.append({"file": pair.name, "lr_size": lr.size, "hr_size": hr.size,
                         "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                         "bicubic": measure(bicubic, reference), "released_model": measure(rec.convert("RGB"), reference)})
    avg = {method: {metric: float(np.mean([r[method][metric] for r in rows])) for metric in ("psnr_db", "ssim")}
           for method in ("bicubic", "released_model")}
    report = {"population": "paired author-distributed example PNGs enumerated by upstream test.py",
              "weights": "released Real-IISR.pth", "training_updates": 0, "n_pairs": len(rows),
              "scope": "official executable smoke check; not the paper's full FLIR-IISR test split",
              "metrics": "RGB 8-bit, full-frame MSE PSNR, Gaussian 11x11 sigma=1.5 SSIM; per-image arithmetic mean",
              "averages": avg, "per_image": rows}
    (EXAMPLES / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pairs": len(rows), "averages": avg}), flush=True)


if __name__ == "__main__":
    main()
