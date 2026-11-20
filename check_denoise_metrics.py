"""Verify denoising metrics against analytic images without loading study data."""
from pathlib import Path

import numpy as np

from evaluate_warmup import prepare_metrics, save_report


def main():
    root = Path(__file__).resolve().parent
    height, width = 20, 28
    valid = np.zeros((height, width), dtype=bool)
    valid[2:18, 2:26] = True
    native = np.zeros_like(valid)
    native[9:11, 13:15] = True
    cleared_background = valid.copy()
    cleared_background[7:13, 11:17] = False
    reference = np.full(valid.shape, 100 / 255, dtype=np.float64)
    reference[native] = 110 / 255
    sample = dict(reference_minus=reference[None], native_mask=native[None],
                  valid_mask=valid[None], background_mask=cleared_background[None])
    measure, coverage = prepare_metrics(sample, extended=True)
    checks = {}

    def check(name, output, expected):
        observed = measure(output)
        for key, value in expected.items():
            if value is None:
                assert observed[key] is None, (name, key, observed[key])
            else:
                np.testing.assert_allclose(observed[key], value, rtol=0, atol=1e-9,
                                           err_msg=f"{name}: {key}")
        checks[name] = {key: observed[key] for key in expected}

    bias = "native_median_ring_contrast_bias_dn"
    contrast = "native_median_ring_contrast_abs_error_dn"
    stripe = "weighted_background_column_residual_dn"
    check("reference", reference, {"whole_mae_dn": 0, "whole_psnr_db": None,
                                   "plan_background_rmse_dn": 0, stripe: 0, bias: 0, contrast: 0})
    for delta in (-7, 12):
        shifted = reference.copy()
        shifted[native] += delta / 255
        check(f"core_{delta:+d}_dn", shifted,
              {bias: delta, contrast: abs(delta), stripe: 0,
               "whole_mae_dn": native.sum() * abs(delta) / valid.sum()})

    # Six central columns have shorter background support because of the excluded core and ring.
    columns = np.arange(width)
    levels = np.where((columns >= 11) & (columns < 17), 6.0, -2.0)
    error = np.broadcast_to(levels, valid.shape)
    check("column_offsets", reference + error / 255,
          {stripe: float(error[cleared_background].std()),
           "plan_background_mean_error_dn": float(error[cleared_background].mean()),
           "whole_psnr_db": float(10 * np.log10(255**2 / np.square(error[valid]).mean()))})

    perturbed = reference.copy()
    perturbed[~valid] = 1
    check("padding_only", perturbed, {"whole_mae_dn": 0, "whole_psnr_db": None,
                                      stripe: 0, bias: 0, contrast: 0})
    assert coverage["eligible_native_components"] == 1
    assert coverage["plan_background_pixels"] == int(cleared_background.sum())
    output = root / "results/local_retrain/metric_checks.json"
    save_report(output, {"data": "analytic fixture only", "cases": checks, "coverage": coverage})
    print(f"Passed {len(checks)} analytic metric checks, saved {output}")


if __name__ == "__main__":
    main()
