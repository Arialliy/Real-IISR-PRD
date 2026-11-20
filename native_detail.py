"""A fixed native-resolution detail branch for grayscale S1 restoration."""
import numpy as np
from scipy.ndimage import convolve1d


VERTICAL_KERNEL = (1/16, 4/16, 6/16, 4/16, 1/16)


def restore_native_detail(noisy_native, s1_native, *, strength=1.0):
    """Replace a fraction of S1's vertical high frequencies with the input's.

    Inputs are native (unpadded) 2D grayscale frames in [0,1]. The filter
    operates along the rows of each column and uses half-sample reflection.
    """
    x, s = (np.asarray(a, dtype=np.float32) for a in (noisy_native, s1_native))
    if x.ndim != 2 or x.shape != s.shape or x.size == 0:
        raise ValueError("Expected two nonempty native grayscale frames of identical shape")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Strength must be a finite scalar between zero and one")
    for frame in (x,s):
        if not np.isfinite(frame).all() or frame.min()<0 or frame.max()>1:
            raise ValueError("Expected finite display-gray samples in [0,1]")
    if strength == 0:
        return s.copy()
    difference = x - s
    average = convolve1d(difference, VERTICAL_KERNEL, axis=0, mode="reflect")
    return np.clip(s + np.float32(strength)*(difference-average), 0, 1)


def iter_methods(noisy_native, s1_native):
    """Generate the primary setting, conventional comparators and fixed controls."""
    from check_column_baseline import correct_columns
    x, s = noisy_native, s1_native
    height, width = x.shape
    yield "input", x
    yield "column", correct_columns(x, 0, 0, width, height)
    yield "s1", s
    yield "detail_half", restore_native_detail(x, s, strength=0.5)
    yield "detail_full", restore_native_detail(x, s)
    yield "full_blend_half", np.float32(0.5)*x + np.float32(0.5)*s


def predict_s1_native(model, gray, device):
    """Run the local S1 protocol on a native frame that fits its 512-pixel canvas."""
    import torch
    from evaluate_retrained import hard_prediction

    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2 or not gray.size or max(gray.shape)>512:
        raise ValueError("The local single-frame S1 protocol supports original grids up to 512 × 512")
    if not np.isfinite(gray).all() or gray.min()<0 or gray.max()>1:
        raise ValueError("Expected finite grayscale input in [0,1]")
    h,w = gray.shape
    top,left = (512-h)//2,(512-w)//2
    canvas = np.zeros((1,512,512),dtype=np.float32)
    window = np.s_[top:top+h,left:left+w]
    canvas[0][window] = gray
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        return hard_prediction(model, canvas, device, greedy=True)[window].copy()


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from PIL import Image
    from evaluate_noise_levels import get_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A display-gray image on its native grid")
    parser.add_argument("output", type=Path, help="Output .npy file with float32 gray in [0,1]")
    parser.add_argument("--strength", type=float, default=1.)
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("Output must end in .npy")
    with Image.open(args.input) as im:
        gray = np.asarray(im.convert("L"),dtype=np.float32)/255
    model,device,_ = get_model("s1")
    prediction = restore_native_detail(gray,predict_s1_native(model,gray,device),strength=args.strength)
    with args.output.open("xb") as handle:
        np.save(handle,prediction,allow_pickle=False)
