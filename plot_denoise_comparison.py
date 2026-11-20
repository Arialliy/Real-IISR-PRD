"""Plot draw 0 for the first training ID using final, native-size float outputs."""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import label

from check_column_baseline import correct_columns
from prd_data import DenoisePairDataset


def main():
    root = Path(__file__).resolve().parent
    out = root / "results/sirst_endpoints"
    sample = DenoisePairDataset(root, seed=0).sample(0, epoch=0, draw_id=0)
    stem = sample["meta"]["sample_id"].split("/")[-1]
    left, top = sample["meta"]["pad_left_top"]
    _, _, w, h = sample["meta"]["crop_xywh"]
    field = np.s_[top:top+h, left:left+w]
    ref = sample["reference_minus"][0][field]
    noisy = sample["noisy_minus"][0][field]
    conventional = correct_columns(noisy, 0, 0, w, h)
    frames = [ref, noisy, conventional] + [np.load(out / f"{stage}_draw0_float" / f"{stem}.npy")
                                           for stage in ("s1", "s2")]
    if not all(a.shape == (h, w) and np.isfinite(a).all() for a in frames):
        raise ValueError("Native float outputs differ in dimensions or finiteness")
    ids, count = label(sample["native_mask"][0][field], np.ones((3, 3), dtype=bool))
    if count == 0:
        raise ValueError("First training image has no annotated region")
    ys, xs = np.where(ids == 1)
    cy, cx = int(np.rint(ys.mean())), int(np.rint(xs.mean()))
    size = 31
    y = int(np.clip(cy-size//2, 0, h-size))
    x = int(np.clip(cx-size//2, 0, w-size))
    window = np.s_[y:y+size, x:x+size]
    candidates = (Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
                  Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    font = next((p for p in candidates if p.exists()), None)
    titles = ("Reference", "Noisy input", "Column correction", "Stage 1", "Stage 2")
    caption = f"{stem}, training draw 0  |  Full frames and the same {size} x {size} pixel window  |  Fixed display range 0–255"
    if font:
        font_manager.fontManager.addfont(str(font))
        mpl.rcParams["font.family"] = [font_manager.FontProperties(fname=str(font)).get_name(), "DejaVu Sans"]
        titles = ("加噪前参考图", "带噪输入", "传统列校正", "第一阶段", "第二阶段")
        caption = f"训练图 {stem} · 第一组噪声  ｜  同一处 {size} × {size} 像素窗口  ｜  全部面板使用 0–255 灰度范围"
    fig, ax = plt.subplots(2, len(frames), figsize=(12, 4.6), layout="constrained",
                           gridspec_kw={"height_ratios": [1, 1]})
    for index, (frame, title) in enumerate(zip(frames, titles)):
        for axis, view in zip(ax[:, index], (frame, frame[window])):
            axis.imshow(view * 255, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axis.set_axis_off()
        ax[0, index].add_patch(Rectangle((x-.5, y-.5), size, size, fill=False,
                                         edgecolor="#f17839", linewidth=1.1))
        ax[0, index].set_title(title, fontsize=12)
    fig.supxlabel(caption, fontsize=10)
    fig.savefig(out / "first_training_image.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
