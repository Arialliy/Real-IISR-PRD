"""Analyze the observations and compare the independent official replays on the training list."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch

from capture import AUDIT, PREVIOUS, PROJECT, UPSTREAM, sha256

sys.path.insert(0, str(UPSTREAM))
from myutils.wavelet_color_fix import adain_color_fix


def write_table(name, rows):
    with (AUDIT / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gray(rgb):
    return rgb.astype(np.float32).mean(axis=2)


def view(array, signed=False):
    values = (array + 32) * (255 / 64) if signed else array
    return Image.fromarray(values.clip(0, 255).round().astype(np.uint8))


def full_scene(path, name, images):
    boxes = (("Model input, DN", images[0], False), ("Before rounding / AdaIN, DN", images[1], False),
             ("Official replay, DN", images[2], False), ("Archived official PNG, DN", images[3], False),
             ("Float minus input, -32 to +32 DN", images[1] - images[0], True),
             ("Official replay minus float, -32 to +32 DN", images[2] - images[1], True))
    canvas = Image.new("RGB", (1536, 1080), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, arr, signed) in enumerate(boxes):
        x, y = (i % 3) * 512, (i // 3) * 540
        draw.text((x + 6, y + 5), f"{name}  |  {label}", fill="black")
        canvas.paste(view(arr, signed), (x, y + 25))
    canvas.save(path, compress_level=2)


def window_at(x, y, extent):
    size = min(512, max(33, int(extent) + 16))
    x0, y0 = (max(0, min(512 - size, v - size // 2)) for v in (x, y))
    return x0, y0, x0 + size, y0 + size


def save_target_view(figure, axes, name, bounds, x, y, imgs):
    x0, y0, x1, y1 = bounds
    for ax in axes:
        ax.clear()
    names = ("Model input", "Before rounding / AdaIN", "Official replay", "Archived official", "Float minus input")
    for i, ax in enumerate(axes[:5]):
        arr = (imgs[i] if i < 4 else imgs[1] - imgs[0])[y0:y1, x0:x1]
        ax.imshow(arr, cmap="gray", vmin=-32 if i == 4 else 0, vmax=32 if i == 4 else 255,
                  interpolation="nearest", extent=(x0 - .5, x1 - .5, y1 - .5, y0 - .5))
        ax.set_title(names[i], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for direction, ax in enumerate(axes[5:]):
        coord = np.arange(x0, x1) if direction == 0 else np.arange(y0, y1)
        for i, (img, color) in enumerate(zip(imgs, ("#222222", "#00844b", "#a83643", "#315a9a"))):
            values = img[y, x0:x1] if direction == 0 else img[y0:y1, x]
            ax.plot(coord, values, color=color, lw=1.4, ls="--" if i == 3 else "-", label=names[i])
        ax.axvline(x if direction == 0 else y, color="gray", lw=0.6)
        ax.set_ylim(0, 255)
        ax.set_xlim(coord[0], coord[-1])
        ax.set_xlabel(f"Horizontal x (y={y})" if direction == 0 else f"Vertical y (x={x})", fontsize=8)
        if direction == 0:
            ax.set_ylabel("Gray DN", fontsize=8)
        ax.tick_params(labelsize=8)
    axes[5].legend(loc="best", fontsize=6, ncol=2)
    figure.suptitle(f"{name}  |  intensity 0–255 DN, signed difference −32 to +32 DN", fontsize=11)
    figure.savefig(AUDIT / "targets" / f"{name}.png", dpi=105)


def contrast(img, core, ring):
    if not ring.any():
        return None
    return float(np.mean(img[core], dtype=np.float64) - np.median(img[ring]))


def nearest(array, shape):
    return np.asarray(Image.fromarray(array).resize(shape, Image.Resampling.NEAREST))


def summary_stats(values):
    return dict(min=float(np.min(values)), mean=float(np.mean(values, dtype=np.float64)),
                max=float(np.max(values)),
                at_zero=float(np.mean(values <= 0)), at_255=float(np.mean(values >= 255)))


def main():
    torch.set_num_threads(2)
    for d in ("scenes", "targets"):
        (AUDIT / d).mkdir(exist_ok=True)
    manifest = json.loads((PREVIOUS / "input_manifest.json").read_text())
    previous = json.loads((PREVIOUS / "run_summary.json").read_text())
    training = [json.loads(s) for s in (PROJECT / "manifests/train.jsonl").read_text().splitlines()]
    assert len(manifest) == len(previous["outputs"]) == len(training) == 213
    scenes, targets, profiles = [], [], []
    figure = plt.figure(figsize=(11, 4.9))
    grid = figure.add_gridspec(2, 10, left=.065, right=.98, bottom=.13, top=.85,
                               hspace=.45, wspace=.85)
    axes = [figure.add_subplot(grid[0, 2*i:2*i+2]) for i in range(5)]
    axes.extend([figure.add_subplot(grid[1, :5]), figure.add_subplot(grid[1, 5:])])
    for row, old, train in zip(manifest, previous["outputs"], training):
        stem = Path(row["official_filename"]).stem
        assert row["id"] == train["source_image_id"] and row["id"] == old["id"]
        src = PROJECT / row["input_png"]
        archive = PREVIOUS / old["output_file"]
        replay = AUDIT / "testset/VARPrediction" / row["official_filename"]
        array_file = AUDIT / "arrays" / f"{stem}.npz"
        assert sha256(src) == row["image_sha256"] and sha256(archive) == old["output_sha256"]
        with Image.open(src) as im:
            size, original_mode = im.size, im.mode
            native = gray(np.asarray(im.convert("RGB")))
            resized = im.convert("RGB").resize((512, 512))
        with np.load(array_file) as record:
            x = record["input_chw"].transpose(1, 2, 0)
            y = record["prediction_chw"].transpose(1, 2, 0)
        # Repeat the official data transformation on the untouched native source.
        expected_x = np.asarray(resized).astype(np.float32) / 255 * 2 - 1
        if not np.array_equal(x, expected_x):
            raise ValueError(f"Actual model input differs from the original official resize: {stem}")
        q = Image.fromarray((y.astype(np.float32) * 255).round().astype(np.uint8))
        regenerated = np.asarray(adain_color_fix(q, resized))
        pub = np.asarray(Image.open(replay).convert("RGB"))
        history = np.asarray(Image.open(archive).convert("RGB"))
        if not np.array_equal(pub, regenerated):
            raise ValueError(f"Official replay differs from postprocessing observed floating output: {stem}")
        x_gray = gray((x.astype(np.float32)+1)*127.5)
        y_gray = gray(y.astype(np.float32)*255)
        imgs = (x_gray, y_gray, gray(pub), gray(history))
        full_scene(AUDIT / "scenes" / f"{stem}.png", row["id"], imgs)
        diff = np.abs(pub.astype(np.int16) - history.astype(np.int16))
        scene = dict(id=row["id"], source=row["input_png"], mask=train["mask_path"], native_width=size[0],
                     native_height=size[1], native_mode=original_mode,
                     tensor_layout="CHW", model_input_shape="3x512x512", model_output_shape="3x512x512",
                     input_dtype=str(x.dtype), output_dtype=str(y.dtype), final_size="512x512 RGB",
                     input_matches_official_resize=True, postprocess_replay_pixel_exact=True,
                     archive_png_bytes_equal=(sha256(replay) == old["output_sha256"]),
                     archive_rgb_pixel_exact=bool(np.array_equal(pub, history)),
                     archive_delta_rgb_mae_dn=float(diff.mean()), archive_delta_rgb_max_dn=int(diff.max()),
                     archive_fraction_changed_pixels=float((diff > 0).any(axis=2).mean()),
                     float_to_quant_rgb_mae_dn=float(np.abs(np.asarray(q).astype(np.float32)-y.astype(np.float32)*255).mean()),
                     float_to_final_gray_mae_dn=float(np.abs(imgs[2]-y_gray).mean()),
                     input_to_float_gray_mae_dn=float(np.abs(x_gray-y_gray).mean()),
                     source_sha256=row["image_sha256"], array_sha256=sha256(array_file),
                     archived_png_sha256=old["output_sha256"], replay_png_sha256=sha256(replay),
                     arrays=str(array_file.relative_to(AUDIT)), overview=f"scenes/{stem}.png")
        for name, img in zip(("native", "input", "float", "replay", "archived"), (native, *imgs)):
            scene.update({f"{name}_gray_{k}": v for k, v in summary_stats(img).items()})
        scenes.append(scene)
        with Image.open(PROJECT / train["mask_path"]) as im:
            native_mask = np.asarray(im.convert("L")) != 0
        assert native_mask.shape == native.shape
        labels, count = ndimage.label(native_mask, structure=np.ones((3, 3), dtype=bool))
        scaled = nearest(labels.astype(np.int32), (512, 512))
        for number in range(1, count+1):
            native_core = labels == number
            native_ring = ndimage.binary_dilation(native_core, structure=np.ones((5,5), bool)) & ~native_mask
            core = scaled == number
            if not core.any():
                raise RuntimeError(f"ROI vanished under projection: {stem} component {number}")
            ring = nearest(native_ring.astype(np.uint8), (512,512)).astype(bool)
            iy, ix = np.nonzero(native_core)
            location = np.argmin((iy-iy.mean())**2 + (ix-ix.mean())**2)
            cy, cx = int(iy[location]), int(ix[location])
            yp, xp = np.nonzero(core)
            anchor = np.argmin((yp - ((cy+.5)*512/size[1]-.5))**2 + (xp - ((cx+.5)*512/size[0]-.5))**2)
            y_line, x_line = int(yp[anchor]), int(xp[anchor])
            bounds = window_at(x_line, y_line, max(2*abs(xp-x_line).max()+1, 2*abs(yp-y_line).max()+1))
            x0, y0, x1, y1 = bounds
            name = f"{stem}_c{number:02d}"
            save_target_view(figure, axes, name, bounds, x_line, y_line, imgs)
            contrasts = [contrast(img, core, ring) for img in imgs]
            target = dict(id=row["id"], component=number, native_anchor_x=cx, native_anchor_y=cy,
                          projected_anchor_x=x_line, projected_anchor_y=y_line, crop_x0=x0, crop_y0=y0,
                          crop_x1_exclusive=x1, crop_y1_exclusive=y1, native_core_pixels=int(native_core.sum()),
                          native_ring_pixels=int(native_ring.sum()), projected_core_pixels=int(core.sum()),
                          projected_ring_pixels=int(ring.sum()), native_contrast_dn=contrast(native, native_core, native_ring),
                          input_contrast_dn=contrasts[0], float_contrast_dn=contrasts[1],
                          final_contrast_dn=contrasts[2], archived_contrast_dn=contrasts[3],
                          float_minus_input_contrast_dn=(contrasts[1]-contrasts[0]) if ring.any() else None,
                          final_minus_input_contrast_dn=(contrasts[2]-contrasts[0]) if ring.any() else None,
                          zoom_and_profiles=f"targets/{name}.png")
            targets.append(target)
            for direction, coord in (("horizontal", range(x0, x1)), ("vertical", range(y0,y1))):
                for at in coord:
                    xpt, ypt = (at, y_line) if direction == "horizontal" else (x_line, at)
                    profiles.append(dict(id=row["id"], component=number, direction=direction, x=xpt, y=ypt,
                                         in_target=int(core[ypt,xpt]), input_dn=float(imgs[0][ypt,xpt]),
                                         float_dn=float(imgs[1][ypt,xpt]), final_dn=float(imgs[2][ypt,xpt]),
                                         archived_dn=float(imgs[3][ypt,xpt])))
        if row["index"] % 32 == 0:
            print(f"Measured {row['index']}/213", flush=True)
    write_table("per_image.csv", scenes)
    write_table("per_target.csv", targets)
    write_table("profiles.csv", profiles)
    result = dict(images=len(scenes), native_components=len(targets), usable_contrast_rows=sum(t["projected_ring_pixels"]>0 for t in targets),
                  exact_inputs=sum(r["input_matches_official_resize"] for r in scenes),
                  exact_repostprocessed_images=sum(r["postprocess_replay_pixel_exact"] for r in scenes),
                  exact_historical_png_bytes=sum(r["archive_png_bytes_equal"] for r in scenes),
                  exact_historical_pixels=sum(r["archive_rgb_pixel_exact"] for r in scenes),
                  mean_per_image_replay_vs_historical_rgb_mae_dn=float(np.mean([r["archive_delta_rgb_mae_dn"] for r in scenes])),
                  largest_replay_vs_historical_rgb_abs_dn=max(r["archive_delta_rgb_max_dn"] for r in scenes),
                  mean_per_image_float_to_final_gray_mae_dn=float(np.mean([r["float_to_final_gray_mae_dn"] for r in scenes])),
                  mean_per_image_input_to_float_gray_mae_dn=float(np.mean([r["input_to_float_gray_mae_dn"] for r in scenes])),
                  mean_rgb_quantization_abs_dn=float(np.mean([r["float_to_quant_rgb_mae_dn"] for r in scenes])),
                  split="NUAA-SIRST/train", checkpoint_sha256=previous["checkpoint_sha256"],
                  source_entry_sha256=previous["test_py_sha256"], original_summary_sha256=sha256(PREVIOUS/"run_summary.json"))
    (AUDIT / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
