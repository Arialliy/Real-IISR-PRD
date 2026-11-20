"""Reproducible training pairs on the native pixel grid of the local SIRST images."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _seed(base, sample_id, epoch, draw_id, purpose):
    parts = json.dumps([base, sample_id, epoch, draw_id, purpose]).encode()
    return int.from_bytes(hashlib.sha256(parts).digest()[:8], "big") >> 1


def _calibration_template(rng):
    """Generate a finite calibration spot and a surrounding two-pixel ring."""
    sigma_x = float(rng.uniform(0.45, 1.5))
    sigma_y = float(rng.uniform(max(0.45, sigma_x / 2), min(1.5, sigma_x * 2)))
    angle = float(rng.uniform(0, np.pi))
    dx, dy = rng.uniform(-0.5, 0.5, 2)
    amplitude_dn = float(rng.choice([1, 2, 4, 6, 8, 12]))
    radius = int(np.ceil(3 * max(sigma_x, sigma_y) + 3))
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    c, s = np.cos(angle), np.sin(angle)
    u = (x - dx) * c + (y - dy) * s
    v = -(x - dx) * s + (y - dy) * c
    core = (u / sigma_x) ** 2 + (v / sigma_y) ** 2 <= 9
    ring = ~core & ((u / (3 * sigma_x + 2)) ** 2 + (v / (3 * sigma_y + 2)) ** 2 <= 1)
    # Four quadrature samples per pixel approximate integration over pixel area.
    shape = np.zeros_like(u)
    for off_x in (-0.25, 0.25):
        for off_y in (-0.25, 0.25):
            ux = (u + c * off_x + s * off_y) / sigma_x
            vy = (v - s * off_x + c * off_y) / sigma_y
            shape += np.exp(-0.5 * (ux ** 2 + vy ** 2))
    shape *= core
    shape *= amplitude_dn / (255 * shape.max())
    parameters = dict(peak_dn=amplitude_dn, sigma_x=sigma_x, sigma_y=sigma_y,
                      angle_rad=angle, offset_x=float(dx), offset_y=float(dy))
    return shape.astype(np.float32), core, ring, parameters


class DenoisePairDataset:
    """One sample per training image; sample(index, epoch, draw_id) makes a new draw.

    Images and masks are NumPy arrays of shape [1, patch_size, patch_size].
    Default indexing uses draw_id=0 and the epoch selected by set_epoch().
    A pair with no admissible injection
    has identical branches and pair_valid=False; its ordinary restoration loss
    can still be evaluated. Masks and metadata are for losses/diagnostics.
    """

    def __init__(self, project_root, patch_size=512, seed=0):
        self.root = Path(project_root)
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.patch_size < 1:
            raise ValueError("patch_size must be positive")
        manifest = self.root / "manifests/train.jsonl"
        self.records = [json.loads(line) for line in manifest.read_text().splitlines()]
        index = Path("datasets/NUAA-SIRST/img_idx/train_NUAA-SIRST.txt")
        expected = (self.root / index).read_text(encoding="utf-8-sig").split()
        if [row["source_image_id"] for row in self.records] != expected:
            raise ValueError("Training manifest differs from the bundled SIRST train IDs or order")
        for row in self.records:
            img_name = row["source_image_id"] + ".png"
            if (row["split"] != "train" or row["dataset"] != "NUAA-SIRST"
                    or row["split_source"] != index.as_posix() or not row["mask_metrics_eligible"]
                    or Path(row["reference_path"]) != index.parent.parent / "images" / img_name
                    or Path(row["mask_path"]) != index.parent.parent / "masks" / img_name):
                raise ValueError(f"Invalid primary training record: {row['sample_id']}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.sample(index, epoch=self.epoch)

    def set_epoch(self, epoch):
        """Set before starting each loader iteration (nonpersistent workers)."""
        self.epoch = int(epoch)

    def sample(self, index, epoch=0, draw_id=0, *, noise_override=None):
        """An optional sigma mapping changes the amplitude of the same noise realization."""
        row = self.records[index]
        seeds = {name: _seed(self.seed, row["sample_id"], int(epoch), int(draw_id), name)
                 for name in ("geometry", "signal", "noise")}
        geometry_rng = np.random.default_rng(seeds["geometry"])
        signal_rng = np.random.default_rng(seeds["signal"])
        noise_rng = np.random.default_rng(seeds["noise"])

        with Image.open(self.root / row["reference_path"]) as image:
            ref = np.asarray(image.convert("L"), dtype=np.float32) / 255
        with Image.open(self.root / row["mask_path"]) as annotation:
            mask = annotation.convert("L")
            native = np.asarray(mask) != 0
            exclude = np.asarray(mask.filter(ImageFilter.MaxFilter(5))) != 0
        if ref.shape != native.shape:
            raise ValueError(f"Image/mask size mismatch: {row['sample_id']}")
        height, width = ref.shape
        stripe_dn, pixel_dn = float(noise_rng.uniform(0, 12)), float(noise_rng.uniform(0, 4))
        if noise_override is not None:
            if set(noise_override) != {"stripe_sigma_dn", "pixel_sigma_dn"}:
                raise ValueError("Supply both stripe_sigma_dn and pixel_sigma_dn")
            stripe_dn = float(noise_override["stripe_sigma_dn"])
            pixel_dn = float(noise_override["pixel_sigma_dn"])
            if not np.isfinite([stripe_dn, pixel_dn]).all() or min(stripe_dn, pixel_dn) < 0:
                raise ValueError("Noise sigmas must be finite and nonnegative")
        # Generate the noise on the entire unpadded original before selecting a crop.
        stripe = noise_rng.standard_normal((1, width), dtype=np.float32) * (stripe_dn / 255)
        noise = noise_rng.standard_normal(ref.shape, dtype=np.float32) * (pixel_dn / 255) + stripe
        raw_noisy = ref + noise
        p = self.patch_size
        w, h = min(p, width), min(p, height)
        x0 = int(geometry_rng.integers(0, width - w + 1))
        y0 = int(geometry_rng.integers(0, height - h + 1))
        crop = np.s_[y0:y0 + h, x0:x0 + w]
        ref, raw_noisy, native, exclude = ref[crop], raw_noisy[crop], native[crop], exclude[crop]
        noisy = np.clip(raw_noisy, 0, 1)

        spot, template_core, template_ring, params = _calibration_template(signal_rng)
        footprint = template_core | template_ring
        radius = spot.shape[0] // 2
        signal, support, ring = np.zeros_like(ref), np.zeros_like(native), np.zeros_like(native)
        cx = cy = -1
        if min(w, h) >= spot.shape[0]:
            for _ in range(128):
                x = int(signal_rng.integers(radius, w - radius))
                y = int(signal_rng.integers(radius, h - radius))
                window = np.s_[y-radius:y+radius+1, x-radius:x+radius+1]
                b, n = ref[window], raw_noisy[window]
                unsafe = exclude[window] | (b <= 0) | (n <= 0) | (b + spot >= 1) | (n + spot >= 1)
                if np.any(unsafe & footprint):
                    continue
                cx, cy = x, y
                signal[window], support[window], ring[window] = spot, template_core, template_ring
                break

        ref_plus = ref + signal
        signal = ref_plus - ref
        arrays = dict(reference_minus=ref, reference_plus=ref_plus, noisy_minus=noisy,
                      noisy_plus=np.clip(raw_noisy + signal, 0, 1), signal=signal,
                      support_mask=support, ring_mask=ring, native_mask=native,
                      valid_mask=np.ones_like(native), background_mask=~(exclude | support | ring))
        left, top = (p-w)//2, (p-h)//2
        padding = ((top, p-h-top), (left, p-w-left))
        sample = {name: np.pad(array, padding, mode="constant")[None, ...]
                  for name, array in arrays.items()}
        sample["pair_valid"] = cx >= 0
        sample["sample_kind"] = "synthetic_pair"
        sample["meta"] = dict(
            sample_id=row["sample_id"], group_id=row["group_id"], split=row["split"],
            epoch=int(epoch), draw_id=int(draw_id), crop_xywh=(x0, y0, w, h), pad_left_top=(left, top),
            center_xy_patch=(cx+left, cy+top) if cx >= 0 else (-1, -1),
            geometry_seed=seeds["geometry"], noise_seed=seeds["noise"], signal_seed=seeds["signal"],
            noise_parameters=dict(stripe_sigma_dn=stripe_dn, pixel_sigma_dn=pixel_dn),
            signal_parameters=params,
        )
        return sample
