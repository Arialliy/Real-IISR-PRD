"""Run the published FLIR-IISR training protocol using verified train/test membership."""

import argparse
import json
import fcntl
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from prd_backbone import UPSTREAM
from prd_original_training import HASHES, SCALES, build_original, digest, distribute_layers, initialize, optimizer_for

# Put upstream ahead of this script's folder so its own 'from train import ...' resolves correctly.
sys.path.insert(0, str(UPSTREAM))
from dataloader.traindataset import TrainDataset
from utils.data_sampler import DistInfiniteBatchSampler
from utils.lr_control import lr_wd_annealing
from utils.misc import MetricLogger


def write_atomic(path, value, binary=False):
    tmp = path.with_suffix(path.suffix + ".partial")
    if binary:
        torch.save(value, tmp)
    else:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_ids(path):
    if not path.is_file():
        raise FileNotFoundError(f"Local fixed split required: {path}")
    lines = [v.strip() for v in path.read_text(encoding="utf-8-sig").splitlines() if v.strip()]
    if any(v != Path(v).name or v in (".", "..") for v in lines):
        raise ValueError(f"Expected one published basename or stem per line in {path}")
    supported = {".bmp", ".png", ".jpg", ".jpeg"}
    ids = [Path(v).stem if Path(v).suffix.lower() in supported else v for v in lines]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty split or duplicate image IDs: {path}")
    return ids


def verified_data(config, run_dir=None):
    root = PROJECT / config["dataset_root"]
    train = read_ids(PROJECT / config["train_ids"])
    test = read_ids(PROJECT / config["test_ids"])
    if len(train) != config["training_pairs"] or len(test) != config["test_pairs"]:
        raise ValueError(f"Configured FLIR split requires {config['training_pairs']} train and {config['test_pairs']} test pairs")
    if set(train) & set(test):
        raise ValueError("Train and test identifiers intersect")
    images = {}
    for folder in ("HR", "LR", "LR_4x"):
        paths = [p for p in (root / folder).iterdir()
                 if p.is_file() and p.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg")]
        by_id = {p.stem: p for p in paths}
        if len(paths) != len(by_id):
            raise ValueError(f"Ambiguous stems in {root / folder}")
        images[folder] = by_id
    population = set(train) | set(test)
    if (len(population) != config["published_pair_count"]
            or any(set(by_id) != population for by_id in images.values())):
        raise ValueError("Local split lists and all three released image folders must match one-to-one")
    from PIL import Image
    for key in train:
        with Image.open(images["HR"][key]) as hr, Image.open(images["LR"][key]) as lr:
            if hr.size != lr.size or min(hr.size) < config["crop_size"]:
                raise ValueError(f"Input pair cannot use the upstream aligned crop: {key}")
    summary = dict(provenance=config["split_provenance"], train_pairs=len(train), test_pairs=len(test),
                   train_sha256=digest(PROJECT / config["train_ids"]),
                   test_sha256=digest(PROJECT / config["test_ids"]),
                   train_examples=2 * len(train))
    if run_dir is None:
        return summary, None
    stage = run_dir / "train_inputs"
    for folder in ("HR", "LR"):
        view = stage / folder
        view.mkdir(parents=True, exist_ok=True)
        expected = {images["HR"][key].name for key in train}
        extra = {p.name for p in view.iterdir()} - expected
        if extra:
            raise ValueError(f"Unexpected files in training input view: {sorted(extra)[:3]}")
        for key in train:
            output = view / images["HR"][key].name
            source = images[folder][key].resolve()
            if output.is_symlink():
                if output.resolve(strict=True) != source:
                    raise ValueError(f"Mismatched staged file: {output}")
            else:
                output.symlink_to(os.path.relpath(source, output.parent))
    ds = TrainDataset(str(stage / "HR"), str(stage / "LR"), crop_size=config["crop_size"],
                      lr_scale=config["sr_scale"])
    if len(ds) != 2 * len(train):
        raise ValueError("Upstream loader did not register exactly two examples per train pair")
    return summary, ds


def check_implementation():
    """Exercise the official data and loss code on a synthetic CPU-only pair."""
    import inspect
    import tempfile
    import numpy as np
    from PIL import Image
    from models.src.loss import ThermalOrderConsistencyLoss
    from trainer import VARTrainer
    from torchvision.transforms import functional as TF
    from torchvision.transforms import InterpolationMode
    with tempfile.TemporaryDirectory(dir=PROJECT / ".runtime/paper_training") as tmp:
        base = Path(tmp)
        y, x = np.indices((512, 512))
        hr = Image.fromarray(((x + 3*y) % 256).astype(np.uint8)).convert("RGB")
        lr = Image.fromarray(((2*x + y) % 256).astype(np.uint8)).convert("RGB")
        for folder, im in (("HR", hr), ("LR", lr)):
            (base / folder).mkdir()
            im.save(base / folder / "fixture.png")
        ds = TrainDataset(str(base / "HR"), str(base / "LR"), crop_size=512, lr_scale=4)
        assert len(ds) == 2
        down = TF.resize(lr, [128, 128], interpolation=InterpolationMode.BICUBIC)
        up = TF.to_tensor(TF.resize(down, [512, 512], interpolation=InterpolationMode.BICUBIC)) * 2 - 1
        for idx, target in ((0, hr), (1, lr)):
            result = ds[idx]
            assert result["label_B"] == idx
            assert torch.equal(result["HR"], TF.to_tensor(target) * 2 - 1)
            assert torch.equal(result["LR"], up)
        toc = ThermalOrderConsistencyLoss(reduction="mean")
        reference = TF.to_tensor(hr).unsqueeze(0)
        assert toc(tar=reference, sr=reference).item() == 0
        reversed_order = toc(tar=reference, sr=1-reference).item()
        assert reversed_order > 0
    return dict(backend="cpu", data="one synthetic HR/LR fixture; zero FLIR study images",
                optimizer_updates=0, branch0_label=0, branch1_label=1,
                crop=512, low_resolution_pixels=128, bicubic_input_exact_for_both_branches=True,
                target_exact_for_both_branches=True, thermal_order_identity=0.0,
                thermal_order_reversed=reversed_order,
                train_step_sha256=digest(Path(inspect.getsourcefile(VARTrainer.train_step))),
                upstream_trainer="VARTrainer.train_step")


def train(config, run_dir, resume):
    from train import NullDDP
    from trainer import VARTrainer
    from utils.amp_sc import AmpOptimizer
    split, ds = verified_data(config, run_dir)
    config_hash = digest(HERE / "config.json")
    latest = run_dir / "last.pt"
    final = run_dir / "final.pt"
    trace = run_dir / "updates.jsonl"
    if final.exists() or resume != latest.exists() or (not resume and trace.exists()):
        raise ValueError("Use the matching unfinished run with --resume or select a fresh run directory")
    for name, expected in HASHES.items():
        if digest(PROJECT / "checkpoints" / name) != expected:
            raise ValueError(f"Incorrect author initialization: {name}")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_original(device)
    if not resume:
        audit = initialize(model)
        write_atomic(run_dir / "load_audit.json", audit)
    model.train()
    model.vae.eval()
    distribute_layers(model)
    # The official trainer computes its ORIGINAL image losses on the input device.
    model.vae.decoder.register_forward_hook(lambda _module, _args, result: result.to(device))
    opt = optimizer_for(model)
    pars = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    amp = AmpOptimizer(mixed_precision=1, optimizer=opt, names=[n for n, p in pars],
                       paras=[p for n, p in pars], grad_clip=config["max_grad_norm"],
                       n_gradient_accumulation=1)
    trainer = VARTrainer(device=device, patch_nums=SCALES, resos=tuple(16*p for p in SCALES),
                         vae_local=model.vae, var_wo_ddp=model, var=NullDDP(model, device_ids=[0]),
                         text_encoder=None, clip_vision=None, exp_name="FLIR-IISR-10000",
                         var_opt=amp, label_smooth=0.0, wandb_flag=False)
    batch_size = config["global_batch"]
    sampler = DistInfiniteBatchSampler(world_size=1, rank=0, dataset_len=len(ds),
                                      glb_batch_size=batch_size, fill_last=True, shuffle=True,
                                      same_seed_for_all_ranks=0)
    target = config["accepted_optimizer_updates"]
    count = 0
    attempts = 0
    # The head receives a CE gradient on every update. Fused AdamW keeps the accepted
    # step count here and reverses its increment when GradScaler flags an overflow.
    step_parameter = model.head.weight
    if resume:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True, mmap=True)
        if (checkpoint.get("format") != "flir_original_super_resolution_v1"
                or checkpoint["split"] != split or checkpoint["config_sha256"] != config_hash
                or checkpoint["initialization"] != HASHES):
            raise ValueError("Checkpoint provenance differs from this run")
        trainer.load_state_dict(checkpoint["trainer"], strict=True)
        count, attempts = checkpoint["updates"], checkpoint["attempts"]
        if (not 0 < count < target or attempts < count
                or int(opt.state[step_parameter]["step"].item()) != count):
            raise ValueError("Accepted updates and restored AdamW step disagree")
        if amp.scaler.state_dict() != checkpoint["trainer"]["var_opt"]["scaler"]:
            raise ValueError("AMP scaler state was not restored")
        del checkpoint
        lines = trace.read_text().splitlines()
        if len(lines) < count or any(json.loads(l)["update"] != i for i,l in enumerate(lines[:count],1)):
            raise ValueError("Committed checkpoint and update history disagree")
        trimmed = trace.with_suffix(".partial")
        trimmed.write_text("\n".join(lines[:count]) + "\n")
        trimmed.replace(trace)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(run_dir / "training_protocol.json", {"config": config, "splits": split, "source": HASHES})
    start = time.monotonic()
    previous_epoch = None
    while count < target:
        epoch, it = divmod(attempts, len(sampler))
        if previous_epoch != epoch:
            sampler.epoch = epoch
            order = sampler.gener_indices()
            previous_epoch = epoch
        batch_indices = order[it * batch_size:(it+1) * batch_size]
        torch.manual_seed(config["seed"] + attempts)
        batch = default_collate([ds[i] for i in batch_indices])
        _, lr, _, _ = lr_wd_annealing(config["schedule"], opt, config["peak_lr"],
                                       config["weight_decay"], config["weight_decay"],
                                       count, config["warmup_epochs"] * len(sampler), target,
                                       config["warmup_start_ratio"], config["final_lr_ratio"])
        metrics = MetricLogger()
        metrics.log_iters = [it]
        before = count

        grad_norm, _ = trainer.train_step(it=it, g_it=attempts, stepping=True, metric_lg=metrics,
             inp_B3HW=batch["HR"].to(device), lr_inp=batch["LR"].to(device),
             label_B=batch["label_B"].to(device), text=None, prog_si=-1, prog_wp_it=1, lr=lr,
             wd=config["weight_decay"])
        attempts += 1
        loss = float(metrics.meters["total_loss"].value)
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError(f"Nonfinite FLIR loss at attempt {attempts}")
        accepted = int(opt.state[step_parameter]["step"].item())
        if accepted == before:
            print(f"AMP overflow at attempt {attempts}; accepted updates still {count}", flush=True)
            if attempts - 100 > count * 2:
                raise FloatingPointError("Repeated AMP skips; inspect loss scaling before resuming")
            continue
        if accepted != before + 1 or not math.isfinite(float(grad_norm)):
            raise RuntimeError("Unexpected AdamW update or gradient norm")
        count = accepted
        row = dict(update=count, attempt=attempts, epoch=epoch+1, lr=lr, grad_norm=float(grad_norm),
                   ids=[Path(p).stem for p in batch["img_path"]], class_labels=batch["label_B"].tolist(),
                   losses={key: float(metrics.meters[key].value) for key in
                           ("CELm", "mse_loss", "order_loss", "total_loss")})
        with trace.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if count == 1 or count % 50 == 0:
            print(f"{count}/{target}, CE={row['losses']['CELm']:.4f}, elapsed={time.monotonic()-start:.1f}s", flush=True)
        if count == 1 or count % 500 == 0 or count == target:
            state = trainer.state_dict()
            checkpoint = dict(format="flir_original_super_resolution_v1", trainer=state, updates=count,
                              attempts=attempts, split=split, initialization=HASHES, config_sha256=config_hash)
            if count == target:
                state.pop("var_opt")
            write_atomic(final if count == target else latest, checkpoint, binary=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-implementation", action="store_true", help="Exercise synthetic CPU-only loader and loss checks")
    parser.add_argument("--validate-data", action="store_true",
                        help="Check fixed local membership and aligned native dimensions on CPU")
    parser.add_argument("--run-dir", type=Path, default=HERE / "flir_10000")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text())
    if args.check_implementation:
        print(json.dumps(check_implementation(), indent=2))
    elif args.validate_data:
        print(json.dumps(verified_data(config)[0], indent=2))
    else:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with (args.run_dir / ".train.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another process owns this FLIR run directory") from exc
            train(config, args.run_dir, args.resume)


if __name__ == "__main__":
    main()
