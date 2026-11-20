"""Train the original global-CAC Real-IISR network on same-size local SIRST pairs."""

import argparse
import json
import time

import torch
from torch import nn
from torch.nn import functional as F

from prd_backbone import PROJECT
from prd_original_training import HASHES, SCALES, build_original, digest, distribute_layers, initialize, optimizer_for
from prd_data import DenoisePairDataset
from train_warmup import atomic_save, valid_order_loss


SEED = 20261120
EPOCHS = 20
BATCH = 4
OUT = PROJECT / "results/local_retrain"
FINAL = PROJECT / "checkpoints/sirst_retrained_baseline.pt"
RESUME = PROJECT / "checkpoints/sirst_retrain_resume.pt"
def load_trained(path, device):
    """Restore our complete locally retrained checkpoint for upstream hard inference."""
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (payload.get("format") != "sirst_local_retrain_v1" or payload["epochs_completed"] != EPOCHS
            or payload["initialization"] != HASHES):
        raise ValueError("Expected the completed local baseline with both verified initializers")
    model = build_original(device)
    model.load_state_dict(payload["trainer"]["var_wo_ddp"], strict=True)
    return model.eval()


class TrainTask(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, noisy, reference, valid):
        model = self.model
        gray = reference
        ref_rgb = gray.mul(2).sub(1).expand(-1, 3, -1, -1)
        with torch.no_grad():
            _, labels = model.vae.img_to_idxBl(ref_rgb)
            target = torch.cat(labels, dim=1)
            prefix = model.vq.idxBl_to_var_input(labels)
            coverage = torch.cat([F.adaptive_avg_pool2d(valid.float(), n).flatten(1)
                                  for n in SCALES], dim=1)
        inp = noisy.mul(2).sub(1).expand(-1, 3, -1, -1)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, h = model(prefix, torch.zeros(len(noisy), device=noisy.device, dtype=torch.long),
                              inp, text_hidden=None)
        ce_tokens = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
        ce = ((ce_tokens * coverage).sum(1) / coverage.sum(1)).mean()
        selected = logits.detach().argmax(-1).split([n * n for n in SCALES], dim=1)
        decoded = model.vae.idxBl_to_img_lora(selected, h.float(), same_shape=True, last_one=True)
        prediction = (decoded + 1) * 0.5
        gray_out = gray.to(prediction.device)
        valid_out = valid.to(prediction.device)
        rgb_error = (prediction - gray_out).square().mean(1, keepdim=True)
        mse = torch.stack([err[mask].mean() for err, mask in zip(rgb_error, valid_out)]).mean().to(ce.device)
        pred_gray = prediction.mean(1, keepdim=True)
        order = torch.stack([valid_order_loss(pred_gray[i:i+1], gray_out[i:i+1], valid_out[i:i+1])
                             for i in range(len(gray))]).mean().to(ce.device)
        total = ce + 0.2 * mse + 0.8 * order
        return total, torch.stack((ce.detach(), mse.detach(), order.detach()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-probe", action="store_true",
                        help="Measure two four-image batches on random weights without saving weights")
    parser.add_argument("--resume", action="store_true", help="Resume exclusively the local baseline")
    args = parser.parse_args()
    if args.resume and args.resource_probe:
        parser.error("A resource probe uses ephemeral random weights")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not args.resource_probe and FINAL.exists():
        raise FileExistsError("This baseline has a final checkpoint")
    if not (args.resource_probe or args.resume) and (OUT / "updates.jsonl").exists():
        raise FileExistsError("This output directory already contains update records")
    if not args.resource_probe and RESUME.exists() != args.resume:
        raise ValueError("A resumable run requires --resume with its matching checkpoint")
    data = DenoisePairDataset(PROJECT, seed=0)
    imgs = [r["sample_id"] for r in data.records]
    if len(imgs) != 213:
        raise ValueError("Expected the bundled SIRST training list")
    steps = (len(imgs) + BATCH - 1) // BATCH
    manifest = digest(PROJECT / "manifests/train.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    if not (args.resource_probe or args.resume):
        for name, expected in HASHES.items():
            if digest(PROJECT / "checkpoints" / name) != expected:
                raise ValueError(f"Wrong initializer: {name}")
    model = build_original(device)
    full_load = None
    if args.resume:
        full_load = torch.load(RESUME, map_location="cpu", weights_only=True, mmap=True)
        if (full_load.get("format") != "sirst_local_retrain_v1" or full_load["manifest_sha256"] != manifest
                or full_load["initialization"] != HASHES):
            raise ValueError("Invalid previous local training run")
        model.load_state_dict(full_load["trainer"]["var_wo_ddp"], strict=True)
    elif not args.resource_probe:
        audit = initialize(model)
        audit["trainable"] = {k: list(p.shape) for k, p in model.named_parameters() if p.requires_grad}
        audit["n_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        atomic_save(OUT / "load_audit.json", audit)
    model.train()
    model.vae.eval()
    wrapper = TrainTask(distribute_layers(model))
    optimizer = optimizer_for(model)
    from utils.lr_control import lr_wd_annealing
    scaler = torch.amp.GradScaler("cuda", init_scale=2048, growth_interval=1000)
    epoch_start = 0
    if full_load is not None:
        optimizer.load_state_dict(full_load["optimizer"])
        scaler.load_state_dict(full_load["scaler"])
        epoch_start = full_load["epochs_completed"]
        del full_load
        log = OUT / "updates.jsonl"
        committed = epoch_start * steps
        lines = log.read_text().splitlines()
        if len(lines) < committed or any(json.loads(line)["update"] != i
                                         for i, line in enumerate(lines[:committed], 1)):
            raise ValueError("Update log disagrees with the committed epoch boundary")
        trimmed = log.with_suffix(".partial")
        trimmed.write_text("\n".join(lines[:committed]) + "\n")
        trimmed.replace(log)
    trainables = [p for p in model.parameters() if p.requires_grad]
    print(f"Single process, batch={BATCH}, parameters={sum(p.numel() for p in trainables):,}, "
          f"epochs={EPOCHS}, steps/epoch={steps}, resource_probe={args.resource_probe}", flush=True)
    start = time.monotonic()
    for epoch in range(epoch_start, EPOCHS):
        generator = torch.Generator().manual_seed(SEED + epoch)
        ids = torch.randperm(len(imgs), generator=generator).tolist()
        ids.extend(ids[:steps * BATCH - len(imgs)])
        if not args.resource_probe:
            atomic_save(OUT / f"epoch_{epoch + 1:02}_schedule.json", [imgs[i] for i in ids])
        for it in range(steps):
            samples = [data.sample(i, epoch=epoch, draw_id=0) for i in ids[it * BATCH:(it+1) * BATCH]]
            inp, ref, valid = [torch.stack([torch.from_numpy(sample[key]) for sample in samples]).to(device)
                               for key in ("noisy_minus", "reference_minus", "valid_mask")]
            step = epoch * steps + it
            _, lr, _, _ = lr_wd_annealing("lin0", optimizer, 5e-5, 0.05, 0.05, step,
                                           0.1 * steps, EPOCHS * steps, wp0=0.005, wpe=0.01)
            for retry in range(12):
                # Replay stochastic layers and running BatchNorm state after an AMP overflow.
                torch.manual_seed(SEED + step)
                bn_snap = [(buf, buf.clone()) for name, buf in model.named_buffers()
                           if name.endswith(("running_mean", "running_var", "num_batches_tracked"))]
                optimizer.zero_grad(set_to_none=True)
                loss, stats = wrapper(inp, ref, valid)
                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError(f"Nonfinite forward at step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_(trainables, 2.0)
                if torch.isfinite(norm):
                    scaler.step(optimizer)
                    scaler.update()
                    break
                for buf, value in bn_snap:
                    buf.copy_(value)
                scaler.update(new_scale=scaler.get_scale() / 2)
            else:
                raise FloatingPointError(f"Nonfinite gradients after 12 scale reductions at step {step}")
            ce, mse, order = stats.tolist()
            row = dict(epoch=epoch + 1, update=step + 1, lr=lr, token_ce=ce, rgb_mse=mse,
                       valid_order=order, grad_norm=float(norm), loss_scale=scaler.get_scale(),
                       retries=retry, elapsed_seconds=round(time.monotonic() - start, 2),
                       peak_allocated_gib=[round(torch.cuda.max_memory_allocated(i) / 2 ** 30, 2) for i in range(4)])
            if not args.resource_probe:
                with (OUT / "updates.jsonl").open("a") as record:
                    record.write(json.dumps(row, allow_nan=False) + "\n")
            if step < 3 or (it + 1) % 10 == 0 or it + 1 == steps:
                print(json.dumps(row), flush=True)
            if args.resource_probe and it == 1:
                return
        if (epoch + 1) % 5 == 0 or epoch == 0:
            payload = dict(format="sirst_local_retrain_v1", epochs_completed=epoch + 1,
                           updates=(epoch + 1) * steps, manifest_sha256=manifest,
                           initialization=HASHES, trainer={"var_wo_ddp": model.state_dict()},
                           config=dict(patch_nums=SCALES, depth=24, global_batch=BATCH,
                                       original_trainable_parameters=sum(p.numel() for p in trainables)))
            if epoch + 1 < EPOCHS:
                payload.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
                atomic_save(RESUME, payload, binary=True)
            else:
                atomic_save(FINAL, payload, binary=True)


if __name__ == "__main__":
    main()
