"""One preregistered pass of free-prefix common adaptation on local SIRST."""
import argparse
import fcntl
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from prd_backbone import PROJECT
from prd_data import DenoisePairDataset
from prd_original_training import build_original, digest, optimizer_for
from prd_rollout import free_rollout
from train_local_baseline import load_trained
from train_warmup import atomic_save, valid_order_loss
from verify_rollout import cached_greedy


OUT = PROJECT / "results/s2_common"
REG = OUT / "preregistration.json"
LOCK_SHA = "e4a47f224c6a7b1b2036c28ac70314c6cbf6cec243b3202aef696f900c8550b7"
FINAL = PROJECT / "checkpoints/sirst_s2_common.pt"
RESUME = PROJECT / "checkpoints/sirst_s2_common_resume.pt"


def load_s2(path, device):
    """Restore the complete S2 endpoint for single-image hard inference."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    spec = json.loads(REG.read_text())
    if (state.get("format") != "sirst_s2_common_v1" or state["updates"] != 54 or state["images"] != 213
            or state["parent_sha256"] != spec["parent_sha256"] or state["manifest_sha256"] != spec["manifest_sha256"]
            or digest(REG) != LOCK_SHA or state["registration_sha256"] != LOCK_SHA):
        raise ValueError("Expected the complete preregistered local S2 endpoint")
    model = build_original(device)
    model.load_state_dict(state["trainer"]["var_wo_ddp"], strict=True)
    return model.eval()


def s2_optimizer(model, spec):
    names = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    original = optimizer_for(model)
    groups = []
    for group in original.param_groups:
        for cac in (False, True):
            params = [p for p in group["params"]
                      if names[id(p)].startswith(("cb_cond.", "vae.quantize.")) == cac]
            if params:
                groups.append(dict(params=params, lr=spec["lr_cac" if cac else "lr_backbone"],
                                   weight_decay=spec["weight_decay"] * group["wd_sc"]))
    return torch.optim.AdamW(groups, betas=tuple(spec["betas"]), eps=spec["eps"], fused=True)


def image_loss(model, sample, check_cached=False):
    device = model.head.weight.device
    reference, inp, valid = [torch.from_numpy(sample[k])[None].to(device)
                             for k in ("reference_minus", "noisy_minus", "valid_mask")]
    with torch.no_grad():
        _, labels = model.vae.img_to_idxBl(reference.mul(2).sub(1).repeat(1, 3, 1, 1))
        coverage = torch.cat([F.adaptive_avg_pool2d(valid.float(), n).flatten(1)
                              for n in model.patch_nums], dim=1)
    inp = inp.mul(2).sub(1).repeat(1, 3, 1, 1)
    if check_cached:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            cached_img, cached_scores, cached_ids = cached_greedy(model, inp)
        del cached_scores
    # A fresh cast of a trainable parameter must participate in autograd on
    # both the first execution and the checkpointed backward recomputation.
    with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
        image, scores, ids = free_rollout(model, inp, cfg=1., temperature=1.)
    if check_cached:
        mismatches = [int((x != y).sum()) for x, y in zip(cached_ids, ids)]
        max_rgb = (cached_img.detach().float() - image.detach().float()).abs().max().item() * 255
        parity = dict(sample_id=sample["meta"]["sample_id"], token_mismatches=mismatches,
                      token_counts=[int(x.numel()) for x in ids], rgb_max_abs_dn=max_rgb)
        atomic_save(OUT / "first_training_forward_parity.json", parity)
        if any(mismatches) or max_rgb != 0:
            raise RuntimeError(f"Training hard forward differs from cached argmax: {parity}")
    ce = F.cross_entropy(torch.cat(scores, dim=1).float().transpose(1, 2), torch.cat(labels, dim=1),
                         reduction="none")
    ce = (ce * coverage).sum() / coverage.sum()
    prediction = image.float()
    mse = (prediction - reference).square().mean(1, keepdim=True)[valid].mean()
    order = valid_order_loss(prediction.mean(1, keepdim=True), reference, valid)
    return ce + 0.2 * mse + 0.8 * order, torch.stack([ce.detach(), mse.detach(), order.detach()])


def run(resume):
    cfg = json.loads(REG.read_text())
    schedule_path = OUT / "schedule.json"
    for path, expected in ((REG, LOCK_SHA), (schedule_path, cfg["schedule_sha256"]),
                           (PROJECT / cfg["parent_path"], cfg["parent_sha256"]),
                           (PROJECT / cfg["manifest_path"], cfg["manifest_sha256"]),
                           (PROJECT / cfg["split_file"], cfg["split_sha256"])):
        if digest(path) != expected:
            raise ValueError(f"Hash mismatch: {path}")
    data = DenoisePairDataset(PROJECT, seed=cfg["data_seed"])
    by_id = {row["sample_id"]: i for i, row in enumerate(data.records)}
    schedule = json.loads(schedule_path.read_text())
    if len(schedule) != cfg["images"] or len(set(schedule)) != len(data) or set(schedule) != set(by_id):
        raise ValueError("Registered schedule does not cover the local training split exactly once")
    logpath = OUT / "updates.jsonl"
    if FINAL.exists() or (not resume and (RESUME.exists() or logpath.exists())):
        raise FileExistsError("Expected a fresh run or an explicit resume with an incomplete endpoint")
    torch.set_num_threads(2)
    torch.manual_seed(20261120)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = load_trained(PROJECT / cfg["parent_path"], device)
    selected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    expected = json.loads((PROJECT / "results/local_retrain/load_audit.json").read_text())["trainable"]
    if {n:list(p.shape) for n, p in selected.items()} != expected:
        raise ValueError("Trainable parameters differ from the original local S1 run")
    optimizer = s2_optimizer(model, cfg["optimizer"])
    if sorted(map(id, selected.values())) != sorted(id(p) for g in optimizer.param_groups for p in g["params"]):
        raise ValueError("Optimizer parameter coverage mismatch")
    scale, start_step = cfg["optimizer"]["loss_scale"], 0
    previous_elapsed = 0.
    if resume:
        if RESUME.exists():
            state = torch.load(RESUME, weights_only=True, map_location="cpu", mmap=True)
            if state["registration_sha256"] != LOCK_SHA:
                raise ValueError("Incompatible optimization snapshot")
            model.load_state_dict(state["trainer"]["var_wo_ddp"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scale, start_step = state["loss_scale"], state["updates"]
            previous_elapsed = state["elapsed_seconds"]
            del state
        lines = logpath.read_text().splitlines() if logpath.exists() else []
        if len(lines) < start_step or any(json.loads(l)["update"] != i for i, l in enumerate(lines[:start_step], 1)):
            raise ValueError("Update log is shorter than or inconsistent with saved state")
        temp = logpath.with_suffix(".partial")
        temp.write_text("".join(line + "\n" for line in lines[:start_step]))
        temp.replace(logpath)
    start = time.monotonic()
    print(f"Original trainables {len(selected)} / {sum(p.numel() for p in selected.values()):,}, "
          f"GPU {torch.cuda.get_device_name(device)}, mode single GPU, start_step={start_step}, schedule={LOCK_SHA}", flush=True)
    rows = []
    for step in range(start_step + 1, cfg["updates"] + 1):
        ids = schedule[(step-1)*4:step*4]
        samples = [data.sample(by_id[k], epoch=cfg["epoch"], draw_id=cfg["draw_id"]) for k in ids]
        for retry in range(12):
            optimizer.zero_grad(set_to_none=True)
            terms = []
            for j, sample in enumerate(samples):
                loss, parts = image_loss(model, sample, check_cached=(step == 1 and j == 0 and retry == 0))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite forward on {ids[j]}")
                (loss * (scale / len(samples))).backward()
                terms.append(parts)
                if step == 1:
                    print(f"backward first group {j+1}/{len(ids)}, "
                          f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
            grads = [p.grad for p in selected.values() if p.grad is not None]
            torch._foreach_mul_(grads, 1. / scale)
            norm = torch.nn.utils.clip_grad_norm_(selected.values(), cfg["optimizer"]["clip"])
            if torch.isfinite(norm):
                break
            scale *= 0.5
            print(f"Retry identical group {step}, lower loss scale {scale}", flush=True)
        else:
            raise FloatingPointError(f"Nonfinite gradients, update {step}")
        if step == 1:
            nonzero = {n:float(p.grad.norm()) if p.grad is not None else None for n, p in selected.items()}
            atomic_save(OUT / "first_update_grad_norms.json", nonzero)
        if len(grads) != len(selected) or not float(norm) > 0:
            raise ValueError("Incomplete gradient path or zero group norm")
        optimizer.step()
        actual_step = int(optimizer.state[model.head.weight]["step"])
        if actual_step != step:
            raise RuntimeError(f"AdamW step counter {actual_step} differs from {step}")
        ce, mse, order = torch.stack(terms).mean(0).tolist()
        elapsed = previous_elapsed + time.monotonic() - start
        row = dict(update=step, sample_ids=ids, images=len(ids), cumulative_images=min(step*4,len(schedule)),
                   token_ce=ce, rgb_mse=mse, valid_order=order, grad_norm=float(norm), loss_scale=scale,
                   gradient_tensors=len(grads), retries=retry, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   elapsed_seconds=round(elapsed,2), optimizer_step=actual_step, lr_backbone=1e-5, lr_cac=5e-5)
        rows.append(row)
        with logpath.open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
        if step in (1, 10, 20, 30, 40, 50, cfg["updates"]):
            payload = dict(format=cfg["format"], parent_sha256=cfg["parent_sha256"],
                           manifest_sha256=cfg["manifest_sha256"], registration_sha256=LOCK_SHA,
                           schedule_sha256=cfg["schedule_sha256"], updates=step, images=row["cumulative_images"],
                           elapsed_seconds=elapsed, trainer={"var_wo_ddp":model.state_dict()})
            if step < cfg["updates"]:
                payload.update(optimizer=optimizer.state_dict(), loss_scale=scale)
                atomic_save(RESUME, payload, binary=True)
            else:
                atomic_save(FINAL, payload, binary=True)
    logged = [json.loads(l) for l in logpath.read_text().splitlines()]
    if [k for r in logged for k in r["sample_ids"]] != schedule:
        raise ValueError("Final committed log differs from the complete schedule")
    summary = dict(format=cfg["format"], parent_sha256=cfg["parent_sha256"], manifest_sha256=cfg["manifest_sha256"],
                   registration_sha256=LOCK_SHA, checkpoint_sha256=digest(FINAL), bytes=FINAL.stat().st_size,
                   images=len(schedule), updates=len(logged), elapsed_seconds=logged[-1]["elapsed_seconds"],
                   optimizer_state_entries=len(optimizer.state), optimizer_steps=sorted({int(s["step"]) for s in optimizer.state.values()}),
                   trainable_tensors=len(selected), trainable_parameters=sum(v.numel() for v in selected.values()),
                   peak_allocated_gib=max(r["peak_allocated_gib"] for r in logged), torch_version=str(torch.__version__),
                   device=torch.cuda.get_device_name(device), device_layout="single GPU", test_images=0,
                   overflow_retries=sum(r["retries"] for r in logged))
    atomic_save(OUT / "training_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with (OUT / ".train.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.resume)
