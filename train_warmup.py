"""S1 teacher-forced adaptation of the released model for same-size image restoration."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prd_backbone import OFFICIAL_SHA256, load_backbone
from prd_data import DenoisePairDataset

ROOT = Path(__file__).resolve().parent


def configure(model):
    model.eval().requires_grad_(False)
    for block in model.blocks:
        block.use_checkpoint = False
    for module in (model.blocks[-1], model.head_nm, model.head, model.cb_cond):
        module.requires_grad_(True)
    for parameter in (model.vq.U, model.vq.V, model.vq.lora_gate):
        parameter.requires_grad_(True)
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def valid_order_loss(pred, ref, valid):
    """Eight-neighbor sign inversions, normalized by valid patch adjacencies."""
    p = F.avg_pool2d(pred, 8, 8)
    r = F.avg_pool2d(ref, 8, 8)
    v = F.avg_pool2d(valid.float(), 8, 8) == 1
    total = p.new_zeros(())
    count = p.new_zeros(())
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        h, w = p.shape[-2:]
        left = (..., slice(0, h-dy), slice(max(0,-dx), w-max(0,dx)))
        right = (..., slice(dy, h), slice(max(0,dx), w-max(0,-dx)))
        keep = v[left] & v[right]
        difference = r[left] - r[right]
        reversal = F.relu(-(p[left] - p[right]) * difference)
        total = total + (reversal * keep * (difference.abs() >= 0.1)).sum()
        count = count + keep.sum()
    return total / count.clamp_min(1)


def branch_loss(model, sample, branch, device):
    to_tensor = lambda value: torch.from_numpy(value).unsqueeze(0).to(device)
    gray = to_tensor(sample['reference_' + branch])
    ref_rgb = gray.mul(2).sub(1).repeat(1,3,1,1)
    inp = to_tensor(sample['noisy_' + branch]).mul(2).sub(1).repeat(1,3,1,1)
    valid = to_tensor(sample['valid_mask'])
    label = torch.zeros(1, device=device, dtype=torch.long)
    with torch.no_grad():
        _, ref_codes = model.vae.img_to_idxBl(ref_rgb)
        prefix = model.vq.idxBl_to_var_input(ref_codes)
        target = torch.cat(ref_codes, dim=1)
        coverage = torch.cat([F.adaptive_avg_pool2d(valid.float(), n).flatten(1)
                              for n in model.patch_nums], dim=1)

    # Inference pools raw image, fused, and class tokens before positional encoding.
    # Capture the pre-position features once while the upstream forward produces CE logits.
    pooled = []
    def capture_condition(module, args, output):
        image_tokens, fused_tokens = args[0], output[0]
        class_token = model.class_emb(label).unsqueeze(1)
        pooled.append(torch.cat((image_tokens, fused_tokens, class_token), 1).mean(dim=1))
    hook = model.cross_attention.register_forward_hook(capture_condition)
    try:
        with torch.autocast('cuda', dtype=torch.float16):
            logits, _ = model(prefix, label, inp, text_hidden=None)
    finally:
        hook.remove()
    ce_by_token = F.cross_entropy(logits.float().transpose(1,2), target, reduction='none')
    ce = (ce_by_token * coverage).sum() / coverage.sum()
    indices = logits.detach().argmax(dim=-1).split([n*n for n in model.patch_nums], dim=1)
    h = model.cb_cond(pooled[0].float())
    with torch.autocast('cuda', enabled=False):
        reconstruction = model.vae.idxBl_to_img_lora(indices, h, same_shape=True, last_one=True)
    pred = reconstruction.float().mean(dim=1, keepdim=True).add(1).mul(0.5)
    mse = (pred - gray).square()[valid].mean()
    order = valid_order_loss(pred, gray, valid)
    return ce + 0.2 * mse + 0.8 * order, torch.stack((ce.detach(), mse.detach(), order.detach()))


def atomic_save(path, value, binary=False):
    tmp = path.with_suffix(path.suffix + '.partial')
    if binary:
        torch.save(value, tmp)
    else:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=213)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--schedule-seed', type=int, default=20261120)
    parser.add_argument('--outdir', type=Path, default=ROOT)
    args = parser.parse_args()
    data = DenoisePairDataset(ROOT, seed=args.seed)
    if not 1 <= args.steps <= len(data):
        parser.error('steps must be between one and the size of the primary training list')
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.schedule_seed)).tolist()[:args.steps]
    cp = args.outdir/'checkpoints/s1_warmup.pt'
    report_path = args.outdir/'results/s1_warmup.json'
    if cp.exists() or report_path.exists():
        raise FileExistsError('Use a fresh outdir to preserve this completed run')
    cp.parent.mkdir(exist_ok=True,parents=True)
    report_path.parent.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    model = load_backbone(ROOT/'checkpoints/Real-IISR.pth', device)
    selected = configure(model)
    groups = dict(var=[], cac=[])
    for name, parameter in selected.items():
        key = 'cac' if name.startswith(('cb_cond.', 'vae.quantize.')) else 'var'
        groups[key].append(parameter)
    opt = torch.optim.AdamW([dict(params=groups['var'], lr=1e-5), dict(params=groups['cac'], lr=5e-5)],
                            weight_decay=0.0, foreach=False)
    starting = {name: param.detach().cpu().clone() for name,param in selected.items()}
    training_ids = [data.records[i]['sample_id'] for i in order]
    config = dict(format='real_iisr_prd_s1_v1', base_sha256=OFFICIAL_SHA256, train_steps=args.steps,
                  training_ids=training_ids, data_seed=args.seed, schedule_seed=args.schedule_seed,
                  manifest_sha256=hashlib.sha256((ROOT/'manifests/train.jsonl').read_bytes()).hexdigest(),
                  epoch=0, draw_id=0, image_size=512, padding_dn=0, branches_per_step=2,
                  training_source='local bundled SIRST train; no test or separate validation lists loaded',
                  stage='S1 teacher-forced warmup; hard argmax decode, deployment-matched CAC conditioning',
                  trainable_names=list(selected), trainable_count=sum(p.numel() for p in selected.values()),
                  frozen='all except final original VAR block, logit head/head norm, cb_cond and original CAC U/V/gate',
                  loss_weights={'valid_coverage_token_ce':1.0,'valid_gray_mse':0.2,'valid_patch_order':0.8},
                  lr={'var':1e-5,'cac':5e-5}, weight_decay=0., gradient_clip_norm=2., loss_scale=128.,
                  precision='FP32 tokenization and CAC/decoder; FP16 autocast teacher-forced VAR',
                  torch_version=str(torch.__version__), device=torch.cuda.get_device_name(device))
    metrics = []
    torch.cuda.reset_peak_memory_stats(device)
    print(f"S1 fixed schedule: {args.steps} pairs, {config['trainable_count']:,} trainable parameters",flush=True)
    start = time.monotonic()
    for step, idx in enumerate(order, start=1):
        sample = data[idx]
        opt.zero_grad(set_to_none=True)
        losses = []
        for branch in ('minus','plus'):
            loss, detail = branch_loss(model, sample, branch, device)
            (loss * (config['loss_scale'] / 2)).backward()
            losses.append(detail)
        for param in selected.values():
            if param.grad is not None:
                param.grad.div_(config['loss_scale'])
        norm = torch.nn.utils.clip_grad_norm_(list(selected.values()), 2., error_if_nonfinite=True)
        if step==1:
            first_grad = {name: float(p.grad.float().norm()) if p.grad is not None else None
                          for name,p in selected.items()}
            print('First backward:', first_grad, flush=True)
            print('CAC starting peak and gate:', model.vq.U.abs().max().item(), model.vq.V.abs().max().item(), model.vq.lora_gate.detach().tolist(), flush=True)
            if not all(any(p.grad is not None and p.grad.abs().max()>0 for p in params)
                       for params in groups.values()):
                raise RuntimeError('One of the expected parameter groups received no gradient')
        opt.step()
        ce,mse,rank=torch.stack(losses).mean(0).tolist()
        metrics.append(dict(step=step,id=sample['meta']['sample_id'],ce=ce,masked_mse=mse,
                            patch_order=rank,gradient_norm=float(norm)))
        if step==1 or step%10==0 or step==args.steps:
            print(f"step {step}/{args.steps} CE={ce:.4f} MSE={mse:.5f} grad={float(norm):.4g} "
                  f"elapsed={time.monotonic()-start:.1f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB",flush=True)
    updated={name: param.detach().cpu().clone() for name,param in selected.items()}
    moves={name: (value-starting[name]).abs().max().item() for name,value in updated.items()}
    if not all(torch.isfinite(v).all() for v in updated.values()) or max(moves.values())<=0:
        raise RuntimeError('Final weights failed update checks')
    atomic_save(cp, dict(**config,trainable_state=updated),binary=True)
    atomic_save(report_path,dict(**config, steps=metrics, elapsed_seconds=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),first_step_parameter_grad_norm=first_grad,
        max_abs_update_by_parameter=moves))
    print(f"Saved {cp} and {report_path}",flush=True)


if __name__ == '__main__':
    main()
