"""Bounded common free-prefix S2 pilot starting from the registered S1 end point."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prd_backbone import OFFICIAL_SHA256, apply_warmup, load_backbone
from prd_data import DenoisePairDataset
from prd_rollout import free_rollout
from train_warmup import ROOT, atomic_save, configure, valid_order_loss


def free_branch_loss(model, sample, side, device):
    ref = torch.from_numpy(sample['reference_' + side])[None].to(device)
    inp = torch.from_numpy(sample['noisy_' + side])[None].to(device)
    valid = torch.from_numpy(sample['valid_mask'])[None].to(device)
    with torch.no_grad():
        _, labels = model.vae.img_to_idxBl(ref.mul(2).sub(1).repeat(1,3,1,1))
        weights = torch.cat([F.adaptive_avg_pool2d(valid.float(), n).flatten(1)
                             for n in model.patch_nums], dim=1)
    with torch.autocast('cuda', dtype=torch.float16):
        rgb, logits, _ = free_rollout(model, inp.mul(2).sub(1).repeat(1,3,1,1))
    # Image loss backpropagates through the hard rollout via its ST surrogate.
    gray = rgb.float().mean(1, keepdim=True)
    mse = (gray - ref).square()[valid].mean()
    order = valid_order_loss(gray, ref, valid)
    ce_tokens = F.cross_entropy(torch.cat(logits,1).float().transpose(1,2),
                                torch.cat(labels,1), reduction='none')
    ce = (ce_tokens * weights).sum() / weights.sum()
    return ce + 0.2*mse + 0.8*order, torch.stack((ce.detach(), mse.detach(), order.detach()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=ROOT)
    args = parser.parse_args()
    cp = args.outdir/'checkpoints/s2_common_pilot.pt'
    report_file = args.outdir/'results/s2_common_pilot.json'
    if cp.exists() or report_file.exists():
        raise FileExistsError('Select a fresh project outdir to preserve completed S2 pilot')
    cp.parent.mkdir(exist_ok=True, parents=True)
    report_file.parent.mkdir(exist_ok=True, parents=True)
    parent = ROOT/'checkpoints/s1_warmup.pt'
    with parent.open('rb') as handle:
        parent_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    parity = json.loads((ROOT/'results/rollout_check_s1_fp16.json').read_text())
    if not parity['exact_hard_parity'] or parity['s1_checkpoint_sha256'] != parent_hash:
        raise ValueError('Free-prefix hard parity record must match this S1 end point')
    data_seed, schedule_seed, draw_id = 0, 20261121, 1
    ds = DenoisePairDataset(ROOT, seed=data_seed)
    updates = torch.randperm(len(ds), generator=torch.Generator().manual_seed(schedule_seed))[:16].tolist()
    torch.manual_seed(data_seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    model = load_backbone(ROOT/'checkpoints/Real-IISR.pth', device)
    s1 = apply_warmup(model, parent)
    selected = configure(model)
    if set(selected) != set(s1['trainable_names']):
        raise ValueError('S2 parameter set differs from S1')
    groups = {'var': [], 'cac': []}
    for name, parameter in selected.items():
        groups['cac' if name.startswith(('cb_cond.','vae.quantize.')) else 'var'].append(parameter)
    lr = s1['lr']
    optimizer = torch.optim.AdamW([dict(params=params,lr=lr[key]) for key,params in groups.items()],
                                  weight_decay=0., foreach=False)
    starting = {k:p.detach().cpu().clone() for k,p in selected.items()}
    config = dict(format='real_iisr_prd_s2_pilot_v1', base_sha256=OFFICIAL_SHA256, parent_sha256=parent_hash,
                  stage='16-step exploratory S2 common free-prefix repair pilot; no P/C losses or local CAC',
                  train_steps=len(updates), trainable_names=list(selected), trainable_count=s1['trainable_count'],
                  training_ids=[ds.records[i]['sample_id'] for i in updates], manifest_sha256=s1['manifest_sha256'],
                  data_seed=data_seed, schedule_seed=schedule_seed, epoch=0, draw_id=draw_id, padding_dn=0,
                  image_size=512, branches_per_step=2, loss_weights=s1['loss_weights'],
                  lr=lr, weight_decay=0., gradient_clip_norm=2., initial_loss_scale=4096., loss_scale_backoff=0.5,
                  train_token_mode='greedy hard forward / softmax ST gradient, tau=1, cfg=1',
                  precision='FP32 reference tokenization, FP16 autocast free rollout incl decoder, FP32 metrics and CE',
                  torch_version=str(torch.__version__), device=torch.cuda.get_device_name(device))
    rows = []
    scale = config['initial_loss_scale']
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    print('S2: 16 fixed paired steps; draw=1, schedule_seed=20261121', flush=True)
    for step, i in enumerate(updates,1):
        sample = ds.sample(i, epoch=0, draw_id=draw_id)
        while True:
            optimizer.zero_grad(set_to_none=True)
            details = []
            for side in ('minus','plus'):
                loss, detail = free_branch_loss(model, sample, side, device)
                (loss * (scale / 2)).backward()
                details.append(detail)
            grads = [param.grad for param in selected.values() if param.grad is not None]
            for grad in grads:
                grad.div_(scale)
            if torch.isfinite(torch.stack([grad.norm() for grad in grads])).all():
                break
            if scale < 2:
                raise FloatingPointError(f'Nonfinite gradients at unit scale on step {step}')
            print(f'nonfinite gradient at step {step}, scale {scale}; retry identical pair', flush=True)
            scale *= config['loss_scale_backoff']
        norm = torch.nn.utils.clip_grad_norm_(list(selected.values()), 2., error_if_nonfinite=True)
        cac_norm = torch.stack([p.grad.norm() for p in groups['cac'] if p.grad is not None]).norm().item()
        if not cac_norm > 0:
            raise RuntimeError(f'CAC image-loss path produced zero gradient at update {step}')
        if step==1:
            first_grad = {k:p.grad.norm().item() if p.grad is not None else None for k,p in selected.items()}
        optimizer.step()
        ce,mse,order=torch.stack(details).mean(0).tolist()
        rows.append(dict(step=step, id=sample['meta']['sample_id'], ce=ce, masked_mse=mse, patch_order=order,
                         gradient_norm=float(norm), loss_scale=scale, clipped_cac_gradient_norm=cac_norm))
        print(f'step {step}/16 CE={ce:.4f} MSE={mse:.5f} grad={float(norm):.4g} CAC={cac_norm:.3g} '
              f'elapsed={time.monotonic()-start:.1f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)
    updated={k:p.detach().cpu().clone() for k,p in selected.items()}
    diffs={k:(v-starting[k]).abs().max().item() for k,v in updated.items()}
    if not all(torch.isfinite(v).all() for v in updated.values()) or max(diffs.values())<=0:
        raise RuntimeError('S2 final parameters failed update check')
    atomic_save(cp, dict(**config,trainable_state=updated), binary=True)
    atomic_save(report_file, dict(**config,steps=rows, max_abs_update_by_parameter=diffs,
                first_step_parameter_grad_norm=first_grad, elapsed_seconds=time.monotonic()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    print(f'Saved {cp} and {report_file}',flush=True)


if __name__ == '__main__':
    main()
