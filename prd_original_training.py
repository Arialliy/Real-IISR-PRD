"""Original Real-IISR model, source initialization and a single-process four-GPU layout."""
import hashlib
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from prd_backbone import PROJECT, UPSTREAM


SCALES = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
HASHES = {"VQVAE.pth": "b4b60aaa47e7cff7fdebb479521138efbbd968faa893fe0d4f8c703fd49ddf45",
          "VARSR.pth": "fd7ebfa6e1d69abdbe5097957c647cbac840cf225355fded2847b6a4d67914db"}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_original(device):
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import models
    if Path(models.__file__).resolve().parent != UPSTREAM / "models":
        raise RuntimeError("Wrong upstream models package")
    classes = (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm,
               nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
    with ExitStack() as scopes:
        for cls in classes:
            scopes.enter_context(patch.object(cls, "reset_parameters", cls.reset_parameters))
        _, model = models.build_var(device=device, patch_nums=SCALES, control_patch_nums=SCALES,
            V=4096, Cvae=32, ch=160, share_quant_resi=4, depth=24, controlnet_depth=24,
            num_classes=2, shared_aln=False, attn_l2_norm=True, flash_if_available=False,
            fused_if_available=False, init_adaln_gamma=1e-4)
    return model


def initialize(model):
    from train import from_pretrained_orig
    vae_checkpoint = torch.load(PROJECT / "checkpoints/VQVAE.pth", weights_only=True,
                                map_location="cpu", mmap=True)
    vae_state = vae_checkpoint["trainer"]["vae_local"]
    skipped = model.vae.load_state_dict(vae_state, strict=False)
    del vae_checkpoint, vae_state
    var_checkpoint = torch.load(PROJECT / "checkpoints/VARSR.pth", weights_only=True,
                                map_location="cpu", mmap=True)
    source = var_checkpoint["trainer"]["var_wo_ddp"]
    # Keep the released training order: the VAR load may also contain VAE keys.
    # from_pretrained_orig does its own remapping and shape rejection before the load.
    traces = {}
    def record_load(module, args):
        state = args[0]
        destinations = module.state_dict()
        matched = sorted(k for k in state if k in destinations and destinations[k].shape == state[k].shape)
        traces.update(var_loaded_keys=matched,
                      var_omitted_destination_keys=sorted(set(destinations) - set(matched)),
                      var_unexpected_source_keys=sorted(state.keys() - destinations.keys()),
                      var_matched_vae_aliases=[k for k in matched if k.startswith(("vae.", "vq."))])
    hook = model.register_load_state_dict_pre_hook(
        lambda module, state, *unused: record_load(module, (state,)))
    try:
        from_pretrained_orig(model, source)
    finally:
        hook.remove()
    traces.update(vq_missing_keys=list(skipped.missing_keys), vq_unexpected_keys=list(skipped.unexpected_keys))
    if not any(k.startswith("blocks.0.") for k in traces["var_loaded_keys"]):
        raise ValueError("VARSR did not initialize the original prediction backbone")
    return traces


def optimizer_for(model):
    nowd = ("cls_token", "start_token", "task_token", "cfg_uncond", "pos_embed", "pos_1LC", "pos_start",
            "lvl_embed", "gamma", "beta", "ada_gss", "moe_bias", "scale_mul")
    groups = {0: [], 1: []}
    for name, param in model.named_parameters():
        if param.requires_grad:
            exempt = param.ndim == 1 or name.endswith("bias") or any(k in name for k in nowd)
            groups[int(not exempt)].append(param)
    return torch.optim.AdamW([{"params": v, "wd_sc": k} for k, v in groups.items()],
                             lr=5e-5, betas=(0.9, 0.95), fused=True)


def distribute_layers(model):
    """Execute one intact four-image batch across local GPUs without gradient collectives.

    The same registered modules provide the model weights and state dict. Hooks only
    move tensors between contiguous stages. The original forward and GradScaler
    handle computation and gradients on each device.
    """
    if torch.cuda.device_count() < 4:
        raise ValueError("Need four visible local GPUs for the full-batch layout")
    primary = torch.device("cuda", 0)
    context = {}

    def begin_batch(module, args):
        context.clear()

    model.register_forward_pre_hook(begin_batch)

    def block_inputs(device):
        def transfer(module, args, kwargs):
            # The upstream model calls all blocks with keyword arguments, referencing
            # the same condition, rotary frequencies and mask for the whole forward.
            if device not in context:
                context[device] = {k: kwargs[k].to(device) for k in ("cond_BD", "freqs_cis", "attn_bias")}
            return args, dict(x=kwargs["x"].to(device), **context[device])
        return transfer

    for index, block in enumerate(model.blocks):
        device = torch.device("cuda", 1 if index < len(model.blocks) // 2 else 2)
        block.to(device)
        block.register_forward_pre_hook(block_inputs(device), with_kwargs=True)
    model.blocks[-1].register_forward_hook(lambda module, args, output: output.to(primary))
    decoder_device = torch.device("cuda", 3)
    model.vae.post_quant_conv.to(decoder_device)
    model.vae.decoder.to(decoder_device)
    model.vae.post_quant_conv.register_forward_pre_hook(
        lambda module, args: (args[0].to(decoder_device),))
    return model


