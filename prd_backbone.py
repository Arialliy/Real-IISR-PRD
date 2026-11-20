"""Load the pinned upstream Real-IISR release and call its unmodified hard forward."""

import hashlib
import sys
import warnings
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


PROJECT = Path(__file__).resolve().parent
UPSTREAM = PROJECT / "vendor" / "Real-IISR"
OFFICIAL_SHA256 = "fe1c2600ec6aeb18bfa674f7726a1dcd0aec45d9fbba25c7f1b9fe266b3ef9a9"


def load_backbone(checkpoint, device):
    """Return the original evaluation-mode model from the published release."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != OFFICIAL_SHA256:
        raise ValueError(f"Expected the published Real-IISR checkpoint: {checkpoint}")

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import models

    if Path(models.__file__).resolve().parent != UPSTREAM / "models":
        raise ImportError(f"Another package is shadowing the pinned models: {models.__file__}")

    release = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    trainer = release["trainer"]
    scales = tuple(trainer["config"]["patch_nums"])
    args = release["args"]
    device = torch.device(device)
    # build_var overrides these class methods process-wide. Scope that change so
    # layers constructed later by the caller still initialize normally.
    classes = (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm,
               nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
    with ExitStack() as scopes:
        if device.type == "cuda":
            scopes.enter_context(torch.cuda.device(device))
        for cls in classes:
            scopes.enter_context(patch.object(cls, "reset_parameters", cls.reset_parameters))
        _, model = models.build_var(
            device=device, patch_nums=scales, control_patch_nums=scales,
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            depth=args["depth"], controlnet_depth=args["depth"], num_classes=2,
            shared_aln=args["saln"], attn_l2_norm=args["anorm"],
            flash_if_available=False, fused_if_available=False,
        )

    state = trainer["var_wo_ddp"]
    keys = model.state_dict().keys()
    extra = state.keys() - keys
    if extra:
        warnings.warn(f"Unused release entries (as in upstream test.py): {sorted(extra)}")
    model.load_state_dict({key: value for key, value in state.items() if key in keys}, strict=True)
    model.prd_base_sha256 = OFFICIAL_SHA256
    return model.eval()


def _read_updates(model, path, expected_format):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if (payload.get("format") != expected_format
            or payload.get("base_sha256") != OFFICIAL_SHA256
            or getattr(model, "prd_base_sha256", None) != OFFICIAL_SHA256):
        raise ValueError("Adaptation checkpoint or base-model identity mismatch")
    if not payload.get("training_ids") or payload.get("train_steps", 0) < 1:
        raise ValueError("Missing update history")
    updates = payload["trainable_state"]
    parameters = dict(model.named_parameters(remove_duplicate=False))
    if not updates or updates.keys() != set(payload.get("trainable_names", [])):
        raise ValueError("Update tensors do not match the recorded parameter list")
    if updates.keys() - parameters.keys():
        raise ValueError("Checkpoint refers to missing parameters")
    destinations = set()
    for name, value in updates.items():
        param = parameters[name]
        if (param.shape != value.shape or param.dtype != value.dtype
                or id(param) in destinations):
            raise ValueError(f"Invalid parameter entry: {name}")
        destinations.add(id(param))
    return payload


@torch.no_grad()
def _apply_updates(model, payload):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    for name, value in payload["trainable_state"].items():
        parameters[name].copy_(value)
    return {key: value for key, value in payload.items() if key != "trainable_state"}


def apply_warmup(model, path):
    """Restore the S1-only parameters on top of the verified original model."""
    return _apply_updates(model, _read_updates(model, path, "real_iisr_prd_s1_v1"))


def apply_s2_pilot(model, warmup_path, pilot_path):
    """Restore S2 absolute weights after verifying the entire S1 parent chain."""
    pilot = _read_updates(model, pilot_path, "real_iisr_prd_s2_pilot_v1")
    with Path(warmup_path).open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != pilot["parent_sha256"]:
            raise ValueError("S2 pilot was initialized from a different S1 checkpoint")
    parent = _read_updates(model, warmup_path, "real_iisr_prd_s1_v1")
    if pilot["trainable_state"].keys() != parent["trainable_state"].keys():
        raise ValueError("S2 pilot must account for every S1 adapted tensor")
    # S2 stores full absolute tensors for exactly the parameters changed since the release.
    return _apply_updates(model, pilot)


def official_inference(model, image):
    """Use upstream test-time options with a shared tie-breaking seed per call."""
    batch = image.shape[0]
    labels = torch.zeros(batch, dtype=torch.long, device=image.device)
    # Upstream places rotary frequencies on the current CUDA device.
    with torch.cuda.device(image.device) if image.is_cuda else nullcontext():
        return model.autoregressive_infer_cfg(
            B=batch, text_hidden=None, lr_inp=image, negative_text=None,
            label_B=labels, cfg=1, top_k=1, top_p=0.75, g_seed=0,
            lr_inp_scale=None, more_smooth=False,
        )
