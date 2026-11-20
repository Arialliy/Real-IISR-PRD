"""Pair supervision for grayscale restoration. All images share the [0, 1] DN scale."""

from typing import Mapping

import torch
from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Per-image mean over a boolean mask. An empty mask contributes zero."""
    count = mask.flatten(1).sum(dim=1)
    return values.masked_fill(~mask, 0).flatten(1).sum(dim=1) / count.clamp_min(1)


def paired_loss(
    pred_minus: Tensor,
    pred_plus: Tensor,
    batch: Mapping[str, Tensor],
    variant: str,
    *,
    ring_weight: float = 0.25,
    amplitude_floor: float = 1 / 255,
) -> Tensor:
    """Return the unweighted P or C term for a minibatch of [B, 1, H, W] images.

    P averages the absolute response error across all valid image pixels.
    C averages it in the signal support and ring, then scales by signal amplitude.
    Both reduce over the same eligible pairs with equal weight per pair.
    """
    if variant not in ("P", "C"):
        raise ValueError("variant must be P or C")
    if amplitude_floor <= 0 or ring_weight < 0:
        raise ValueError("amplitude_floor must be positive and ring_weight nonnegative")
    if pred_minus.ndim != 4 or pred_minus.shape[1] != 1:
        raise ValueError("expected [B, 1, H, W] grayscale outputs")
    for field in ("signal", "valid_mask", "support_mask", "ring_mask"):
        if batch[field].shape != pred_minus.shape:
            raise ValueError(f"{field} shape does not match the model outputs")
    if pred_minus.shape != pred_plus.shape:
        raise ValueError("paired outputs must have identical shapes")
    signal = batch["signal"].float()
    error = (pred_plus.float() - pred_minus.float() - signal).abs()
    valid = batch["valid_mask"].bool()
    support = valid & batch["support_mask"].bool()
    eligible = batch["pair_valid"].reshape(-1).bool()
    if eligible.numel() != len(pred_minus):
        raise ValueError("pair_valid must contain one flag per image pair")
    eligible = eligible & support.flatten(1).any(dim=1)

    if variant == "P":
        per_pair = _masked_mean(error, valid)
    else:
        ring = valid & batch["ring_mask"].bool()
        amplitude = signal.abs().masked_fill(~valid, 0).flatten(1).amax(dim=1)
        local = _masked_mean(error, support) + ring_weight * _masked_mean(error, ring)
        per_pair = local / amplitude.clamp_min(amplitude_floor)
    return per_pair.masked_fill(~eligible, 0).sum() / eligible.sum().clamp_min(1)
