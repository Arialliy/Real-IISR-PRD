"""Differentiable free-prefix generation through the original global Real-IISR codebook."""

import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _condition(model, image):
    from models.src.Heatmap import compute_thermal_response_map
    from models.src.sobel import sobel_torch

    feat, _ = model.con_embedding(image)
    spatial = feat.flatten(2).transpose(1, 2)
    heat, _ = model.heatmap_embedding(compute_thermal_response_map(image.mean(1, keepdim=True)))
    edge, _ = model.edge_embedding(sobel_torch(image))
    fused = model.fusion_block(heat, edge).flatten(2).transpose(1, 2)
    cross, _ = model.cross_attention(spatial, fused)
    tokens = torch.cat((spatial, cross), dim=1).repeat(2, 1, 1)
    labels = torch.zeros(len(image), dtype=torch.long, device=image.device)
    classes = model.class_emb(torch.cat((labels, torch.full_like(labels, model.num_classes - 1))))
    tokens = torch.cat((tokens, classes[:, None]), dim=1)
    if tokens.shape[1] != model.context_token:
        raise ValueError("The input dimensions do not match the release's condition grid")
    return tokens, classes, model._h_from_sos(tokens[:len(image)])


def _embedding_pair(quant, logits, h, temperature):
    ids = logits.argmax(-1)
    feedback = quant.embedding(ids)
    final = quant.embedding_lora(ids, h)
    # Lookups above carry the exact hard forward and true table/condition gradients.
    # These two identically-zero terms supply only the soft surrogate logits gradient.
    if torch.is_grad_enabled() and logits.requires_grad:
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probs = (logits.float() / temperature).softmax(-1)
            tangent = probs - probs.detach()
            base = tangent @ quant.embedding.weight.detach().float()
            correction = base
            if quant.use_codebook_lora:
                low_rank = (tangent @ quant.U.detach().float()) * h.detach().float()[:, None]
                correction = base + quant.lora_gate.detach().float().tanh() * (low_rank @ quant.V.detach().float().T)
            feedback = feedback + base.to(feedback.dtype)
            final = final + correction.to(final.dtype)
    return feedback, final, ids


def _prefix_block(block, hidden, classes, freqs, lengths):
    """Recompute one full prefix by scale using upstream SDPA kernel shapes.

    K/V live only within this function call. No K/V are stored by a module or
    carried across generation steps. Each query sees its own scale and all
    earlier scales, exactly as in the cached deployment path.
    """
    from models.basic_var import apply_rotary_emb, slow_attn

    attn = block.attn
    if attn.using_flash or attn.using_xform or block.shared_aln or not block.ada_lin_flag:
        raise ValueError("Expected the pinned release with the project's SDPA configuration")
    gamma1, gamma2, scale1, scale2, shift1, shift2 = (
        block.ada_lin(classes).view(-1, 1, 6, block.C).unbind(2)
    )
    biases = torch.cat((attn.q_bias, attn.zero_k_bias, attn.v_bias))
    keys, values, outputs = [], [], []
    offset = 0
    for x in hidden.split(lengths, dim=1):
        batch, length, width = x.shape
        x = x.contiguous()
        pre = block.ln_wo_grad(x) * (scale1 + 1) + shift1
        qkv = F.linear(pre, attn.mat_qkv.weight, biases)
        q, k, v = qkv.view(batch, length, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        if attn.attn_l2_norm:
            scale = attn.scale_mul_1H11.clamp_max(attn.max_scale_mul).exp()
            q = F.normalize(q, dim=-1) * scale
            k = F.normalize(k, dim=-1)
        q, k = apply_rotary_emb(q, k, freqs[offset:offset + length])
        offset += length
        keys.append(k)
        values.append(v)
        # For the first segment, preserve the original stride/layout of the
        # unbound value tensor; concatenation from the second segment is dense.
        k_prefix = k if len(keys) == 1 else torch.cat(keys, dim=2)
        v_prefix = v if len(values) == 1 else torch.cat(values, dim=2)
        update = slow_attn(query=q, key=k_prefix, value=v_prefix, scale=attn.scale,
                           attn_mask=None, dropout_p=0.)
        update = update.transpose(1, 2).reshape(batch, length, width)
        x = x + block.drop_path(attn.proj_drop(attn.proj(update)) * gamma1)
        pre = block.ln_wo_grad(x) * (scale2 + 1) + shift2
        outputs.append(x + block.drop_path(block.ffn(pre) * gamma2))
    return torch.cat(outputs, dim=1)


def _prefix_logits(model, prefix, classes, freqs, lengths):
    hidden = prefix
    for block in model.blocks:
        if torch.is_grad_enabled():
            hidden = checkpoint(_prefix_block, block, hidden, classes, freqs, lengths,
                                use_reentrant=False, preserve_rng_state=False)
        else:
            hidden = _prefix_block(block, hidden, classes, freqs, lengths)
    hidden = hidden[:, -lengths[-1]:].contiguous()
    logits = model.head(model.head_nm.forward_part(hidden.float(), classes).float()).float()
    return logits[:, -1:] if len(lengths) == 1 else logits


def free_rollout(model, image, *, cfg=1., temperature=1.):
    """Return (RGB clipped to [0,1], logits per scale, greedy IDs per scale).

    `image` uses the release RGB [-1,1] convention. Call model.eval() to freeze
    batch statistics and stochastic layers. Autograd remains controlled by the caller.
    """
    if model.training or any(block.attn.caching for block in model.blocks):
        raise ValueError("Free-prefix rollout requires evaluation mode with idle upstream caches")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(cfg):
        raise ValueError("cfg must be finite")
    b = len(image)
    tokens, classes, h = _condition(model, image)
    levels = model.lvl_embed(model.lvl_1L)
    prefix = tokens + model.pos_start + levels[:, :model.first_l]
    freqs = model.freqs_cis.to(image.device)
    last = model.patch_nums[-1]
    quant = model.vq
    accumulated = tokens.new_zeros((b, model.Cvae, last, last))
    decoded = torch.zeros_like(accumulated, dtype=torch.float32)
    logits_list, ids_list = [], []

    for index, size in enumerate(model.patch_nums):
        length = prefix.shape[1]
        lengths = (model.first_l, *(n*n for n in model.patch_nums[1:index+1]))
        # The first prefix contains only conditioning tokens. Its last position predicts scale 0.
        def score(inputs, cond, rotary, *, segments=lengths):
            return _prefix_logits(model, inputs, cond, rotary, segments)
        if torch.is_grad_enabled():
            raw = checkpoint(score, prefix, classes, freqs[:length],
                             use_reentrant=False, preserve_rng_state=False)
        else:
            raw = score(prefix, classes, freqs[:length])
        weight = cfg * (index / model.num_stages_minus_1)
        logits = (1 + weight) * raw[:b] - weight * raw[b:]
        feedback, final, ids = _embedding_pair(quant, logits, h, temperature)
        logits_list.append(logits)
        ids_list.append(ids)
        ratio = index / model.num_stages_minus_1
        to_grid = (b, model.Cvae, size, size)
        output_grid = final.transpose(1, 2).reshape(to_grid)
        if size != last:
            output_grid = F.interpolate(output_grid, (last, last), mode="bicubic")
        decoded = decoded + quant.quant_resi[ratio](output_grid).float()
        if index < model.num_stages_minus_1:
            feedback_grid = feedback.transpose(1, 2).reshape(to_grid)
            up = F.interpolate(feedback_grid, (last, last), mode="bicubic")
            accumulated = accumulated + quant.quant_resi[ratio](up).to(accumulated.dtype)
            next_size = model.patch_nums[index + 1]
            next_map = F.interpolate(accumulated, (next_size, next_size), mode="area").flatten(2).transpose(1, 2)
            embedded = model.word_embed(next_map) + levels[:, length:length + next_size ** 2]
            prefix = torch.cat((prefix, embedded.repeat(2, 1, 1)), dim=1)

    # Same accumulation and clamp as the original final CAC decode. Keep decoder -> input graph.
    fn = model.vae.fhat_to_img
    if torch.is_grad_enabled():
        result = checkpoint(fn, decoded, use_reentrant=False, preserve_rng_state=False)
    else:
        result = fn(decoded)
    return (result + 1) * 0.5, logits_list, ids_list
