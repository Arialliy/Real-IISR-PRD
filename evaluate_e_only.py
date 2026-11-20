"""Capture and replay formal SIRST endpoints with identical tokens and two final decoders."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from evaluate_sirst_endpoints import MANIFEST, OUT, S1, S2
from evaluate_warmup import file_hash, sample_digest, save_report
from prd_backbone import PROJECT
from prd_data import DenoisePairDataset
from probe_backbone import model_input, to_gray_01
from verify_rollout import cached_greedy


DEST = PROJECT / "results/denoise_v2/token_sources"
SOURCES = ("predicted", "input", "reference")
SETTINGS = ("0", "clean", "1")


def lineage(model):
    """Hash the actual floating state that determines VQ tokenization or pixel reconstruction."""
    def module_hash(modules):
        digest = hashlib.sha256()
        for prefix, module in modules.items():
            for name, tensor in sorted(module.state_dict().items()):
                arr = tensor.detach().cpu().contiguous().numpy()
                digest.update(json.dumps([prefix, name, str(arr.dtype), arr.shape]).encode())
                digest.update(memoryview(arr).cast("B"))
        return digest.hexdigest()

    quant = model.vae.quantize
    return {"tokenizer_sha256": module_hash({"encoder": model.vae.encoder, "quant_conv": model.vae.quant_conv,
                                            "embedding": quant.embedding, "quant_resi": quant.quant_resi}),
            "decoder_sha256": module_hash({"quant_resi": quant.quant_resi, "post_quant_conv": model.vae.post_quant_conv,
                                          "decoder": model.vae.decoder})}


def samples(data, draw, controls):
    for i, (record, prior) in enumerate(zip(data.records, controls, strict=True)):
        sample = data.sample(i, epoch=0, draw_id=0 if draw == "clean" else int(draw))
        if draw == "clean":
            sample["noisy_minus"] = sample["reference_minus"]
        w,h = record["reference_size"]
        left, top = sample["meta"]["pad_left_top"]
        if (record["sample_id"] != prior["sample_id"] or sample_digest(sample) != prior["data_sha256"]
                or sample["meta"]["crop_xywh"] != (0, 0, w, h) or sample["valid_mask"].sum() != w*h):
            raise ValueError(f"Training input or native extent differs: {record['sample_id']}")
        yield sample, prior, np.s_[top:top+h, left:left+w]


def encode(model, gray, device):
    with torch.autocast("cuda", enabled=False):
        _, codes = model.vae.img_to_idxBl(model_input(gray, device))
    return codes


def capture(model, device, data, controls, endpoint, draw, identity):
    stages = np.asarray(model.patch_nums, dtype=np.int64)
    offsets = np.r_[0, np.cumsum(stages**2)]
    packed = {s: [] for s in SOURCES}
    global_conditions, ids, digests = [], [], []
    old_probe = OUT / f"{endpoint}_draw0_token_sources.json"
    old_rows = json.loads(old_probe.read_text())["samples"] if draw == "0" else None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for idx, (sample, prior, native_slice) in enumerate(samples(data, draw, controls)):
            noisy = sample["noisy_minus"]
            observations = []
            decode = model.vae.idxBl_to_img_lora

            def observe(tokens, h, *args, **kwargs):
                observations.append((tokens, h))
                return decode(tokens, h, *args, **kwargs)

            with patch.object(model.vae, "idxBl_to_img_lora", observe):
                output, _, sampler_tokens = cached_greedy(model, model_input(noisy, device))
            if len(observations) != 1:
                raise ValueError("Expected one official final global CAC decode")
            predicted, h = observations[0]
            if any(not torch.equal(a, b) for a,b in zip(predicted, sampler_tokens, strict=True)):
                raise ValueError("Cached greedy tokens differ from the decoder input")
            archived = OUT / prior["native_gray_npy"]
            gray = to_gray_01(output).numpy()[native_slice]
            if (file_hash(archived) != prior["native_gray_sha256"]
                    or not np.array_equal(gray, np.load(archived, allow_pickle=False))):
                raise ValueError(f"Official native prediction did not replay: {prior['sample_id']}")
            tokens = (predicted, encode(model, noisy, device), encode(model, sample["reference_minus"], device))
            for key, codes in zip(SOURCES, tokens, strict=True):
                if [c.numel() for c in codes] != list(stages**2):
                    raise ValueError("Unexpected token scales")
                flat = torch.cat(tuple(codes), dim=1)[0].cpu().numpy()
                if flat.min() < 0 or flat.max() >= model.vq.embedding.num_embeddings or flat.max() > 65535:
                    raise ValueError("Token index outside the lossless uint16 range")
                packed[key].append(flat.astype(np.uint16))
            if draw == "clean" and not np.array_equal(packed["reference"][-1], packed["input"][-1]):
                raise ValueError("Clean-reference and input encodings differ on the identical image")
            if old_rows is not None:
                previous = old_rows[idx]
                assert (previous["sample_id"], previous["data_sha256"]) == (prior["sample_id"], prior["data_sha256"])
                for j, key in enumerate(("pred_tokens_noisy_h", "noisy_tokens_noisy_h")):
                    diff = [int((a != b).sum()) for a,b in zip(tokens[j], tokens[-1], strict=True)]
                    assert previous["different_from_ref_tokens_by_scale"][key] == diff
            ids.append(prior["sample_id"])
            digests.append(prior["data_sha256"])
            global_conditions.append(h[0].cpu().numpy().copy())
            if len(ids) % 65 == 0:
                print(f"{endpoint} draw {draw}: captured {len(ids)}/213", flush=True)
    stem = DEST / f"{endpoint}_{draw}_tokens"
    temp = stem.with_suffix(".pending")
    with temp.open("wb") as f:
        np.savez_compressed(f, ids=np.asarray(ids), input_sha256=np.asarray(digests), scales=stages,
                            offsets=offsets, h=np.stack(global_conditions), **{k:np.stack(v) for k,v in packed.items()})
    archive = stem.with_suffix(".npz")
    temp.replace(archive)
    report = dict(**identity, endpoint=endpoint, draw=draw, n_images=len(ids), token_archive=archive.name,
                  tokens_and_h_sha256=file_hash(archive), verified_greedy_rasters=len(ids),
                  clean_independent_encodings_verified=len(ids) if draw == "clean" else 0,
                  historical_draw0_token_mismatch_counts_verified=len(ids) if old_rows else 0,
                  scales=stages.tolist(), scale_offsets=offsets.tolist(), global_h_dtype=str(global_conditions[0].dtype))
    save_report(stem.with_suffix(".json"), report)
    print(f"{endpoint} draw {draw}: captured all indices and exact noisy h", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["capture", "score"])
    parser.add_argument("--endpoints", nargs="+", choices=["s1","s2"], default=["s1","s2"])
    parser.add_argument("--draws", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    data = DenoisePairDataset(PROJECT, seed=0)
    assert len(data) == 213 and len({r["sample_id"] for r in data.records}) == 213
    assert all(r["split"] == "train" and r["dataset"] == "NUAA-SIRST" for r in data.records)
    DEST.mkdir(parents=True, exist_ok=True)
    for endpoint in args.endpoints:
        if endpoint == "s1":
            from train_local_baseline import load_trained
            model, asset = load_trained(S1, device), S1
        else:
            from train_s2_common import load_s2
            model, asset = load_s2(S2, device), S2
        common = dict(checkpoint_sha256=file_hash(asset), manifest_sha256=file_hash(MANIFEST), **lineage(model))
        for draw in args.draws:
            path = OUT / f"{endpoint}_draw{draw}.json"
            control = json.loads(path.read_text())
            assert (control["endpoint"], control["draw"], control["data_seed"], control["epoch"],
                    control["split"]) == (endpoint, draw, 0, 0, "NUAA-SIRST/train")
            assert control["checkpoint_sha256"] == common["checkpoint_sha256"]
            assert control["manifest_sha256"] == common["manifest_sha256"]
            identity = dict(**common, endpoint_control_sha256=file_hash(path))
            if args.operation == "capture":
                capture(model, device, data, control["samples"], endpoint, draw, identity)
            else:
                from score_e_only import score
                score(model, device, data, control["samples"], endpoint, draw, identity)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
