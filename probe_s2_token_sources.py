"""Compare four token-source routes on eight fixed local training images at S2 step 16."""

import json

import torch

from evaluate_warmup import file_hash, prepare_metrics, sample_digest, save_report, summarize
from prd_backbone import OFFICIAL_SHA256, PROJECT, apply_s2_pilot, load_backbone, official_inference
from prd_data import DenoisePairDataset
from probe_backbone import VENDOR_REVISION, model_input, run_branch, to_gray_01


METRICS = ("valid_mae_dn", "background_rmse_dn")
ROUTES = {
    "noisy_input": {"uses_reference_to_generate_output": False, "source": "uncorrected input"},
    "pred_tokens_noisy_h": {"uses_reference_to_generate_output": False, "source": "official cached AR hard prediction from the noisy image"},
    "noisy_tokens_noisy_h": {"uses_reference_to_generate_output": False, "source": "quantize the noisy image using the frozen VQ encoder and global base codebook"},
    "ref_tokens_noisy_h": {"uses_reference_to_generate_output": True, "source": "quantize the pre-corruption reference using the same VQ encoder and global base codebook"},
}


def main():
    checkpoint_dir = PROJECT / "checkpoints"
    warmup = checkpoint_dir / "s1_warmup.pt"
    pilot = checkpoint_dir / "s2_common_pilot.pt"
    formal_path = PROJECT / "results/s2_pilot_paired_eval.json"
    formal = json.loads(formal_path.read_text())
    dataset = DenoisePairDataset(PROJECT, seed=0)
    device = torch.device("cuda:0")
    model = load_backbone(checkpoint_dir / "Real-IISR.pth", device)
    pilot_meta = apply_s2_pilot(model, warmup, pilot)
    pilot_hash = file_hash(pilot)
    if pilot_meta["train_steps"] != 16 or pilot_hash != formal["s2_pilot"]["checkpoint_sha256"]:
        raise ValueError("Expected the S2 step 16 checkpoint in the official paired trace")
    rows = []
    expected_sizes = [n * n for n in model.patch_nums]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for index in range(8):
            sample = dataset.sample(index, epoch=0, draw_id=0)
            prior = formal["samples"][index]
            if sample["meta"]["sample_id"] != prior["sample_id"] or sample_digest(sample) != prior["data_sha256"]:
                raise ValueError("Input differs from the reference evaluation trace")
            noisy = sample["noisy_minus"]
            displays, ref_codes, pred_codes, noisy_h = run_branch(
                model, sample["reference_minus"], noisy, device, official_inference
            )
            with torch.autocast("cuda", enabled=False):
                _, noisy_codes = model.vae.img_to_idxBl(model_input(noisy, device))
            if [tokens.shape[1] for tokens in noisy_codes] != expected_sizes:
                raise ValueError("Unexpected VQ scale order")
            decoded = model.vae.idxBl_to_img_lora(noisy_codes, noisy_h, same_shape=True, last_one=True)
            displays["noisy_tokens_noisy_h"] = to_gray_01((decoded + 1) * 0.5)
            measure, coverage = prepare_metrics(sample)
            metrics = {}
            for route in ROUTES:
                measured = measure(displays[route].numpy())
                metrics[route] = {key: measured[key] for key in METRICS}
            for route, prior_route in (("noisy_input", "input"), ("pred_tokens_noisy_h", "s2_pilot")):
                for key in METRICS:
                    if abs(metrics[route][key] - prior["metrics"][prior_route][key]) > 1e-5:
                        raise ValueError(f"Official control failed to replay {route}/{key} on {prior['sample_id']}")
            rows.append({
                "sample_id": prior["sample_id"], "data_sha256": prior["data_sha256"],
                "used_for_s1_update": prior["used_for_s1_update"],
                "used_for_s2_update": prior["used_for_s2_update"],
                "valid_pixels": coverage["valid_pixels"], "background_pixels": coverage["background_pixels"],
                "metrics": metrics,
                "different_from_reference_token_counts_by_scale": {
                    name: [torch.count_nonzero(a != b).item() for a, b in zip(codes, ref_codes)]
                    for name, codes in (("noisy_encoded", noisy_codes), ("freely_predicted", pred_codes))
                },
            })
    report = {
        "selection": "first eight IDs in bundled training-list order", "split": "train",
        "training_manifest_sha256": file_hash(PROJECT / "manifests/train.jsonl"),
        "data_seed": 0, "epoch": 0, "draw_id": 0, "branch": "noisy_minus", "padding_dn": 0,
        "source_revision": VENDOR_REVISION, "base_checkpoint_sha256": OFFICIAL_SHA256,
        "s1_checkpoint_sha256": file_hash(warmup), "s2_checkpoint_sha256": pilot_hash,
        "s2_steps": pilot_meta["train_steps"], "formal_trace_sha256": file_hash(formal_path),
        "patch_nums": list(model.patch_nums), "token_counts_by_scale": expected_sizes,
        "routes": ROUTES,
        "shared_decode": "all three token routes use the identical noisy h captured from the upstream cached call, the global CAC and the same decoder",
        "hard_prediction": "upstream cached cfg=1 top_k=1 top_p=0.75, g_seed=0 reset for each image",
        "precision": "FP32 noisy/reference VQ tokenization; FP16 CUDA autocast for hard prediction and final VQ decodes; FP32 RGB mean and FP64 metrics",
        "evaluation_reference": "pre-corruption reference_minus, used for scoring all four routes",
        "aggregation": "mean and median of per-image scores in grayscale DN",
        "samples": rows, "by_image": summarize(rows),
    }
    save_report(PROJECT / "results/s2_token_source_probe.json", report)
    print(json.dumps(report["by_image"], indent=2))


if __name__ == "__main__":
    main()
