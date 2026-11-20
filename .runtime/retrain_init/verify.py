from pathlib import Path
import hashlib
import json
import os
import zipfile
import torch
from datetime import datetime
from zoneinfo import ZoneInfo

root = Path("/project")
revision = "a0e478f5e1640a7e048f1303781b6d34ff3296ca"
records = {
    "VQVAE.pth": (1400076874,"b4b60aaa47e7cff7fdebb479521138efbbd968faa893fe0d4f8c703fd49ddf45","vae_local"),
    "VARSR.pth": (13702361426,"fd7ebfa6e1d69abdbe5097957c647cbac840cf225355fded2847b6a4d67914db","var_wo_ddp")
}
collected = {}
for name,(expected_bytes, expected_sha, state_name) in records.items():
    path = root/"checkpoints"/name
    assert path.stat().st_size == expected_bytes
    with path.open("rb") as file:
        actual_sha = hashlib.file_digest(file,"sha256").hexdigest()
    assert actual_sha == expected_sha,(name,actual_sha)
    state = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
    assert set(state) == {"epoch","iter","trainer","args"}
    trainer = state["trainer"]
    tensors = trainer[state_name]
    assert isinstance(tensors, dict)
    assert all(isinstance(v,torch.Tensor) for v in tensors.values())
    probes = ("quantize.embedding.weight","encoder.conv_in.weight","decoder.conv_out.weight") if name=="VQVAE.pth" else ("class_emb.weight","pos_start","blocks.0.attn.mat_qkv.weight","head.weight")
    sampled = {}
    for key in probes:
        value = tensors[key].detach().float()
        assert torch.isfinite(value).all().item()
        sampled[key] = list(value.shape)
    collected[name] = {
        "path": str(path.relative_to(root)),
        "source_repository":"https://huggingface.co/qyp2000/VARSR",
        "revision":revision,
        "url": f"https://huggingface.co/qyp2000/VARSR/resolve/{revision}/{name}",
        "size_bytes":path.stat().st_size,
        "sha256":actual_sha,
        "matches_pinned_lfs_oid":True,
        "torch_load": {
            "weights_only": True, "mmap":True, "map_location":"cpu",
            "pytorch_version":str(torch.__version__), "success":True,
            "zip_archive": zipfile.is_zipfile(path),
            "root_keys":list(state),"trainer_keys":list(trainer),
            "source_tensor_dict":f"trainer.{state_name}", "tensor_entries":len(tensors),
            "source_patch_nums":list(trainer["config"]["patch_nums"]),
            "finite_tensor_probes":sampled
        },
    }
    print(name,actual_sha,len(tensors), flush=True)
    del value,tensors,trainer,state
report={
    "source_readme":"vendor/Real-IISR/README.md",
    "upstream_reference":"vendor/Real-IISR/train.py:128",
    "training_initialization":"Published VQ-VAE and VARSR source initialization for an independent local retraining run",
    "verified_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    "source_metadata_url":f"https://huggingface.co/api/models/qyp2000/VARSR/tree/{revision}",
    "files":collected,
}
out = root/"results/local_retrain/initialization.json"
out.parent.mkdir(parents=True,exist_ok=True)
tmp=out.with_suffix(".json.tmp")
tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
os.replace(tmp,out)
