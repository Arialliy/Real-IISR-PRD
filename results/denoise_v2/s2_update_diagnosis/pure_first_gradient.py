"""Twice run the original pre-update loss, changing only its audit output directory."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch


OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[2]
sys.path.insert(0,str(PROJECT))
import train_s2_common as original
from evaluate_warmup import save_report


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261120)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0)
    cfg=json.loads(original.REG.read_text())
    old=json.loads((original.OUT/"first_update_grad_norms.json").read_text())
    logged=json.loads((original.OUT/"updates.jsonl").read_text().splitlines()[0])
    ids=logged["sample_ids"]
    data=original.DenoisePairDataset(PROJECT,seed=0)
    indices={r["sample_id"]:i for i,r in enumerate(data.records)}
    model=original.load_trained(PROJECT/cfg["parent_path"],torch.device("cuda:0"))
    named={n:p for n,p in model.named_parameters() if p.requires_grad}
    opt=original.s2_optimizer(model,cfg["optimizer"])
    outs=[]
    base=OUT/"pure_first_gradient"
    base.mkdir(exist_ok=True)
    for repeat in (1,2):
        opt.zero_grad(set_to_none=True)
        vals=[]
        with patch.object(original,"OUT",base):
            for j,k in enumerate(ids):
                sample=data.sample(indices[k],epoch=20,draw_id=2)
                loss,parts=original.image_loss(model,sample,check_cached=(j==0))
                (loss*(512./4)).backward()
                vals.append(parts)
        torch._foreach_mul_([p.grad for p in named.values()],1./512.)
        norm=float(torch.nn.utils.clip_grad_norm_(named.values(),2.))
        norms={n:float(p.grad.norm()) for n,p in named.items()}
        outs.append(dict(repeat=repeat,global_norm=norm,parts=torch.stack(vals).mean(0).tolist(),
                         equal_original_postclip_grad_norms=sum(old[n]==v for n,v in norms.items()),
                         max_abs_diff_original_postclip_grad_norms=max(abs(old[n]-v) for n,v in norms.items()),
                         postclip_grad_norms=norms))
    prior_replay=json.loads((OUT/"T-ALL/updates.jsonl").read_text().splitlines()[0])
    stats=dict(accepted_updates=0,official_loss_implementation="train_s2_common.image_loss",physical_gpu=3,
               original_published_log_norm=logged["grad_norm"],instrumented_replay_norm=prior_replay["grad_norm"],
               objective="full original S2 combined; first four original samples; same seed, original registered scale 512 and clip 2",
               repeated_global_norm_difference=abs(outs[0]["global_norm"]-outs[1]["global_norm"]),replicates=outs)
    save_report(base/"repeatability.json",stats)
    print([r["global_norm"] for r in outs],flush=True)


if __name__=="__main__":
    main()
