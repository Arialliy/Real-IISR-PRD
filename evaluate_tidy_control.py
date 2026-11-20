"""External thermal denoiser control on NUAA train, and a paired SCaN-TIR sequence pilot."""
import argparse
import csv
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from check_column_baseline import correct_columns
from evaluate_noise_levels import FIELDS, dataset, get_model, write_csv
from evaluate_protocol_v2 import measure_native, native_window, pixel_sha
from evaluate_retrained import hard_prediction
from evaluate_warmup import file_hash, save_report

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results/denoise_v2/tidy_control"
UPSTREAM = ROOT/".runtime/tidy_official"
TIDY = ROOT/"checkpoints/tidy.pth"
TIDY_HASH = "37a45473779afcd73abd941e71787caac658325d813342a1bbf787612180906b"
REVISION = "ebb20eaa615cc3ae3fbad6d25ee48de46e7034c0"
SCAN = ROOT/"datasets/SCaN-TIR/300_floor5.tar.gz"
SCAN_HASH = "06a495daec5db7593335138e6fee8e5c8f9b21f4ff7ed83a7070a094523f533b"
SCAN_IDS = [f"frame_{k:05d}.png" for k in np.rint(np.linspace(0,493,32)).astype(int)]


def load_tidy():
    if file_hash(TIDY) != TIDY_HASH:
        raise ValueError("Incorrect TIDY release file")
    sys.path.insert(0,str(UPSTREAM))
    from tidy.backends import PyTorchBackend
    return PyTorchBackend(TIDY,torch.device("cuda"),fp16=False)


def tidy_image(backend, gray):
    tensor = torch.from_numpy(np.ascontiguousarray(gray))[None,None].repeat(1,3,1,1)
    with torch.inference_mode():
        out = backend.infer(tensor).float().clamp(0,1).mean(1)[0].cpu().numpy()
    if out.shape != gray.shape or not np.isfinite(out).all():
        raise ValueError("Invalid native TIDY output")
    return out


def means(rows, keys):
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys
            if any(r[k] is not None for r in rows)}


def nuaa():
    with (ROOT/"results/protocol_v2/per_image.csv").open(newline="") as handle:
        main={(r["corruption"],r["sample_id"]):r for r in csv.DictReader(handle) if r["method"]=="input"}
    backend = load_tidy()
    data = dataset()
    images,targets,summary=[],[],{}
    for c, draw in (("clean",0),("draw0",0),("draw1",1)):
        for i, rec in enumerate(data.records):
            sample=data.sample(i,epoch=0,draw_id=draw)
            gray=sample["reference_minus" if c=="clean" else "noisy_minus"][0]
            box=native_window(sample)
            digest=pixel_sha(gray[box])
            assert digest == main[c,rec["sample_id"]]["input_pixels_sha256"]
            output=tidy_image(backend,gray[box])
            pred=np.zeros_like(gray)
            pred[box]=output
            m,objs=measure_native(sample,pred,details=True)
            key=dict(sample_id=rec["sample_id"],corruption=c,method="TIDY_published")
            images.append(dict(**key,input_sha256=digest,output_sha256=pixel_sha(output),**{k:m[k] for k in FIELDS}))
            targets.extend(key|obj for obj in objs)
        summary[c]=means([p for p in images if p["corruption"]==c],FIELDS)
        print(c,summary[c],flush=True)
    write_csv(OUT/"nuaa_per_image.csv",images)
    write_csv(OUT/"nuaa_per_target.csv",targets)
    save_report(OUT/"nuaa.json",dict(per_condition=summary,population=213,conditions=3,
                train_manifest=file_hash(ROOT/"manifests/train.jsonl"),local_tidy_updates=0,
                checkpoint=TIDY_HASH,source_commit=REVISION,tensors_loaded=backend.loaded_tensors,
                native_float_inputs_matched=len(images),precision="official PyTorch FP32, TF32 off"))


def scan_frames():
    if file_hash(SCAN) != SCAN_HASH:
        raise ValueError("Unexpected SCaN-TIR archive")
    wanted={f"300_floor5/{side}/{frame}" for side in ("left_clean","right_noisy") for frame in SCAN_IDS}
    received={}
    with tarfile.open(SCAN,"r|gz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            if member.name in received or not member.isfile():
                raise ValueError("Unexpected duplicate or non-file image")
            with Image.open(io.BytesIO(archive.extractfile(member).read())) as im:
                rgb=np.asarray(im.convert("RGB"))
                if im.size != (640,512) or not np.array_equal(rgb[...,0],rgb[...,1]) or not np.array_equal(rgb[...,0],rgb[...,2]):
                    raise ValueError("Expected identical grayscale RGB channels of size 640 × 512")
                received[member.name]=rgb[:,64:576,0].astype(np.float32)/255
    if set(received)!=wanted:
        raise ValueError("Incomplete pilot subset")
    return [(k,received[f"300_floor5/right_noisy/{k}"], received[f"300_floor5/left_clean/{k}"]) for k in SCAN_IDS]


def scan(method):
    frames=scan_frames()
    backend=load_tidy() if method=="tidy" else None
    if method=="s1":
        model,device,sha=get_model("s1")
    rows=[]
    for frame,x,y in frames:
        if method=="tidy":
            pred=tidy_image(backend,x)
        elif method=="s1":
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                pred=hard_prediction(model,x[None],device,greedy=True)
        elif method=="column":
            pred=correct_columns(x,0,0,512,512)
        else:
            pred=x
        d=255*(pred.astype(np.float64)-y)
        rmse=np.sqrt(np.square(d).mean())
        rows.append(dict(frame=frame,method=method,input_sha256=pixel_sha(x),reference_sha256=pixel_sha(y),
                         output_sha256=pixel_sha(pred),mae_dn=np.abs(d).mean(),mean_signed_dn=d.mean(),
                         rmse_dn=rmse,col_residual_dn=d.mean(axis=0).std(),psnr_db=20*np.log10(255/rmse),
                         ssim=structural_similarity(y.astype(np.float64),pred.astype(np.float64),
                              data_range=1,win_size=11,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)))
    write_csv(OUT/f"scan_{method}_frames.csv",rows)
    save_report(OUT/f"scan_{method}.json",dict(means=means(rows,("mae_dn","mean_signed_dn","rmse_dn","col_residual_dn","psnr_db","ssim")),
                frames=len(rows),sequence="300_floor5",archive_sha256=SCAN_HASH,
                checkpoint_sha256=TIDY_HASH if method=="tidy" else sha if method=="s1" else None,
                frames_selected=SCAN_IDS,crop_xywh=[64,0,512,512],input_stream="right_noisy",reference_stream="left_clean"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["nuaa","scan"])
    parser.add_argument("--method",choices=["input","column","tidy","s1"],default="tidy")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.action=="nuaa":
        nuaa()
    else:
        scan(args.method)
