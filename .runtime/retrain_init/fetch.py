from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
import hashlib
import os
import requests
import time
import random

root = Path(__file__).resolve().parents[2]
revision = "a0e478f5e1640a7e048f1303781b6d34ff3296ca"
files = {
    "VQVAE.pth": (1400076874, "b4b60aaa47e7cff7fdebb479521138efbbd968faa893fe0d4f8c703fd49ddf45"),
    "VARSR.pth": (13702361426, "fd7ebfa6e1d69abdbe5097957c647cbac840cf225355fded2847b6a4d67914db"),
}
bs = 16 * 1024**2
start_time = time.monotonic()
lock = Lock()

def download(file, start):
    size = files[file][0]
    dest = root / ".runtime/retrain_init" / file / str(start)
    end = min(start + bs, size) - 1
    want = end - start + 1
    if dest.exists() and dest.stat().st_size == want:
        return file, want
    for attempt in range(8):
        host = "huggingface.co" if attempt % 2 == 0 else "hf-mirror.com"
        url = f"https://{host}/qyp2000/VARSR/resolve/{revision}/{file}"
        try:
            params={"download":"true","part":start}
            if attempt: params['retry']=attempt
            with requests.get(url,headers={"Range":f"bytes={start}-{end}"},params=params,timeout=(15,40),stream=True) as response:
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                    raise ValueError(f"HTTP {response.status_code} for {start} with range {response.headers.get('Content-Range')}")
                partial = dest.with_suffix(".downloading")
                with partial.open('wb') as f:
                    for chunk in response.iter_content(512 * 1024):
                        f.write(chunk)
                if partial.stat().st_size!=want:
                    raise ValueError(f"unexpected bytes {partial.stat().st_size}")
                partial.replace(dest)
            return file, want
        except Exception as e:
            print(file,start,attempt,type(e).__name__,str(e).split('?')[0][:200],flush=True)
            if attempt==7: raise
            time.sleep(0.5+random.random()*2)

def assemble(file):
    size,checksum = files[file]
    dest = root / "checkpoints" / file
    partial = dest.with_suffix(".pth.assembling")
    sha = hashlib.sha256()
    with partial.open('wb') as out:
        for start in range(0,size,bs):
            part = root / ".runtime/retrain_init" / file / str(start)
            with part.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2), b''):
                    sha.update(block)
                    out.write(block)
    assert sha.hexdigest()==checksum and partial.stat().st_size==size
    partial.replace(dest)
    for p in (root / ".runtime/retrain_init" / file).iterdir():
        if p.name.isdigit(): p.unlink()
    print("CHECKSUM_VERIFIED",file,checksum,flush=True)

if __name__=='__main__':
    jobs=[]; totals={}
    for file,(size,sha) in files.items():
        dest=root/'checkpoints'/file
        if dest.exists(): 
            assert hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()==sha
            continue
        (root/'.runtime/retrain_init'/file).mkdir(exist_ok=True)
        totals[file]=0
        jobs.extend((file,start) for start in range(0,size,bs))
    with ThreadPoolExecutor(max_workers=32) as pool, ThreadPoolExecutor(max_workers=2) as assemblers:
        reads=[pool.submit(download,file,start) for file,start in jobs]
        joins=[]
        for i,f in enumerate(as_completed(reads),1):
            file,n=f.result()
            totals[file]+=n
            if i%8==0 or totals[file]==files[file][0]:
                print("complete_MB", {k:v//1048576 for k,v in totals.items()}, "seconds",int(time.monotonic()-start_time),flush=True)
            if totals[file]==files[file][0]:
                joins.append(assemblers.submit(assemble,file))
        for j in joins: j.result()
