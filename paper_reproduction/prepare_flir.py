"""Fetch the pinned author release and inventory the paired FLIR-IISR images."""
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import requests


PROJECT = Path(__file__).resolve().parents[1]
REVISION = "db77fe45d0b103cfc1f435d550446045041b3043"
URL = f"https://huggingface.co/datasets/yuanzsz/FLIR-IISR/resolve/{REVISION}/FLIR-IISR.zip"
SIZE = 1052929565
CHECKSUM = "bbf265e50344c4151b45f21a4148242d956921e116d5efc10abdcd0c7f8c08d4"
CHUNK = 16 * 1024**2


def fetch_piece(folder, start):
    end = min(start + CHUNK, SIZE) - 1
    dest = folder / f"{start:012}"
    if dest.is_file() and dest.stat().st_size == end - start + 1:
        return dest
    for attempt in range(5):
        try:
            with requests.get(URL, params={"part": start, "attempt": attempt}, stream=True,
                              headers={"Range": f"bytes={start}-{end}"}, timeout=(10, 45)) as response:
                response.raise_for_status()
                if response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                    raise ValueError("Unexpected download range")
                temp = dest.with_suffix(".partial")
                with temp.open("wb") as out:
                    for block in response.iter_content(512 * 1024):
                        out.write(block)
                if temp.stat().st_size != end - start + 1:
                    raise ValueError("Incomplete range")
                temp.replace(dest)
                return dest
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(1 + attempt)


def get_archive():
    cache = PROJECT / ".runtime" / "flir_release"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "FLIR-IISR.zip"
    if not archive.exists():
        with ThreadPoolExecutor(max_workers=20) as pool:
            parts = list(pool.map(lambda start: fetch_piece(cache, start), range(0, SIZE, CHUNK)))
        temp = archive.with_suffix(".assembling")
        with temp.open("wb") as out:
            for part in parts:
                with part.open("rb") as src:
                    shutil.copyfileobj(src, out)
        with temp.open("rb") as src:
            if hashlib.file_digest(src, "sha256").hexdigest() != CHECKSUM:
                raise ValueError("Assembled release checksum mismatch")
        temp.replace(archive)
        for part in parts:
            part.unlink()
    with archive.open("rb") as src:
        if archive.stat().st_size != SIZE or hashlib.file_digest(src, "sha256").hexdigest() != CHECKSUM:
            raise ValueError(f"Invalid cached archive: {archive}")
    return archive


def main():
    archive = get_archive()
    output = PROJECT / "datasets" / "FLIR-IISR"
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            name = PurePosixPath(item.filename)
            if (name.is_absolute() or ".." in name.parts or name.parts[:1] != ("FLIR-IISR",)
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"Unexpected release member {name}")
        bad = z.testzip()
        if bad:
            raise ValueError(f"Release member CRC failure: {bad}")
        if not output.exists():
            with tempfile.TemporaryDirectory(dir=output.parent) as folder:
                z.extractall(folder)
                os.rename(Path(folder) / "FLIR-IISR", output)
        samples = {}
        for name in ("HR", "LR", "LR_4x"):
            samples[name] = sorted(PurePosixPath(x.filename).name for x in z.infolist()
                                   if not x.is_dir() and PurePosixPath(x.filename).parent == PurePosixPath("FLIR-IISR", name))
            if samples[name] != sorted(p.name for p in (output / name).iterdir() if p.is_file()):
                raise ValueError(f"Extracted membership differs for {name}")
        if not (samples["HR"] == samples["LR"] == samples["LR_4x"] and len(samples["HR"]) == 1457):
            raise ValueError("Published image sets differ")
        metadata = {p.name: sorted(json.loads(p.read_text(encoding="utf-8-sig"))) for p in sorted(output.glob("*.json"))
                    if isinstance(json.loads(p.read_text(encoding="utf-8-sig")), dict)}
        summary = {
            "source": URL, "release_sha256": CHECKSUM, "archive_bytes": SIZE,
            "total_image_pairs": len(samples["HR"]), "directories": {k: len(v) for k, v in samples.items()},
            "release_root_files": sorted(p.name for p in output.iterdir() if p.is_file()),
            "metadata_top_level_keys": metadata, "all_image_files": samples["HR"],
            "release_directories": sorted(p.name for p in output.iterdir() if p.is_dir()),
        }
    path = PROJECT / "paper_reproduction" / "flir_release_inventory.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)
    print("Verified FLIR-IISR release:", summary["directories"], summary["release_root_files"], flush=True)


if __name__ == "__main__":
    main()
