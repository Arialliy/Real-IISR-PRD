"""Build the fixed local FLIR-IISR split from the published full-data release."""

import hashlib
import json
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT / "datasets/FLIR-IISR"
VIEWS = ("HR", "LR", "LR_4x")
SEED = 20261120
TEST_COUNT = 265


def pixel_hashes(name):
    hashes = {}
    for view in VIEWS:
        with Image.open(DATA / view / name) as image:
            rgb = image.convert("RGB")
            digest = hashlib.sha256(struct.pack(">II", *rgb.size))
            digest.update(rgb.tobytes())
            hashes[view] = digest.hexdigest()
    return name, hashes


def grouped_ids(pixels):
    parent = {key: key for key in pixels}
    owners = {}

    def representative(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for key, views in pixels.items():
        for digest in views.values():
            owner = owners.setdefault(digest, key)
            parent[representative(key)] = representative(owner)
    groups = defaultdict(list)
    for key in sorted(parent):
        groups[representative(key)].append(key)
    return list(groups.values())


def choose_test(groups, labels):
    counts = Counter(labels.values())
    if set(counts) != {0, 1}:
        raise ValueError("Expected released optical and motion classes 0 and 1")
    size = len(labels)
    quota = {k: counts[k] * TEST_COUNT // size for k in counts}
    remainder_order = sorted(counts, key=lambda k: (-(counts[k] * TEST_COUNT % size), k))
    for k in remainder_order[:TEST_COUNT - sum(quota.values())]:
        quota[k] += 1
    target = (quota[0], quota[1])

    # Assign each connected component a deterministic pseudorandom priority.
    def priority(group):
        key = str(SEED) + "\n" + "\n".join(group)
        return hashlib.sha256(key.encode()).digest()

    groups = sorted(groups, key=priority)
    # Each count state remembers the first exact solution in priority order.
    parents = {(0, 0): None}
    for group_index, group in enumerate(groups):
        optical = sum(labels[key] == 0 for key in group)
        motion = len(group) - optical
        for state in tuple(parents):
            after = (state[0] + optical, state[1] + motion)
            if after[0] <= target[0] and after[1] <= target[1] and after not in parents:
                parents[after] = (state, group_index)
        if target in parents:
            break
    if target not in parents:
        raise ValueError("Required stratified counts conflict with duplicate groups")
    chosen = set()
    state = target
    while parents[state] is not None:
        state, group_index = parents[state]
        chosen.update(groups[group_index])
    return chosen, quota


def main():
    degradation_file = DATA / "degradation.json"
    metadata = json.loads(degradation_file.read_text())
    labels = {key: value["degradation_label"] for key, value in metadata["data"].items()}
    ids = sorted(labels)
    if len(ids) != 1457 or set(json.loads((DATA / "scene.json").read_text())["data"]) != set(ids):
        raise ValueError("Released annotations must cover all 1,457 IDs")
    for view in VIEWS:
        if {p.name for p in (DATA / view).iterdir() if p.is_file()} != set(ids):
            raise ValueError(f"Released {view} and annotation IDs differ")
    with ThreadPoolExecutor(max_workers=12) as workers:
        pixels = dict(workers.map(pixel_hashes, ids))
    groups = grouped_ids(pixels)
    test, quota = choose_test(groups, labels)
    train = set(ids) - test
    if len(train) != 1192 or len(test) != 265 or Counter(labels[k] for k in test) != quota:
        raise ValueError("Invalid pair counts or degradation quotas")
    train_pixels = {h for key in train for h in pixels[key].values()}
    test_pixels = {h for key in test for h in pixels[key].values()}
    if train_pixels & test_pixels:
        raise ValueError("An identical decoded input or target crosses the split")
    split_files = {name + ".txt": "".join(key + "\n" for key in sorted(subset)).encode()
                   for name, subset in (("train", train), ("test", test))}
    release = json.loads((OUT.parent / "flir_release_inventory.json").read_text())
    report = {
        "split_source": "Local seeded split authorized for FLIR-IISR; paper's original ID membership unavailable.",
        "archive_url": release["source"], "archive_sha256": release["release_sha256"],
        "seed": SEED, "filenames": "sorted original BMP basenames, UTF-8 with LF",
        "degradation_annotation_sha256": hashlib.sha256(degradation_file.read_bytes()).hexdigest(),
        "degradation_labels": metadata["labels"],
        "all_degradation_counts": dict(Counter(labels.values())),
        "train_degradation_counts": dict(Counter(labels[k] for k in train)),
        "test_degradation_counts": dict(Counter(labels[k] for k in test)),
        "partitions": {"train": len(train), "test": len(test)}, "train_test_id_overlap": len(train & test),
        "pixel_digest": "SHA-256 over width,height as two big-endian uint32, then RGB8 row-major pixels",
        "grouping": "IDs linked by any equal pixel digest across HR, LR, LR_4x are assigned together; includes transitive links.",
        "component_count": len(groups), "component_size_histogram": dict(sorted(Counter(map(len, groups)).items())),
        "non_singleton_components": sorted(g for g in groups if len(g) > 1),
        "identical_pixel_hashes_across_splits": len(train_pixels & test_pixels),
        "scope": "Exact pixel identity was checked. Scene independence and near-duplicate exclusion were not established.",
        "sampling": "Largest remainder allocates class quotas. Rank components by SHA256(seed LF sorted IDs separated by LF). Subset-sum chooses first feasible exact per-class size in this ordering.",
        "manifest_sha256": {name: hashlib.sha256(buf).hexdigest() for name, buf in split_files.items()},
        "reproduce": "python3 paper_reproduction/splits/make_split.py",
    }
    outputs = {**split_files, "provenance.json": (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode()}
    for name, buf in outputs.items():
        dst = OUT / name
        if dst.exists() and dst.read_bytes() != buf:
            raise ValueError(f"Existing {dst.name} differs; retain it for investigation")
    for name, buf in outputs.items():
        dst = OUT / name
        if not dst.exists():
            staging = dst.with_suffix(".tmp")
            staging.write_bytes(buf)
            staging.replace(dst)
    print(json.dumps({k: report[k] for k in ("partitions", "train_degradation_counts", "test_degradation_counts",
                                            "component_count", "component_size_histogram", "manifest_sha256")}))


if __name__ == "__main__":
    main()
