"""Build the eval plan set: skv4 labels (room types, walls, windows) + full-res COCO images.

Inputs (gitignored, downloaded beforehand):
  data/raw/skv4/{test,train}.jsonl          from HF Tristan12138/cubicasa5k-skv4
  data/raw/coco/*.parquet                   from HF phungpx/cubicassa5k-coco
Outputs:
  data/plans/{test,dev}/{plan_id}.png       full-res image
  data/plans/{test,dev}/{plan_id}.json      {plan_id, split, coco_file_name, width, height,
                                             skv4_gt (1024 frame, NOT yet aligned), skv4_roles,
                                             coco_annotations}
  eval/splits.json                          seed + plan ids (safe to commit: ids only)

Alignment of skv4 coords to image pixels is a separate step (see the eval plan).
Usage: .venv/bin/python eval/prepare_data.py
"""
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

SEED = 42
N_TEST = 50
N_DEV = 10
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "plans")
NUM_RE = re.compile(r"_(\d+)_png")


def load_skv4(name):
    rows = [json.loads(l) for l in open(os.path.join(RAW, "skv4", name))]
    return [r for r in rows if r["watertight"]]


def index_coco():
    """plan number -> (parquet path, row index, file_name)."""
    idx = {}
    for path in sorted(glob.glob(os.path.join(RAW, "coco", "*.parquet"))):
        names = pq.read_table(path, columns=["file_name"]).column("file_name").to_pylist()
        for i, fn in enumerate(names):
            m = NUM_RE.search(fn)
            if m:
                idx.setdefault(m.group(1), (path, i, fn))
    return idx


def pick(rows, n, coco_idx, rng):
    rows = rows[:]
    rng.shuffle(rows)
    chosen, missing = [], 0
    for r in rows:
        pid = r["id"].split("/")[1]
        if pid in coco_idx:
            chosen.append(r)
        else:
            missing += 1
        if len(chosen) == n:
            break
    return chosen, missing


def main():
    rng = random.Random(SEED)
    coco_idx = index_coco()
    print(f"indexed {len(coco_idx)} COCO images")

    test, miss_t = pick(load_skv4("test.jsonl"), N_TEST, coco_idx, rng)
    dev, miss_d = pick(load_skv4("train.jsonl"), N_DEV, coco_idx, rng)
    print(f"test={len(test)} (skipped {miss_t} not in COCO), dev={len(dev)} (skipped {miss_d})")

    # group wanted rows by parquet file so each file is read once
    wanted = {}
    for split, rows in (("test", test), ("dev", dev)):
        for r in rows:
            pid = r["id"].split("/")[1]
            path, i, fn = coco_idx[pid]
            wanted.setdefault(path, []).append((i, split, pid, fn, r))

    for path, items in wanted.items():
        t = pq.read_table(path).take([i for i, *_ in items]).to_pylist()
        for row, (_, split, pid, fn, r) in zip(t, items):
            d = os.path.join(OUT, split)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{pid}.png"), "wb") as f:
                f.write(row["image"]["bytes"])
            meta = {
                "plan_id": pid, "split": split, "coco_file_name": fn,
                "width": row["width"], "height": row["height"],
                "skv4_gt": r["gt"], "skv4_roles": r["roles"], "skv4_frame": r["frame"],
                "coco_annotations": row["annotations"],
            }
            with open(os.path.join(d, f"{pid}.json"), "w") as f:
                json.dump(meta, f)
        print(f"wrote {len(items)} plans from {os.path.basename(path)}")

    splits = {
        "seed": SEED,
        "source": {"labels": "Tristan12138/cubicasa5k-skv4 (watertight only)",
                   "images": "phungpx/cubicassa5k-coco"},
        "test": [r["id"].split("/")[1] for r in test],
        "dev": [r["id"].split("/")[1] for r in dev],
    }
    with open(os.path.join(ROOT, "eval", "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    print("wrote eval/splits.json")


if __name__ == "__main__":
    main()
