"""Messy copies of TEST plans with the ground truth transformed identically.

Variants (fixed seed): phone_photo (perspective warp + uneven light + blur + JPEG 70),
rotated (90° clockwise), lowres (600 px long side + JPEG 30).
Writes data/plans/{variant}/{id}.png + {id}.gt.json and adds the ids to eval/splits.json under the variant name.
Usage: .venv/bin/python eval/make_messy.py [--n 15]
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANS = os.path.join(ROOT, "data", "plans")
SEED = 7


def apply_pts(M, pts):
    a = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(a, M).reshape(-1, 2).tolist()


def transform_gt(gt, M, w, h):
    g = json.loads(json.dumps(gt))
    g["image"] = {"w": w, "h": h}
    for r in g["rooms"]:
        r["polygon"] = apply_pts(M, r["polygon"])
    for key in ("windows", "doors"):
        for o in g[key]:
            o["center"] = apply_pts(M, [o["center"]])[0]
    return g


def phone_photo(img, rng):
    h, w = img.shape[:2]
    j = lambda s: rng.uniform(0.03, 0.08) * s
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[j(w), j(h)], [w - j(w), j(h) * 0.5], [w - j(w) * 0.5, h - j(h)], [j(w) * 0.5, h - j(h) * 0.7]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h), borderValue=(205, 200, 190))
    gx = np.linspace(0.75, 1.05, w)[None, :, None] * np.linspace(1.05, 0.85, h)[:, None, None]  # uneven light
    out = np.clip(out.astype(np.float32) * gx, 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), 1.0)
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])[1], cv2.IMREAD_COLOR)
    return out, M


def rotated(img, rng):
    h, w = img.shape[:2]
    M = np.float64([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]])  # 90° clockwise: (x, y) -> (h-1-y, x)
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), M


def lowres(img, rng):
    h, w = img.shape[:2]
    s = 600 / max(h, w)
    out = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], cv2.IMREAD_COLOR)
    return out, np.float64([[s, 0, 0], [0, s, 0], [0, 0, 1]])


VARIANTS = {"phone_photo": phone_photo, "rotated": rotated, "lowres": lowres}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()
    sp_path = os.path.join(ROOT, "eval", "splits.json")
    splits = json.load(open(sp_path))
    ids = splits["test"][: args.n]
    for name, fn in VARIANTS.items():
        rng = random.Random(SEED)
        d = os.path.join(PLANS, name)
        os.makedirs(d, exist_ok=True)
        for pid in ids:
            img = cv2.imread(os.path.join(PLANS, "test", f"{pid}.png"))
            gt = json.load(open(os.path.join(PLANS, "test", f"{pid}.gt.json")))
            out, M = fn(img, rng)
            cv2.imwrite(os.path.join(d, f"{pid}.png"), out)
            json.dump(transform_gt(gt, M, out.shape[1], out.shape[0]), open(os.path.join(d, f"{pid}.gt.json"), "w"))
        splits[name] = ids
        print(f"{name}: {len(ids)} plans")
    splits["messy_note"] = f"first {args.n} TEST plans, seed {SEED}; GT transformed with the same matrix"
    json.dump(splits, open(sp_path, "w"), indent=2)


if __name__ == "__main__":
    main()
