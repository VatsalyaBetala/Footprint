"""Build eval ground truth per plan: COCO room polygons (geometry) + skv4 room types / exterior windows.

For each plan in data/plans/{test,dev}/:
  1. fit skv4 (1024 frame) -> image px: uniform scale s (grid) + offset (FFT cross-correlation of skv4 wall
     centrelines against the COCO wall mask). wall_overlap = share of centreline pixels on a COCO wall.
  2. geometry GT = COCO room annotations (cats 1/4/5, area > 0.2% of image): already in image px and traced
     along inner wall edges, the same convention as our tracer, so IoU/count don't depend on step 1.
  3. types + has_exterior_window from the matching skv4 room (IoU >= 0.5), only when the fit is ok;
     else type from the COCO class (bathroom/kitchen) or null (excluded from type/window metrics).
Writes data/plans/{split}/{id}.gt.json and overlays for the first N test plans to data/overlays/.
Usage: .venv/bin/python eval/align.py [--overlays 3]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANS = os.path.join(ROOT, "data", "plans")
OVERLAYS = os.path.join(ROOT, "data", "overlays")
COCO_ROOM_CATS = {1: "Bath", 4: "Kitchen", 5: None}
TYPE_MAP = {"bedroom": "Bedroom", "living": "Living", "kitchen": "Kitchen", "bath": "Bath",
            "corridor": "Hall", "foyer": "Hall", "wic": "Storage", "store": "Storage",
            "utility": "Storage", "balcony": "Balcony"}  # "undefined" -> dropped
MIN_WALL_OVERLAP = 0.6  # fraction of skv4 wall centreline pixels that must land on a COCO wall
COLORS = {"Bedroom": (60, 60, 220), "Living": (60, 170, 60), "Kitchen": (0, 160, 230), "Bath": (220, 140, 40),
          "Hall": (150, 150, 150), "Storage": (170, 90, 170), "Balcony": (120, 200, 200)}


def bbox(poly):
    a = np.asarray(poly, float)
    return [a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()]


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def tf(pt, f):
    return [f["sx"] * pt[0] + f["tx"], f["sy"] * pt[1] + f["ty"]]


def coco_rooms(meta):
    a, out = meta["coco_annotations"], []
    min_area = 0.002 * meta["width"] * meta["height"]
    for cat, seg, area in zip(a["category_id"], a["segmentation"], a["area"]):
        if cat in COCO_ROOM_CATS and area > min_area and seg and len(seg[0]) >= 6:
            poly = np.asarray(seg[0], float).reshape(-1, 2).tolist()
            out.append({"cat": cat, "polygon": poly, "bbox": bbox(poly)})
    return out


def match(sk_boxes, co_boxes, min_iou=0.3):
    if not sk_boxes or not co_boxes:
        return []
    cost = np.array([[-box_iou(s, c) for c in co_boxes] for s in sk_boxes])
    rows, cols = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(rows, cols) if -cost[r, c] >= min_iou]


def wall_mask(meta, ds):
    """COCO wall annotations (cat 7) rasterised at 1/ds resolution."""
    a, W, H = meta["coco_annotations"], meta["width"], meta["height"]
    m = np.zeros((H // ds + 1, W // ds + 1), np.float32)
    for cat, seg in zip(a["category_id"], a["segmentation"]):
        if cat == 7:
            for s in seg:
                if len(s) >= 6:
                    cv2.fillPoly(m, [np.int32(np.asarray(s).reshape(-1, 2) / ds)], 1.0)
    return m


def render_walls(walls, s, ds, shape):
    """skv4 wall centrelines at scale s (no offset), 1/ds resolution, on a canvas padded by `shape`."""
    img = np.zeros(shape, np.float32)
    for w in walls:
        a = np.array(w["start"]) * s / ds
        b = np.array(w["end"]) * s / ds
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), 1.0, 1)
    return img


def fit_plan(meta, ds=4):
    """Uniform scale s + offset (tx, ty): grid over s, FFT cross-correlation of skv4 wall centrelines
    against the COCO wall mask for the offset. Score = fraction of centreline pixels landing on a COCO wall."""
    from scipy.signal import fftconvolve
    walls = meta["skv4_gt"]["walls"]
    k = max(meta["width"], meta["height"]) / meta["skv4_frame"]
    mask = cv2.dilate(wall_mask(meta, ds), np.ones((3, 3), np.uint8))
    best = None
    for s in k * np.arange(0.70, 1.32, 0.01):
        H, W = int(meta["skv4_frame"] * s / ds) + 2, int(meta["skv4_frame"] * s / ds) + 2
        lines = render_walls(walls, s, ds, (H, W))
        total = lines.sum()
        if total == 0:
            continue
        corr = fftconvolve(mask, lines[::-1, ::-1], mode="full")  # corr[dy + H - 1, dx + W - 1] = overlap at offset (dx, dy)
        iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
        score = corr[iy, ix] / total
        if best is None or score > best[0]:
            best = (score, s, (ix - (W - 1)) * ds, (iy - (H - 1)) * ds)
    if best is None:
        return {"sx": k, "sy": k, "tx": 0.0, "ty": 0.0, "wall_overlap": 0.0, "ok": False}
    score, s, tx, ty = best
    return {"sx": float(s), "sy": float(s), "tx": float(tx), "ty": float(ty), "wall_overlap": float(score),
            "ok": bool(score >= MIN_WALL_OVERLAP and len(walls) >= 8)}


def poly_iou(a, b):
    try:
        pa, pb = Polygon(a).buffer(0), Polygon(b).buffer(0)
        u = pa.union(pb).area
        return pa.intersection(pb).area / u if u > 0 else 0.0
    except Exception:
        return 0.0


def build_gt(meta):
    """Geometry GT = COCO room polygons (already in image px, inner wall edges, same convention as our tracer).
    Types + exterior windows come from skv4, only when the wall fit is good and the skv4 room matches the COCO
    room (IoU >= 0.5 against the skv4 polygon shrunk to inner edges); otherwise type comes from the COCO class
    (bathroom / kitchen) or stays null (excluded from type metrics)."""
    f = fit_plan(meta)
    walls = {w["id"]: w for w in meta["skv4_gt"]["walls"]}
    t_px = float(np.median([w["thickness"] for w in walls.values()])) * f["sx"] if walls else 0.0
    sk = []
    if f["ok"]:
        for r in meta["skv4_gt"]["rooms"]:
            if not r.get("polygon") or len(r["polygon"]) < 3:
                continue
            poly = [tf(p, f) for p in r["polygon"]]
            inner = Polygon(poly).buffer(-t_px / 2, join_style=2)
            ext = any(o["type"] == "window" for wid in r.get("walls", []) if walls.get(wid, {}).get("type") == "exterior"
                      for o in walls[wid].get("openings", []))
            sk.append({"type": TYPE_MAP.get(r["label"].lower()), "label": r["label"], "inner": inner, "ext": ext})
    rooms = []
    co = coco_rooms(meta)
    cost = np.array([[-poly_iou(c["polygon"], list(s["inner"].exterior.coords)) if s["inner"].geom_type == "Polygon" else 0
                      for s in sk] for c in co]) if co and sk else np.zeros((len(co), 0))
    pairs = {}
    if cost.size:
        for i, j in zip(*linear_sum_assignment(cost)):
            if -cost[i, j] >= 0.5:
                pairs[i] = j
    for i, c in enumerate(co):
        s = sk[pairs[i]] if i in pairs else None
        # COCO's bathroom class beats skv4; skv4 "corridor"/"foyer" is dropped: on TEST it was attached to
        # saunas, washrooms, closets and bedrooms (87 of 92 type "errors" were GT Hall), so it's not reliable GT.
        sk_type = s["type"] if s and s["type"] and s["type"] != "Hall" else None
        typ = "Bath" if c["cat"] == 1 else sk_type or COCO_ROOM_CATS[c["cat"]]
        rooms.append({"type": typ, "type_source": "coco" if c["cat"] == 1 or (typ and not sk_type) else ("skv4" if typ else None),
                      "polygon": [list(map(float, p)) for p in c["polygon"]],
                      "has_exterior_window": s["ext"] if s else None})
    windows, doors = [], []
    if f["ok"]:
        for w in walls.values():
            for o in w.get("openings", []):
                item = {"center": tf(o["center"], f), "width": o["width"] * f["sx"]}
                if o["type"] == "window":
                    windows.append({**item, "exterior": w["type"] == "exterior"})
                elif o["type"] == "door":
                    doors.append(item)
    f.update(n_coco_rooms=len(co), n_skv4_rooms=len(meta["skv4_gt"]["rooms"]), n_typed_from_skv4=len(pairs))
    return {"plan_id": meta["plan_id"], "split": meta["split"], "image": {"w": meta["width"], "h": meta["height"]},
            "wall_thickness_px": t_px, "rooms": rooms, "windows": windows, "doors": doors, "fit": f}


def draw_overlay(gt, meta, img_path, out_path):
    """GT rooms filled by type (grey = untyped), skv4 walls in red at the fitted transform, windows (blue = exterior)."""
    img = cv2.imread(img_path)
    over = img.copy()
    for r in gt["rooms"]:
        cv2.fillPoly(over, [np.int32(r["polygon"])], COLORS.get(r["type"], (200, 200, 200)))
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    f = gt["fit"]
    for w in meta["skv4_gt"]["walls"]:
        cv2.line(img, tuple(int(v) for v in tf(w["start"], f)), tuple(int(v) for v in tf(w["end"], f)), (0, 0, 255), 2)
    for r in gt["rooms"]:
        x, y = np.mean(np.asarray(r["polygon"]), axis=0).astype(int)
        lab = (r["type"] or "?") + (" +win" if r["type"] == "Bedroom" and r["has_exterior_window"] else "")
        cv2.putText(img, lab, (int(x) - 30, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    for w in gt["windows"]:
        cv2.circle(img, tuple(int(v) for v in w["center"]), 7, (255, 120, 0) if w["exterior"] else (160, 160, 160), -1)
    cv2.putText(img, f"wall overlap {f['wall_overlap']:.2f} ok={f['ok']} typed {f['n_typed_from_skv4']}/{f['n_coco_rooms']}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OVERLAYS, exist_ok=True)
    bad, typed, total, bedrooms = [], 0, 0, 0
    for split in ("test", "dev"):
        files = sorted(f for f in glob.glob(os.path.join(PLANS, split, "*.json")) if not f.endswith(".gt.json"))
        for n, path in enumerate(files):
            meta = json.load(open(path))
            gt = build_gt(meta)
            json.dump(gt, open(path.replace(".json", ".gt.json"), "w"))
            f = gt["fit"]
            if not f["ok"]:
                bad.append(f"{split}/{meta['plan_id']}")
            t = sum(r["type"] is not None for r in gt["rooms"])
            typed += t; total += len(gt["rooms"])
            bedrooms += sum(r["type"] == "Bedroom" and r["has_exterior_window"] is not None for r in gt["rooms"])
            k = max(meta["width"], meta["height"]) / meta["skv4_frame"]
            print(f"{split}/{meta['plan_id']:>6}  overlap {f['wall_overlap']:.2f}  s/k {f['sx'] / k:.3f}  t ({f['tx']:5.0f},{f['ty']:5.0f})  "
                  f"rooms {len(gt['rooms']):2d}  typed {t:2d}  {'ok' if f['ok'] else 'BAD FIT (types from COCO class only)'}")
            if split == "test" and n < args.overlays:
                draw_overlay(gt, meta, path.replace(".json", ".png"), os.path.join(OVERLAYS, f"{meta['plan_id']}.png"))
    print(f"\nwall fit ok {60 - len(bad)}/60; bad: {bad}")
    print(f"rooms typed {typed}/{total}; bedrooms with window GT {bedrooms}")


if __name__ == "__main__":
    main()
