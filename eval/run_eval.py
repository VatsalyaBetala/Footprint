"""Footprint eval: system (POST /analyze) vs baseline B1 (one Carbon call -> room polygons).

Ground truth: data/plans/{split}/{id}.gt.json from eval/align.py (COCO room polygons + skv4 types/windows).
Every model output is cached under data/cache/{system}/{split}/{id}.json, so re-scoring is free.

Usage:
  .venv/bin/python eval/run_eval.py --split dev --systems B1,v1            # run (cached) + score
  .venv/bin/python eval/run_eval.py --split test --systems B1,v1 --workers 4
  .venv/bin/python eval/run_eval.py --split dev --systems B1,v1 --score-only
Systems: B1 = baseline; v1 / v2 = whatever pipeline the server at --api is running (label it yourself).
Writes eval/results/{split}_{system}.json (per plan), eval/failures_{split}_{system}.jsonl,
and eval/results.json (Proof-panel format) for the split you ran.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANS = os.path.join(ROOT, "data", "plans")
CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "eval", "results")
load_dotenv(os.path.join(ROOT, ".env"))
load_dotenv(os.path.join(os.path.dirname(ROOT), ".env"))

CARBON = "gemini-4-carbon"
CARBON_PRICE = (2.00, 10.00)  # USD per 1M tokens in/out, same ⚠️ estimate as backend/server.py
BASELINE_PROMPT = """You are reading an apartment floor plan image. Trace every room.

COORDINATES: every point is [x, y] normalized to 0-1000 over the FULL image.
x = distance from the LEFT edge, y = distance from the TOP edge.

Return ONLY JSON:
{"rooms": [{"label": "Bedroom", "polygon": [[x, y], ...], "confidence": 0.0-1.0,
            "has_exterior_window": true or false or null}]}

Rules:
- One entry per room or labeled space (bedrooms, living, kitchen, bath, closets, halls, balconies).
- label = the printed room name; if unlabeled, your best room type.
- polygon = the room's floor outline along the INNER faces of its walls, as many vertices as needed
  (at least 4), in order around the room.
- confidence = how sure you are the polygon is correct.
- has_exterior_window = for bedrooms: is there a window on an exterior wall of this room? null for other rooms.
"""

# room label -> eval type. Short tokens are matched exactly (Finnish plan abbreviations included),
# long keywords as substrings. Applied identically to every system.
ABBR = {"mh": "Bedroom", "h": "Bedroom", "alk": "Bedroom", "oh": "Living", "tupa": "Living", "k": "Kitchen",
        "kk": "Kitchen", "kt": "Kitchen", "tupak": "Kitchen", "kh": "Bath", "ph": "Bath", "wc": "Bath", "s": "Bath",
        "pkh": "Bath", "et": "Hall", "aula": "Hall", "vh": "Storage", "var": "Storage", "tk": "Storage",
        "sk": "Storage", "khh": "Storage", "lvk": "Storage", "vk": "Storage", "pk": "Storage", "ak": "Storage",
        "lp": "Balcony", "parv": "Balcony", "kph": "Bath", "oloh": "Living", "huone": "Bedroom", "ruok": "Living",
        "avokuisti": "Balcony", "psh": "Bath", "lh": "Bath", "e": "Hall", "tuulikaappi": "Hall", "tk": "Storage", "kuisti": "Balcony", "pukuh": "Bath", "ruokailu": "Living", "br": "Bedroom", "bdr": "Bedroom", "lr": "Living", "dr": "Living"}
KEYWORDS = [("bed", "Bedroom"), ("master", "Bedroom"), ("makuu", "Bedroom"), ("living", "Living"),
            ("lounge", "Living"), ("family", "Living"), ("dining", "Living"), ("olohuone", "Living"),
            ("kitchen", "Kitchen"), ("keitti", "Kitchen"), ("bath", "Bath"), ("toilet", "Bath"),
            ("shower", "Bath"), ("sauna", "Bath"), ("powder", "Bath"), ("kylpy", "Bath"), ("pesu", "Bath"),
            ("hall", "Hall"), ("corridor", "Hall"), ("foyer", "Hall"), ("entry", "Hall"), ("eteinen", "Hall"),
            ("closet", "Storage"), ("storage", "Storage"), ("wardrobe", "Storage"), ("utility", "Storage"),
            ("laundry", "Storage"), ("pantry", "Storage"), ("varasto", "Storage"), ("komero", "Storage"),
            ("vaate", "Storage"), ("balcony", "Balcony"), ("terrace", "Balcony"), ("patio", "Balcony"),
            ("parveke", "Balcony"), ("terassi", "Balcony")]
IOU_CORRECT = 0.7   # a room is "right" at IoU >= 0.7 (and the right type, when GT type is known)
IOU_MATCH = 0.1     # below this a GT room counts as missed


def label_type(label):
    s = (label or "").lower()
    for tok in re.findall(r"[a-zäöå]+", s):
        if tok in ABBR:
            return ABBR[tok]
    for kw, t in KEYWORDS:
        if kw in s:
            return t
    return None


def poly(p):
    try:
        g = Polygon(p).buffer(0)
        return g if g.area > 0 else None
    except Exception:
        return None


# ---------------- systems ----------------
def run_system(api, img_path, timeout=180):
    t = time.perf_counter()
    with open(img_path, "rb") as f:
        r = httpx.post(f"{api}/analyze", files={"file": (os.path.basename(img_path), f, "image/png")}, timeout=timeout)
    m = r.json()
    # the server answers with a partial model at its 45 s deadline and keeps working; fetch the full one
    deadline = time.time() + 240
    while m.get("partial") and m.get("model_id") and time.time() < deadline:
        time.sleep(5)
        try:
            full = httpx.get(f"{api}/model/{m['model_id']}", timeout=15).json()
            if not full.get("partial"):
                full["_was_partial"] = True
                m = full
        except Exception:
            pass
    m["_wall_s"] = time.perf_counter() - t
    m["_http"] = r.status_code
    return m


_client = None


def run_baseline(img_path):
    global _client
    from google import genai
    from google.genai import types
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={"api_version": "v1beta"})
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    png = cv2.imencode(".png", img)[1].tobytes()
    out = {"rooms": [], "error": None, "latency_s": None, "cost_usd": None}
    for attempt in (1, 2):
        t = time.perf_counter()
        try:
            resp = _client.models.generate_content(
                model=CARBON, contents=[types.Part.from_bytes(data=png, mime_type="image/png"), BASELINE_PROMPT],
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   http_options=types.HttpOptions(timeout=120_000),
                                                   thinking_config=types.ThinkingConfig(thinking_level="LOW")))
            data = json.loads(resp.text)
            out["latency_s"] = time.perf_counter() - t
            u = resp.usage_metadata
            if u:
                out["cost_usd"] = ((u.prompt_token_count or 0) * CARBON_PRICE[0] +
                                   ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) * CARBON_PRICE[1]) / 1e6
            for r in data.get("rooms", []):
                pts = [[p[0] * W / 1000, p[1] * H / 1000] for p in r.get("polygon") or []
                       if isinstance(p, (list, tuple)) and len(p) == 2]
                out["rooms"].append({"label": r.get("label"), "polygon": pts, "confidence": r.get("confidence"),
                                     "has_exterior_window": r.get("has_exterior_window")})
            out["error"] = None
            return out
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            out["latency_s"] = time.perf_counter() - t
    return out


def cached(system, split, pid, fn):
    path = os.path.join(CACHE, system, split, f"{pid}.json")
    if os.path.exists(path):
        return json.load(open(path))
    res = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"))
    return res


# ---------------- normalise predictions ----------------
def preds_from(system, out):
    """-> list of {label, type, geom, confidence, verified, window}, plus plan-level latency/cost/error."""
    rooms = []
    if system == "B1":
        for r in out.get("rooms", []):
            g = poly(r["polygon"]) if len(r["polygon"]) >= 3 else None
            if g is None:
                continue
            c = r.get("confidence")
            c = float(c) if isinstance(c, (int, float)) else None
            rooms.append({"label": r.get("label"), "type": label_type(r.get("label")), "geom": g, "confidence": c,
                          "verified": c is not None and c >= 0.8, "window": r.get("has_exterior_window"), "flags": []})
        return rooms, out.get("latency_s"), out.get("cost_usd"), out.get("error")
    legal = {b["room"]: b for b in (out.get("answers") or {}).get("legal_bedrooms") or []}
    for r in out.get("rooms") or []:
        g = poly(r["polygon"]) if r.get("polygon") and len(r["polygon"]) >= 3 else None
        if g is None:
            continue
        w = legal.get(r.get("id"), {}).get("checks", {}).get("exterior_window")
        rooms.append({"label": r.get("label"), "type": label_type(r.get("label")), "geom": g,
                      "confidence": r.get("confidence"), "verified": r.get("verified") is True,
                      "window": w if isinstance(w, bool) else None, "flags": r.get("flags") or []})
    t = out.get("timings_ms") or {}
    lat = sum(v for v in t.values() if isinstance(v, (int, float))) / 1000 if t else out.get("_wall_s")
    err = None
    if out.get("_http", 200) != 200:
        err = f"HTTP {out.get('_http')}: {str(out.get('error') or out.get('detail'))[:150]}"
    elif out.get("partial"):
        err = "partial (deadline hit)"
    elif out.get("errors"):
        err = "; ".join(f"{e.get('stage')}: {str(e.get('error'))[:80]}" for e in out["errors"])
    return rooms, lat, out.get("cost_usd"), err


# ---------------- scoring ----------------
def score_plan(gt, preds):
    G = [{"type": r["type"], "geom": poly(r["polygon"]), "window": r["has_exterior_window"]} for r in gt["rooms"]]
    G = [g for g in G if g["geom"] is not None]
    n, m = len(G), len(preds)
    iou = np.zeros((n, m))
    for i, g in enumerate(G):
        for j, p in enumerate(preds):
            inter = g["geom"].intersection(p["geom"]).area
            if inter > 0:
                iou[i, j] = inter / g["geom"].union(p["geom"]).area
    pairs = {}
    if n and m:
        for i, j in zip(*linear_sum_assignment(-iou)):
            if iou[i, j] >= IOU_MATCH:
                pairs[i] = j
    inv = {j: i for i, j in pairs.items()}
    s = {"n_gt": n, "n_pred": m, "room_iou": float(sum(iou[i, j] for i, j in pairs.items()) / n) if n else None,
         "count_exact": n == m, "count_err": abs(n - m), "type_ok": 0, "type_n": 0, "win_ok": 0, "win_n": 0,
         "win_answered": 0, "rooms": [], "failures": []}
    for i, g in enumerate(G):
        if i not in pairs:
            s["failures"].append({"category": "missed_room", "detail": f"GT {g['type'] or 'room'} not traced"})
            continue
        p = preds[pairs[i]]
        if g["type"] and iou[i, pairs[i]] >= 0.5:
            s["type_n"] += 1
            if p["type"] == g["type"]:
                s["type_ok"] += 1
            else:
                s["failures"].append({"category": "wrong_type", "detail": f"GT {g['type']} labelled '{p['label']}' ({p['type']})"})
        if g["type"] == "Bedroom" and g["window"] is not None and iou[i, pairs[i]] >= 0.5:
            s["win_n"] += 1
            if p["window"] is not None:
                s["win_answered"] += 1
                s["win_ok"] += int(p["window"] == g["window"])
        ratio = p["geom"].area / g["geom"].area
        if iou[i, pairs[i]] < 0.5:
            cat = "leak_oversize" if ratio > 1.3 else "undersize" if ratio < 0.7 else "misplaced"
            s["failures"].append({"category": cat, "detail": f"{g['type'] or 'room'}: IoU {iou[i, pairs[i]]:.2f}, area x{ratio:.2f}"})
    for j, p in enumerate(preds):
        covered = [i for i in range(n) if G[i]["geom"].intersection(p["geom"]).area >= 0.25 * G[i]["geom"].area]
        if len(covered) >= 2:
            s["failures"].append({"category": "open_plan_merge", "detail": f"'{p['label']}' spans {len(covered)} GT rooms"})
        if j not in inv:
            s["failures"].append({"category": "extra_room", "detail": f"'{p['label']}' matches no GT room"})
        i = inv.get(j)
        right = i is not None and iou[i, j] >= IOU_CORRECT and (G[i]["type"] is None or p["type"] == G[i]["type"])
        s["rooms"].append({"confidence": p["confidence"], "verified": p["verified"], "correct": bool(right),
                           "iou": float(iou[i, j]) if i is not None else 0.0, "flags": p["flags"]})
    return s


def aggregate(plans):
    ok = [p for p in plans if p["score"]["room_iou"] is not None]
    rooms = [r for p in ok for r in p["score"]["rooms"]]
    ver = [r for r in rooms if r["verified"]]
    tn, wn = sum(p["score"]["type_n"] for p in ok), sum(p["score"]["win_n"] for p in ok)
    wa = sum(p["score"]["win_answered"] for p in ok)
    lat = [p["latency_s"] for p in plans if p["latency_s"] is not None]
    cost = [p["cost_usd"] for p in plans if p["cost_usd"] is not None]
    conf = sorted([r for r in rooms if r["confidence"] is not None], key=lambda r: -r["confidence"])
    curve = []
    for c in (0.2, 0.4, 0.6, 0.8, 1.0):
        top = conf[:max(1, round(c * len(conf)))] if conf else []
        if top:
            curve.append([c, round(sum(r["correct"] for r in top) / len(top), 3)])
    return {
        "n_plans": len(plans), "n_errors": sum(p["error"] is not None for p in plans),
        "room_iou": statistics.mean(p["score"]["room_iou"] for p in ok) if ok else None,
        "count_acc": statistics.mean(p["score"]["count_exact"] for p in ok) if ok else None,
        "count_mae": statistics.mean(p["score"]["count_err"] for p in ok) if ok else None,
        "type_acc": sum(p["score"]["type_ok"] for p in ok) / tn if tn else None,
        "window_acc": sum(p["score"]["win_ok"] for p in ok) / wa if wa else None,
        "window_coverage": wa / wn if wn else None,
        "verified_precision": sum(r["correct"] for r in ver) / len(ver) if ver else None,
        "verified_share": len(ver) / len(rooms) if rooms else None,
        "room_precision_all": sum(r["correct"] for r in rooms) / len(rooms) if rooms else None,
        "latency_p50_s": statistics.median(lat) if lat else None,
        "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
        "cost_usd": statistics.mean(cost) if cost else None,
        "coverage_curve": curve,
    }


def failure_summary(plans):
    counts, examples = {}, {}
    for p in plans:
        if p["error"]:
            cat = "timeout" if "partial" in p["error"] or "Timeout" in p["error"] else "pipeline_error"
            counts[cat] = counts.get(cat, 0) + 1
            examples.setdefault(cat, f"{p['plan_id']}: {p['error'][:120]}")
        for f in p["score"]["failures"]:
            counts[f["category"]] = counts.get(f["category"], 0) + 1
            examples.setdefault(f["category"], f"{p['plan_id']}: {f['detail']}")
    return [{"category": c, "count": n, "example": examples[c]} for c, n in sorted(counts.items(), key=lambda x: -x[1])]


METRICS = [("room_iou", "Room shape IoU", "ratio", True), ("count_acc", "Room count exact", "pct", True),
           ("count_mae", "Room count error", "num", False), ("type_acc", "Room type accuracy", "pct", True),
           ("window_acc", "Bedroom window check (when answered)", "pct", True),
           ("window_coverage", "Bedroom window answered", "pct", True),
           ("verified_precision", "Right when VERIFIED", "pct", True),
           ("room_precision_all", "Right, all rooms", "pct", True),
           ("latency_p50_s", "Latency p50", "sec", False), ("cost_usd", "Cost per plan", "usd", False)]
SYSTEM_DESC = {"B1": ("Baseline B1", "Single Carbon call -> room polygons"),
               "v1": ("Footprint v1", "Carbon + OpenCV + Decisions verify + Neon repair"),
               "v2": ("Footprint v2", "v1 + fixes for the top DEV failure categories")}


VARIANTS = ("phone_photo", "rotated", "lowres")


def variant_rows():
    """By-input-type rows from eval/results/{variant}_{system}.json (messy copies of TEST plans)."""
    rows = []
    for v in VARIANTS:
        row = {"variant": v, "n": 0}
        for path in glob.glob(os.path.join(OUT, f"{v}_*.json")):
            system = os.path.basename(path)[len(v) + 1:-5]
            a = aggregate(json.load(open(path)))
            row[system] = {"room_iou": a["room_iou"], "verified_precision": a["verified_precision"],
                           "count_acc": a["count_acc"], "n_errors": a["n_errors"]}
            row["n"] = max(row["n"], a["n_plans"])
        if row["n"]:
            rows.append(row)
    return rows


def write_results(split, agg, fails, n, seed):
    systems = [s for s in ("B0", "B1", "B2", "v1", "v2") if s in agg]
    best = "v2" if "v2" in agg else "v1" if "v1" in agg else systems[-1]
    res = {"placeholder": False, "split": split, "seed": seed,
           "dataset": "CubiCasa5K test plans (skv4 types/windows + phungpx COCO room polygons)",
           "n_test": n, "models": {"carbon": CARBON}, "iou_headline": "raw (GT = COCO inner-edge room polygons)",
           "systems": {s: {"name": SYSTEM_DESC.get(s, (s,))[0], "desc": SYSTEM_DESC.get(s, ("", ""))[1]} for s in systems},
           "headline": {"verified_precision": {s: agg[s]["verified_precision"] for s in systems}},
           "metrics": [{"key": k, "label": l, "fmt": f, "higher_is_better": h, **{s: agg[s][k] for s in systems}}
                       for k, l, f, h in METRICS],
           "by_variant": [{"variant": "clean", "n": n, **{s: {"room_iou": agg[s]["room_iou"],
                                                            "verified_precision": agg[s]["verified_precision"]} for s in systems}}]
                         + (variant_rows() if split == "test" else []),
           "coverage_curve": {s: agg[s]["coverage_curve"] for s in systems},
           "failures": fails.get(best, [])[:8], "failures_baseline": fails.get("B1", [])[:8],
           "detail": agg, "generated_at": time.strftime("%Y-%m-%d %H:%M")}
    if split != "test":
        res["dataset"] = res["dataset"].replace("test plans", f"{split.upper()} plans")
    path = os.path.join(ROOT, "eval", "results.json" if split == "test" else f"results_{split}.json")
    json.dump(res, open(path, "w"), indent=1)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")  # dev | test | phone_photo | rotated | lowres
    ap.add_argument("--systems", default="B1,v1")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    splits = json.load(open(os.path.join(ROOT, "eval", "splits.json")))
    ids = splits[args.split][: args.limit or None]
    os.makedirs(OUT, exist_ok=True)
    agg, fails = {}, {}
    for system in args.systems.split(","):
        def one(pid):
            img = os.path.join(PLANS, args.split, f"{pid}.png")
            cpath = os.path.join(CACHE, system, args.split, f"{pid}.json")
            if args.score_only and not os.path.exists(cpath):
                return None
            try:
                out = cached(system, args.split, pid, (lambda: run_baseline(img)) if system == "B1" else (lambda: run_system(args.api, img)))
            except Exception as e:
                out = {"_http": 0, "error": f"{type(e).__name__}: {str(e)[:150]}", "rooms": []}
            preds, lat, cost, err = preds_from(system, out)
            gt = json.load(open(os.path.join(PLANS, args.split, f"{pid}.gt.json")))
            s = score_plan(gt, preds)
            print(f"  {system} {pid:>6}  iou {s['room_iou'] or 0:.2f}  rooms {s['n_pred']:2d}/{s['n_gt']:2d}  "
                  f"lat {lat or 0:5.1f}s  {'ERR ' + err[:60] if err else ''}", flush=True)
            return {"plan_id": pid, "latency_s": lat, "cost_usd": cost, "error": err, "score": s}
        t0 = time.time()
        print(f"[{system}] {len(ids)} {args.split} plans")
        with ThreadPoolExecutor(args.workers) as ex:
            plans = [p for p in ex.map(one, ids) if p]
        json.dump(plans, open(os.path.join(OUT, f"{args.split}_{system}.json"), "w"))
        with open(os.path.join(ROOT, "eval", f"failures_{args.split}_{system}.jsonl"), "w") as f:
            for p in plans:
                for x in p["score"]["failures"]:
                    f.write(json.dumps({"plan_id": p["plan_id"], **x}) + "\n")
        agg[system], fails[system] = aggregate(plans), failure_summary(plans)
        a = agg[system]
        print(f"[{system}] done in {time.time() - t0:.0f}s: IoU {a['room_iou']}, count {a['count_acc']}, type {a['type_acc']}, "
              f"verified-precision {a['verified_precision']} ({a['verified_share']} of rooms verified), errors {a['n_errors']}")
    # merge with previously scored systems for this split so results.json has every column
    for path in glob.glob(os.path.join(OUT, f"{args.split}_*.json")):
        system = os.path.basename(path)[len(args.split) + 1:-5]
        if system not in agg:
            plans = json.load(open(path))
            agg[system], fails[system] = aggregate(plans), failure_summary(plans)
    print("wrote", write_results(args.split, agg, fails, len(ids), splits["seed"]))
    if args.split in VARIANTS and glob.glob(os.path.join(OUT, "test_*.json")):
        tagg, tf = {}, {}
        for path in glob.glob(os.path.join(OUT, "test_*.json")):
            system = os.path.basename(path)[5:-5]
            plans = json.load(open(path))
            tagg[system], tf[system] = aggregate(plans), failure_summary(plans)
        print("rebuilt", write_results("test", tagg, tf, len(splits["test"]), splits["seed"]))
    print(json.dumps({s: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items() if k != "coverage_curve"}
                      for s, a in agg.items()}, indent=1))
    for s, f in fails.items():
        print(f"failures {s}:", [(x["category"], x["count"]) for x in f[:8]])


if __name__ == "__main__":
    sys.exit(main())
