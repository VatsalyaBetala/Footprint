"""verify.py — Decisions API checks every room outline against the plan; Neon ranks repairs from a fixed menu.

verify_and_repair(img, parsed, geo) -> {"rooms", "alignment_score_history", "decision_checks", "windows",
                                       "repairs", "neon", "usage", "cost_usd", "timing_ms"}
Loop: check all rooms (iteration 0) -> Neon ranks the repair menu once per failing room -> each round applies
every failing room's next-ranked repair, re-checks only the changed rooms, and KEEPS a repair only if the
area-weighted alignment score rises (or ties with higher confidence); otherwise it is rolled back. So the
score history never goes down. Models never output coordinates: Decisions returns probabilities, Neon returns
option names, every repair is applied deterministically here (or by re-running geometry.py).
"usage" is keyed by model name ({"input_tokens", "output_tokens", "thought_tokens", "calls"}) for server
pricing; cost_usd is None here because prices live in server.py.
CLI: python backend/verify.py <plan> [...]   (reuses scratch/parse_<name>.json if present)
"""
import base64
import json
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from shapely.geometry import MultiPolygon, Polygon, box as shapely_box
from shapely.ops import unary_union

import geometry

CONTEXT = 'Crop of an apartment floor plan. A candidate outline for the room labeled "{label}" is drawn in thick red.'
Q_ONE_ROOM = ("The red outline covers most of exactly ONE room: not just a thin strip of a room, "
              "and not two or more rooms.")
Q_LABEL = ('The room\'s label text "{label}" is inside the red outline. If the plan prints no label for this room, '
           'answer whether the feature it names (for example stairs) is inside the outline.')
Q_WALLS = ("The red outline follows this room's own walls and does not cut across any wall into a neighboring "
           "room or outside the apartment. An outline drawn on top of the room's own walls counts as following them.")
Q_ALIGN = "How well does the red outline align with the walls of the room it outlines?"
ALIGN_LEVELS = ["0 - unrelated to the room's walls", "1 - right area, but large parts misaligned",
                "2 - mostly follows the walls, small deviations", "3 - traces the room's walls closely"]

WINDOW_CONTEXT = "Crop of an apartment floor plan. One window is marked with a thick red circle."
Q_EXTERIOR = ("The window marked by the red circle is on the OUTER boundary wall of the floor plan, the wall that "
              "separates the apartment from the outside (the blank area or open space beyond the plan), not on an "
              "interior wall between two rooms.")

NEON_PROMPT = """You repair one room outline in a floor-plan digitizer. The image is a crop of the floor plan
with the current outline for the room "{label}" drawn in red{partner_note}.
Verification failed on: {failed}.
How the outline was made: {method}. Flags: {flags}.
Repairs available:
{options}
Rank ALL of these repairs from best to worst for this outline. Repairs are tried in your order and each one is
kept only if verification improves; anything ranked below "keep" is never tried.
Return ONLY JSON: {{"ranking": ["<option name>", ...]}}"""

REPAIR_HELP = {
    "use_box": "replace the outline with the room's detected bounding box (when the trace leaked or was cut short)",
    "reseed": "re-trace the walls from a new start point in the most open part of the room (when the trace got stuck in a pocket)",
    "split": "this outline overlaps room {other} (blue); cut the shared area along the line halfway between the two rooms (when two outlines double-count floor)",
    "merge": "join this outline with neighbouring room {other} into one room (only if they are truly one open space with no wall between)",
    "keep": "leave the outline as it is (when it is actually correct or no option helps)",
}

DECISIONS = "models/gemini-decision-1"
NEON = "gemini-4-neon"
NEON_THINKING = "LOW"     # lowest level Neon accepts (MINIMAL -> 400); auto-dropped if the API rejects it
TIMEOUT_S = 15
MAX_WORKERS = 8
MAX_ITERS = 3             # repair rounds after iteration 0 (Neon is asked once; rounds walk its ranking)
BUDGET_S = 20             # no new repair round starts after this much verify time
PASS_NOUL, PASS_SCORE = 0.7, 1.5
PAD_FT, WINDOW_PAD_FT = 1.5, 5.0
EXTERIOR_YES, EXTERIOR_NO = 0.7, 0.3
NOUL_KEYS = ("one_room", "label_inside", "within_walls")

log = logging.getLogger("footprint.verify")
load_dotenv()
_client = None
_client_lock = threading.Lock()
_neon_thinking = True


def _get_client():
    """One shared client. Created under a lock: racing threads must not build (and GC-close) extra clients."""
    global _client
    with _client_lock:
        if _client is None:
            # ⚠️ uncertain — attempts=1 turns off the Decisions SDK's built-in 4x backoff retry; we retry once ourselves
            _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
                                   http_options=types.HttpOptions(api_version="v1beta",
                                                                  retry_options=types.HttpRetryOptions(attempts=1)))
            _client.decisions  # build the lazy sub-client now, not concurrently inside worker threads
    return _client


def _with_retry(name, fn):
    """fn() with one retry; logs latency. Returns (result, ms_of_successful_attempt). Raises after 2 failures."""
    last = None
    for attempt in (1, 2):
        t = time.perf_counter()
        try:
            out = fn()
            ms = (time.perf_counter() - t) * 1000
            log.info("%s ok attempt=%d %.0f ms", name, attempt, ms)
            return out, ms
        except Exception as e:
            log.warning("%s failed attempt=%d %.0f ms: %s: %s", name, attempt, (time.perf_counter() - t) * 1000,
                        type(e).__name__, str(e)[:200])
            last = e
    raise last


def _b64(img):
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def _crop(vis, bounds, pad):
    H, W = vis.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bounds)
    return vis[max(0, y1 - pad):min(H, y2 + pad + 1), max(0, x1 - pad):min(W, x2 + pad + 1)]


def _room_crop(img, room, pad, thick, partner=None):
    vis = img.copy()
    if partner is not None and partner["polygon"]:
        cv2.polylines(vis, [np.array(partner["polygon"], np.int32)], True, (255, 80, 0), thick)
    cv2.polylines(vis, [np.array(room["polygon"], np.int32)], True, (0, 0, 255), thick)
    return _crop(vis, room["geom"].bounds, pad)


def _check_room(img, room, pad, thick):
    crop = _room_crop(img, room, pad, thick)
    label = room["label"]
    inp = [{"type": "image", "data": _b64(crop), "mime_type": "image/png", "resolution": "low"},
           {"type": "text", "text": CONTEXT.format(label=label)}]
    qs = {"one_room": {"type": "noul", "instructions": Q_ONE_ROOM},
          "label_inside": {"type": "noul", "instructions": Q_LABEL.format(label=label)},
          "within_walls": {"type": "noul", "instructions": Q_WALLS},
          "alignment": {"type": "score", "instructions": Q_ALIGN, "criteria": ALIGN_LEVELS}}
    try:
        resp, ms = _with_retry(f"decisions {room['id']}", lambda: _get_client().decisions.create(
            model=DECISIONS, input=inp, questions=qs, timeout=TIMEOUT_S))
        checks = {k: float(resp.answers[k].noul) for k in NOUL_KEYS}
        score = float(resp.answers["alignment"].score)
        ok = all(p >= PASS_NOUL for p in checks.values()) and score >= PASS_SCORE
        return {"checks": checks, "score": score, "verified": ok, "confidence": round(min(checks.values()), 3),
                "latency_ms": round(ms), "tokens": (resp.usage.total_input_tokens or 0) if resp.usage else 0,
                "error": None, "crop": crop}
    except Exception as e:
        return {"checks": {}, "score": None, "verified": False, "confidence": None, "latency_ms": None, "tokens": 0,
                "error": f"{type(e).__name__}: {str(e)[:200]}", "crop": crop}


def _check_window(img, w, pad, radius, thick):
    vis = img.copy()
    x, y = w["center"]
    cv2.circle(vis, (x, y), radius, (0, 0, 255), thick)
    crop = _crop(vis, (x, y, x, y), pad)
    inp = [{"type": "image", "data": _b64(crop), "mime_type": "image/png", "resolution": "low"},
           {"type": "text", "text": WINDOW_CONTEXT}]
    try:
        resp, ms = _with_retry(f"decisions {w['id']}", lambda: _get_client().decisions.create(
            model=DECISIONS, input=inp, questions={"exterior": {"type": "noul", "instructions": Q_EXTERIOR}},
            timeout=TIMEOUT_S))
        return {"p": float(resp.answers["exterior"].noul), "latency_ms": round(ms),
                "tokens": (resp.usage.total_input_tokens or 0) if resp.usage else 0, "error": None}
    except Exception as e:
        return {"p": None, "latency_ms": None, "tokens": 0, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def _rule_ranking(options):
    return [o for o in ("use_box", "split", "reseed") if o in options] + ["keep"]


def _ask_neon(room, res, options, partners, crop):
    """Neon ranks the option names. Returns (ranking, source, ms, usage). Falls back to a fixed rule on failure."""
    failed = [f"{k} (p={v:.2f})" for k, v in res["checks"].items() if v < PASS_NOUL]
    if res["score"] is not None and res["score"] < PASS_SCORE:
        failed.append(f"alignment (score {res['score']:.1f} of 3)")
    prompt = NEON_PROMPT.format(
        label=room["label"], partner_note=f", and room {partners['split']} in blue" if "split" in partners else "",
        failed=", ".join(failed) or "unknown", method=room["method"], flags=", ".join(room["flags"]) or "none",
        options="\n".join(f"- {o}: {REPAIR_HELP[o].format(other=partners.get(o, ''))}" for o in options))
    schema = {"type": "OBJECT", "required": ["ranking"],
              "properties": {"ranking": {"type": "ARRAY", "items": {"type": "STRING", "enum": options}}}}

    def call():
        global _neon_thinking
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema,  # ⚠️ uncertain — enum schema on Neon
            http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000),
            thinking_config=types.ThinkingConfig(thinking_level=NEON_THINKING) if _neon_thinking else None)
        try:
            resp = _get_client().models.generate_content(
                model=NEON, contents=[types.Part.from_bytes(data=base64.b64decode(_b64(crop)), mime_type="image/png"),
                                      prompt], config=cfg)
        except Exception as e:
            if _neon_thinking and "thinking" in str(e).lower():
                _neon_thinking = False  # parameter rejected -> drop it
            raise
        ranking = list(dict.fromkeys(o for o in json.loads(resp.text).get("ranking", []) if o in options))
        if not ranking:
            raise ValueError(f"neon returned no valid option: {resp.text[:120]!r}")
        return ranking + ([] if "keep" in ranking else ["keep"]), resp.usage_metadata

    try:
        (ranking, u), ms = _with_retry(f"neon {room['id']}", call)
        usage = {"input_tokens": u.prompt_token_count or 0, "output_tokens": u.candidates_token_count or 0,
                 "thought_tokens": u.thoughts_token_count or 0} if u else None
        return ranking, "neon", round(ms), usage
    except Exception:
        return _rule_ranking(options), "rule_fallback", None, None


def _set_geom(room, geom):
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    if not isinstance(geom, Polygon) or geom.is_empty:
        return False
    room["geom"] = geom
    room["polygon"] = [[int(round(x)), int(round(y))] for x, y in geom.exterior.coords[:-1]]
    return True


def _overlap_partner(room, rooms):
    best, best_a = None, 0.0
    for o in rooms:
        if o is room or o["geom"] is None:
            continue
        a = room["geom"].intersection(o["geom"]).area
        if a > 0.02 * min(room["geom"].area, o["geom"].area) and a > best_a:
            best, best_a = o, a
    return best


def _merge_partner(room, rooms, gap):
    best, best_a = None, 0.0
    grown = room["geom"].buffer(gap)
    for o in rooms:
        if o is room or o["geom"] is None:
            continue
        a = grown.intersection(o["geom"].buffer(gap)).area
        if a > best_a:
            best, best_a = o, a
    return best


def _options(room, rooms, gap):
    opts, partners = [], {}
    if room["box"] and room["method"] not in ("box_fallback", "box_only"):
        opts.append("use_box")
    if room["box"] and room["method"] != "box_only":
        opts.append("reseed")
    p = _overlap_partner(room, rooms)
    if p:
        opts.append("split")
        partners["split"] = p["id"]
    m = _merge_partner(room, rooms, gap)
    if m:
        opts.append("merge")
        partners["merge"] = m["id"]
    return opts + ["keep"], partners


def _seed_of(r):
    return r["seed"] or [int(v) for v in r["geom"].representative_point().coords[0]]


def _split(a, b):
    """Cut the overlap of a and b along the perpendicular bisector of their seeds."""
    inter = a["geom"].intersection(b["geom"])
    if inter.area <= 0:
        return False
    (ax, ay), (bx, by) = _seed_of(a), _seed_of(b)
    n = math.hypot(bx - ax, by - ay) or 1.0
    dx, dy, mx, my, L = (bx - ax) / n, (by - ay) / n, (ax + bx) / 2, (ay + by) / 2, 1e5
    nx, ny = -dy, dx
    half_a = Polygon([(mx + nx * L, my + ny * L), (mx + nx * L - dx * L, my + ny * L - dy * L),
                      (mx - nx * L - dx * L, my - ny * L - dy * L), (mx - nx * L, my - ny * L)])
    return (_set_geom(a, a["geom"].difference(inter.difference(half_a)))
            and _set_geom(b, b["geom"].difference(inter.intersection(half_a))))


def _open_point(room, geo):
    """Point inside the room's box farthest from any wall/furniture pixel of the current mask."""
    x1, y1, x2, y2 = room["box"]
    masks = geo.get("_masks") or {}
    mask = masks.get(room["method"]) if masks.get(room["method"]) is not None else masks.get("color")
    if mask is None:
        mask = masks.get("adaptive")
    if mask is None or x2 - x1 < 3 or y2 - y1 < 3:
        return [(x1 + x2) // 2, (y1 + y2) // 2]
    free = (mask[y1:y2, x1:x2] == 0).astype(np.uint8)
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    yy, xx = np.unravel_index(int(dt.argmax()), dt.shape)
    return [int(x1 + xx), int(y1 + yy)]


def _no_new_overlap(room, geom, rooms):
    """Trim floor that this repair would newly take from other rooms (overlap that already existed is left)."""
    for o in rooms:
        if o is room or o["geom"] is None:
            continue
        taken = geom.intersection(o["geom"]).difference(room["geom"])
        if taken.area > 0:
            geom = geom.difference(taken)
    return geom


def _apply(choice, room, partner, rooms, img, parsed, geo, gap):
    """Apply one repair deterministically (mutates room/partner/rooms). Returns ids whose outline changed.
    use_box / reseed never create new double-counted floor with another room (e.g. never undo a split)."""
    if choice == "use_box":
        if not _set_geom(room, _no_new_overlap(room, shapely_box(*room["box"]), rooms)):
            return set()
        room["method"] = "box_fallback"
    elif choice == "reseed":
        seed = _open_point(room, geo)
        p2 = {**parsed, "rooms": [{**r, "seed": seed} if r["id"] == room["id"] else r for r in parsed["rooms"]]}
        nr = next(r for r in geometry.build_rooms(img, p2, (geo.get("_masks") or {}).get("adaptive"))["rooms"]
                  if r["id"] == room["id"])
        if nr["geom"] is None:
            return set()
        new_geom = _no_new_overlap(room, nr["geom"], rooms)
        room["seed"], room["method"] = seed, nr["method"]
        room["flags"] = list(dict.fromkeys(room["flags"] + nr["flags"]))
        if not _set_geom(room, new_geom):
            return set()
    elif choice == "split" and partner is not None:
        if not _split(room, partner):
            return set()
        partner["flags"].append("repaired_split")
        room["flags"].append("repaired_split")
        return {room["id"], partner["id"]}
    elif choice == "merge" and partner is not None:
        big, small = (room, partner) if room["geom"].area >= partner["geom"].area else (partner, room)
        _set_geom(big, unary_union([big["geom"], small["geom"]]).buffer(gap).buffer(-gap))
        big["label"] = f"{big['label']} + {small['label']}"
        big["flags"].append(f"merged_{small['id']}")
        rooms[:] = [o for o in rooms if o is not small]
        return {big["id"]}
    else:
        return set()
    room["flags"].append(f"repaired_{choice}")
    return {room["id"]}


def _alignment_score(rooms, results, weights):
    """Share of room area verified. Weights = each room's iteration-0 area, fixed for the whole run, so a repair
    moves the score only by flipping verification, never by growing/shrinking a still-failing outline."""
    tot = sum(weights.get(r["id"], 0.0) for r in rooms)
    ok = sum(weights.get(r["id"], 0.0) for r in rooms if results.get(r["id"], {}).get("verified"))
    return round(ok / tot, 4) if tot else 0.0


def _mean_conf(res_list):
    v = [(r["confidence"] or 0.0) for r in res_list if r]
    return sum(v) / len(v) if v else 0.0


def _copy(r):
    return {**r, "flags": list(r["flags"]), "verify_log": list(r["verify_log"])}


def verify_and_repair(img, parsed, geo):
    t0 = time.perf_counter()
    H, W = img.shape[:2]
    scale = geo.get("scale_px_per_ft")
    px_ft = scale or 0.025 * max(W, H)  # only for crop padding / drawing when scale is undetermined
    pad, thick = max(8, round(PAD_FT * px_ft)), max(2, round(min(H, W) / 250))
    gap = max(3, round(0.75 * px_ft))  # wall thickness tolerance for merge adjacency
    rooms = [{**r, "flags": list(r.get("flags") or []), "verify_log": []} for r in geo["rooms"]]
    for r in rooms:
        if r.get("geom") is None and r.get("polygon"):
            r["geom"] = Polygon(r["polygon"])
    results, repairs, history, neon_log = {}, [], [], []
    dec_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    neon_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    _get_client()  # create the shared client before worker threads start

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        def run_checks(room_list, iteration):
            """One Decisions call per room, in parallel. Returns {id: result}; logs usage and verify_log."""
            todo = [r for r in room_list if r["geom"] is not None]
            out = {}
            for r, res in zip(todo, ex.map(lambda r: _check_room(img, r, pad, thick), todo)):
                out[r["id"]] = res
                dec_usage["calls"] += 1
                dec_usage["input_tokens"] += res["tokens"]
                r["verify_log"].append({"iteration": iteration, "verified": res["verified"],
                                        "confidence": res["confidence"], "method": r["method"],
                                        "alignment": round(res["score"], 2) if res["score"] is not None else None})
            return out

        # iteration 0: all rooms + all windows in one parallel batch
        wins = [w for w in parsed.get("windows") or [] if w.get("center")]
        f_wins = [ex.submit(_check_window, img, w, max(10, round(WINDOW_PAD_FT * px_ft)),
                            max(8, round(1.5 * px_ft)), max(1, thick - 1)) for w in wins]  # wide thin ring
        results.update(run_checks(rooms, 0))
        weights = {r["id"]: r["geom"].area for r in rooms if r["geom"] is not None}
        history.append(_alignment_score(rooms, results, weights))
        ranking, tried, settled = {}, {}, set()

        for it in range(1, MAX_ITERS + 1):
            if time.perf_counter() - t0 > BUDGET_S:
                log.warning("verify budget %ss reached; no further repair rounds", BUDGET_S)
                break
            failing = [r for r in rooms if r["id"] in results and not results[r["id"]]["verified"]
                       and results[r["id"]]["error"] is None and r["id"] not in settled]
            if not failing:
                break

            # Neon ranks the menu once per failing room (in parallel)
            need = [r for r in failing if r["id"] not in ranking]
            if need:
                plans = []
                for r in need:
                    opts, partners = _options(r, rooms, gap)
                    blue = next((o for o in rooms if o["id"] == partners.get("split")), None)
                    plans.append((r, opts, partners, _room_crop(img, r, pad, thick, blue)))
                answers = ex.map(lambda p: _ask_neon(p[0], results[p[0]["id"]], p[1], p[2], p[3]), plans)
                for (r, *_), (rank, source, ms, usage) in zip(plans, answers):
                    ranking[r["id"]] = rank
                    neon_log.append({"room": r["id"], "ranking": rank, "source": source, "latency_ms": ms})
                    if usage:
                        neon_usage["calls"] += 1
                        for k in ("input_tokens", "output_tokens", "thought_tokens"):
                            neon_usage[k] += usage[k]

            # apply each failing room's next untried repair on copies
            work_list = [_copy(r) for r in rooms]
            work = {r["id"]: r for r in work_list}
            attempts, busy = [], set()
            for r in failing:
                rid = r["id"]
                opts, partners = _options(work[rid], work_list, gap)
                tried.setdefault(rid, set())
                choice = next((c for c in ranking[rid] if c in opts and c not in tried[rid]), "keep")
                if choice == "keep":
                    settled.add(rid)
                    continue
                pid = partners.get(choice)
                if rid in busy or (pid is not None and pid in busy):
                    continue  # touches a room already being repaired this round; try next round
                tried[rid].add(choice)
                before = {o["id"] for o in work_list}
                changed = _apply(choice, work[rid], work.get(pid), work_list, img, parsed, geo, gap)
                removed = before - {o["id"] for o in work_list}
                busy |= changed | removed | {rid}
                if changed:
                    attempts.append({"room": rid, "choice": choice, "partner": pid, "changed": changed,
                                     "removed": removed})
                else:
                    repairs.append({"iteration": it, "room": rid, "choice": choice, "partner": pid,
                                    "accepted": False, "reason": "not applicable"})
            if not attempts:
                continue

            # re-check only changed rooms; keep a repair only if the score rises (or ties with higher confidence)
            alive = {o["id"]: o for o in work_list}
            new_res = run_checks([alive[i] for a in attempts for i in a["changed"] if i in alive], it)
            pre, pre_weights = _alignment_score(rooms, results, weights), weights
            keep_changed, keep_removed = set(), set()
            for a in attempts:
                ids, gone = a["changed"], a["removed"]
                cand = [alive[o["id"]] if o["id"] in ids else o for o in rooms if o["id"] not in gone]
                cres = {k: v for k, v in results.items() if k not in gone}
                cres.update({i: new_res[i] for i in ids if i in new_res})
                cw = dict(weights)
                for g in gone:  # a merged-away room's weight moves to the surviving room
                    cw[next(iter(ids))] = cw.get(next(iter(ids)), 0.0) + cw.pop(g, 0.0)
                s_new = _alignment_score(cand, cres, cw)
                c_pre, c_new = _mean_conf([results.get(i) for i in ids | gone]), _mean_conf([new_res.get(i) for i in ids])
                # split removes double-counted floor (a geometric fix in itself): keep it unless it costs verification
                ok = s_new > pre + 1e-6 or (abs(s_new - pre) <= 1e-6 and (c_new > c_pre + 1e-6 or a["choice"] == "split"))
                if ok:
                    keep_changed |= ids
                    keep_removed |= gone
                    weights = cw if gone else weights
                repairs.append({"iteration": it, "room": a["room"], "choice": a["choice"], "partner": a["partner"],
                                "accepted": ok, "score_before": round(pre, 3), "score_after": round(s_new, 3),
                                "confidence_before": round(c_pre, 3), "confidence_after": round(c_new, 3)})
            new_rooms = [alive[o["id"]] if o["id"] in keep_changed else o for o in rooms if o["id"] not in keep_removed]
            new_results = {k: v for k, v in results.items() if k not in keep_removed}
            new_results.update({i: new_res[i] for i in keep_changed if i in new_res})
            if _alignment_score(new_rooms, new_results, weights) + 1e-6 >= pre:  # guard: combined repairs never lower it
                rooms, results = new_rooms, new_results
            else:
                weights = pre_weights
            history.append(_alignment_score(rooms, results, weights))

        win_res = [f.result() for f in f_wins]

    # final room fields
    checks_out = []
    for r in rooms:
        res = results.get(r["id"])
        r["verified"] = bool(res and res["verified"])
        r["confidence"] = res["confidence"] if res else None
        r["alignment"] = round(res["score"], 2) if res and res["score"] is not None else None
        r["checks"] = {k: round(v, 3) for k, v in res["checks"].items()} if res else {}
        r["area_sqft"] = round(r["geom"].area / scale ** 2, 1) if (scale and r["geom"] is not None) else None
        if res is None:
            r["flags"].append("not_verified_no_polygon")
        elif res["error"]:
            r["flags"].append("verify_call_failed")
        if res and not res["error"]:
            label = r["label"]
            for k, q in (("one_room", Q_ONE_ROOM), ("label_inside", Q_LABEL.format(label=label)),
                         ("within_walls", Q_WALLS)):
                checks_out.append({"room": r["id"], "key": k, "question": q, "p_yes": round(res["checks"][k], 3),
                                   "latency_ms": res["latency_ms"]})
            checks_out.append({"room": r["id"], "key": "alignment", "question": Q_ALIGN, "p_yes": None,
                               "score": round(res["score"], 2), "latency_ms": res["latency_ms"]})

    windows = []
    for w, wr in zip(wins, win_res):
        dec_usage["calls"] += 1
        dec_usage["input_tokens"] += wr["tokens"]
        p, carbon = wr["p"], w.get("on_exterior_wall")
        # Decisions confirms; it never overturns Carbon alone. Disagreement or a middling p -> null (unknown), so a
        # legal-bedroom verdict is never built on a contested window. Failed call -> Carbon's reading, unverified.
        if p is None:
            ext = carbon
        elif p >= EXTERIOR_YES:
            ext = True
        elif p <= EXTERIOR_NO and carbon is not True:
            ext = False
        else:
            ext = None
        windows.append({**w, "on_exterior_wall": ext, "on_exterior_wall_carbon": carbon,
                        "p_exterior": round(p, 3) if p is not None else None, "exterior_verified": p is not None})
        if p is not None:
            checks_out.append({"room": None, "window": w["id"], "key": "window_exterior", "question": Q_EXTERIOR,
                               "p_yes": round(p, 3), "latency_ms": wr["latency_ms"]})
    windows += [w for w in parsed.get("windows") or [] if not w.get("center")]

    history = [round(h, 3) for h in history]
    ms = (time.perf_counter() - t0) * 1000
    log.info("verify %.0f ms history=%s repairs=%s", ms, history,
             [(x["room"], x["choice"], x["accepted"]) for x in repairs])
    return {"rooms": rooms, "alignment_score_history": history, "decision_checks": checks_out, "windows": windows,
            "repairs": repairs, "neon": neon_log, "cost_usd": None, "timing_ms": round(ms),
            "usage": {"gemini-decision-1": dec_usage, NEON: neon_usage},
            "_crops": {rid: res["crop"] for rid, res in results.items()}}


def draw_debug(img, ver, out_path):
    vis = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in ver["rooms"]:
        if not r["polygon"]:
            continue
        c = (40, 170, 40) if r["verified"] else (0, 0, 230)
        pts = np.array(r["polygon"], np.int32)
        cv2.polylines(vis, [pts], True, c, 2)
        cx, cy = (int(v) for v in r["geom"].representative_point().coords[0])
        for j, s in enumerate([f"{r['id']} {r['label'][:16]}",
                               f"{'OK' if r['verified'] else 'FAIL'} c={r['confidence']} a={r['alignment']}"]):
            cv2.putText(vis, s, (cx - 45, cy + 14 * j), font, 0.4, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(vis, s, (cx - 45, cy + 14 * j), font, 0.4, c, 1, cv2.LINE_AA)
    head = f"alignment history {ver['alignment_score_history']}  verify {ver['timing_ms']} ms"
    cv2.putText(vis, head, (6, 14), font, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, head, (6, 14), font, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "httpx2", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from parse import parse_plan

    scratch = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratch"))
    os.makedirs(os.path.join(scratch, "verify_crops"), exist_ok=True)
    fresh = "--fresh" in sys.argv  # --fresh: new Carbon call instead of scratch/parse_<name>.json
    for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
        name = os.path.splitext(os.path.basename(path))[0]
        cache = os.path.join(scratch, f"parse_{name}.json")
        img = cv2.imread(path)
        parsed = json.load(open(cache)) if (os.path.exists(cache) and not fresh) else parse_plan(path)
        geo = geometry.build_rooms(img, parsed)
        ver = verify_and_repair(img, parsed, geo)
        draw_debug(img, ver, os.path.join(scratch, f"verify_{name}.png"))
        for rid, crop in ver.pop("_crops").items():
            cv2.imwrite(os.path.join(scratch, "verify_crops", f"{name}_{rid}.png"), crop)
        print(f"\n== {path}  verify {ver['timing_ms']} ms  alignment history {ver['alignment_score_history']}")
        for n in ver["neon"]:
            print(f"  neon ranking {n['room']}: {n['ranking']} ({n['source']}, {n['latency_ms']} ms)")
        for x in ver["repairs"]:
            extra = (f" score {x['score_before']}->{x['score_after']} conf {x['confidence_before']}->{x['confidence_after']}"
                     if "score_before" in x else f" ({x.get('reason')})")
            print(f"  it{x['iteration']}: {x['room']} {x['choice']}{' with ' + x['partner'] if x['partner'] and x['choice'] in ('split', 'merge') else ''}"
                  f" -> {'KEPT' if x['accepted'] else 'rolled back'}{extra}")
        for r in ver["rooms"]:
            trail = " -> ".join(f"it{v['iteration']}:{v['method']}:{'OK' if v['verified'] else 'x'}({v['confidence']})"
                                for v in r["verify_log"])
            print(f"  {r['id']:3s} {r['label'][:24]:24s} {'VERIFIED' if r['verified'] else 'failed  '} "
                  f"conf={r['confidence']} align={r['alignment']} area={r['area_sqft']}  trail: {trail}")
        for w in ver["windows"]:
            print(f"  window {w['id']} p_exterior={w['p_exterior']} -> on_exterior_wall={w['on_exterior_wall']}")
        print(f"  usage: {ver['usage']}")
