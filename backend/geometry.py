"""geometry.py — OpenCV traces exact room shapes from Carbon's seeds; computes scale.

wall_mask(img) -> uint8 mask        needs no model output: run it in a thread while Carbon runs
build_rooms(img, parsed, walls=None) -> {"rooms", "scale_px_per_ft", "scale_status", "door_kernel_px", "flags", "timing_ms"}

Per room, wall masks are tried in order until one gives a clean region:
  adaptive      grayscale adaptive threshold + open (classic black-line plans)
  color         pixels matching the plan's wall color, sampled along Carbon's box edges (rendered plans)
  color_strong  color OR dark outlines, stronger opening, drop small blobs (furniture / tiles)
  box_fallback  Carbon's box polygon
Non-habitable spaces (stairs, closets, outdoor) skip tracing: box polygon, flag "box_only".
CLI: python backend/geometry.py <plan> [...] [--cached]   (--cached reuses scratch/parse_<name>.json)
"""
import json
import logging
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box as shapely_box

DOOR_FT_DEFAULT = 2.67
CLOSE_MARGIN = 1.25        # close kernel = door width px * this (entrances run ~3 ft)
LEAK_AREA_RATIO = 2.5      # region > this x Carbon box area -> leaked
MIN_AREA_RATIO = 0.6       # region < this x box area -> cut short by furniture/text
SMALL_BLOB = 0.6           # mask blobs shorter than this x door width (text, chairs) are dropped before closing
THICK_FT = 1.2             # color_strong: blobs thicker than this are furniture, never walls
CONFINE_FT = 0.5           # tracing is confined to the union of Carbon boxes grown by this margin
MIN_INSIDE_BOX = 0.75      # at least this share of a traced region must lie inside its own (grown) box
SEAL_MIN_FT = 0.2          # only structures thicker than this (walls, not door leaves/text) are bridged when sealing doors
SMOOTH_FT = 0.1            # morphological close on traced polygons: removes slits around door leaves
SCALE_OUTLIER = 0.10
COLOR_TOL = 14.0           # Lab distance (OpenCV 8-bit Lab units) to the sampled wall color
FLOOR_EXCLUDE = 10.0       # ring pixels this close to a room's floor color are not wall samples
DARK_GRAY = 110
BOX_ONLY_RE = re.compile(r"stair|closet|\bclo\b|\bcl\b|w\.?i\.?c\b|wardrobe|dress|pantry|storage|store\b|utility|laundry"
                         r"|w/d|shaft|duct|elevator|lift|balcony|terrace|veranda|porch|patio|deck|garage|(^|\s)open$", re.I)
METHODS = ("adaptive", "color", "color_strong")

log = logging.getLogger("footprint.geometry")


def wall_mask(img):
    """Adaptive-threshold wall mask. Independent of Carbon, so it can run in parallel with parse_plan."""
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 3)
    block = max(15, (min(img.shape[:2]) // 15) | 1)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 10)
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _wall_color_lab(img, rooms):
    """Most common non-floor, non-white color in a band along Carbon's room-box edges."""
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    t = max(3, round(min(H, W) * 0.015))
    ring = np.zeros((H, W), np.uint8)
    floors = []
    for r in rooms:
        if not r["box"]:
            continue
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(ring, (x1, y1), (x2, y2), 255, t)
        ix, iy = (x2 - x1) // 4, (y2 - y1) // 4
        inner = lab[y1 + iy:y2 - iy, x1 + ix:x2 - ix].reshape(-1, 3)
        if len(inner):
            floors.append(np.median(inner, 0))
    px = lab[ring > 0].reshape(-1, 3).astype(np.float32)
    keep = ~((px[:, 0] > 245) & (np.abs(px[:, 1] - 128) < 6) & (np.abs(px[:, 2] - 128) < 6))  # white
    for f in floors:
        keep &= np.linalg.norm(px - f, axis=1) > FLOOR_EXCLUDE
    px = px[keep]
    if len(px) < 50:
        return None
    q = (px // 8).astype(np.int32)
    keys, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    return np.median(px[inv.ravel() == counts.argmax()], 0)


def color_mask(img, rooms, strong=False, thick_px=None):
    wc = _wall_color_lab(img, rooms)
    if wc is None:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = (np.linalg.norm(lab - wc, axis=2) < COLOR_TOL).astype(np.uint8) * 255
    if not strong:
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m |= (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < DARK_GRAY).astype(np.uint8) * 255  # fuse fill with its outlines
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    d = max(9, int(thick_px or 0.03 * min(img.shape[:2]))) | 1  # walls are thinner than this; furniture is not
    solid = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))  # speckled beds/counters -> solid
    thick = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    return cv2.bitwise_and(m, cv2.bitwise_not(cv2.dilate(thick, np.ones((5, 5), np.uint8))))


def _drop_small(mask, min_len):
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= min_len]
    return np.isin(cc, keep).astype(np.uint8) * 255


def _close_doors(mask, length):
    """Seal door gaps: union of horizontal and vertical line closings."""
    length = max(3, int(length))
    h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    return h | v


def _box_area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def _rough_scale(rooms):
    s = [((_box_area(r["box"])) / (r["printed_dims_ft"][0] * r["printed_dims_ft"][1])) ** 0.5
         for r in rooms if r["box"] and r["printed_dims_ft"]]
    return statistics.median(s) if s else None


def _region_polygon(region, smooth_px):
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, max(1.5, 0.004 * cv2.arcLength(c, True)), True).reshape(-1, 2)
    if len(c) < 3:
        return None
    p = Polygon(c)
    p = p if p.is_valid else p.buffer(0)
    p = p.buffer(smooth_px, join_style="mitre").buffer(-smooth_px, join_style="mitre")  # close thin slits
    if isinstance(p, MultiPolygon):
        p = max(p.geoms, key=lambda g: g.area)
    return p.simplify(1.0) if isinstance(p, Polygon) and not p.is_empty else None


def _trace(labels, room, others, W, H, grow, smooth_px):
    """Region containing the room's seed in a labeled free-space map. Returns (polygon|None, status, flags)."""
    x, y = room["seed"]
    lab = labels[y, x]
    flags = []
    if lab == 0:  # seed on a wall/furniture pixel: take the largest free component inside the box
        if not room["box"]:
            return None, "seed_on_wall", flags
        x1, y1, x2, y2 = room["box"]
        ids, counts = np.unique(labels[y1:y2, x1:x2], return_counts=True)
        counts, ids = counts[ids > 0], ids[ids > 0]
        if not len(ids):
            return None, "seed_on_wall", flags
        lab = ids[counts.argmax()]
        flags.append("seed_snapped")
    region = (labels == lab).astype(np.uint8)
    ys, xs = np.nonzero(region)
    if xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1:
        return None, "leaked", flags  # escaped to the outside of the plan
    if any(labels[o["seed"][1], o["seed"][0]] == lab for o in others):
        return None, "leaked", flags  # swallowed another room
    area = len(xs)
    if room["box"] and area > LEAK_AREA_RATIO * _box_area(room["box"]):
        return None, "leaked", flags
    if room["box"] and area < MIN_AREA_RATIO * _box_area(room["box"]):
        return None, "undersized", flags
    if room["box"]:
        x1, y1, x2, y2 = room["box"]
        inside = ((xs >= x1 - grow) & (xs <= x2 + grow) & (ys >= y1 - grow) & (ys <= y2 + grow)).mean()
        if inside < MIN_INSIDE_BOX:
            return None, "spilled", flags  # ran into an unlabeled/open area outside its own box
    poly = _region_polygon(region, smooth_px)
    return (poly, "ok", flags) if poly is not None and poly.area > 0 else (None, "no_polygon", flags)


def build_rooms(img, parsed, walls=None):
    t0 = time.perf_counter()
    H, W = img.shape[:2]
    rooms_in = parsed["rooms"]
    flags = []

    # door width in px -> close kernel
    widths = [d["width_ft"] for d in parsed["doors"] if d["width_ft"]]
    door_ft = statistics.median(widths) if widths else DOOR_FT_DEFAULT
    if not widths:
        flags.append("door_width_default_2.67ft")
    rough = _rough_scale(rooms_in)
    if rough:
        door_px = door_ft * rough
    else:
        door_px = 0.05 * max(W, H)
        flags.append("door_kernel_heuristic_no_dims")
    close_len = door_px * CLOSE_MARGIN
    px_per_ft = rough or door_px / DOOR_FT_DEFAULT

    # non-habitable spaces: box only, and act as obstacles so neighbours can't leak into them.
    # Everything outside the union of Carbon boxes (+margin) is also an obstacle: no escaping through windows.
    box_only = {r["id"] for r in rooms_in if BOX_ONLY_RE.search(r["label"]) and r["box"]}
    obstacles = np.full((H, W), 255, np.uint8)
    grow = max(2, round(CONFINE_FT * px_per_ft))
    for r in rooms_in:
        if r["box"]:
            x1, y1, x2, y2 = r["box"]
            obstacles[max(0, y1 - grow):y2 + grow + 1, max(0, x1 - grow):x2 + grow + 1] = 0
    for r in rooms_in:
        if r["id"] in box_only:
            x1, y1, x2, y2 = r["box"]
            obstacles[y1:y2 + 1, x1:x2 + 1] = 255
    traced = [r for r in rooms_in if r["seed"] and r["id"] not in box_only]

    masks = {"adaptive": walls if walls is not None else wall_mask(img)}
    label_maps = {}

    def labels_for(method):
        if method not in label_maps:
            if method not in masks:
                masks[method] = color_mask(img, rooms_in, strong=(method == "color_strong"), thick_px=THICK_FT * px_per_ft)
            m = masks[method]
            if m is None:
                label_maps[method] = None
            else:
                base = _drop_small(m, SMALL_BLOB * door_px)
                # seal doors by bridging thick structures only, so thin door leaves / text never get bridged to walls
                s = int(SEAL_MIN_FT * px_per_ft)
                k = max(3, s if s % 2 else s - 1)  # largest odd kernel <= SEAL_MIN_FT
                thick =_drop_small(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)), SMALL_BLOB * door_px)
                free = cv2.bitwise_not(base | _close_doors(thick, close_len) | obstacles)
                label_maps[method] = cv2.connectedComponents(free, connectivity=4)[1]
        return label_maps[method]

    out_rooms = []
    for r in rooms_in:
        room = {**r, "flags": list(r["flags"]), "polygon": None, "geom": None, "method": None, "attempts": {}}
        if r["id"] in box_only:
            room["geom"], room["method"] = shapely_box(*r["box"]), "box_only"
            room["flags"].append("box_only")
        elif r["seed"]:
            others = [o for o in traced if o["id"] != r["id"]]
            for method in METHODS:
                labels = labels_for(method)
                if labels is None:
                    room["attempts"][method] = "no_mask"
                    continue
                poly, status, f = _trace(labels, r, others, W, H, grow, max(2.0, SMOOTH_FT * px_per_ft))
                room["attempts"][method] = status
                if poly is not None:
                    room["geom"], room["method"] = poly, method
                    room["flags"] += f
                    break
            if room["geom"] is None and r["box"]:
                room["geom"], room["method"] = shapely_box(*r["box"]), "box_fallback"
                room["flags"].append("box_fallback")
                if {"leaked", "spilled"} & set(room["attempts"].values()):
                    room["flags"].append("leaked_region")
        elif r["box"]:
            room["geom"], room["method"] = shapely_box(*r["box"]), "box_fallback"
            room["flags"].append("box_fallback")
        if room["geom"] is not None:
            room["polygon"] = [[int(round(x)), int(round(y))] for x, y in room["geom"].exterior.coords[:-1]]
        out_rooms.append(room)

    # scale: per room with printed dims, sqrt(pixel_area / (w_ft*h_ft)); median; flag outliers
    per_room = {r["id"]: (r["geom"].area / (r["printed_dims_ft"][0] * r["printed_dims_ft"][1])) ** 0.5
                for r in out_rooms if r["geom"] is not None and r["printed_dims_ft"]}
    if not per_room:
        scale, status = None, "undetermined"
    else:
        scale = statistics.median(per_room.values())
        outliers = [rid for rid, s in per_room.items() if abs(s - scale) / scale > SCALE_OUTLIER]
        for r in out_rooms:
            if r["id"] in outliers:
                r["flags"].append("scale_outlier")
        status = "low_confidence" if len(per_room) < 2 or len(outliers) * 2 >= len(per_room) else "ok"
    for r in out_rooms:
        r["scale_px_per_ft"] = round(per_room[r["id"]], 2) if r["id"] in per_room else None
        r["area_sqft"] = round(r["geom"].area / scale ** 2, 1) if (scale and r["geom"] is not None) else None

    ms = (time.perf_counter() - t0) * 1000
    log.info("geometry %.0f ms scale=%s (%s) close=%dpx methods=%s", ms, scale and round(scale, 2), status,
             close_len, {r["id"]: r["method"] for r in out_rooms})
    return {"rooms": out_rooms, "scale_px_per_ft": round(scale, 2) if scale else None, "scale_status": status,
            "door_kernel_px": int(close_len), "flags": flags, "timing_ms": round(ms), "_masks": masks}


COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 200, 200), (240, 50, 230), (128, 128, 0), (0, 128, 128), (170, 110, 40)]


def draw_debug(img, geo, out_path):
    vis = img.copy()
    overlay = img.copy()
    for i, r in enumerate(geo["rooms"]):
        if r["polygon"]:
            pts = np.array(r["polygon"], np.int32)
            cv2.fillPoly(overlay, [pts], COLORS[i % len(COLORS)])
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, r in enumerate(geo["rooms"]):
        c = COLORS[i % len(COLORS)]
        if r["polygon"]:
            pts = np.array(r["polygon"], np.int32)
            cv2.polylines(vis, [pts], True, c, 2)
            cx, cy = (int(v) for v in r["geom"].representative_point().coords[0])
            area = f"{r['area_sqft']:g}sf" if r["area_sqft"] is not None else "?sf"
            for j, s in enumerate([f"{r['id']} {r['label'][:14]}", f"{r['method']} {area}"]):
                cv2.putText(vis, s, (cx - 40, cy + 14 * j), font, 0.4, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(vis, s, (cx - 40, cy + 14 * j), font, 0.4, c, 1, cv2.LINE_AA)
        if r["seed"]:
            cv2.circle(vis, tuple(r["seed"]), 4, (0, 0, 0), -1)
    head = f"scale={geo['scale_px_per_ft']} px/ft ({geo['scale_status']})  close={geo['door_kernel_px']}px  {geo['flags']}"
    cv2.putText(vis, head, (6, 14), font, 0.42, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, head, (6, 14), font, 0.42, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from parse import parse_plan

    scratch = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratch"))
    cached = "--cached" in sys.argv
    for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
        name = os.path.splitext(os.path.basename(path))[0]
        cache = os.path.join(scratch, f"parse_{name}.json")
        img = cv2.imread(path)
        t = time.perf_counter()
        with ThreadPoolExecutor(2) as ex:  # wall mask runs while Carbon thinks
            f_walls = ex.submit(wall_mask, img)
            if cached and os.path.exists(cache):
                parsed = json.load(open(cache))
            else:
                parsed = ex.submit(parse_plan, path).result()
                json.dump(parsed, open(cache, "w"), indent=1)
            walls = f_walls.result()
        log.info("parse+walls %.0f ms (cached=%s)", (time.perf_counter() - t) * 1000, cached and os.path.exists(cache))
        if parsed["error"]:
            print("parse error:", parsed["error"])
            continue
        geo = build_rooms(img, parsed, walls)
        draw_debug(img, geo, os.path.join(scratch, f"geometry_{name}.png"))
        m = geo.pop("_masks")
        cv2.imwrite(os.path.join(scratch, f"geometry_{name}_masks.png"),
                    np.hstack([m.get(k) if m.get(k) is not None else np.zeros(img.shape[:2], np.uint8) for k in METHODS]))
        print(f"\n== {path}: scale={geo['scale_px_per_ft']} px/ft ({geo['scale_status']}) "
              f"close={geo['door_kernel_px']}px flags={geo['flags']} {geo['timing_ms']} ms")
        for r in geo["rooms"]:
            print(f"  {r['id']:3s} {r['label'][:18]:18s} method={r['method']:13s} area={r['area_sqft']} sqft "
                  f"dims={r['printed_dims_ft']} room_scale={r['scale_px_per_ft']} flags={r['flags']} attempts={r['attempts']}")
        print(f"  debug: scratch/geometry_{name}.png  masks: scratch/geometry_{name}_masks.png")
