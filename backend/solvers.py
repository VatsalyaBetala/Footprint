"""Deterministic solvers: area, legal bedrooms, furniture fit, clear path.

Pure geometry (shapely + numpy), no model calls. Coordinates are image pixels
[x, y], origin top-left. `scale` is px per foot and may be None; anything that
needs feet then comes back as not checkable instead of being guessed.
"""
import math
import re
from collections import deque

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

AREA_AGREE_TOL = 0.05          # printed vs geometry area must be within 5%
MIN_BEDROOM_SQFT = 80
MIN_BEDROOM_WIDTH_FT = 8
MEASURE_AGREE_TOL = 0.15       # printed vs traced room area; beyond this an unverified room isn't judged
DISAGREE = "measurements disagree; not judged"
FIT_STEP_FT = 0.25             # 3 in grid-search step
DOOR_NEAR_FT = 1.0             # a door belongs to a room if its centre is within 1 ft of it
DEFAULT_DOOR_FT = 2.67         # ⚠️ assumed (standard 32 in door) when Carbon gives no width
DOUBLE_DOOR_FT = 4.0           # ⚠️ judgement call: an opening wider than this is a double door (two half-width leaves)
SWING_BLOCKS_CLEARANCE = False  # ⚠️ judgement call: a swing may cross the walkway, never the object
PERSON_WIDTH_FT = 1.5          # 18 in shoulder width → shrink free space 9 in from every wall
DOOR_OPENING_DEPTH_FT = 2.0    # opening rectangle bridges walls up to ~2 ft thick
PATH_CELL_FT = 0.25            # 3 in raster cells for the path search
SNAP_FT = 3.0                  # entrance usually sits in the wall; snap to floor within 3 ft
NO_SCALE = "not_checkable_no_scale"
NO_POLYGON = "not_checkable_no_polygon"
# ⚠️ spec says label contains "bed"; BDRM / BR added for NYC listing shorthand ("2BR", "BR 1")
BEDROOM_RE = re.compile(r"bed|bdrm|(?<![a-z])br(?![a-z])")
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


# ---------- public ----------

def compute_area(rooms, scale):
    """Total sq ft two independent ways: printed w×h per room, and polygon area / scale².

    None for any part that can't be computed. `agree` compares like with like: printed
    total vs geometry over the SAME rooms (rooms without printed dims are left out of
    both). printed_coverage = [rooms with printed dims, all rooms], so a partial printed
    total is never passed off as the whole apartment.
    """
    printed_rooms = [r for r in rooms if _dims(r)]
    printed = sum(_dims(r)[0] * _dims(r)[1] for r in printed_rooms) if printed_rooms else None
    geometry = sum(_poly(r).area for r in rooms) / scale ** 2 if scale and rooms else None
    agree = None
    if printed and scale:
        same = sum(_poly(r).area for r in printed_rooms) / scale ** 2
        agree = abs(same - printed) <= AREA_AGREE_TOL * printed
    return {"from_printed": None if printed is None else round(printed, 1),
            "from_geometry": None if geometry is None else round(geometry, 1),
            "agree": agree,
            "printed_coverage": [len(printed_rooms), len(rooms)]}


def legal_bedrooms(rooms, windows, scale):
    """Legal-bedroom checks for every room labelled as a bedroom: area >= 80 sq ft and min width >= 8 ft.
    (Window and ceiling height are not bedroom rules here; `windows` is accepted for call compatibility only.)

    Area and min width come from printed_dims_ft when present (measurement_source
    "printed"), else from the traced polygon ("geometry"). The polygon cross-checks the
    printed dims: >15% apart on an unverified room -> legal None, "measurements disagree".
    Each check is True/False or "not_checkable_*". legal = False if either check fails,
    True only if both pass, otherwise None (unknown).
    """
    out = []
    for r in filter(_is_bedroom, rooms):
        poly = _poly(r)
        area, width, source, traced = _measure(r, poly, scale)
        agree = (abs(traced - area) <= MEASURE_AGREE_TOL * area) if source == "printed" and traced else None
        checks, fails, unknown = {}, [], []
        if area is None:
            checks["area_ge_80"] = checks["min_width_ge_8ft"] = NO_SCALE if not scale else NO_POLYGON
            unknown.append(f"area/width not checkable ({'no scale' if not scale else 'no polygon'})")
        else:
            checks["area_ge_80"] = area >= MIN_BEDROOM_SQFT
            checks["min_width_ge_8ft"] = width >= MIN_BEDROOM_WIDTH_FT
            if not checks["area_ge_80"]:
                fails.append(f"{_floor(area)} sq ft < {MIN_BEDROOM_SQFT} sq ft minimum")
            if not checks["min_width_ge_8ft"]:
                fails.append(f"{_floor(width, 1)} ft wide < {MIN_BEDROOM_WIDTH_FT} ft minimum")
        vals = [checks["area_ge_80"], checks["min_width_ge_8ft"]]
        legal = False if any(v is False for v in vals) else (True if all(v is True for v in vals) else None)
        reason = "; ".join(fails + unknown) if legal is not True else (
            f"{area:.0f} sq ft, {width:.1f} ft min width ({source} dims)")
        if agree is False and r.get("verified") is not True:
            legal, reason = None, DISAGREE   # one of the two measurements is wrong; don't guess which
        out.append({"room": r["id"], "label": r.get("label"), "legal": legal, "checks": checks,
                    "reason": reason, "measurement_source": source,
                    "measurements": {"printed_sqft": _r1(area) if source == "printed" else None,
                                     "geometry_sqft": _r1(traced), "agree": agree}})
    return out


def fit(room, w_ft, d_ft, clearance_ft, doors, scale):
    """Place a w×d object (back to a wall) with clearance on its other 3 sides.

    Grid search in 3 in steps over 8 orientations: back to the top/bottom/left/right wall with the
    w side along the wall ("standard"), then the same four turned 90° so the d side runs along the wall
    ("sideways"; e.g. a 7×3 sofa needs 13×6 ft standard but only 9×10 ft sideways). Valid = object +
    clearance inside the room and object clear of door swings. Preference: standard against a wall,
    then sideways against a wall, then anything valid. placement = object rectangle in px (no clearance).
    """
    poly = _poly(room)
    if not scale or poly.is_empty:
        return {"fits": None, "placement": None, "reason": NO_SCALE if not scale else NO_POLYGON}
    step = FIT_STEP_FT * scale
    inside = prep(poly.buffer(0.02 * scale))       # ~1/4 in tolerance for exact fits
    swings = _door_swings(poly, doors, scale)
    swing = prep(unary_union(swings)) if swings else None
    wall = poly.boundary.buffer(step)
    x0, y0, x1, y1 = poly.bounds
    fallback, door_blocked = None, False
    w, d, c = w_ft * scale, d_ft * scale, clearance_ft * scale
    layouts = [(lay, "standard") for lay in _layouts(w, d, c)]
    if abs(w_ft - d_ft) > 1e-9:  # a square object turned 90° is the same footprint
        layouts += [(lay, "sideways") for lay in _layouts(d, w, c)]
    turned = f" (turned sideways: its {d_ft:g} ft side along the wall)"
    for (fw, fh, ob, (b0, b1)), orient in layouts:
        for x in _steps(x0, x1 - fw, step):
            for y in _steps(y0, y1 - fh, step):
                foot = box(x, y, x + fw, y + fh)
                if not inside.contains(foot):
                    continue
                obj = box(x + ob[0], y + ob[1], x + ob[2], y + ob[3])
                if swing and swing.intersects(foot if SWING_BLOCKS_CLEARANCE else obj):
                    door_blocked = True
                    continue
                back = LineString([(x + b0[0], y + b0[1]), (x + b1[0], y + b1[1])])
                if back.difference(wall).length <= 0.1 * back.length:
                    return {"fits": True, "placement": _coords(obj), "orientation": orient,
                            "orientations_tried": len(layouts),
                            "reason": f"fits against a wall with {clearance_ft:g} ft clearance on 3 sides"
                                      + (turned if orient == "sideways" else "")}
                fallback = (obj, orient) if fallback is None else fallback
    if fallback is not None:
        return {"fits": True, "placement": _coords(fallback[0]), "orientation": fallback[1],
                "orientations_tried": len(layouts),
                "reason": f"fits with {clearance_ft:g} ft clearance on 3 sides, but not against a wall"
                          + (turned if fallback[1] == "sideways" else "")}
    long_, short = (s / scale for s in _rect_sides(poly))
    need = f"needs {w_ft + 2 * clearance_ft:.1f} × {d_ft + clearance_ft:.1f} ft"
    if len(layouts) > 4:
        need += f", or {d_ft + 2 * clearance_ft:.1f} × {w_ft + clearance_ft:.1f} ft turned sideways"
    why = (f"every spot that fits inside the walls is blocked by a door swing (tried {len(layouts)} orientations)"
           if door_blocked else
           f"tried {len(layouts)} orientations: {need} ({w_ft:g} × {d_ft:g} ft + {clearance_ft:g} ft clearance"
           f" on 3 sides); room is {long_:.1f} × {short:.1f} ft")
    return {"fits": False, "placement": None, "orientation": None, "orientations_tried": len(layouts), "reason": why}


def fit_candidates(room, w_ft, d_ft, clearance_ft, doors, scale, max_n=24, spacing_ft=1.5):
    """Every distinct valid spot for a w×d object over all 8 orientations (same rules as fit()), best first:
    against a wall before free-standing, standard before sideways; spots closer than spacing_ft to an
    already-kept spot with the same footprint are skipped. Each: {placement (px), orientation, against_wall}.
    The caller decides between them (e.g. the first one that keeps the way out clear)."""
    poly = _poly(room)
    if not scale or poly.is_empty:
        return []
    step = FIT_STEP_FT * scale
    inside = prep(poly.buffer(0.02 * scale))
    swings = _door_swings(poly, doors, scale)
    swing = prep(unary_union(swings)) if swings else None
    wall = poly.boundary.buffer(step)
    x0, y0, x1, y1 = poly.bounds
    w, d, c = w_ft * scale, d_ft * scale, clearance_ft * scale
    layouts = [(lay, "standard") for lay in _layouts(w, d, c)]
    if abs(w_ft - d_ft) > 1e-9:
        layouts += [(lay, "sideways") for lay in _layouts(d, w, c)]
    found = []
    for (fw, fh, ob, (b0, b1)), orient in layouts:
        for x in _steps(x0, x1 - fw, step):
            for y in _steps(y0, y1 - fh, step):
                if not inside.contains(box(x, y, x + fw, y + fh)):
                    continue
                obj = box(x + ob[0], y + ob[1], x + ob[2], y + ob[3])
                if swing and swing.intersects(box(x, y, x + fw, y + fh) if SWING_BLOCKS_CLEARANCE else obj):
                    continue
                back = LineString([(x + b0[0], y + b0[1]), (x + b1[0], y + b1[1])])
                against = back.difference(wall).length <= 0.1 * back.length
                found.append((not against, orient != "standard", obj, orient, against))
    found.sort(key=lambda f: (f[0], f[1]))
    kept, gap = [], spacing_ft * scale
    for _, _, obj, orient, against in found:
        bx = obj.bounds
        shape = round(bx[2] - bx[0])                          # footprint width in px: distinguishes turned copies
        if any(k[3] == shape and obj.centroid.distance(k[0].centroid) < gap for k in kept):
            continue
        kept.append((obj, orient, against, shape))
        if len(kept) >= max_n:
            break
    return [{"placement": _coords(o), "orientation": orient, "against_wall": a} for o, orient, a, _ in kept]


def closest_spot(room, w_ft, d_ft, doors, scale):
    """When a w×d object fits nowhere: the spot where it pokes out of the walls least (both orientations,
    3 in grid over the room's box), out of door swings when possible, pushed into a corner on ties.
    For SHOWING how badly it misses, never as a fit. {placement, rotation, overflow_sqft, too_big_ft: [x, y]}."""
    poly = _poly(room)
    if not scale or poly.is_empty:
        return None
    swings = unary_union(_door_swings(poly, doors, scale) or [Polygon()])
    x0, y0, x1, y1 = poly.bounds
    step, best = FIT_STEP_FT * scale, None
    for fw, fh, rot in ((w_ft * scale, d_ft * scale, 0), (d_ft * scale, w_ft * scale, 90)):
        xs = _steps(x0, x1 - fw, step) if x1 - x0 >= fw else [(x0 + x1 - fw) / 2]
        ys = _steps(y0, y1 - fh, step) if y1 - y0 >= fh else [(y0 + y1 - fh) / 2]
        for x in xs:
            for y in ys:
                obj = box(x, y, x + fw, y + fh)
                out = obj.area - obj.intersection(poly).area
                key = (round(out / scale ** 2, 1), round(obj.intersection(swings).area / scale ** 2, 1),
                       round(min(x - x0, x1 - x - fw) + min(y - y0, y1 - y - fh)))
                if best is None or key < best[0]:
                    best = (key, obj, rot, fw, fh)
    (out, _, _), obj, rot, fw, fh = best
    return {"placement": _coords(obj), "rotation": rot, "overflow_sqft": out,
            "too_big_ft": [round(max(0, fw - (x1 - x0)) / scale, 1), round(max(0, fh - (y1 - y0)) / scale, 1)]}


def check_spot(room, placement, doors, scale, max_ft=15.0):
    """A placement chosen by hand (px polygon, 4 corners in order): inside the room? clear of door swings?
    Free floor beyond each side, in ft (0 = against a wall, capped at max_ft). Side names are plan
    directions (top = smaller y)."""
    poly = _poly(room)
    if not scale or poly.is_empty:
        return {"inside": None, "door_blocked": None, "sides": [], "reason": NO_SCALE if not scale else NO_POLYGON}
    pts = np.asarray(placement, float)[:4]
    obj = Polygon(pts)
    room_in = prep(poly.buffer(0.02 * scale))
    inside = room_in.contains(obj)
    door_blocked = any(obj.intersects(s) for s in _door_swings(poly, doors, scale))
    c, sides = pts.mean(0), []
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        e = b - a
        n = np.array([-e[1], e[0]]) / (np.hypot(*e) or 1)
        if np.dot(n, (a + b) / 2 - c) < 0:
            n = -n                                          # outward normal
        lo, hi = 0.0, max_ft
        if not inside or room_in.contains(Polygon([a, b, b + n * hi * scale, a + n * hi * scale])):
            lo = hi if inside else 0.0
        else:
            while hi - lo > 0.05:                           # strip containment is monotone in depth: bisect
                mid = (lo + hi) / 2
                if room_in.contains(Polygon([a, b, b + n * mid * scale, a + n * mid * scale])):
                    lo = mid
                else:
                    hi = mid
        name = ("right" if n[0] > 0 else "left") if abs(n[0]) >= abs(n[1]) else ("bottom" if n[1] > 0 else "top")
        sides.append({"side": name, "gap_ft": round(lo, 1), "length_ft": round(float(np.hypot(*e)) / scale, 2)})
    return {"inside": inside, "door_blocked": door_blocked, "sides": sides}


def clear_path(rooms, doors, entrance, scale, obstacles=None):
    """Per bedroom: can an 18 in wide person walk from it to the entrance?

    Free space = rooms ∪ door openings − obstacles (optional px polygons, e.g. a placed
    bed), shrunk 9 in from every wall, rasterised at 3 in, BFS from the entrance.
    path = [[x, y], ...] px from the bedroom to the entrance, or None.
    """
    def res(r, ok, path, why):
        return {"from": r["id"], "to": "entrance", "ok": ok, "path": path, "reason": why}

    beds = [r for r in rooms if _is_bedroom(r)]
    if not scale or entrance is None:
        return [res(r, None, None, NO_SCALE if not scale else "entrance not detected") for r in beds]
    free = _walkable(rooms, doors, scale, obstacles)
    if free.is_empty:
        return [res(r, False, None, "no walkable floor") for r in beds]
    cell = PATH_CELL_FT * scale
    grid, cx, cy = _raster(free, cell)
    goal = _nearest_cell(grid, cx, cy, entrance, SNAP_FT * scale)
    parent = _bfs(grid, goal) if goal else {}
    out = []
    for r in beds:
        poly, name = _poly(r), r.get("label") or r["id"]
        if poly.is_empty:
            out.append(res(r, None, None, NO_POLYGON))
            continue
        c = poly.centroid
        start = _nearest_cell(grid & shapely.contains_xy(poly, cx, cy), cx, cy, (c.x, c.y))
        if goal is None:
            out.append(res(r, False, None, f"no walkable floor within {SNAP_FT:g} ft of the entrance"))
        elif start is None:
            out.append(res(r, False, None, f"no free floor left in {name}"))
        elif start not in parent:
            out.append(res(r, False, None, f"no {PERSON_WIDTH_FT * 12:.0f} in wide route from {name} to the entrance"))
        else:
            out.append(res(r, True, _trace(parent, start, cx, cy, entrance, cell),
                           f"clear {PERSON_WIDTH_FT * 12:.0f} in wide route to the entrance"))
    return out


# ---------- helpers ----------

def _poly(room):
    """Room polygon as a valid shapely Polygon (largest piece if self-intersecting); empty if missing."""
    pts = room.get("polygon") or []
    if len(pts) < 3:
        return Polygon()
    p = Polygon(pts)
    if not p.is_valid:
        p = p.buffer(0)
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
    return p


def _dims(room):
    """printed_dims_ft if it is two positive numbers, else None."""
    d = room.get("printed_dims_ft")
    ok = isinstance(d, (list, tuple)) and len(d) == 2 and all(
        isinstance(v, (int, float)) and v > 0 for v in d)
    return d if ok else None


def _is_bedroom(room):
    return bool(BEDROOM_RE.search((room.get("label") or "").lower()))


def _measure(room, poly, scale):
    """(area_sqft, min_width_ft, source, traced_sqft): printed dims first, traced polygon second.
    Printed dims need no scale; the traced area (the cross-check) does."""
    traced = poly.area / scale ** 2 if scale and not poly.is_empty else None
    d = _dims(room)
    if d:
        return d[0] * d[1], min(d), "printed", traced
    if traced is not None:
        return traced, _rect_sides(poly)[1] / scale, "geometry", traced
    return None, None, None, None


def _r1(x):
    return None if x is None else round(x, 1)


def _rect_sides(poly):
    """(long, short) side of the minimum rotated rectangle, in px.
    ⚠️ For L-shaped rooms the short side overstates the true clear width (it's the bounding width)."""
    r = poly.minimum_rotated_rectangle
    if r.is_empty or r.geom_type != "Polygon":
        return 0.0, 0.0
    c = np.asarray(r.exterior.coords)
    a, b = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])
    return float(max(a, b)), float(min(a, b))


def _floor(x, nd=0):
    """Round DOWN for '< minimum' messages so 79.6 never prints as '80 < 80'."""
    f = 10 ** nd
    return f"{math.floor(x * f + 1e-6) / f:.{nd}f}"


def _wall_axis(pt, poly):
    """Unit vector along the room edge nearest pt, snapped to x or y.
    ⚠️ assumes axis-aligned walls (true for almost all plans; wrong for angled walls)."""
    c = np.asarray(poly.exterior.coords)
    a, ab = c[:-1], np.diff(c, axis=0)
    t = np.clip(((pt - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
    i = np.linalg.norm(a + t[:, None] * ab - pt, axis=1).argmin()
    return np.array([1.0, 0.0]) if abs(ab[i, 0]) >= abs(ab[i, 1]) else np.array([0.0, 1.0])


def _door_swings(poly, doors, scale):
    """Quarter circle (radius = door width, centred on the hinge) swinging INTO this room,
    for each door whose centre is within 1 ft of the room. Unknown hinge → both sides."""
    out = []
    for d in doors or []:
        if d.get("center") is None:
            continue
        c = np.asarray(d["center"], float)
        if poly.distance(Point(c)) > DOOR_NEAR_FT * scale:
            continue
        r = (d.get("width_ft") or DEFAULT_DOOR_FT) * scale
        u = _wall_axis(c, poly)
        n = np.array([-u[1], u[0]])
        if poly.distance(Point(c + n * r / 2)) > poly.distance(Point(c - n * r / 2)):
            n = -n                                          # n now points into the room
        # ⚠️ uncertain — verify against parse.py's prompt: "left" = hinge on your left
        # when standing inside this room facing the door.
        left = np.array([-n[1], n[0]])
        t = np.linspace(0, np.pi / 2, 12)[:, None]
        double = r > DOUBLE_DOOR_FT * scale      # two leaves of half the width, hinged at both ends
        for s in [1, -1] if double else {"left": [1], "right": [-1]}.get(d.get("hinge"), [1, -1]):
            hinge = c + s * left * r / 2
            arc = hinge + (r / 2 if double else r) * (np.cos(t) * (-s * left) + np.sin(t) * n)
            out.append(Polygon([hinge, *arc]))
    return out


def _layouts(w, d, c):
    """(footprint w, h, object box, back edge) relative to the footprint's top-left,
    for the object's back against the top / bottom / left / right wall."""
    W, D = w + 2 * c, d + c
    return [
        (W, D, (c, 0, c + w, d), ((c, 0), (c + w, 0))),    # back to top wall
        (W, D, (c, c, c + w, D), ((c, D), (c + w, D))),    # back to bottom wall
        (D, W, (0, c, d, c + w), ((0, c), (0, c + w))),    # back to left wall
        (D, W, (c, c, D, c + w), ((D, c), (D, c + w))),    # back to right wall
    ]


def _steps(lo, hi, step):
    """lo, lo+step, ... plus exactly hi, so placements can sit flush against both walls."""
    return [] if hi < lo - 1e-9 else [*np.arange(lo, hi, step), hi]


def _coords(geom):
    return [[round(float(x), 1), round(float(y), 1)] for x, y in geom.exterior.coords[:-1]]


def _door_opening(door, polys, scale):
    """Rectangle across the wall at a door: door width along the wall, 2 ft through it."""
    c = np.asarray(door["center"], float)
    near = min(polys, key=lambda p: p.distance(Point(c)))
    u = _wall_axis(c, near)
    n = np.array([-u[1], u[0]])
    hw = (door.get("width_ft") or DEFAULT_DOOR_FT) * scale / 2
    hd = DOOR_OPENING_DEPTH_FT * scale / 2
    return Polygon([c - hw * u - hd * n, c + hw * u - hd * n, c + hw * u + hd * n, c - hw * u + hd * n])


def _walkable(rooms, doors, scale, obstacles):
    """Rooms ∪ door openings − obstacles, shrunk so any passage narrower than 18 in closes."""
    polys = [p for p in map(_poly, rooms) if not p.is_empty]
    if not polys:
        return Polygon()
    openings = [_door_opening(d, polys, scale) for d in doors or [] if d.get("center") is not None]
    free = unary_union(polys + openings)
    if obstacles:
        free = free.difference(unary_union([Polygon(o) for o in obstacles]))
    # ⚠️ "erode by 18 in" read as an 18 in wide person (9 in each side). An 18 in RADIUS
    # would need 36 in passages and would seal every standard 30–32 in door.
    return free.buffer(-PERSON_WIDTH_FT / 2 * scale)


def _raster(geom, cell):
    """Boolean grid of cells whose centres lie in geom, plus the centres' px x/y arrays."""
    x0, y0, x1, y1 = geom.bounds
    cx, cy = np.meshgrid(np.arange(x0 + cell / 2, x1, cell), np.arange(y0 + cell / 2, y1, cell))
    shapely.prepare(geom)
    return shapely.contains_xy(geom, cx, cy), cx, cy


def _nearest_cell(mask, cx, cy, pt, max_px=None):
    """(row, col) of the True cell closest to pt, or None if there is none within max_px."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    d2 = (cx[rows, cols] - pt[0]) ** 2 + (cy[rows, cols] - pt[1]) ** 2
    i = int(d2.argmin())
    if max_px is not None and d2[i] > max_px ** 2:
        return None
    return int(rows[i]), int(cols[i])


def _bfs(grid, start):
    """8-connected BFS over True cells; {cell: parent} for every cell reachable from start."""
    g, (h, w) = grid.tolist(), grid.shape
    parent, q = {start: None}, deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in NBRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr][nc] and (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return parent


def _trace(parent, cell, cx, cy, end, tol):
    """Follow parents from cell back to the BFS root, append the true entrance, simplify."""
    pts = []
    while cell is not None:
        pts.append((float(cx[cell]), float(cy[cell])))
        cell = parent[cell]
    pts.append((float(end[0]), float(end[1])))
    return [[round(x, 1), round(y, 1)] for x, y in LineString(pts).simplify(tol).coords]
