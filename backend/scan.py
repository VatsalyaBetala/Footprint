"""scan.py — phone video of ONE room -> estimated floor plan in the model.json shape.

scan_video(path) -> dict   (never raises: failures come back as errors + flags in a valid result)

1. OpenCV: validate (5–60 s), 4 evenly spaced keyframes (JPEG, longest side 768 px).
2. Files API upload, poll until ACTIVE.
3. In parallel: CARBON reads walls (lengths + turns), openings, furniture, ceiling (pydantic-validated,
   one corrective retry); NEON estimates width × depth only; DECISIONS checks the keyframes for scale
   references (door, corners, floor + ceiling).
4. Deterministic: walk Carbon's walls into a polygon and close it (or fall back to Neon's rectangle),
   cross-check against Neon, widen ranges when evidence is weak, render a plan PNG at 20 px/ft.
Models only estimate lengths/angles from visual references; every coordinate is computed here.
"""
import base64
import logging
import math
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError, field_validator, model_validator

CARBON_PROMPT = """You are a surveyor. The camera pans around ONE room. Ignore furniture when finding walls. Use standard references to estimate scale: interior door 6'8" tall x 2'8" wide, outlet plate 4.5" tall, switch height 48", chair seat 18", countertop 36", typical ceiling 8–9 ft. Return JSON only:
{walls:[{id, length_ft:{low,best,high}, turn_deg_to_next: number (usually 90 or -90)}],   // in order, clockwise from the wall with the main door
 openings:[{type:'door'|'window', wall_id, offset_ft_from_wall_start, width_ft, exterior: true|false|null}],
 furniture:[{label, w_ft, d_ft, h_ft, wall_id|null, offset_ft, distance_from_wall_ft}],
 ceiling_ft:{low,best,high}, references_used:[...], notes}"""
CARBON_RETRY_SUFFIX = "\n\nYour previous answer failed validation:\n{error}\nReturn the corrected JSON only."
NEON_PROMPT = """You are a surveyor. The camera pans around ONE room. Ignore furniture. Estimate the room's floor
dimensions wall to wall, using standard references for scale: interior door 6'8" tall x 2'8" wide, outlet plate
4.5" tall, switch height 48", chair seat 18", countertop 36", typical ceiling 8–9 ft. Return JSON only:
{"width_ft": {"low": n, "best": n, "high": n}, "depth_ft": {"low": n, "best": n, "high": n}, "references_used": [...]}"""
DECISIONS_CONTEXT = "Four frames from a phone video panning around one room, in time order."
DECISION_QS = {
    "full_height_door_visible": "At least one frame shows a complete interior door, from the floor to its top edge.",
    "at_least_two_room_corners_visible": "Across the frames, at least two corners of the room (where two walls meet) are visible.",
    "floor_and_ceiling_both_visible": "Across the frames, both the floor and the ceiling of the room are visible.",
}
WEAK_FLAG = {"full_height_door_visible": "weak_reference_door",
             "at_least_two_room_corners_visible": "weak_reference_corners",
             "floor_and_ceiling_both_visible": "weak_reference_floor_ceiling"}

CARBON = "gemini-4-carbon"
NEON = "gemini-4-neon"
DECISIONS = "models/gemini-decision-1"
THINKING_LEVEL = "LOW"         # auto-dropped per model if the API rejects it
MIN_S, MAX_S = 5, 60           # accepted video length; the advice we give is 15–30 s
N_KEYFRAMES, KEYFRAME_SIDE = 4, 768
UPLOAD_TIMEOUT_MS, CARBON_TIMEOUT_MS, NEON_TIMEOUT_MS, GET_TIMEOUT_MS = 120_000, 120_000, 90_000, 15_000
DECISIONS_TIMEOUT_S = 30
ACTIVE_WAIT_S, POLL_S = 90, 2
PX_PER_FT, MARGIN_FT = 20, 1
CLOSE_TOL, VERIFY_CLOSE_TOL = 0.15, 0.05   # closure error / perimeter
AGREE_TOL, WIDEN, NOUL_MIN, TURN_TOL_DEG = 0.15, 0.15, 0.6, 15
FALLBACK_CONF = 0.5            # ⚠️ judgement call: geometry confidence when we fall back to a rectangle
VIDEO_MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
              ".avi": "video/x-msvideo", ".mkv": "video/x-matroska", ".3gp": "video/3gpp"}

log = logging.getLogger("footprint.scan")
load_dotenv()
_client = None                 # shared-client pattern: server.py may set scan._client = its client
_client_lock = threading.Lock()
_no_thinking = set()           # models that rejected thinking_config


def _get_client():
    """Lazy, thread-safe: without the lock two threads can each build a client, and the one that loses the race
    is garbage-collected, closing its connection mid-request ("client has been closed")."""
    global _client
    with _client_lock:
        if _client is None:
            _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options=types.HttpOptions(
                api_version="v1beta", retry_options=types.HttpRetryOptions(attempts=1)))  # we retry once ourselves
        return _client


# ---------- response schemas ----------

class Range(BaseModel):
    low: float
    best: float
    high: float

    @model_validator(mode="after")
    def order_and_check(self):
        self.low, self.best, self.high = sorted((self.low, self.best, self.high))
        if self.best <= 0:
            raise ValueError("best must be > 0")
        return self


def _as_str(v):
    return None if v is None else str(v)


class Wall(BaseModel):
    id: str
    length_ft: Range
    turn_deg_to_next: float

    @field_validator("id", mode="before")
    @classmethod
    def id_to_str(cls, v):
        return _as_str(v)


class Opening(BaseModel):
    type: Literal["door", "window"]
    wall_id: str
    offset_ft_from_wall_start: float = 0.0
    width_ft: float
    exterior: bool | None = None

    @field_validator("wall_id", mode="before")
    @classmethod
    def wall_id_to_str(cls, v):
        return _as_str(v)

    @field_validator("type", mode="before")
    @classmethod
    def type_lower(cls, v):
        return str(v).strip().lower()


class Furniture(BaseModel):
    label: str
    w_ft: float
    d_ft: float
    h_ft: float | None = None
    wall_id: str | None = None
    offset_ft: float | None = None
    distance_from_wall_ft: float | None = None

    @field_validator("wall_id", mode="before")
    @classmethod
    def wall_id_to_str(cls, v):
        return _as_str(v)


class CarbonScan(BaseModel):
    walls: list[Wall]
    openings: list[Opening] = []
    furniture: list[Furniture] = []
    ceiling_ft: Range | None = None
    references_used: list[str] = []
    notes: str | None = None


class NeonDims(BaseModel):
    width_ft: Range
    depth_ft: Range
    references_used: list[str] = []


# ---------- entry point ----------

def scan_video(path):
    out = _empty()
    t0 = time.perf_counter()
    try:
        duration, frames = _probe(path)
        out["scan"]["duration_s"] = None if duration is None else round(duration, 1)
        if duration is None or not MIN_S <= duration <= MAX_S or len(frames) < N_KEYFRAMES:
            got = "unknown length" if duration is None else f"{duration:.0f} s"
            _fail(out, "validate", f"video is {got}; record 15–30 s of one room", "bad_video")
        else:
            jpegs = [_jpeg(f) for f in frames]
            _models(out, path, jpegs)
            t = time.perf_counter()
            _build(out)
            out["timings_ms"]["build"] = round((time.perf_counter() - t) * 1000)
    except Exception as e:  # nothing escapes: a valid result with the error recorded
        log.exception("scan failed")
        _fail(out, "scan", f"{type(e).__name__}: {e}"[:300], "scan_failed")
    for k in ("_carbon", "_neon", "_decisions"):
        out["scan"].pop(k, None)  # pydantic objects stay internal; raw content is in carbon_walls_raw / neon_dims
    out["timings_ms"]["total"] = round((time.perf_counter() - t0) * 1000)
    return out


def _empty():
    return {"image": {"w": None, "h": None, "png_b64": None}, "scale_px_per_ft": PX_PER_FT, "scale_status": "estimated",
            "rooms": [], "doors": [], "windows": [], "entrance": None, "furniture": [],
            "area": {"from_printed": None, "from_geometry": None, "listing_claim": None, "claim_scope": None},
            "answers": {"king_bed_fits": None, "clear_path": None, "legal_bedrooms": [], "custom_query": None},
            "alignment_score_history": [], "decision_checks": [],
            "scan": {"duration_s": None, "closure_error_ft": None, "perimeter_ft": None, "references_used": [],
                     "ceiling_ft": None, "carbon_walls_raw": None, "neon_dims": None, "polygon_source": None,
                     "_carbon": None, "_neon": None, "_decisions": None},
            "usage": {}, "cost_usd": None,
            "timings_ms": {"upload": 0, "carbon": 0, "neon": 0, "decisions": 0, "build": 0},
            "flags": [], "errors": []}


def _fail(out, stage, msg, flag):
    out["errors"].append({"stage": stage, "error": msg})
    if flag not in out["flags"]:
        out["flags"].append(flag)
    return out


# ---------- video ----------

def _probe(path):
    """(duration_s or None, N_KEYFRAMES evenly spaced BGR frames). Raises if OpenCV can't open it."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("OpenCV can't open this file as a video")
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 0.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n / fps if fps > 0 and n > 0 else None
    want = [int((i + 0.5) / N_KEYFRAMES * n) for i in range(N_KEYFRAMES)] if duration else []
    frames = []
    for idx in want:  # fast path: seek straight to each keyframe
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if len(frames) == N_KEYFRAMES:
        return duration, frames
    return _probe_sequential(path, fps)


def _probe_sequential(path, fps):
    """Slow path for files without a usable frame count / seek (e.g. browser webm): decode through once."""
    cap, n = cv2.VideoCapture(path), 0
    while cap.grab():
        n += 1
    duration = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000) or (n / fps if fps > 0 else None)
    cap.release()
    want, frames, i = {int((k + 0.5) / N_KEYFRAMES * n) for k in range(N_KEYFRAMES)}, [], 0
    cap = cv2.VideoCapture(path)
    while len(frames) < len(want) and cap.grab():
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f)
        i += 1
    cap.release()
    return (duration if n else None), frames


def _jpeg(frame):
    h, w = frame.shape[:2]
    s = KEYFRAME_SIDE / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()


# ---------- model calls ----------

def _with_retry(name, fn):
    """fn() with one retry; logs latency. Returns (result, ms of the successful attempt). Raises after 2 failures."""
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


def _models(out, path, jpegs):
    """Decisions starts at once (keyframes only); Carbon + Neon start when the upload is ACTIVE."""
    with ThreadPoolExecutor(3) as ex:
        f_dec = ex.submit(_decisions, jpegs)
        t = time.perf_counter()
        try:
            file = _upload(path)
        except Exception as e:
            file = None
            _fail(out, "upload", f"{type(e).__name__}: {e}"[:300], "upload_failed")
        out["timings_ms"]["upload"] = round((time.perf_counter() - t) * 1000)
        futs = {"carbon": ex.submit(_carbon, file), "neon": ex.submit(_neon, file)} if file else {}
        futs["decisions"] = f_dec
        for stage, f in futs.items():
            try:
                result, usage, ms = f.result()
                out["scan"][f"_{stage}"] = result
                out["usage"][usage[0]] = usage[1]
                out["timings_ms"][stage] = round(ms)
            except Exception as e:
                _fail(out, stage, f"{type(e).__name__}: {e}"[:300], f"{stage}_failed")
    if file is not None:
        threading.Thread(target=_delete, args=(file.name,), daemon=True).start()  # don't keep user video


def _upload(path):
    mime = VIDEO_MIME.get(os.path.splitext(path)[1].lower(), "video/mp4")
    f, _ = _with_retry("upload", lambda: _get_client().files.upload(file=path, config=types.UploadFileConfig(
        mime_type=mime, http_options=types.HttpOptions(timeout=UPLOAD_TIMEOUT_MS))))
    start = time.perf_counter()
    while _state(f) != "ACTIVE":   # File.state is a FileState enum: PROCESSING / ACTIVE / FAILED (SDK 3.6.1)
        if _state(f) == "FAILED":
            raise RuntimeError(f"video processing failed: {f.error}")
        if time.perf_counter() - start > ACTIVE_WAIT_S:
            raise TimeoutError(f"video not ACTIVE after {ACTIVE_WAIT_S} s (state {_state(f)})")
        time.sleep(POLL_S)
        f, _ = _with_retry("files.get", lambda name=f.name: _get_client().files.get(
            name=name, config=types.GetFileConfig(http_options=types.HttpOptions(timeout=GET_TIMEOUT_MS))))
    return f


def _state(f):
    s = getattr(f, "state", None)
    return getattr(s, "name", None) or (str(s) if s is not None else None)


def _delete(name):
    try:
        _with_retry("files.delete", lambda: _get_client().files.delete(
            name=name, config=types.DeleteFileConfig(http_options=types.HttpOptions(timeout=GET_TIMEOUT_MS))))
    except Exception:
        pass  # files expire on their own after 48 h


def _generate(model, file, prompt, timeout_ms):
    """One generate_content call on the uploaded video (JSON out). thinking_config dropped if rejected."""
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json", http_options=types.HttpOptions(timeout=timeout_ms),
        thinking_config=None if model in _no_thinking else types.ThinkingConfig(thinking_level=THINKING_LEVEL))
    try:
        return _get_client().models.generate_content(
            model=model, contents=[types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type), prompt], config=cfg)
    except Exception as e:
        if "thinking" in str(e).lower():
            _no_thinking.add(model)
        raise


def _carbon(file):
    """Walls/openings/furniture/ceiling. One corrective retry if the JSON fails validation; raises if it fails twice."""
    usage, total_ms = _usage0(), 0.0
    prompt = CARBON_PROMPT
    for attempt in (1, 2):
        resp, ms = _with_retry("carbon", lambda p=prompt: _generate(CARBON, file, p, CARBON_TIMEOUT_MS))
        total_ms += ms
        _add_usage(usage, resp)
        try:
            return CarbonScan.model_validate_json(resp.text or ""), (CARBON, usage), total_ms
        except ValidationError as e:
            log.warning("carbon validation failed attempt=%d: %s", attempt, str(e)[:300])
            prompt = CARBON_PROMPT + CARBON_RETRY_SUFFIX.replace("{error}", str(e)[:1500])
    raise ValueError("Carbon output failed validation twice")


def _neon(file):
    usage = _usage0()

    def call():
        resp = _generate(NEON, file, NEON_PROMPT, NEON_TIMEOUT_MS)
        _add_usage(usage, resp)
        return NeonDims.model_validate_json(resp.text or "")  # invalid JSON -> exception -> the one retry

    dims, ms = _with_retry("neon", call)
    return dims, (NEON, usage), ms


def _decisions(jpegs):
    inp = [{"type": "image", "data": b, "mime_type": "image/jpeg", "resolution": "low"} for b in jpegs]
    inp.append({"type": "text", "text": DECISIONS_CONTEXT})
    qs = {k: {"type": "noul", "instructions": q} for k, q in DECISION_QS.items()}
    resp, ms = _with_retry("decisions", lambda: _get_client().decisions.create(
        model=DECISIONS, input=inp, questions=qs, timeout=DECISIONS_TIMEOUT_S))
    probs = {k: float(resp.answers[k].noul) for k in DECISION_QS}
    tokens = (resp.usage.total_input_tokens or 0) if getattr(resp, "usage", None) else 0
    return probs, ("gemini-decision-1", {"input_tokens": tokens, "output_tokens": 0, "thought_tokens": 0, "calls": 1}), ms


def _usage0():
    return {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "calls": 0}


def _add_usage(usage, resp):
    u = getattr(resp, "usage_metadata", None)
    usage["calls"] += 1
    if u:
        usage["input_tokens"] += u.prompt_token_count or 0
        usage["output_tokens"] += u.candidates_token_count or 0
        usage["thought_tokens"] += u.thoughts_token_count or 0


# ---------- geometry (deterministic) ----------

def _walk(walls):
    """Vertices (ft, y down) walking each wall's best length from (0,0) heading east, turning after each wall.
    ⚠️ the turn sign convention isn't pinned down: all-(-90) walks a mirror image (same dims and area)."""
    pts, heading = [(0.0, 0.0)], 0.0
    for w in walls:
        x, y = pts[-1]
        a = math.radians(heading)
        pts.append((x + w.length_ft.best * math.cos(a), y + w.length_ft.best * math.sin(a)))
        heading += w.turn_deg_to_next
    return pts


def _close(pts, perimeter):
    """Spread the closure error linearly along the perimeter so the loop ends where it started."""
    ex, ey = pts[-1]
    out, run = [], 0.0
    for i, (x, y) in enumerate(pts[:-1]):
        if i:
            run += math.dist(pts[i - 1], pts[i])
        out.append((x - ex * run / perimeter, y - ey * run / perimeter))
    return out


def _polygon(out):
    """(vertices ft, source, confidence_before). Carbon's walls if they close; else Neon's (or Carbon's) rectangle."""
    sc, car, neo = out["scan"], out["scan"]["_carbon"], out["scan"]["_neon"]
    walls = car.walls if car else []
    if walls:
        pts = _walk(walls)
        per = sum(w.length_ft.best for w in walls)
        closure = math.hypot(*pts[-1])
        sc["closure_error_ft"], sc["perimeter_ft"] = round(closure, 2), round(per, 2)
        right_angles = all(abs(abs(w.turn_deg_to_next) - 90) <= TURN_TOL_DEG for w in walls)
        if len(walls) >= 4 and right_angles and closure <= CLOSE_TOL * per:
            return _close(pts, per), "carbon_walls", 1 - closure / per
    if neo:
        w, d = neo.width_ft.best, neo.depth_ft.best
    elif walls:
        xs, ys = zip(*_walk(walls))
        w, d = max(xs) - min(xs), max(ys) - min(ys)
    else:
        return None, None, 0.0
    if w <= 0 or d <= 0:
        return None, None, 0.0
    out["flags"].append("fallback_rectangle")
    return [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)], "fallback_rectangle", FALLBACK_CONF


def _axis_ranges(out, verts, source):
    """dims {w, d}: best = polygon bounding box; low/high = envelope of Carbon's wall ranges and Neon's."""
    xs, ys = zip(*verts)
    best = {"w": max(xs) - min(xs), "d": max(ys) - min(ys)}
    rng = {k: [v, v] for k, v in best.items()}
    car, neo = out["scan"]["_carbon"], out["scan"]["_neon"]
    if car and source == "carbon_walls":
        for axis, horizontal in (("w", True), ("d", False)):
            heading, lows, highs = 0.0, [], []
            for w in car.walls:
                if (abs(math.cos(math.radians(heading))) > 0.7) == horizontal:
                    lows.append(w.length_ft.low / w.length_ft.best)
                    highs.append(w.length_ft.high / w.length_ft.best)
                heading += w.turn_deg_to_next
            if lows:
                rng[axis] = [best[axis] * min(lows), best[axis] * max(highs)]
    if neo:  # Neon's width/depth aren't tied to our axes: match larger to larger
        big, small = sorted(("w", "d"), key=lambda a: -best[a])
        n_big, n_small = sorted((neo.width_ft, neo.depth_ft), key=lambda r: -r.best)
        for axis, r in ((big, n_big), (small, n_small)):
            rng[axis] = [min(rng[axis][0], r.low), max(rng[axis][1], r.high)]
    return {k: {"low": min(rng[k][0], best[k]), "best": best[k], "high": max(rng[k][1], best[k])} for k in best}


def _cross_check(out, dims, source):
    """True if the two models agree within 15% on both dimensions (orientation-free); else flag + False."""
    car, neo = out["scan"]["_carbon"], out["scan"]["_neon"]
    if source == "carbon_walls" and neo:
        other = (neo.width_ft.best, neo.depth_ft.best)
    elif source == "fallback_rectangle" and neo and car and car.walls:
        xs, ys = zip(*_walk(car.walls))
        other = (max(xs) - min(xs), max(ys) - min(ys))
    else:
        out["flags"].append("cross_check_unavailable")
        return False
    mine = sorted((dims["w"]["best"], dims["d"]["best"]))
    other = sorted(other)
    ok = all(o > 0 and abs(m - o) <= AGREE_TOL * o for m, o in zip(mine, other))
    out["flags"].append("two_model_agree" if ok else "two_model_disagree")
    return ok


def _widen(r):
    return None if r is None else {"low": r["low"] * (1 - WIDEN), "best": r["best"], "high": r["high"] * (1 + WIDEN)}


def _build(out):
    sc, flags = out["scan"], out["flags"]
    verts, source, conf = _polygon(out)
    sc["polygon_source"] = source
    car, neo, dec = sc["_carbon"], sc["_neon"], sc["_decisions"]
    sc["carbon_walls_raw"] = [w.model_dump() for w in car.walls] if car else None
    sc["neon_dims"] = neo.model_dump() if neo else None
    sc["references_used"] = sorted(set((car.references_used if car else []) + (neo.references_used if neo else [])))
    ceiling = car.ceiling_ft.model_dump() if car and car.ceiling_ft else None
    if verts is None:
        return _fail(out, "build", "no usable geometry from Carbon or Neon", "no_geometry")
    dims = _axis_ranges(out, verts, source)
    agree = _cross_check(out, dims, source)
    conf_after = conf
    if not agree:
        dims, ceiling, conf_after = {k: _widen(v) for k, v in dims.items()}, _widen(ceiling), conf_after * (1 - WIDEN)
    weak = [k for k, p in (dec or {}).items() if p < NOUL_MIN]
    if dec is None:
        flags.append("decisions_unavailable")
    if weak:
        flags.extend(WEAK_FLAG[k] for k in weak)
        dims, ceiling, conf_after = {k: _widen(v) for k, v in dims.items()}, _widen(ceiling), conf_after * (1 - WIDEN)
    closure_ok = source == "carbon_walls" and sc["closure_error_ft"] <= VERIFY_CLOSE_TOL * sc["perimeter_ft"]
    verified = agree and closure_ok and dec is not None and not weak
    sc["ceiling_ft"] = _round_range(ceiling)
    out["alignment_score_history"] = [round(conf, 3), round(conf_after, 3)]
    out["decision_checks"] = [{"room": "s1", "key": k, "question": DECISION_QS[k], "p_yes": round(p, 3)}
                              for k, p in (dec or {}).items()]
    _render(out, verts, dims, car if source == "carbon_walls" else None, car, verified, conf_after)
    return out


def _round_range(r):
    return None if r is None else {k: round(v, 2) for k, v in r.items()}


# ---------- placement + render ----------

def _segments(verts):
    """(start, unit direction, length, inward normal) per wall, in ft."""
    cx, cy = np.mean(verts, axis=0)
    segs = []
    for i, a in enumerate(verts):
        b = verts[(i + 1) % len(verts)]
        L = math.dist(a, b) or 1e-9
        u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        n = (-u[1], u[0])
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if (cx - mid[0]) * n[0] + (cy - mid[1]) * n[1] < 0:
            n = (-n[0], -n[1])
        segs.append((a, u, L, n))
    return segs


def _wall_index(car, wall_id, n_segs):
    """Carbon wall id -> segment index. On a fallback rectangle Carbon's i-th wall maps to side i % 4 (approximate)."""
    if car is None or wall_id is None:
        return None
    ids = [w.id for w in car.walls]
    return ids.index(wall_id) % n_segs if wall_id in ids else None


def _render(out, verts, dims, car_geom, car_any, verified, conf):
    xs, ys = zip(*verts)
    ox, oy = min(xs) - MARGIN_FT, min(ys) - MARGIN_FT
    px = lambda p: [round((p[0] - ox) * PX_PER_FT, 1), round((p[1] - oy) * PX_PER_FT, 1)]  # noqa: E731
    W = int(math.ceil((max(xs) - ox + MARGIN_FT) * PX_PER_FT))
    H = int(math.ceil((max(ys) - oy + MARGIN_FT) * PX_PER_FT))
    img = np.full((H, W, 3), 255, np.uint8)
    poly_px = [px(v) for v in verts]
    cv2.polylines(img, [np.round(poly_px).astype(np.int32)], True, (0, 0, 0), 4, cv2.LINE_AA)
    segs = _segments(verts)
    car = car_geom or car_any
    if car_any and car_geom is None and car_any.openings:
        out["flags"].append("openings_approximate")
    _openings(out, img, segs, car, px)
    _furniture(out, img, segs, car, verts, px)
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(verts, verts[1:] + verts[:1]))) / 2
    text = f"~ {dims['w']['best']:.1f} x {dims['d']['best']:.1f} ft"   # in the top margin, never over furniture
    scale_txt = min(0.5, 0.9 * W / max(1, cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]))
    cv2.putText(img, text, (6, int(PX_PER_FT * MARGIN_FT * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale_txt, (60, 60, 60), 1, cv2.LINE_AA)
    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="footprint_scan_")
    os.close(fd)
    cv2.imwrite(tmp, img)
    with open(tmp, "rb") as f:
        png = f.read()
    os.unlink(tmp)
    out["image"] = {"w": W, "h": H, "png_b64": base64.b64encode(png).decode()}
    out["area"]["from_geometry"] = round(area, 1)
    out["rooms"] = [{"id": "s1", "label": "Scanned room", "polygon": poly_px, "area_sqft": round(area, 1),
                     "dims_ft": {k: _round_range(v) for k, v in dims.items()}, "printed_dims_ft": None,
                     "verified": verified, "confidence": round(conf, 3), "method": "video_scan",
                     "flags": list(out["flags"])}]


def _openings(out, img, segs, car, px):
    for k, o in enumerate(car.openings if car else []):
        i = _wall_index(car, o.wall_id, len(segs))
        if i is None:
            out["flags"].append("opening_on_unknown_wall")
            continue
        a, u, L, n = segs[i]
        width = min(max(o.width_ft, 0.5), L)
        off = min(max(o.offset_ft_from_wall_start, 0.0), L - width)
        s = (a[0] + u[0] * off, a[1] + u[1] * off)
        e = (s[0] + u[0] * width, s[1] + u[1] * width)
        center = px(((s[0] + e[0]) / 2, (s[1] + e[1]) / 2))
        S, E = tuple(int(round(v)) for v in px(s)), tuple(int(round(v)) for v in px(e))  # cv2 wants plain ints
        cv2.line(img, S, E, (255, 255, 255), 6)  # gap in the wall
        if o.type == "door":
            r = width * PX_PER_FT
            arc = [(S[0] + r * (math.cos(t) * u[0] + math.sin(t) * n[0]), S[1] + r * (math.cos(t) * u[1] + math.sin(t) * n[1]))
                   for t in np.linspace(0, math.pi / 2, 16)]
            cv2.polylines(img, [np.round(arc).astype(np.int32)], False, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.line(img, S, (int(S[0] + r * n[0]), int(S[1] + r * n[1])), (40, 40, 40), 2, cv2.LINE_AA)
            out["doors"].append({"id": f"d{k + 1}", "center": center, "width_ft": round(width, 2), "hinge": None})
        else:
            for sgn in (-1, 1):
                d = (round(sgn * 3 * n[0]), round(sgn * 3 * n[1]))
                cv2.line(img, (S[0] + d[0], S[1] + d[1]), (E[0] + d[0], E[1] + d[1]), (0, 0, 0), 1, cv2.LINE_AA)
            out["windows"].append({"id": f"w{k + 1}", "center": center, "width_ft": round(width, 2), "on_exterior_wall": o.exterior})
    out["entrance"] = out["doors"][0]["center"] if out["doors"] else None


def _furniture(out, img, segs, car, verts, px):
    for f in car.furniture if car else []:
        if f.w_ft <= 0 or f.d_ft <= 0:
            continue
        i = _wall_index(car, f.wall_id, len(segs))
        if i is None:  # free-standing: centred, axis-aligned
            cx, cy = np.mean(verts, axis=0)
            corners = [(cx - f.w_ft / 2, cy - f.d_ft / 2), (cx + f.w_ft / 2, cy - f.d_ft / 2),
                       (cx + f.w_ft / 2, cy + f.d_ft / 2), (cx - f.w_ft / 2, cy + f.d_ft / 2)]
        else:
            a, u, L, n = segs[i]
            w = min(f.w_ft, L)
            o = min(max(f.offset_ft or 0.0, 0.0), L - w)
            g = max(f.distance_from_wall_ft or 0.0, 0.0)
            p = lambda s, t: (a[0] + u[0] * s + n[0] * t, a[1] + u[1] * s + n[1] * t)  # noqa: E731
            corners = [p(o, g), p(o + w, g), p(o + w, g + f.d_ft), p(o, g + f.d_ft)]
        poly = [px(c) for c in corners]
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(img, [pts], (205, 205, 205))
        cv2.polylines(img, [pts], True, (120, 120, 120), 1, cv2.LINE_AA)
        # no text on furniture: at 20 px/ft names don't fit and ran off the image (the UI shows them on hover)
        out["furniture"].append({"label": f.label, "polygon": poly, "h_ft": f.h_ft})
