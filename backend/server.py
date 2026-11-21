"""server.py — FastAPI: floor plan image -> model.json (CLAUDE.md contract + additions).

Run from repo root:  ./run_server.sh            (RELOAD=1 for dev auto-reload)
Deps: requirements.txt (python-multipart is needed for uploads).

POST /analyze   multipart "file" = plan image -> model.json  (hard deadline 45 s -> partial model)
POST /query     JSON {"model_id", "w_ft", "d_ft", "clearance_ft", "room_label"} -> fit + clear path
POST /voice     multipart "audio" (wav/ogg/webm) + "model_id" -> Hydrogen intent -> same as /query
GET  /model/{model_id}, GET /health (modules, startup warm-up per model)

Stage interfaces (img = decoded BGR array from cv2, the same pixels parse.py sent to Carbon).
A missing or failing module -> stub below, logged, listed in model["stubs_used"]:
  geometry.build_rooms(img, parsed) -> {"rooms": [parsed room + "polygon", "method", "flags"],
      "scale_px_per_ft": float|None, "scale_status": "ok|low_confidence|undetermined", "flags": [...]}
  verify.verify_and_repair(img, parsed, geo) -> {"rooms": [geo room + "verified", "confidence",
      "alignment", "flags"], "alignment_score_history": [...], "decision_checks": [...],
      "repairs": [...], "windows": [... + verified on_exterior_wall, null if unsure],
      "usage": {model name: {"input_tokens", "output_tokens", "thought_tokens", "calls"}} (priced
      here), "cost_usd": optional extra cost verify priced itself}   (matches verify.py as of 12:40)
verify gets geo exactly as geometry returned it (incl. "_masks" and per-room shapely "geom");
only explicit, JSON-safe fields reach model.json.
Failure policy: a stage failure goes to model["errors"] and the pipeline continues; anything
unhandled still comes back as JSON (never an HTML/plain 500 page).
"""
import base64
import copy
import glob
import hashlib
import importlib
import json
import logging
import math
import os
import re
import statistics
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # works from repo root or backend/
import solvers  # noqa: E402  (pure, no model calls; must exist)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("footprint.server")

CARBON = "gemini-4-carbon"
NEON = "gemini-4-neon"
HYDROGEN = "gemini-4-hydrogen"
DECISIONS = "models/gemini-decision-1"
ANALYZE_TIMEOUT_S = float(os.environ.get("ANALYZE_TIMEOUT_S", 45))  # hard deadline for /analyze -> partial model
HYDROGEN_TIMEOUT_MS = 30_000
WARMUP_TIMEOUT_MS = 20_000
VOICE_PROMPT = """You turn ONE spoken question about fitting furniture in an apartment into JSON.
Rooms on this floor plan (room_label must be copied exactly from this list): {rooms}

Return ONLY JSON with exactly these keys:
{{"transcript": "what the speaker said, verbatim",
  "object": short name, e.g. "king bed", "queen bed", "full bed", "sofa", "desk",
  "w_ft": number or null, "d_ft": number or null, "clearance_ft": number or null,
  "room_label": one label from the list above, or null}}

Rules:
- w_ft = width along the wall, d_ft = depth out from the wall, in feet. Convert inches / meters.
- Presets if no size is spoken: king bed 6.33 x 6.67, queen bed 5 x 6.67, full bed 4.5 x 6.25,
  sofa 7 x 3, desk 5 x 2.5. A spoken length overrides the preset width ("7-foot sofa" -> w_ft 7).
- clearance_ft: only if the speaker says one, else null.
- room_label: the listed room the speaker means ("living room" may be printed as "DRAWING / DINING",
  "the bedroom" -> the bedroom's label). null if no listed room matches.
"""
# object keyword -> (w_ft, d_ft); checked in this order ("sofa bed" is a sofa)
PRESETS = {"sofa": (7.0, 3.0), "couch": (7.0, 3.0), "desk": (5.0, 2.5),
           "king": (6.33, 6.67), "queen": (5.0, 6.67), "full": (4.5, 6.25), "double": (4.5, 6.25)}
BED_CLEARANCE_FT, OTHER_CLEARANCE_FT = 3.0, 2.5
KING_BED_FT = PRESETS["king"]
AUDIO_MIME = {".wav": "audio/wav", ".ogg": "audio/ogg", ".oga": "audio/ogg", ".webm": "audio/webm",
              ".mp3": "audio/mp3", ".flac": "audio/flac", ".m4a": "audio/aac", ".aac": "audio/aac"}
# ⚠️ uncertain — third-party reported prices (USD per 1M tokens, input/output), not Google's page.
# Carbon intro price is through Nov 30, 2026 ($4/$20 after). Thinking tokens billed as output.
# Decisions: $0.10 per 1M input, no output charge. Hydrogen price unknown -> its cost is null, never guessed.
PRICE_PER_MTOK = {CARBON: (2.00, 10.00), NEON: (3.00, 15.00), "gemini-decision-1": (0.10, 0.0)}
MAX_MODELS = 200              # in-memory models kept (oldest dropped)
MODELS = {}
HASH_CACHE = {}               # image-hash cache key -> model_id (also on disk)
# disk copy so a restart / reload doesn't lose models or the upload cache mid-demo
MODEL_DIR = os.path.join(tempfile.gettempdir(), "footprint_models")
# cache keys include a hash of the backend code, so any pipeline change invalidates old results
PIPELINE_VERSION = hashlib.sha1(b"".join(open(p, "rb").read() for p in sorted(glob.glob(os.path.join(HERE, "*.py"))))
                                ).hexdigest()[:8]
WARMUP = {"status": "not_started", "models": {}}


def _try_import(name, fn_name):
    """(function or None, why-not string). Any import error (missing, syntax error mid-edit) -> stub."""
    try:
        fn = getattr(importlib.import_module(name), fn_name, None)
        if fn is None:
            raise AttributeError(f"{name}.py has no {fn_name}()")
        log.info("%s.%s loaded", name, fn_name)
        return fn, None
    except Exception as e:
        log.warning("%s.%s unavailable (%s: %s) -> stub will be used", name, fn_name, type(e).__name__, e)
        return None, f"{type(e).__name__}: {e}"


def _make_client():
    """ONE genai client for Carbon, Decisions, Neon and Hydrogen (one connection pool, warmed at startup).
    Same options verify.py uses: v1beta, SDK auto-retry off — every caller retries once itself."""
    try:
        from google import genai
        from google.genai import types
        return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options=types.HttpOptions(
            api_version="v1beta", retry_options=types.HttpRetryOptions(attempts=1)))
    except Exception as e:
        log.warning("could not create genai client: %s: %s", type(e).__name__, e)
        return None


PARSE_FN, PARSE_WHY = _try_import("parse", "parse_plan")
GEOMETRY_FN, GEOMETRY_WHY = _try_import("geometry", "build_rooms")
VERIFY_FN, VERIFY_WHY = _try_import("verify", "verify_and_repair")
CLIENT = _make_client()
for _name in ("parse", "verify"):
    _mod = sys.modules.get(_name)
    if CLIENT is not None and _mod is not None and hasattr(_mod, "_client"):
        _mod._client = CLIENT  # ⚠️ relies on their lazy module-level `_client`; their files are untouched


@asynccontextmanager
async def _lifespan(app):
    threading.Thread(target=_warm_up, daemon=True, name="warmup").start()  # never delays startup
    yield


app = FastAPI(title="Footprint", lifespan=_lifespan)


@app.middleware("http")
async def _json_errors(request: Request, call_next):
    """Last resort: an unhandled exception becomes JSON, not a plain-text 500 page.
    Registered before CORS so CORS wraps it and error responses still carry CORS headers."""
    try:
        return await call_next(request)
    except Exception as e:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        err = f"{type(e).__name__}: {e}"[:300]
        return JSONResponse({"error": err, "errors": [{"stage": "server", "error": err}]}, status_code=500)


# Any origin, incl. "null" (frontend/index.html opened from file://) and http://localhost:<port>.
# No credentials are used, so "*" is allowed by browsers for all of these.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    model_id: str
    w_ft: float = Field(gt=0)
    d_ft: float = Field(gt=0)
    clearance_ft: float | None = Field(default=BED_CLEARANCE_FT, ge=0)
    room_label: str


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"ok": True, "models_in_memory": len(MODELS), "pipeline_version": PIPELINE_VERSION,
            "modules": {"parse": PARSE_WHY or "ok", "geometry": GEOMETRY_WHY or "ok",
                        "verify": VERIFY_WHY or "ok"},
            "warmup": WARMUP}


@app.post("/analyze")
def analyze(file: UploadFile | None = File(None), image: UploadFile | None = File(None)):
    """Plan image as multipart field "file" (contract) or "image" (what frontend/index.html sends)."""
    upload = file or image
    if upload is None:
        model = _empty_model(uuid.uuid4().hex[:12])
        model["errors"].append({"stage": "upload", "error": "send the plan as multipart field 'file' (or 'image')"})
        return JSONResponse(model, status_code=400)
    data = upload.file.read()
    key = _cache_key(data) if data else None
    hit = _cache_get(key) if key else None
    if hit is not None:  # same image bytes + same pipeline code as an earlier clean live run
        log.info("analyze cache hit %s -> %s", key, hit["model_id"])
        return JSONResponse({**hit, "cached": True, "answers": {**hit["answers"], "custom_query": None}})
    model = _empty_model(uuid.uuid4().hex[:12])
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        model["errors"].append({"stage": "upload", "error": "upload is not a readable image"})
        return JSONResponse(_jsonable(model), status_code=400)
    fd, path = tempfile.mkstemp(suffix=".img")  # cv2 detects format from content, not extension
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    out = _run_with_deadline(model, path, img, key)
    log.info("analyze %s: %d rooms, partial=%s, timings %s, errors %d, stubs %s", out["model_id"], len(out["rooms"]),
             out["partial"], out["timings_ms"], len(out["errors"]), [s["stage"] for s in out["stubs_used"]])
    return JSONResponse(out)


@app.get("/model/{model_id}")
def get_model(model_id: str):
    m = _get_model(model_id)
    return m if m else JSONResponse({"error": f"unknown model_id {model_id}"}, status_code=404)


@app.post("/query")
def query(q: Query):
    """Fit a w×d object in the named room, then re-check bedroom→entrance paths with it placed."""
    model = _get_model(q.model_id)
    if model is None:
        return JSONResponse({"error": f"unknown model_id {q.model_id}"}, status_code=404)
    clearance = BED_CLEARANCE_FT if q.clearance_ft is None else q.clearance_ft
    out = _jsonable({"query": q.model_dump(), **_fit_and_path(model, q.w_ft, q.d_ft, clearance, q.room_label)})
    model["answers"]["custom_query"] = out
    _save_model(model)
    return JSONResponse(out)


@app.post("/voice")
def voice(model_id: str = Form(...), audio: UploadFile = File(...)):
    """Spoken question -> Hydrogen JSON intent -> presets/defaults -> same fit + path as /query."""
    model = _get_model(model_id)
    if model is None:
        return JSONResponse({"error": f"unknown model_id {model_id}"}, status_code=404)
    data = audio.file.read()
    out = {"parsed_query": None, "transcript_guess": None, "fit": None, "path": None, "room": None,
           "label": None, "errors": [], "timings_ms": {}, "cost_usd": None, "cost_is_estimate": True}
    if not data:
        out["errors"].append({"stage": "upload", "error": "empty audio upload"})
        return JSONResponse(out, status_code=400)
    labels = [r.get("label") for r in model["rooms"]]
    res = _run(out, "hydrogen", _call_hydrogen, data, _audio_mime(audio), labels)
    if res is None:
        return JSONResponse(_jsonable(out))
    intent, usage = res
    q = _intent_to_query(intent)
    out.update(parsed_query=q, transcript_guess=intent.get("transcript"), cost_usd=_token_cost(HYDROGEN, usage))
    if not (q["w_ft"] and q["d_ft"] and q["room_label"]):
        out["errors"].append({"stage": "intent", "error": "couldn't tell the object size or the room from the audio"})
        return JSONResponse(_jsonable(out))
    r = _fit_and_path(model, q["w_ft"], q["d_ft"], q["clearance_ft"], q["room_label"])
    out.update(room=r["room"], label=r["label"], fit=r["fit"], path=r["clear_path"])
    out["errors"] += r["errors"]
    out["timings_ms"].update(r["timings_ms"])
    out = _jsonable(out)
    model["answers"]["custom_query"] = {**out, "query": q}
    _save_model(model)
    return JSONResponse(out)


# ---------- pipeline ----------

def _empty_model(model_id):
    """Every contract key present from the start, so a partial result is still a valid model.json."""
    return {"model_id": model_id, "cached": False, "partial": False, "image": None,
            "scale_px_per_ft": None, "scale_status": "undetermined",
            "rooms": [], "doors": [], "windows": [], "entrance": None, "listing_claim": None,
            "area": {"from_printed": None, "from_geometry": None, "listing_claim": None, "claim_scope": None},
            "alignment_score_history": [],
            "answers": {"king_bed_fits": None, "clear_path": None, "legal_bedrooms": [], "custom_query": None},
            "decision_checks": [], "repairs": [], "cost_usd": 0.0, "cost_is_estimate": True, "cost_breakdown": {},
            "timings_ms": {"parse": 0, "geometry": 0, "verify": 0, "solvers": 0},
            "geometry_flags": [], "stubs_used": [], "errors": []}


def _run_with_deadline(model, path, img, key):
    """Run the pipeline in a worker thread. It publishes a JSON snapshot after each stage; if it isn't done
    in ANALYZE_TIMEOUT_S, the latest snapshot goes back as a partial model (with a timeout error). The
    worker keeps going and, when done, replaces the stored model (same id) with the full result."""
    lock = threading.Lock()
    state = {"snap": _jsonable(model), "final": None, "late": False}

    def publish(m):
        snap = _jsonable(m)  # only the worker writes these dicts, so copying here is race-free
        with lock:
            state["snap"] = snap

    def work():
        try:
            _pipeline(model, path, img, publish)
        except Exception as e:  # bug in the glue itself: keep what we have
            log.exception("pipeline failed")
            model["errors"].append({"stage": "pipeline", "error": f"{type(e).__name__}: {e}"[:300]})
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        final = _jsonable(model)
        with lock:
            state["final"] = final
            if state["late"]:  # the request already got a partial: upgrade the stored copy
                _store(final, key)
                log.info("analyze %s finished late (%s); stored full model", final["model_id"], final["timings_ms"])

    worker = threading.Thread(target=work, daemon=True, name=f"analyze-{model['model_id']}")
    worker.start()
    worker.join(ANALYZE_TIMEOUT_S)
    with lock:
        if state["final"] is not None:
            out = state["final"]
        else:
            state["late"] = True
            out = {**state["snap"], "partial": True}
            out["errors"] = out["errors"] + [{"stage": "timeout", "error": (
                f"analysis exceeded {ANALYZE_TIMEOUT_S} s; returning the stages finished so far. The full "
                f"result replaces this model at GET /model/{out['model_id']} when it completes.")}]
        _store(out, key)
    return out


def _pipeline(model, path, img, publish):
    parsed = _parse(model, path, img)
    model["image"] = parsed.get("image") or {"w": img.shape[1], "h": img.shape[0]}
    model["listing_claim"] = parsed.get("listing_claim")
    publish(model)
    geo = _with_fallback(model, "geometry", GEOMETRY_FN, GEOMETRY_WHY, _stub_geometry, img, parsed)
    model["geometry_flags"] = _jsonable(geo.get("flags") or [])
    # provisional answer from unverified geometry: what a timeout during verify returns
    prov = copy.deepcopy(model)
    _assemble(prov, parsed, geo, _stub_verify(img, parsed, geo))
    _solve(prov, parsed, verify_ran=False)
    publish(prov)
    ver = _with_fallback(model, "verify", VERIFY_FN, VERIFY_WHY, _stub_verify, img, parsed, geo)
    _assemble(model, parsed, geo, ver)
    _solve(model, parsed, verify_ran=not any(s["stage"] == "verify" for s in model["stubs_used"]))
    publish(model)


def _assemble(model, parsed, geo, ver):
    """model.json fields from the stage outputs (explicit, JSON-safe fields only)."""
    scale = geo.get("scale_px_per_ft")
    scale = float(scale) if isinstance(scale, (int, float)) and scale > 0 else None
    model["scale_px_per_ft"] = round(scale, 3) if scale else None
    model["scale_status"] = (geo.get("scale_status") or "low_confidence") if scale else "undetermined"
    model["rooms"] = [_room_out(r, scale) for r in ver["rooms"]]
    model["doors"] = _jsonable(geo.get("doors") or parsed.get("doors") or [])
    model["windows"] = _jsonable(ver.get("windows") or parsed.get("windows") or [])
    model["entrance"] = _jsonable(geo.get("entrance") or parsed.get("entrance"))
    model["alignment_score_history"] = _jsonable(ver.get("alignment_score_history") or [])
    model["decision_checks"] = _jsonable(ver.get("decision_checks") or [])
    model["repairs"] = _jsonable(ver.get("repairs") or [])
    parts = {"parse": _token_cost(parsed.get("model"), parsed.get("usage"))}
    usage = ver.get("usage") if isinstance(ver.get("usage"), dict) else {}
    for name, u in usage.items():  # per model, e.g. gemini-decision-1, gemini-4-neon
        parts[f"verify:{name}"] = _token_cost(name, u) if isinstance(u, dict) else None
    if isinstance(ver.get("cost_usd"), (int, float)):
        parts["verify"] = ver["cost_usd"]
    model["cost_breakdown"] = parts  # null part = unknown; cost_usd sums the known parts
    model["cost_usd"] = round(sum(v for v in parts.values() if isinstance(v, (int, float))), 5)


def _parse(model, path, img):
    parsed = {"image": {"w": img.shape[1], "h": img.shape[0]}, "rooms": [], "doors": [], "windows": [],
              "entrance": None, "listing_claim": None, "model": None, "usage": None, "error": None}
    if PARSE_FN is None:
        model["errors"].append({"stage": "parse", "error": f"parse.py unavailable: {PARSE_WHY}"})
        return parsed
    parsed.update(_jsonable(_run(model, "parse", PARSE_FN, path) or {}))
    if parsed.get("error"):  # parse_plan reports Carbon failures instead of raising
        model["errors"].append({"stage": "parse", "error": parsed["error"]})
    return parsed


def _with_fallback(model, stage, real_fn, why_missing, stub_fn, *args):
    """Real stage if importable and it returns {"rooms": [...]}; otherwise the stub (logged)."""
    out = _run(model, stage, real_fn, *args) if real_fn else None
    if out is not None and not (isinstance(out, dict) and isinstance(out.get("rooms"), list)):
        model["errors"].append({"stage": stage, "error": f"returned {type(out).__name__}, need dict with 'rooms'"})
        out = None
    if out is None:
        why = f"{stage}.py unavailable ({why_missing})" if not real_fn else f"{stage}.py failed (see errors)"
        log.warning("%s: using STUB — %s", stage, why)
        model["stubs_used"].append({"stage": stage, "reason": why})
        out = _run(model, stage, stub_fn, *args) or {"rooms": []}
    return out  # raw on purpose: verify may need geometry's masks / shapely geoms


def _stub_geometry(img, parsed):
    """Fallback: Carbon's boxes as room polygons (flag carbon_box), scale from printed dims."""
    rooms = []
    for r in parsed.get("rooms") or []:
        b = r.get("box")
        poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]] if b else None
        rooms.append({**r, "polygon": poly, "method": "stub_carbon_box",
                      "flags": list(r.get("flags") or []) + ["carbon_box"]})
    scale = _scale_from_dims(rooms)
    # Carbon boxes are model-estimated coordinates -> never better than low_confidence
    return {"rooms": rooms, "scale_px_per_ft": scale, "scale_status": "low_confidence" if scale else "undetermined",
            "flags": ["geometry_stub"]}


def _scale_from_dims(rooms):
    """CLAUDE.md rule: median over rooms of sqrt(px_area / (w_ft*h_ft)); rooms >10% off get flagged."""
    est = {}
    for r in rooms:
        d, area = solvers._dims(r), solvers._poly(r).area
        if d and area > 0:
            est[r["id"]] = math.sqrt(area / (d[0] * d[1]))
    if not est:
        return None
    s = statistics.median(est.values())
    for r in rooms:
        if r["id"] in est and abs(est[r["id"]] - s) > 0.1 * s:
            r["flags"].append("scale_outlier")
    return s


def _stub_verify(img, parsed, geo):
    """Fallback: nothing verified, no Decisions / Neon calls, no repairs."""
    rooms = [{**r, "verified": False, "confidence": None} for r in geo.get("rooms") or []]
    return {"rooms": rooms, "alignment_score_history": [], "decision_checks": [], "repairs": [], "usage": {}}


def _room_out(r, scale):
    """Contract room fields (+ seed, method, alignment), JSON-safe: shapely geom etc. stay behind."""
    poly = _jsonable(r.get("polygon"))  # numpy array would break `polygon or []` in solvers
    area = round(solvers._poly({"polygon": poly}).area / scale ** 2, 1) if scale and poly else None
    return {"id": r["id"], "label": r.get("label"), "polygon": poly,
            "printed_dims_ft": _jsonable(r.get("printed_dims_ft")), "area_sqft": area,
            "verified": r.get("verified") is True, "confidence": _jsonable(r.get("confidence")),
            "alignment": _jsonable(r.get("alignment")),  # Decisions 0-3 wall-alignment score
            "flags": _jsonable(list(r.get("flags") or [])), "seed": _jsonable(r.get("seed")),
            "method": r.get("method")}


def _solve(model, parsed, verify_ran):
    """Each solver runs on its own: one failing never blocks the others.
    verify_ran=False (stub / provisional snapshot): windows are Carbon's unchecked guesses -> neutral headline."""
    rooms, doors, scale, ans = model["rooms"], model["doors"], model["scale_px_per_ft"], model["answers"]
    area = _run(model, "solvers.compute_area", solvers.compute_area, rooms, scale) or {}
    claim = (parsed.get("listing_claim") or {}).get("sqft")
    scope = parsed.get("claim_scope")
    if scope not in ("same_floor", "ambiguous"):
        scope = "ambiguous" if claim is not None else None  # ⚠️ never "same_floor" unless parse.py says so
    model["area"] = {"from_printed": area.get("from_printed"), "from_geometry": area.get("from_geometry"),
                     "listing_claim": claim, "claim_scope": scope, "agree": area.get("agree"),
                     "printed_coverage": area.get("printed_coverage")}
    ans["legal_bedrooms"] = _run(model, "solvers.legal_bedrooms", solvers.legal_bedrooms,
                                 rooms, model["windows"], scale) or []
    paths = _run(model, "solvers.clear_path", solvers.clear_path, rooms, doors, model["entrance"], scale)
    ans["clear_path"] = _path_summary(paths) if paths is not None else None
    ans["king_bed_fits"] = _run(model, "solvers.king_bed_fits", _king_bed, rooms, doors, scale)
    ans["bedroom_count"] = _run(model, "solvers.headline", _bedroom_headline,
                                ans["legal_bedrooms"], rooms, parsed.get("listing_claim"), verify_ran)


def _king_bed(rooms, doors, scale):
    """King bed with 3 ft clearance in the largest bedroom."""
    beds = [r for r in rooms if solvers._is_bedroom(r)]
    if not beds:
        return {"room": None, "fits": None, "placement": None, "reason": "no bedroom on plan"}
    r = max(beds, key=lambda b: solvers._poly(b).area)
    return {"room": r["id"], **solvers.fit(r, *KING_BED_FT, BED_CLEARANCE_FT, doors, scale)}


def _path_summary(per_room):
    """Contract shape {ok, from, to} for the worst bedroom, plus every bedroom under per_room."""
    if not per_room:
        return {"ok": None, "from": None, "to": "entrance", "path": None, "reason": "no bedroom on plan",
                "per_room": []}
    oks = [p["ok"] for p in per_room]
    ok = False if False in oks else (True if all(v is True for v in oks) else None)
    head = next((p for p in per_room if p["ok"] is ok), per_room[0])
    return {"ok": ok, "from": head["from"], "to": "entrance", "path": head["path"], "reason": head["reason"],
            "per_room": per_room}


def _bedroom_headline(legal, rooms, claim, verify_ran):
    """'Listed 2BR → legally 1BR + home office' ONLY if verify really ran (windows checked by Decisions)
    and every bedroom's verdict is trustworthy: judged (legal not None) AND measured from printed dims
    or traced-and-verified. Otherwise a neutral line that states sizes and makes no legal claim."""
    by_id = {r["id"]: r for r in rooms}
    listed = (claim or {}).get("bedrooms")
    trusted = [b for b in legal if b["legal"] is not None and (
        b.get("measurement_source") == "printed" or by_id.get(b["room"], {}).get("verified") is True)]
    out = {"listed": listed, "labelled": len(legal), "legal": None, "not_legal": None,
           "not_judged": len(legal) - len(trusted)}
    if verify_ran and legal and len(trusted) == len(legal):
        n_ok, n_bad = sum(b["legal"] is True for b in legal), sum(b["legal"] is False for b in legal)
        head = (f"Listed {listed}BR" if listed is not None else f"Plan shows {len(legal)}BR") + f" → legally {n_ok}BR"
        if n_bad:
            head += " + home office" if n_bad == 1 else f" + {n_bad} non-legal rooms"
        return {**out, "legal": n_ok, "not_legal": n_bad, "kind": "legal", "headline": head}
    sizes = [s for s in (_size_text(by_id.get(b["room"], {})) for b in legal) if s]
    head = f"Plan shows {len(legal) or 'no'} bedroom{'' if len(legal) == 1 else 's'}" + "".join(", " + s for s in sizes)
    if listed is not None:
        head = f"Listed {listed}BR; plan shows" + head[len("Plan shows"):]
    return {**out, "kind": "neutral", "headline": head}


def _size_text(room):
    """'10×11 ft' from printed dims, else '~101 sq ft' traced, else ''."""
    d = solvers._dims(room)
    if d:
        return f"{d[0]:g}×{d[1]:g} ft"
    return f"~{room['area_sqft']:.0f} sq ft" if room.get("area_sqft") else ""


def _fit_and_path(model, w_ft, d_ft, clearance_ft, room_label):
    """Shared by /query and /voice: fit in the matched room, then clear_path with the object placed."""
    out = {"room": None, "label": None, "fit": None, "clear_path": None, "errors": [], "timings_ms": {}}
    room = _find_room(model["rooms"], room_label)
    if room is None:
        labels = [r.get("label") for r in model["rooms"]]
        out["errors"].append({"stage": "room", "error": f"no room matching {room_label!r}; rooms: {labels}"})
        return out
    out["room"], out["label"] = room["id"], room.get("label")
    scale, doors = model["scale_px_per_ft"], model["doors"]
    out["fit"] = _with_pose(_run(out, "fit", solvers.fit, room, w_ft, d_ft, clearance_ft, doors, scale), w_ft, d_ft, scale)
    placed = (out["fit"] or {}).get("placement")
    paths = _run(out, "clear_path", solvers.clear_path, model["rooms"], doors, model["entrance"], scale,
                 [placed] if placed else None)
    out["clear_path"] = _path_summary(paths) if paths is not None else None
    return out


def _with_pose(fit, w_ft, d_ft, scale):
    """Add center_px + rotation_deg (what the frontend draws: a w×d rect, w along x at 0°, rotated about
    its centre) to a fit that has a placement. rotation_deg = 90 when the solver turned the object."""
    p = (fit or {}).get("placement")
    if not p or not scale:
        return fit
    xs, ys = [q[0] for q in p], [q[1] for q in p]
    x_extent = max(xs) - min(xs)
    rot = 0 if abs(x_extent - w_ft * scale) <= abs(x_extent - d_ft * scale) else 90
    return {**fit, "center_px": [round(sum(xs) / len(xs), 1), round(sum(ys) / len(ys), 1)], "rotation_deg": rot}


def _find_room(rooms, label):
    """Exact label/id match first, else most query words in the label; ties -> largest room."""
    q = (label or "").strip().lower()
    words = [w for w in re.findall(r"[a-z0-9]+", q) if w not in {"the", "a", "my", "room"}]

    def score(r):
        lab = (r.get("label") or "").lower()
        return (lab == q or r["id"].lower() == q, sum(w in lab for w in words), solvers._poly(r).area)

    best = max(rooms, key=score, default=None)
    return best if best is not None and any(score(best)[:2]) else None


# ---------- models: warm-up + voice ----------

def _genai():
    global CLIENT
    if CLIENT is None:
        CLIENT = _make_client()
        if CLIENT is None:
            raise RuntimeError("google-genai client unavailable")
    return CLIENT


def _warm_up():
    """One tiny call per model at startup, in parallel: opens connections and shows in /health which
    models answer. Runs in a background thread; failures are only reported, never raised."""
    from google.genai import types
    WARMUP["status"] = "running"
    tiny_png = base64.b64encode(cv2.imencode(".png", np.full((16, 16, 3), 255, np.uint8))[1]).decode()
    cfg = types.GenerateContentConfig(http_options=types.HttpOptions(timeout=WARMUP_TIMEOUT_MS))

    def gen(name):
        return lambda: _genai().models.generate_content(model=name, contents="Reply with the word OK.", config=cfg)

    def decide():
        return _genai().decisions.create(
            model=DECISIONS, timeout=WARMUP_TIMEOUT_MS / 1000,
            input=[{"type": "image", "data": tiny_png, "mime_type": "image/png", "resolution": "low"},
                   {"type": "text", "text": "Warm-up image."}],
            questions={"ping": {"type": "noul", "instructions": "The image is mostly white."}})

    calls = {CARBON: gen(CARBON), "gemini-decision-1": decide, NEON: gen(NEON), HYDROGEN: gen(HYDROGEN)}
    with ThreadPoolExecutor(len(calls)) as ex:
        futs = {name: ex.submit(_timed, fn) for name, fn in calls.items()}
        for name, f in futs.items():
            WARMUP["models"][name] = f.result()
    WARMUP["status"] = "done"
    log.info("warm-up: %s", {k: (v["ok"], v["ms"]) for k, v in WARMUP["models"].items()})


def _timed(fn):
    t = time.perf_counter()
    try:
        fn()
        return {"ok": True, "ms": round((time.perf_counter() - t) * 1000), "error": None}
    except Exception as e:
        return {"ok": False, "ms": round((time.perf_counter() - t) * 1000), "error": f"{type(e).__name__}: {e}"[:200]}


def _audio_mime(upload):
    """Browser content type if it's audio (params like ;codecs=opus dropped), else from the extension.
    ⚠️ uncertain — audio/webm (Chrome MediaRecorder) isn't on Gemini's documented audio list
    (wav, mp3, aiff, aac, ogg, flac); if Hydrogen rejects it, record WAV in the browser."""
    ct = (upload.content_type or "").split(";")[0].strip().lower()
    if ct.startswith("audio/"):
        return "audio/wav" if ct in ("audio/x-wav", "audio/wave") else ct
    return AUDIO_MIME.get(os.path.splitext(upload.filename or "")[1].lower(), "audio/wav")


def _call_hydrogen(audio, mime, labels):
    """One Hydrogen call + one retry, latency logged. Returns (intent dict, usage dict|None)."""
    from google.genai import types
    last = None
    for attempt in (1, 2):
        t = time.perf_counter()
        try:
            resp = _genai().models.generate_content(
                model=HYDROGEN,
                contents=[types.Part.from_bytes(data=audio, mime_type=mime),
                          VOICE_PROMPT.format(rooms=json.dumps(labels))],
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   http_options=types.HttpOptions(timeout=HYDROGEN_TIMEOUT_MS)))
            intent = json.loads(resp.text)
            if not isinstance(intent, dict):
                raise ValueError(f"Hydrogen returned {type(intent).__name__}, not an object")
            log.info("hydrogen ok attempt=%d %.0f ms mime=%s", attempt, (time.perf_counter() - t) * 1000, mime)
            u = resp.usage_metadata
            usage = {"input_tokens": u.prompt_token_count, "output_tokens": u.candidates_token_count,
                     "thought_tokens": u.thoughts_token_count} if u else None
            return intent, usage
        except Exception as e:
            log.warning("hydrogen failed attempt=%d %.0f ms: %s: %s", attempt,
                        (time.perf_counter() - t) * 1000, type(e).__name__, str(e)[:300])
            last = e
    raise last


def _intent_to_query(intent):
    """Hydrogen intent -> {object, w_ft, d_ft, clearance_ft, room_label}; presets / defaults applied here,
    deterministically, so a spoken '7-foot sofa' and a bare 'sofa' both get sizes."""
    obj = str(intent.get("object") or "").strip().lower()
    preset = next((v for k, v in PRESETS.items() if k in obj), None)

    def num(k, default=None):
        v = intent.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else default

    w, d = num("w_ft", preset and preset[0]), num("d_ft", preset and preset[1])
    c = intent.get("clearance_ft")
    c = float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) and c >= 0 else (
        BED_CLEARANCE_FT if "bed" in obj else OTHER_CLEARANCE_FT)
    return {"object": obj or None, "w_ft": w, "d_ft": d, "clearance_ft": c, "room_label": intent.get("room_label")}


# ---------- storage ----------

def _cache_key(data):
    return f"{hashlib.sha256(data).hexdigest()[:24]}_{PIPELINE_VERSION}"


def _cache_get(key):
    """Model for an earlier clean live run of the same image bytes + same pipeline code, else None."""
    mid = HASH_CACHE.get(key)
    if mid is None:
        try:
            with open(os.path.join(MODEL_DIR, f"hash_{key}.txt")) as f:
                mid = f.read().strip()
        except OSError:
            return None
    m = _get_model(mid)
    if m is not None:
        HASH_CACHE[key] = mid
    return m


def _store(model, key):
    """Save the model; register it in the upload cache only if it is a clean, complete, live result."""
    _save_model(model)
    if key and not model["errors"] and not model["partial"] and not model["stubs_used"]:
        HASH_CACHE[key] = model["model_id"]
        try:
            with open(os.path.join(MODEL_DIR, f"hash_{key}.txt"), "w") as f:
                f.write(model["model_id"])
        except OSError as e:
            log.warning("could not write upload cache entry: %s", e)


def _save_model(model):
    """Keep in memory (newest MAX_MODELS) and on disk; a failed disk write only logs."""
    MODELS[model["model_id"]] = model
    while len(MODELS) > MAX_MODELS:
        MODELS.pop(next(iter(MODELS)))
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, f"{model['model_id']}.json"), "w") as f:
            json.dump(model, f)
    except OSError as e:
        log.warning("could not save model %s to disk: %s", model["model_id"], e)


def _get_model(model_id):
    """Memory first, then the disk copy. Only 12-hex ids reach the filesystem (no path tricks)."""
    if model_id in MODELS:
        return MODELS[model_id]
    if not re.fullmatch(r"[0-9a-f]{12}", model_id or ""):
        return None
    try:
        with open(os.path.join(MODEL_DIR, f"{model_id}.json")) as f:
            MODELS[model_id] = json.load(f)
        return MODELS[model_id]
    except (OSError, ValueError):
        return None


# ---------- utils ----------

def _token_cost(model_name, usage):
    """USD for one call from token usage; None if usage or price unknown."""
    price = PRICE_PER_MTOK.get(model_name)
    if usage is None or not price:
        return None
    out_tok = (usage.get("output_tokens") or 0) + (usage.get("thought_tokens") or 0)
    return round(((usage.get("input_tokens") or 0) * price[0] + out_tok * price[1]) / 1e6, 6)


def _run(target, stage, fn, *args):
    """fn(*args); on exception log it, add {stage, error} to target["errors"], return None.
    Adds elapsed ms to target["timings_ms"][stage before the first '.']."""
    t = time.perf_counter()
    try:
        return fn(*args)
    except Exception as e:
        log.exception("%s failed", stage)
        target["errors"].append({"stage": stage, "error": f"{type(e).__name__}: {e}"[:300]})
        return None
    finally:
        k = stage.split(".")[0]
        target["timings_ms"][k] = target["timings_ms"].get(k, 0) + round((time.perf_counter() - t) * 1000)


def _jsonable(o):
    """Plain-JSON copy: numpy -> python, tuples -> lists, NaN/inf -> None (Starlette rejects NaN)."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o if o is None or isinstance(o, (str, int, float, bool)) else str(o)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
