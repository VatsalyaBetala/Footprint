"""parse.py — Carbon reads a floor plan image -> semantics with pixel coordinates.

parse_plan(image_path) -> dict  (never raises on model failure; sets "error" instead)
CLI: python backend/parse.py <plan.png> [...]  -> prints JSON, writes scratch/parse_<name>.png
"""
import json
import logging
import os
import sys
import time

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types

CARBON_PROMPT = """You are reading an apartment floor plan image. Extract semantics only.

COORDINATES: every point is [x, y] and every box is [x1, y1, x2, y2], normalized to 0-1000
over the FULL image. x = horizontal distance from the LEFT edge, y = vertical distance from
the TOP edge. x always comes first.

Return ONLY JSON with exactly these keys:
{
  "rooms": [{"label": "Bedroom", "seed": [x, y], "box": [x1, y1, x2, y2], "printed_dims_ft": [a, b] or null}],
  "doors": [{"center": [x, y], "width_ft": number or null, "hinge": "left" or "right" or null}],
  "windows": [{"center": [x, y], "on_exterior_wall": true or false or null}],
  "entrance": [x, y] or null,
  "listing_claim": {"bedrooms": integer or null, "bathrooms": number or null, "sqft": number or null} or null,
  "dims_unit_printed": "imperial" or "metric" or null
}

Rules:
- rooms: one entry per room or labeled space (bedrooms, living, kitchen, bath, closets, halls,
  balconies). label = the printed name; if unlabeled, your best room type.
  seed = a point clearly INSIDE the room on empty floor: not on a wall, text, door swing, or fixture.
  box = tight box around the room's interior floor, wall to wall.
  printed_dims_ft = the dimensions PRINTED for that room, in decimal feet (11'3" -> 11.25;
  meters x 3.2808). null if no dimensions are printed for it. Never estimate dimensions.
- doors: every door opening, including the entrance door. center = middle of the opening.
  width_ft = opening width in feet if you can tell, else null.
  hinge = "left" if the door is hinged at the left/top end of the opening (in image coordinates),
  "right" if at the right/bottom end, null if no swing is drawn.
- windows: every window. on_exterior_wall = true if it sits on the outer wall of the apartment.
- entrance = center of the main entry door of the apartment, null if you cannot tell.
- listing_claim = only from text printed on the image (e.g. "2 BR / 1 BA, 850 SQ FT");
  convert m2 to sq ft. null if no such text is printed.
"""

CARBON = "gemini-4-carbon"
THINKING_LEVEL = "LOW"  # set None to use the model default; auto-dropped if the API rejects it
TIMEOUT_MS = 60_000

log = logging.getLogger("footprint.parse")
load_dotenv()
_client = None
_use_thinking = True


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={"api_version": "v1beta"})
    return _client


def _call_carbon(png_bytes):
    """One Carbon call with one retry. Returns (parsed_json, response, latency_ms). Raises on 2 failures."""
    global _use_thinking
    last_err = None
    for attempt in (1, 2):
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=TIMEOUT_MS),
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL) if (THINKING_LEVEL and _use_thinking) else None,
        )
        t = time.perf_counter()
        try:
            resp = _get_client().models.generate_content(
                model=CARBON,
                contents=[types.Part.from_bytes(data=png_bytes, mime_type="image/png"), CARBON_PROMPT],
                config=cfg,
            )
            data = json.loads(resp.text)
            ms = (time.perf_counter() - t) * 1000
            log.info("carbon ok attempt=%d %.0f ms thinking=%s", attempt, ms, cfg.thinking_config is not None)
            return data, resp, ms
        except Exception as e:
            ms = (time.perf_counter() - t) * 1000
            log.warning("carbon failed attempt=%d %.0f ms: %s: %s", attempt, ms, type(e).__name__, str(e)[:300])
            if cfg.thinking_config is not None and "thinking" in str(e).lower():
                _use_thinking = False  # parameter rejected -> drop it for this and later calls
            last_err = e
    raise last_err


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _pt(p, W, H):
    """Normalized [x, y] (0-1000) -> pixel [x, y] ints, clamped. None if malformed."""
    if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(_num(v) is not None for v in p)):
        return None
    x, y = (min(max(float(v), 0.0), 1000.0) for v in p)
    return [round(x * W / 1000), round(y * H / 1000)]


def _box(b, W, H):
    if not (isinstance(b, (list, tuple)) and len(b) == 4 and all(_num(v) is not None for v in b)):
        return None
    p1, p2 = _pt(b[:2], W, H), _pt(b[2:], W, H)
    x1, x2 = sorted([p1[0], p2[0]])
    y1, y2 = sorted([p1[1], p2[1]])
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _normalize(data, W, H):
    rooms = []
    for i, r in enumerate(data.get("rooms") or [], 1):
        seed, box = _pt(r.get("seed"), W, H), _box(r.get("box"), W, H)
        dims = r.get("printed_dims_ft")
        dims = [float(d) for d in dims] if (isinstance(dims, list) and len(dims) == 2
                                            and all((_num(d) or 0) > 0 for d in dims)) else None
        flags = []
        if seed is None and box is not None:
            seed = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
            flags.append("seed_from_box")
        elif seed and box and not (box[0] <= seed[0] <= box[2] and box[1] <= seed[1] <= box[3]):
            flags.append("seed_outside_box")
        rooms.append({"id": f"r{i}", "label": str(r.get("label") or "Room"), "seed": seed, "box": box,
                      "printed_dims_ft": dims, "flags": flags})
    doors = [{"id": f"d{i}", "center": _pt(d.get("center"), W, H), "width_ft": _num(d.get("width_ft")),
              "hinge": d.get("hinge") if d.get("hinge") in ("left", "right") else None}
             for i, d in enumerate(data.get("doors") or [], 1)]
    windows = [{"id": f"w{i}", "center": _pt(w.get("center"), W, H),
                "on_exterior_wall": w.get("on_exterior_wall") if isinstance(w.get("on_exterior_wall"), bool) else None}
               for i, w in enumerate(data.get("windows") or [], 1)]
    lc = data.get("listing_claim") if isinstance(data.get("listing_claim"), dict) else {}
    claim = {"bedrooms": _num(lc.get("bedrooms")), "bathrooms": _num(lc.get("bathrooms")), "sqft": _num(lc.get("sqft"))}
    if claim["bedrooms"] is not None:
        claim["bedrooms"] = int(claim["bedrooms"])
    unit = data.get("dims_unit_printed")
    return {
        "rooms": rooms, "doors": doors, "windows": windows,
        "entrance": _pt(data.get("entrance"), W, H),
        "listing_claim": claim if any(v is not None for v in claim.values()) else None,
        "dims_unit_printed": unit if unit in ("imperial", "metric") else None,
    }


def parse_plan(image_path):
    img = cv2.imread(image_path)  # applies EXIF orientation; we send exactly these pixels
    if img is None:
        raise ValueError(f"could not read image: {image_path}")
    H, W = img.shape[:2]
    out = {"image": {"w": W, "h": H}, "rooms": [], "doors": [], "windows": [], "entrance": None,
           "listing_claim": None, "dims_unit_printed": None, "model": CARBON, "latency_ms": None,
           "usage": None, "error": None}
    png = cv2.imencode(".png", img)[1].tobytes()
    try:
        data, resp, ms = _call_carbon(png)
        out.update(_normalize(data, W, H))
        u = resp.usage_metadata
        out["latency_ms"] = round(ms)
        out["usage"] = {"input_tokens": u.prompt_token_count, "output_tokens": u.candidates_token_count,
                        "thought_tokens": u.thoughts_token_count} if u else None
    except Exception as e:
        out["error"] = f"carbon_failed: {type(e).__name__}: {str(e)[:300]}"
    return out


COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (128, 128, 0), (0, 128, 128), (170, 110, 40)]


def draw_debug(image_path, parsed, out_path):
    img = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, r in enumerate(parsed["rooms"]):
        c = COLORS[i % len(COLORS)]
        if r["box"]:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            dims = "x".join(f"{d:g}" for d in r["printed_dims_ft"]) if r["printed_dims_ft"] else "no dims"
            cv2.putText(img, f"{r['id']} {r['label']} [{dims}]", (x1 + 4, y1 + 16), font, 0.45, c, 1, cv2.LINE_AA)
        if r["seed"]:
            cv2.circle(img, tuple(r["seed"]), 7, (255, 255, 255), -1)
            cv2.circle(img, tuple(r["seed"]), 5, c, -1)
    for d in parsed["doors"]:
        if d["center"]:
            cv2.circle(img, tuple(d["center"]), 9, (0, 140, 255), 2)
            w = f"{d['width_ft']:g}ft" if d["width_ft"] else "?ft"
            cv2.putText(img, f"{d['id']} {w} {d['hinge'] or '-'}", (d["center"][0] + 10, d["center"][1] - 6),
                        font, 0.4, (0, 110, 230), 1, cv2.LINE_AA)
    for w in parsed["windows"]:
        if w["center"]:
            x, y = w["center"]
            cv2.rectangle(img, (x - 7, y - 7), (x + 7, y + 7), (255, 120, 0), -1 if w["on_exterior_wall"] else 2)
    if parsed["entrance"]:
        cv2.drawMarker(img, tuple(parsed["entrance"]), (0, 170, 0), cv2.MARKER_STAR, 28, 2)
        cv2.putText(img, "ENTRANCE", (parsed["entrance"][0] + 14, parsed["entrance"][1] + 22), font, 0.5, (0, 150, 0), 2)
    head = f"claim={parsed['listing_claim']}  {parsed['latency_ms']} ms  err={parsed['error']}"
    cv2.putText(img, head[:140], (8, 18), font, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    scratch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratch")
    os.makedirs(scratch, exist_ok=True)
    for path in sys.argv[1:]:
        parsed = parse_plan(path)
        print(json.dumps(parsed, indent=1))
        out = os.path.join(scratch, f"parse_{os.path.splitext(os.path.basename(path))[0]}.png")
        draw_debug(path, parsed, out)
        print(f"debug image: {os.path.normpath(out)}")
