"""agent.py — turn a plain-English furniture request into placement ACTIONS for the CAD view.

plan_actions(client, text, walls, furniture) -> {"actions": [...], "say": str, "usage": {...}, "latency_ms": int}
The model never outputs coordinates (CLAUDE.md hard rule): it picks a wall id, a position along that wall
(start / middle / end), a gap from the wall and a size. frontend/cad3d.js turns that into exact placement
and checks overlaps and walkway gaps deterministically.
"""
import base64
import json
import logging
import re
import time
from typing import Literal

from google.genai import types
from pydantic import BaseModel, field_validator

AGENT_PROMPT = """You arrange furniture in a 3D model of ONE room. Turn the user's request into actions.

Walls (clockwise; "along" runs from the wall's start corner to its end corner):
{walls}
Furniture already in the room:
{furniture}

{shot}{issues}
User request: "{text}"

Return ONLY JSON:
{{"actions": [ACTION, ...],      // changes to make right now (may be empty)
  "options": [{{"name": short title, "why": one sentence, "actions": [ACTION, ...]}}],
        // ONLY if the user asks for other layouts / ideas / arrangements / transformations: 2-3 complete alternatives,
        // each starting from the CURRENT furniture (move / rotate / remove / add). Otherwise [].
  "suggestions": [{{"text": one sentence fix, "actions": [ACTION, ...]}}],
        // ONLY if problems are listed above: up to 3 concrete fixes. Otherwise [].
  "say": one short, friendly sentence}}
ACTION = {{"op": "add" | "move" | "rotate" | "remove",
          "target": exact label of an existing piece (for move / rotate / remove) or null,
          "label": short name for a new piece (for add) or null,
          "w_ft": width ALONG the wall, "d_ft": depth OUT from the wall, "h_ft": height  (add only; numbers),
          "wall": wall id (integer) or null for the middle of the room,
          "position": "start" | "middle" | "end",
          "gap_ft": distance from that wall, 0 = against it}}

Rules:
- Never output coordinates. Choose a wall, a position along it and a gap.
- "by the window" / "under the window" = the wall that has the window; "by the door" = the wall with the door,
  at the end of the wall nearest the door is fine but never block the door: if the door is in the middle, use start or end.
- Sizes when not given (ft, w x d x h): king bed 6.33x6.67x2, queen bed 5x6.67x2, full bed 4.5x6.25x2, twin bed 3.25x6.25x2,
  sofa 7x3x2.8, loveseat 5x3x2.8, armchair 3x3x3, desk 5x2.5x2.5, dining table 5x3x2.5, coffee table 4x2x1.5,
  dresser 5x1.6x2.8, nightstand 1.6x1.4x2, bookshelf 3x1x6, TV stand 5x1.5x2, wardrobe 4x2x6.5, crib 2.5x4.5x3.
- Beds: headboard against the wall (gap 0). Use "remove" before re-adding something the user wants replaced.
- If the request is not about furniture in this room, return no actions and say so.
"""

HYDROGEN = "gemini-4-hydrogen"   # the fast intent model; this is exactly its job
TIMEOUT_MS = 45_000        # screenshot + options/suggestions schema: replies take 10–25 s
log = logging.getLogger("footprint.agent")


class Action(BaseModel):
    op: Literal["add", "move", "rotate", "remove"]
    target: str | None = None
    label: str | None = None
    w_ft: float | None = None
    d_ft: float | None = None
    h_ft: float | None = None
    wall: int | None = None
    position: Literal["start", "middle", "end"] = "middle"
    gap_ft: float = 0.0

    @field_validator("op", "position", mode="before")
    @classmethod
    def lower(cls, v):
        return str(v).strip().lower() if v is not None else v

    @field_validator("gap_ft", mode="before")
    @classmethod
    def gap_default(cls, v):
        return 0.0 if v is None else v


class Plan(BaseModel):
    actions: list[Action] = []
    say: str = ""
    options: list[dict] = []
    suggestions: list[dict] = []


def _describe_walls(walls):
    lines = []
    for w in walls:
        things = [t for t, on in (("door", w.get("door")), ("window", w.get("window"))) if on]
        lines.append(f"- wall {w.get('id')}: {float(w.get('length_ft', 0)):.1f} ft long" + (f", has the {' and '.join(things)}" if things else ""))
    return "\n".join(lines) or "- (unknown)"


def _describe_furniture(furniture):
    return "\n".join(f"- \"{f.get('label')}\" {float(f.get('w_ft', 0)):.1f} x {float(f.get('d_ft', 0)):.1f} ft" for f in furniture) or "- (none)"


OP_SYN = {"place": "add", "put": "add", "insert": "add", "create": "add", "delete": "remove", "take": "remove",
          "turn": "rotate", "spin": "rotate", "shift": "move", "slide": "move", "relocate": "move"}
POS_SYN = {"left": "start", "beginning": "start", "first": "start", "center": "middle", "centre": "middle", "mid": "middle",
           "right": "end", "last": "end"}


def _num(v):
    """5, 5.0, "5", "5 ft", "5'" -> 5.0; anything else -> None."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    m = re.search(r"-?\d+(\.\d+)?", str(v)) if v is not None else None
    return float(m.group()) if m else None


def _clean(a):
    """Lenient normalisation of one action dict before validation (synonyms, "wall 0", "5 ft")."""
    if not isinstance(a, dict):
        return None
    a = dict(a)
    op = str(a.get("op") or "").strip().lower()
    a["op"] = OP_SYN.get(op, op)
    pos = str(a.get("position") or "middle").strip().lower()
    a["position"] = POS_SYN.get(pos, pos) if POS_SYN.get(pos, pos) in ("start", "middle", "end") else "middle"
    w = _num(a.get("wall"))
    a["wall"] = int(w) if w is not None else None
    for k in ("w_ft", "d_ft", "h_ft", "gap_ft"):
        a[k] = _num(a.get(k))
    a["gap_ft"] = a["gap_ft"] or 0.0
    return a


def _actions(items):
    """Valid actions from a list, dropping (and logging) only the broken ones."""
    out, dropped = [], []
    for a in items or []:
        try:
            out.append(Action.model_validate(_clean(a)))
        except Exception as e:
            dropped.append(f"{a!r:.120} -> {type(e).__name__}")
    if dropped:
        log.warning("agent dropped %d invalid action(s): %s", len(dropped), dropped)
    return out


def _parse(raw):
    """Plan from raw JSON text. Raises only if it isn't JSON at all (that triggers the retry)."""
    data = json.loads(raw or "")
    if isinstance(data, list):
        data = {"actions": data}
    plan = Plan(actions=_actions(data.get("actions")), say=str(data.get("say") or ""))
    plan.options = [{"name": str(o.get("name") or f"Option {i + 1}"), "why": str(o.get("why") or ""),
                     "actions": [a.model_dump() for a in _actions(o.get("actions"))]}
                    for i, o in enumerate(data.get("options") or []) if isinstance(o, dict)]
    plan.suggestions = [{"text": str(s.get("text") or "Suggested fix"), "actions": [a.model_dump() for a in _actions(s.get("actions"))]}
                        for s in data.get("suggestions") or [] if isinstance(s, dict)]
    plan.options = [o for o in plan.options if o["actions"]][:3]
    plan.suggestions = [s for s in plan.suggestions if s["actions"]][:3]
    return plan


def plan_actions(client, text, walls, furniture, image_b64=None, issues=None):
    """One Hydrogen call (+ one retry: without the screenshot, in case the image was the problem). Raises after two failures.
    image_b64: JPEG screenshot of the current 3D view (reference only). issues: problems found by the deterministic checker."""
    issues = [str(i) for i in (issues or [])][:12]
    prompt = AGENT_PROMPT.format(
        walls=_describe_walls(walls), furniture=_describe_furniture(furniture), text=text.replace('"', "'")[:500],
        shot="A screenshot of the current 3D view is attached for reference (blue labels = wall lengths).\n" if image_b64 else "",
        issues=("Problems the checker found in the current layout:\n" + "\n".join(f"- {i}" for i in issues) +
                "\nPut up to 3 fixes in \"suggestions\": move / rotate, swap to a smaller size (e.g. queen instead of king), or remove. "
                "Keep what the user asked for.\n") if issues else "")
    cfg = types.GenerateContentConfig(response_mime_type="application/json", http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    log.info("agent request: %r (walls=%d, furniture=%d, screenshot=%s, issues=%d)", text[:200], len(walls), len(furniture), bool(image_b64), len(issues))
    last = None
    for attempt in (1, 2):
        t = time.perf_counter()
        raw = None
        try:
            contents = prompt
            if image_b64 and attempt == 1:
                contents = [types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg"), prompt]
            resp = client.models.generate_content(model=HYDROGEN, contents=contents, config=cfg)
            raw = resp.text
            plan = _parse(raw)
            ms = round((time.perf_counter() - t) * 1000)
            log.info("agent ok attempt=%d %d ms say=%r actions=%s", attempt, ms, plan.say[:120],
                     [(a.op, a.label or a.target, a.wall, a.position) for a in plan.actions])
            u = resp.usage_metadata
            usage = {"input_tokens": u.prompt_token_count, "output_tokens": u.candidates_token_count,
                     "thought_tokens": u.thoughts_token_count} if u else None
            return {"actions": [a.model_dump() for a in plan.actions], "options": plan.options, "suggestions": plan.suggestions,
                    "say": plan.say, "usage": usage, "latency_ms": ms, "used_screenshot": bool(image_b64) and attempt == 1}
        except Exception as e:
            log.warning("agent failed attempt=%d %.0f ms: %s: %s | raw reply: %r", attempt, (time.perf_counter() - t) * 1000,
                        type(e).__name__, str(e)[:200], (raw or "")[:400])
            last = e
    raise last


if __name__ == "__main__":  # quick manual check: python backend/agent.py "put a queen bed under the window"
    import sys
    from google import genai
    from dotenv import load_dotenv
    import os
    load_dotenv()
    c = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={"api_version": "v1beta"})
    demo_walls = [{"id": 0, "length_ft": 12, "door": True}, {"id": 1, "length_ft": 10}, {"id": 2, "length_ft": 12, "window": True}, {"id": 3, "length_ft": 10}]
    print(json.dumps(plan_actions(c, " ".join(sys.argv[1:]) or "put a queen bed under the window and a desk by the door", demo_walls, []), indent=1))
