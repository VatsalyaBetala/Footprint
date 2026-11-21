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

Walls (clockwise; positions along a wall are measured in ft from its start corner):
{walls}
Furniture in the room RIGHT NOW (where each piece stands):
{furniture}
{history}
{shot}{issues}
User request: "{text}"

Return ONLY JSON:
{{"actions": [ACTION, ...],      // the changes to make now (may be empty)
  "suggestions": [{{"text": one sentence fix, "actions": [ACTION, ...]}}],
        // ONLY if problems are listed above: up to 2 fixes, each fixing as many of the problems as it can. Otherwise [].
  "say": one short, friendly sentence saying what you did}}
ACTION = {{"op": "add" | "move" | "rotate" | "remove",
          "target": exact label of an existing piece (move / rotate / remove) or null,
          "label": short name for a new piece (add) or null,
          "w_ft": width ALONG the wall, "d_ft": depth OUT from the wall, "h_ft": height  (add only; numbers),
          "wall": wall id (integer) or null for the open floor,
          "position": "start" | "middle" | "end"   (where along that wall),
          "next_to": exact label of an existing piece to stand right beside (same wall), or null,
          "gap_ft": distance from that wall, 0 = against it}}

Rules:
- Never output coordinates. Choose a wall, a position along it (or next_to) and a gap.
- The furniture list above is the current state: it already includes everything from earlier requests. Leave every piece
  you weren't asked to change where it is. "it" / "that" means the piece from the most recent request.
- Never stack pieces: put a piece on a free span of a wall that is at least as long as the piece, and pick the position
  (start / middle / end) that falls inside that span. If the wall the user named has no room, use the nearest wall that
  does and say so; only if no wall has room, use the open floor (wall null). Do it rather than refuse.
  (If a spot is taken anyway, the app slides the piece to the nearest free spot.)
- If the room already has a piece of the kind asked for (e.g. a bed when asked to "put a queen bed ..."), move / replace
  that one instead of adding a second, unless the user says "another", "second", "two" or similar.
- "by the window" / "under the window" = the wall that has the window; "by the door" = the wall with the door, never
  in front of the door itself. "next to X" / "beside X" = next_to X.
- If the user asks for another layout / arrangement / ideas: give ONE complete, different arrangement as "actions"
  (move / rotate the existing pieces; don't add or remove unless asked) and describe it in "say".
- Sizes when not given (ft, w x d x h): king bed 6.33x6.67x2, queen bed 5x6.67x2, full bed 4.5x6.25x2, twin bed 3.25x6.25x2,
  sofa 7x3x2.8, loveseat 5x3x2.8, armchair 3x3x3, desk 5x2.5x2.5, dining table 5x3x2.5, coffee table 4x2x1.5,
  dresser 5x1.6x2.8, nightstand 1.6x1.4x2, bookshelf 3x1x6, TV stand 5x1.5x2, wardrobe 4x2x6.5, crib 2.5x4.5x3.
- Beds: headboard against the wall (gap 0). Use "remove" before re-adding something the user wants replaced.
- If the request is not about furniture in this room, return no actions and say so.
"""

HYDROGEN = "gemini-4-hydrogen"   # the fast intent model; this is exactly its job
TIMEOUT_MS = 30_000        # first try (with the screenshot); low thinking replies in 5–10 s
RETRY_TIMEOUT_MS = 30_000  # second try, text only
# default (high) thinking spent ~5k thought tokens = 28–60 s and timed out; "low" answers in 5–7 s (measured Nov 21).
# The deterministic resolver + tested fixes in cad3d.js catch what a quicker answer gets wrong.
THINKING = types.ThinkingConfig(thinking_level="low")   # ⚠️ "minimal" is rejected for this model (400)
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
    next_to: str | None = None
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


def _spans(spans):
    return ", ".join(f"{float(a):.1f}-{float(b):.1f}" for a, b in spans if b - a >= 1) or "none"


def _describe_walls(walls):
    lines = []
    for w in walls:
        things = [t for t, on in (("door", w.get("door")), ("window", w.get("window"))) if on]
        free = w.get("free")
        lines.append(f"- wall {w.get('id')}: {float(w.get('length_ft', 0)):.1f} ft long"
                     + (f", has the {' and '.join(things)}" if things else "")
                     + (f"; free spans (ft from start): {_spans(free)}" if free is not None else ""))
    return "\n".join(lines) or "- (unknown)"


def _describe_furniture(furniture):
    out = []
    for f in furniture:
        line = f"- \"{f.get('label')}\" {float(f.get('w_ft') or 0):.1f} x {float(f.get('d_ft') or 0):.1f} ft"
        if f.get("wall") is not None and f.get("span_ft"):
            a, b = f["span_ft"]
            line += f": against wall {f['wall']}" if not f.get("gap_ft") else f": {float(f['gap_ft']):.1f} ft from wall {f['wall']}"
            line += f", occupying {float(a):.1f}-{float(b):.1f} ft along it"
        elif "wall" in f:
            line += ": in the open floor"
        out.append(line)
    return "\n".join(out) or "- (none)"


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
    nt = a.get("next_to")
    a["next_to"] = str(nt).strip() or None if isinstance(nt, (str, int, float)) and not isinstance(nt, bool) else None
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
    plan.options = [o for o in plan.options if o["actions"]][:1]
    if not plan.actions and plan.options:      # one layout only: an "option" reply becomes the actions
        plan.actions = _actions(plan.options[0]["actions"])
        plan.say = plan.say or f"{plan.options[0]['name']}: {plan.options[0]['why']}"
    plan.options = []
    plan.suggestions = [s for s in plan.suggestions if s["actions"]][:2]
    return plan


def plan_actions(client, text, walls, furniture, image_b64=None, issues=None, history=None):
    """One Hydrogen call (+ one retry: without the screenshot, in case the image was the problem). Raises after two failures.
    image_b64: JPEG screenshot of the current 3D view (reference only). issues: problems found by the deterministic checker."""
    issues = [str(i) for i in (issues or [])][:12]
    history = [str(h)[:300] for h in (history or [])][-6:]
    prompt = AGENT_PROMPT.format(
        history=("Earlier in this conversation (oldest first):\n" + "\n".join(f"- {h}" for h in history) + "\n") if history else "",
        walls=_describe_walls(walls), furniture=_describe_furniture(furniture), text=text.replace('"', "'")[:500],
        shot="A screenshot of the current 3D view is attached for reference (blue labels = wall lengths).\n" if image_b64 else "",
        issues=("Problems the checker found in the current layout:\n" + "\n".join(f"- {i}" for i in issues) +
                "\nPut up to 2 fixes in \"suggestions\": move / rotate to a free span, swap to a smaller size (e.g. queen instead of king), or remove. "
                "Keep what the user asked for.\n") if issues else "")
    cfgs = {n: types.GenerateContentConfig(response_mime_type="application/json", thinking_config=THINKING,
                                           http_options=types.HttpOptions(timeout=ms))
            for n, ms in ((1, TIMEOUT_MS), (2, RETRY_TIMEOUT_MS))}
    log.info("agent request: %r (walls=%d, furniture=%d, screenshot=%s, issues=%d, history=%d)", text[:200], len(walls),
             len(furniture), bool(image_b64), len(issues), len(history))
    last = None
    for attempt in (1, 2):
        t = time.perf_counter()
        raw = None
        try:
            contents = prompt
            if image_b64 and attempt == 1:
                contents = [types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg"), prompt]
            resp = client.models.generate_content(model=HYDROGEN, contents=contents, config=cfgs[attempt])
            raw = resp.text
            plan = _parse(raw)
            ms = round((time.perf_counter() - t) * 1000)
            log.info("agent ok attempt=%d %d ms say=%r actions=%s", attempt, ms, plan.say[:120],
                     [(a.op, a.label or a.target, a.wall, a.position, a.next_to) for a in plan.actions])
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
