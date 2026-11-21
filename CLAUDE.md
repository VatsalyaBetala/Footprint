# Footprint — Gemini 4 NYC Hackathon

## Idea
Any apartment floor plan image → a verified digital model you can query.
Carbon reads labels/dimensions/doors, OpenCV traces exact room shapes, the
Decisions API verifies each room against the original image, Neon picks repairs
for failures, and an alignment score climbs live. Deterministic solvers then
answer: true square footage (two independent methods), furniture fit with
clearance, clear path to the entrance. 3D view in Three.js. Optional: Shopify
products that actually fit.

Problem Statement 1 (5 Sigma). Eval on 50 random CubiCasa5K plans vs a
single-Carbon-call baseline.

## Hard rules
- All project code written during the event (after 10:30 AM, Nov 21).
- Repo is public. Never commit `.env`, `data/`, dataset files, or API keys.
- Feature freeze 4:15 PM. Submission due 5:00 PM.
- Models never output final coordinates. Geometry comes from OpenCV/shapely;
  models read semantics, verify, and choose repairs from a fixed menu.
- No silent guessing: if scale can't be determined, say so in the output.

## Ownership (never edit the other person's folder)
- Person A — `backend/`: parse.py, geometry.py, verify.py, solvers.py, server.py
- Person B — `frontend/`, `eval/`: UI, overlay, score, 3D, eval + baseline, Shopify
- Shared — `CLAUDE.md`, `samples/`. Change the contract only together.

## Models (fill in from client.models.list())
- CARBON = "gemini-4-carbon"        # ⚠️ verify ID — plan reading, semantics
- NEON = "gemini-4-neon"            # ⚠️ verify ID — repair choice
- DECISIONS = "gemini-decision-1"   # per-room verification (noul/choice/score)
- SDK: google-genai >= 3.2.0, client with http_options={"api_version": "v1beta"}
- Decisions API is AI Studio (Developer API) only, not Vertex.

## Pipeline
1. parse.py — Carbon(image) → rooms[{label, seed[x,y], printed_dims_ft}],
   doors[{center, width_ft, hinge}], entrance[x,y], listing_sqft|null
2. geometry.py — adaptive threshold → open (drop thin lines) → close (seal door
   gaps) → flood fill from each seed → polygon. Scale = median over rooms of
   sqrt(pixel_area / (w_ft*h_ft)). Rooms >10% off median scale get a flag.
3. verify.py — draw outline on plan, crop per room, Decisions API (parallel,
   one call per room): single room? label inside? stops at walls? alignment
   score. Failures → Neon picks one repair: reseed | tighten_close | split |
   merge | use_carbon_box. Re-verify. Max 3 iterations. Record score history.
4. solvers.py — area from printed dims and from geometry; bed fit (grid search,
   0/90°, 3 ft clearance on 3 sides, avoid door swing arcs); clear path (erode
   free space by 18 in, check bedroom↔entrance connectivity).
5. server.py — FastAPI POST /analyze (image) → model.json (contract below).

## Contract: model.json
```json
{
  "image": {"w": 1200, "h": 900},
  "scale_px_per_ft": 38.2,
  "scale_status": "ok | low_confidence | undetermined",
  "rooms": [
    {"id": "r1", "label": "Bedroom", "polygon": [[0,0],[1,0],[1,1]],
     "printed_dims_ft": [11.3, 10.0], "area_sqft": 112.4,
     "verified": true, "confidence": 0.93, "flags": []}
  ],
  "doors": [{"id": "d1", "center": [0,0], "width_ft": 2.67, "hinge": "left"}],
  "entrance": [0, 0],
  "area": {"from_printed": 712, "from_geometry": 698, "listing_claim": 850},
  "alignment_score_history": [0.61, 0.88, 0.94],
  "answers": {
    "king_bed_fits": {"room": "r1", "fits": false, "reason": "..."},
    "clear_path": {"ok": true, "from": "r1", "to": "entrance"}
  },
  "timings_ms": {"parse": 0, "geometry": 0, "verify": 0, "solvers": 0}
}
```
Person B builds against `samples/mock_model.json` until /analyze works.

## Eval (eval/)
- 50 random CubiCasa5K plans (fixed seed, recorded in results).
- System vs baseline (single Carbon call returning room polygons).
- Metrics: room polygon IoU, room count accuracy, room type accuracy,
  door detection, alignment score per iteration, latency, cost per plan.
- Log every failure category; fix one, re-run, report before/after.

## Schedule
- 10:30–12:00 parse + geometry on 1 plan (A) | UI + overlay on mock (B)
- 12:30 checkpoint: OpenCV failing on 3/5 samples → fall back to Carbon boxes
- 12:00–2:30 verify loop + solvers (A) | 3D + score + baseline script (B)
- 2:30–3:30 fix eval failures (A) | run eval, results slide (B)
- 3:30–4:15 Managed Agent wrapper (A) | Shopify fit if ahead (B)
- 4:15–5:00 freeze, 1-min video, submit

## Coding rules for agents
- Simplest thing that works. No abstractions, frameworks or classes unless needed.
- Flag uncertain API usage inline: `# ⚠️ uncertain — verify this`.
- Every model call: timeout, one retry, log latency. One failed call must never crash the demo.
- Prompts live as constants at the top of the module that uses them.
- Coordinates: image pixels, origin top-left, [x, y]. Unknown values = null, never omitted.
- `samples/mock_model.json` must always match the contract.

## Demo (3 min)
1. Judge gives any listing floor plan.
2. Rooms appear, alignment score climbs.
3. Lead with a surprise answer (area overstated / bed doesn't fit).
4. 3D orbit.
5. One slide: system vs baseline + where it breaks + what we fixed.
