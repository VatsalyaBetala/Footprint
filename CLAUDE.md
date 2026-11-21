# Footprint — Gemini 4 NYC Hackathon

## Idea
HEADLINE: "Is that 2BR actually a 2BR?" — NYC listings, checked against reality.
One flow, three steps (frontend/index.html is a single page in this order):
1. CHECK THE LISTING — floor plan image → rooms, real sq ft vs claim, every bedroom
   checked against NYC minimums (80 sq ft, 8 ft wide, exterior window). One verdict.
2. MEASURE A ROOM — 15–30 s phone video at the viewing (backend/scan.py, POST /scan)
   → width × depth ranges + ceiling height, compared with the plan's printed size.
   Fills the check a plan can't answer (ceiling ≥ 8 ft).
3. FIT YOUR FURNITURE — any plan or scanned room → furniture placed with walkway
   clearance, out of door swings, path to the entrance still clear (POST /query,
   voice via POST /voice). 2D + 3D view.
Pipeline (unchanged): Carbon reads semantics, OpenCV traces shapes, Decisions checks
outlines, Neon picks repairs, deterministic solvers answer. AI internals (decision
probabilities, alignment score) stay in model.json for eval but are NOT shown in the
UI; the UI shows one trust mark per room (confirmed / estimate) and honest ranges.

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

## Models (verified via client.models.list() + live calls, Nov 21)
- CARBON = "gemini-4-carbon"        # ✅ verified — plan reading, semantics
- NEON = "gemini-4-neon"            # ✅ verified — repair choice
- HYDROGEN = "gemini-4-hydrogen"    # ✅ verified — voice intent parser
- DECISIONS = "gemini-decision-1"   # ✅ verified — per-room verification (noul/choice/score)
- Also listed, unused: "gemini-4-argon"
- SDK: google-genai >= 3.2.0 (3.6.1 installed), client with http_options={"api_version": "v1beta"}
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
  "area": {"from_printed": 712, "from_geometry": 698, "listing_claim": 850,
           "claim_scope": "same_floor | ambiguous | null"},
  "alignment_score_history": [0.61, 0.88, 0.94],
  "answers": {
    "king_bed_fits": {"room": "r1", "fits": false, "reason": "..."},
    "clear_path": {"ok": true, "from": "r1", "to": "entrance"},
    "legal_bedrooms": [{"room": "r2", "legal": false,
      "checks": {"area_ge_80": false, "min_width_ge_8ft": true},
      "reason": "71 sq ft < 80 sq ft minimum"}],
    "custom_query": null
  },
  "decision_checks": [{"room": "r1", "question": "...", "p_yes": 0.94}],
  "cost_usd": 0.0,
  "timings_ms": {"parse": 0, "geometry": 0, "verify": 0, "solvers": 0}
}
```
Person B builds against `samples/mock_model.json` until /analyze works.
`claim_scope`: "same_floor" only when the claim clearly covers the floor shown; "ambiguous"
(e.g. one floor of a multi-floor listing) → show both numbers, no overstated/understated verdict;
null when there is no claim. Never call area overstated unless claim_scope = "same_floor".

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
1. Step 1: judge's floor plan (or samples/synthetic_2br.png) → verdict + bedroom cards.
2. Step 2: room video (scratch/videos/cad_bedroom.mp4 = 12×10 ft) → "Bedroom 1 matches
   the plan", ceiling height fills the bedroom card.
3. Step 3: king bed → doesn't fit (needs 12.3 × 9.7 ft) → 3D dollhouse; ask by voice.
4. One slide: measured accuracy (room scan: 12% mean error, truth inside range 3/3 on
   CAD rooms of known size) + where it breaks (no printed dims → no scale; Finnish labels).

## Impact upgrades (priority order)
1. Legal bedroom check (solvers.py, deterministic): every room labeled bedroom must have
   area >= 80 sq ft and min clear width >= 8 ft (shapely: shortest side of minimum rotated rectangle).
   Those two are the only bedroom rules: window and ceiling height are NOT checked (removed Nov 21).
   Headline output: "Listed 2BR → legally 1BR + home office".
2. Voice query: judge speaks "will a 7-foot sofa fit in the living room?" → audio to HYDROGEN →
   JSON {object, w_ft, d_ft, clearance_ft, room} → same fit solver → drawn on plan + 3D.
   HYDROGEN = "gemini-4-hydrogen"  # ⚠️ verify ID — fast intent parser, its only job.
3. Decisions panel in UI: per-room questions with live probabilities, plus latency + cost badge.
4. Proof slide line: "When Footprint says VERIFIED, it is right X% of the time (baseline: Y%)."
Stretch, in this order: Managed Agent wrapper, then Shopify fit, then napkin-sketch input.
