# Footprint

**Is that 2BR actually a 2BR?** Footprint checks an apartment listing against reality. Give it the listing's
floor plan and it rebuilds the apartment and checks every bedroom against NYC minimums. At the viewing,
film a room on your phone and it measures the room and builds an interactive 3D model, where you can
place your furniture by dragging or by asking in plain English.

Built at the Gemini 4 NYC Hackathon (Problem Statement 1).

## What you can do

| Step | You give it | You get |
|---|---|---|
| **1. Check the listing** | The floor plan image from the listing | Every room traced and measured; real square footage vs. the listing's claim; each bedroom checked against NYC minimums (at least 80 sq ft and 8 ft wide) |
| **2. Measure a room** | A 15–30 s phone video of one room (no floor plan needed) | Width × depth with honest ranges, ceiling height, a plan drawn from the video, and a comparison with the plan's printed size |
| **3. Fit your furniture** | A room, plus a bed, sofa or desk (presets, custom size, or by voice) | Every spot in all 8 orientations is tried; you get the best one that keeps the walkway, stays out of door swings and leaves the way to the entrance clear. If it only fits with a narrower walkway, it says by how much. Drag the piece, turn it or step through the other spots, and each one is re-checked; 2D and 3D views |

For a filmed room, the **interactive CAD view** shows every wall dimension and the ceiling height. Drag
furniture to see live gaps to the walls, or type requests such as *"put a queen bed under the window and
a desk by the door"* or *"show me other layouts"*. An agent places the pieces, a checker flags overlaps,
blocked door swings and walkways under 2 ft, and the agent suggests fixes. A fix is shown only if it actually reduces the problems when
tested on a copy of the layout.

## How it works

Models read and judge. **They never output final coordinates.** All geometry comes from OpenCV, shapely
and deterministic code.

**Floor plan** (`POST /analyze`)
1. `parse.py`: **Gemini 4 Carbon** reads room labels, printed dimensions, doors, windows and the listing claim.
2. `geometry.py`: OpenCV traces each room's exact outline from Carbon's seed points; scale comes from printed dimensions.
3. `verify.py`: the **Gemini Decisions API** checks each outline against the image; **Gemini 4 Neon** picks a repair from a fixed menu for failures. A repair is kept only if it improves the score.
4. `solvers.py`: deterministic answers: area two ways, legal-bedroom checks, furniture fit, clear path to the entrance.

**Room video** (`POST /scan`, `scan.py`)
- Carbon reads the wall sequence, openings and furniture from the video, using standard references (a 6'8" door, 4.5" outlet plates, a 36" countertop).
- Neon independently estimates width × depth.
- Decisions checks whether the keyframes show those references.
- The polygon is then built deterministically. It must close, it is cross-checked against Neon, and its ranges widen when the evidence is weak.

**Furniture agent** (`POST /cad_agent`, `agent.py`)
- **Gemini 4 Hydrogen** (low thinking, 5–15 s) reads the request, a screenshot of the 3D view, where every piece stands
  (wall + span), each wall's free spans, and the last few requests. For each piece it picks a wall, a position along it
  (or "next to" another piece) and a gap.
- `cad3d.js` computes the exact placement. If that spot overlaps a piece, a door swing or a wall, it slides the piece to the
  nearest free spot: the same wall first, then other walls, then turned, then open floor. It then runs the checks.

**Voice** (`POST /voice`): Hydrogen turns a spoken question into a fit query.

## Results

**Floor plans.** 50 held-out CubiCasa5K test plans (seed 42). Baseline B1 is a single Carbon call that
returns room polygons. Full results are in `eval/results.json`; the harness is `eval/run_eval.py`.

| Metric | Baseline B1 | Footprint v1 |
|---|---|---|
| Right when a room is marked VERIFIED | 78.6% | **82.0%** |
| Room count exactly right | 26% | **38%** |
| Room count error (mean) | 1.4 | **1.1** |
| Room type accuracy | 97.8% | 97.8% |
| Room shape IoU | **0.87** | 0.81 |
| Right, all rooms | **76.9%** | 73.0% |
| Latency (median) | **10.7 s** | 36.6 s |
| Cost per plan | **$0.012** | $0.13 |

The verify-and-repair loop makes VERIFIED trustworthy and fixes room counts. It doesn't make outlines
tighter, and it costs about 3× the time and 11× the money. On messy inputs (phone photos, rotated or
low-resolution plans) VERIFIED precision drops below the baseline; see `by_variant` in the results.

**Room videos.** Three rendered rooms of exactly known size, filmed from a corner with the door in frame:
mean error **12%**, with the true size inside the reported range in **3 of 3** rooms. The scanner tends
to underestimate, by 12% on larger rooms and 20% on a small office. Real-room footage needs its own
validation.

**Where it breaks**
- **No printed dimensions:** most CubiCasa plans have none, so there's no scale and no square footage. Footprint says "can't measure" instead of guessing.
- **Unjudged labels:** rooms labelled in other languages (e.g. Finnish *MH*) are recognised as bedrooms but not judged ("NOT CHECKED").
- **Listing claims:** a listing's square footage may cover other floors. Footprint shows both numbers and gives no verdict unless the claim clearly covers the floor shown.

## Run it

Requirements: Python 3.14 (any recent 3.x should work), a Gemini API key, and a modern browser.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # then set GEMINI_API_KEY=...
./run_server.sh                 # backend on http://localhost:8000 (RELOAD=1 for dev auto-reload)
python3 -m http.server 5500     # in a second terminal, from the repo root
open http://localhost:5500/frontend/
```

- **Try it:** "Try a sample" loads `samples/synthetic_2br.png`, a synthetic 2-bedroom plan.
- **Without a backend:** `?mock=1` loads the mock model.
- **Another backend:** `?api=http://host:port` points the page at it.

**Demo tip.** Results are cached by exact upload bytes plus a hash of the backend code. Repeat uploads
are instant (and survive restarts) until a `backend/*.py` file changes. After the last code change,
restart the backend and run:

```bash
./warm_demo.sh [plan.png] [room_video.mp4]   # defaults: samples/plan1.png, scratch/videos/hotel_room.mp4
```

### API

| Endpoint | Input | Output |
|---|---|---|
| `POST /analyze` | multipart `file`: floor plan image | `model.json` (contract in `CLAUDE.md`) |
| `POST /scan` | multipart `file`: room video (5–60 s) | `model.json` for the filmed room (dimension ranges, ceiling, plan PNG) |
| `POST /query` | JSON `{model_id, w_ft, d_ft, clearance_ft, room_label}` | verdict (`fits` / `tight` / `blocks_path` / `no_fit`), best spot, up to 12 other spots, clear path |
| `POST /check_placement` | JSON `{model_id, room_label, w_ft, d_ft, center_px, rotation_deg, clearance_ft}` | verdict for a spot you chose: inside the walls, door swings, walkway per side, clear path |
| `POST /voice` | multipart `audio` + `model_id` | spoken question → the same fit |
| `POST /cad_agent` | JSON `{text, walls, furniture, image_b64?, issues?, history?}` | placement actions (one layout at a time), fix suggestions |
| `GET /model/{id}`, `GET /health` | | stored model; module status and model warm-up |

Every failure comes back as JSON with an `errors` list, never as a 500 page. `/analyze` has a 45 s hard
deadline and returns a partial model if it's hit.

## Repo layout

```
backend/   parse.py · geometry.py · verify.py · solvers.py · scan.py · agent.py · server.py
frontend/  index.html (single page) · three3d.js (apartment dollhouse) · cad3d.js (interactive room CAD)
eval/      run_eval.py, results*.json, failure logs
samples/   demo plans + mock models (see samples/ATTRIBUTION.md)
```

## Data and licenses

- `samples/plan1.png` is from Wikimedia Commons ("Floor plan 1395 Sq.ft." by SmartVilla), CC BY-SA 4.0; see `samples/ATTRIBUTION.md`.
- CubiCasa5K (CC BY-NC 4.0) is used for evaluation only and is not included in this repo.
