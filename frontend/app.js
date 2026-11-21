// Footprint UI — renders model.json (see CLAUDE.md contract) over the plan image.
// Serve from repo root: `python3 -m http.server 5500` → http://localhost:5500/frontend/
// Backend URL: ?api=http://host:port (default http://localhost:8000). ?mock=1 skips the backend.

const params = new URLSearchParams(location.search);
const API = params.get("api") || "http://localhost:8000";
const MOCK_MODEL = "../samples/mock_model.json";
const MOCK_IMAGE = "mock_plan.svg"; // drawn to match mock_model.json (760×600); plan1.png is a different plan
const SVG_NS = "http://www.w3.org/2000/svg";

let model = null, imageUrl = MOCK_IMAGE, timers = [];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (n, d = 0) => (n == null ? "—" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d }));
const pill = (cls, text) => `<span class="pill ${cls}">${esc(text)}</span>`;
const roomById = (id) => model.rooms.find((r) => r.id === id);
const roomName = (id) => roomById(id)?.label || id;

// ---------- loading ----------
async function loadMock() {
  const res = await fetch(MOCK_MODEL);
  model = await res.json();
  imageUrl = MOCK_IMAGE;
  $("source").textContent = "mock model";
  render();
}

async function analyze(file) {
  imageUrl = URL.createObjectURL(file);
  $("source").textContent = "analyzing…";
  $("headline").textContent = "Reading the plan…";
  const form = new FormData();
  form.append("file", file); // matches server.py: analyze(file: UploadFile)
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 90000);
    const res = await fetch(`${API}/analyze`, { method: "POST", body: form, signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    model = await res.json();
    const errs = (model.errors || []).length;
    $("source").textContent = `live · ${API}${model.partial ? " · PARTIAL (deadline hit)" : ""}${errs ? ` · ${errs} stage error(s)` : ""}`;
  } catch (err) {
    console.warn("analyze failed, showing mock", err);
    model = await (await fetch(MOCK_MODEL)).json();
    imageUrl = MOCK_IMAGE;
    $("source").textContent = `backend unavailable (${err.message}) — showing mock`;
  }
  render();
}

// ---------- render ----------
function render() {
  timers.forEach(clearTimeout); timers = [];
  drawPlan();
  animateScore();
  renderHeadline();
  renderArea();
  renderAnswers();
  renderLegal();
  renderRooms();
  renderDecisions();
  renderMeta();
  // 3D is a separate ES module; if it hasn't loaded yet it picks this up on init.
  // A 3D failure (e.g. CDN blocked) must never take down the 2D page.
  try {
    if (window.Footprint3D) window.Footprint3D.build(model);
    else window.__pending3d = model;
  } catch (err) { console.warn("3D build failed", err); }
}

function el(tag, attrs, parent) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function roomStatus(r) {
  const bedFail = model.answers?.king_bed_fits?.fits === false && model.answers.king_bed_fits.room === r.id;
  const illegal = (model.answers?.legal_bedrooms || []).some((b) => b.room === r.id && b.legal === false);
  if (!r.verified) return "unverified";
  return bedFail || illegal ? "fail" : "ok";
}

function centroid(poly) {
  let x = 0, y = 0;
  poly.forEach(([px, py]) => { x += px; y += py; });
  return [x / poly.length, y / poly.length];
}

function drawPlan() {
  const svg = $("plan");
  svg.innerHTML = "";
  const { w, h } = model.image;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  el("image", { href: imageUrl, x: 0, y: 0, width: w, height: h }, svg);

  const steps = Math.max(1, (model.alignment_score_history || []).length);
  const stagger = Math.min(250, (steps * 700) / Math.max(1, model.rooms.length));

  model.rooms.forEach((r, i) => {
    if (!r.polygon?.length) return;
    const g = el("g", {}, svg);
    const poly = el("polygon", {
      points: r.polygon.map((p) => p.join(",")).join(" "),
      class: `room ${roomStatus(r)}`,
      style: `animation-delay:${i * stagger}ms`,
      "data-id": r.id,
    }, g);
    const [cx, cy] = centroid(r.polygon);
    const t = el("text", { x: cx, y: cy, "text-anchor": "middle", class: "room-label",
                           style: `opacity:0;animation:pop .35s ${i * stagger}ms forwards` }, g);
    t.textContent = r.label;
    poly.addEventListener("mousemove", (e) => showTip(e, r));
    poly.addEventListener("mouseleave", () => ($("tip").hidden = true));
  });

  (model.doors || []).forEach((d) => {
    if (d.center) el("circle", { cx: d.center[0], cy: d.center[1], r: 4, class: "door" }, svg);
  });
  if (model.entrance) el("circle", { cx: model.entrance[0], cy: model.entrance[1], r: 9, class: "entrance" }, svg);
}

function showTip(e, r) {
  const tip = $("tip");
  const wrap = tip.parentElement.getBoundingClientRect();
  const dims = r.printed_dims_ft ? `${r.printed_dims_ft[0]} × ${r.printed_dims_ft[1]} ft printed` : "no printed dims";
  tip.innerHTML = `<b>${esc(r.label)}</b><br>${fmt(r.area_sqft, 1)} sq ft measured · ${dims}<br>
    ${r.verified ? pill("ok", "verified") : pill("warn", "unverified")} conf ${fmt(r.confidence, 2)}
    ${r.flags?.length ? `<br><span class="reason">${esc(r.flags.join(", "))}</span>` : ""}`;
  tip.hidden = false;
  tip.style.left = `${Math.min(e.clientX - wrap.left + 14, wrap.width - 250)}px`;
  tip.style.top = `${e.clientY - wrap.top + 14}px`;
}

function highlight(id, on) {
  document.querySelectorAll(`.room[data-id="${id}"]`).forEach((p) => p.classList.toggle("hl", on));
}

function animateScore() {
  const hist = model.alignment_score_history || [];
  $("score-fill").style.width = "0"; $("score-num").textContent = "–"; $("score-steps").textContent = "";
  hist.forEach((s, i) => {
    timers.push(setTimeout(() => {
      $("score-fill").style.width = `${Math.round(s * 100)}%`;
      $("score-num").textContent = s.toFixed(2);
      $("score-steps").textContent = hist.slice(0, i + 1).map((v, j) => `iter ${j + 1}: ${v.toFixed(2)}`).join("  →  ");
    }, 300 + i * 900));
  });
}

// Surprise answer first: illegal bedroom > overstated area > bed doesn't fit.
function renderHeadline() {
  const a = model.answers || {}, area = model.area || {};
  const bedrooms = a.legal_bedrooms || [];
  const illegal = bedrooms.filter((b) => b.legal === false);
  const measured = area.from_geometry ?? area.from_printed;
  let html = "";
  if (illegal.length) {
    const n = bedrooms.length;
    html = `Listed ${n}BR → <span class="bad">legally ${n - illegal.length}BR</span> + ${illegal.length} other room${illegal.length > 1 ? "s" : ""}`;
  } else if (area.claim_scope === "same_floor" && area.listing_claim && measured && area.listing_claim > measured * 1.1) {
    const pct = Math.round((area.listing_claim / measured - 1) * 100);
    html = `Listed ${fmt(area.listing_claim)} sq ft → <span class="bad">measured ${fmt(measured)} sq ft</span> (${pct}% overstated)`;
  } else if (a.king_bed_fits?.fits === false) {
    html = `<span class="bad">A king bed doesn't fit</span> in the ${esc(roomName(a.king_bed_fits.room))}`;
  } else {
    html = `<span class="ok">Plan checks out</span>`;
  }
  timers.push(setTimeout(() => ($("headline").innerHTML = html), 300 + (model.alignment_score_history?.length || 1) * 900));
  $("headline").textContent = "";
}

function renderArea() {
  const a = model.area || {};
  let verdict = "";
  const measured = a.from_geometry ?? a.from_printed;
  if (model.scale_status === "undetermined") {
    verdict = pill("warn", "scale undetermined — geometry area not reported");
  } else if (a.listing_claim == null || a.claim_scope == null) {
    verdict = `<span class="reason">No listing claim on the plan.</span>`;
  } else if (a.claim_scope === "ambiguous") {
    verdict = `<span class="reason">Claim may cover more than this floor — no verdict.</span>`;
  } else if (measured) {
    const diff = (a.listing_claim - measured) / measured;
    verdict = diff > 0.1 ? pill("bad", `claim ${Math.round(diff * 100)}% over measured`)
            : diff < -0.1 ? pill("warn", `claim ${Math.round(-diff * 100)}% under measured`)
            : pill("ok", "claim matches");
  }
  $("area").innerHTML = `
    <div class="row"><span class="k">From printed dims</span><span class="big">${fmt(a.from_printed)}</span></div>
    <div class="row"><span class="k">From traced geometry</span><span class="big">${fmt(a.from_geometry)}</span></div>
    <div class="row"><span class="k">Listing claims</span><span class="big">${fmt(a.listing_claim)}</span></div>
    <div>${verdict}</div>
    <div class="reason">Scale ${fmt(model.scale_px_per_ft, 1)} px/ft · ${esc(model.scale_status)}</div>`;
}

function renderAnswers() {
  const a = model.answers || {};
  let html = "";
  if (a.king_bed_fits) {
    const b = a.king_bed_fits;
    html += `<div class="row" data-room="${esc(b.room)}"><span>King bed in ${esc(roomName(b.room))}</span>
      ${b.fits ? pill("ok", "fits") : pill("bad", "doesn't fit")}</div><div class="reason">${esc(b.reason)}</div>`;
  }
  if (a.clear_path) {
    const c = a.clear_path;
    html += `<div class="row" data-room="${esc(c.from)}"><span>Clear path ${esc(roomName(c.from))} → ${esc(c.to)}</span>
      ${c.ok ? pill("ok", "clear") : pill("bad", "blocked")}</div>`;
  }
  if (a.custom_query) html += `<div class="reason">${esc(JSON.stringify(a.custom_query))}</div>`;
  $("answers").innerHTML = html || `<span class="reason">No answers.</span>`;
  wireHover("answers");
}

function renderLegal() {
  const list = model.answers?.legal_bedrooms || [];
  const label = { area_ge_80: "≥ 80 sq ft", min_width_ge_8ft: "≥ 8 ft wide", exterior_window: "exterior window", ceiling_height: "ceiling height" };
  $("legal").innerHTML = list.length ? list.map((b) => `
    <div class="row" data-room="${esc(b.room)}"><span>${esc(roomName(b.room))}</span>${
      b.legal === true ? pill("ok", "legal") : b.legal === false ? pill("bad", "not legal") : pill("warn", "can't confirm")}</div>
    <div class="reason">${Object.entries(b.checks || {}).map(([k, v]) =>
      `${v === true ? "✓" : v === false ? "✗" : "?"} ${label[k] || k}${typeof v === "string" ? ` (${v.replaceAll("_", " ")})` : ""}`).join(" · ")}
      <br>${esc(b.reason)}</div>`).join("") : `<span class="reason">No rooms labeled bedroom.</span>`;
  wireHover("legal");
}

function renderRooms() {
  $("rooms").innerHTML = model.rooms.map((r) => `
    <div class="row" data-room="${esc(r.id)}"><span>${esc(r.label)}</span>
      <span>${fmt(r.area_sqft, 0)} sq ft ${r.verified ? pill("ok", "✓") : pill("warn", "?")}</span></div>`).join("");
  wireHover("rooms");
}

function renderDecisions() {
  const checks = model.decision_checks || [];
  const color = (p) => (p >= 0.8 ? "var(--ok)" : p >= 0.6 ? "var(--warn)" : "var(--bad)");
  $("decisions").innerHTML = checks.length ? checks.map((c) => `
    <div class="prob" data-room="${esc(c.room)}"><span>${esc(roomName(c.room))}: ${esc(c.question)}</span>
      <b>${Math.round(c.p_yes * 100)}%</b>
      <div class="bar"><div style="width:${c.p_yes * 100}%;background:${color(c.p_yes)}"></div></div></div>`).join("")
    : `<span class="reason">No checks.</span>`;
  wireHover("decisions");
}

function renderMeta() {
  const t = model.timings_ms || {};
  const total = Object.values(t).reduce((s, v) => s + (v || 0), 0);
  $("meta").innerHTML = `${(total / 1000).toFixed(1)} s total (${Object.entries(t).map(([k, v]) => `${k} ${fmt(v)} ms`).join(" · ")})
    <br>Cost $${(model.cost_usd ?? 0).toFixed(4)} · ${model.rooms.length} rooms · ${(model.doors || []).length} doors`;
}

function wireHover(containerId) {
  $(containerId).querySelectorAll("[data-room]").forEach((n) => {
    n.addEventListener("mouseenter", () => highlight(n.dataset.room, true));
    n.addEventListener("mouseleave", () => highlight(n.dataset.room, false));
  });
}

// ---------- wiring ----------
$("file").addEventListener("change", (e) => e.target.files[0] && analyze(e.target.files[0]));
$("replay").addEventListener("click", () => model && render());
loadMock();
