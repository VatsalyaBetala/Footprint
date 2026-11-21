// Footprint CAD — interactive 3D model of a room measured from video (POST /scan result).
// Exposes window.FootprintCAD = { show(model, fit), add({label, w, d, h}) }.
// Walls with openings, 1 ft floor grid, every wall dimensioned, ceiling height; drag furniture (double-click rotates)
// with live distances to walls / furniture on all four sides. Units: feet. World x = px / scale, z = py / scale, y up.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const WALL_T = 0.35, DOOR_H = 6.67, SILL = 3, HEAD = 7, SNAP = 0.1;
let R, L, scene, cam, ctl, el, hud;
let poly = [], H = 8.5, pieces = [], sel = null, drag = null, liveDims = null, built = null, wallInfo = [], doorZones = [];
function dispose(obj) {        // remove from the scene AND drop its HTML labels (CSS2D elements outlive their objects)
  if (!obj) return; obj.traverse((o) => o.element?.remove()); obj.parent?.remove(obj);
}
const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), ray = new THREE.Raycaster(), ptr = new THREE.Vector2();

function init() {
  el = document.getElementById("cad3d"); hud = document.getElementById("cad-hud");
  R = new THREE.WebGLRenderer({ antialias: true }); R.setPixelRatio(Math.min(devicePixelRatio, 2)); R.shadowMap.enabled = true;
  el.prepend(R.domElement);
  L = new CSS2DRenderer(); L.domElement.className = "labels3d"; el.appendChild(L.domElement);
  cam = new THREE.PerspectiveCamera(45, 1, 0.1, 500);
  ctl = new OrbitControls(cam, R.domElement); ctl.enableDamping = true; ctl.maxPolarAngle = Math.PI / 2 - 0.03;
  new ResizeObserver(() => { const w = el.clientWidth, h = el.clientHeight; if (!w || !h) return;
    R.setSize(w, h); L.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix(); }).observe(el);
  R.domElement.addEventListener("pointerdown", onDown); R.domElement.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp); R.domElement.addEventListener("dblclick", onDbl);
  R.setAnimationLoop(() => {
    if (!scene || !el.offsetParent) return;      // hidden: don't burn the GPU
    ctl.update(); cutaway(); R.render(scene, cam); L.render(scene, cam);
  });
}
function cutaway() {   // dollhouse: hide the walls between the camera and the room instead of see-through walls (no flicker)
  for (const w of wallGroups) w.g.visible = (cam.position.x - w.mid[0]) * w.out[0] + (cam.position.z - w.mid[1]) * w.out[1] < 0.3;
}

// ---------- build ----------
function show(model, fit) {
  if (!model?.rooms?.[0]?.polygon) return;
  if (scene && model === built) return setFit(fit);   // same room: keep the camera and your arrangement, just update the fit
  if (scene) scene.traverse((o) => { o.geometry?.dispose(); if (o.material && o.material !== wallMat) [].concat(o.material).forEach((m) => m.dispose()); });
  const s = model.scale_px_per_ft || 20, r = model.rooms[0];
  const raw = r.polygon.map(([x, y]) => [x / s, y / s]);
  const c = raw.reduce((a, p) => [a[0] + p[0] / raw.length, a[1] + p[1] / raw.length], [0, 0]);
  const toW = ([x, y]) => [x / s - c[0], y / s - c[1]];
  poly = raw.map(([x, y]) => [x - c[0], y - c[1]]);
  H = model.scan?.ceiling_ft?.best || 8.5;
  scene = new THREE.Scene(); scene.background = new THREE.Color(0xf6f7f9);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa3ad, 1.7));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1); sun.position.set(12, 25, 8); sun.castShadow = true; scene.add(sun);
  L.domElement.replaceChildren(); pieces = []; sel = null; liveDims = null;
  floor(); walls(model, toW); dimensions();
  (model.furniture || []).forEach((f) => { const pts = (f.polygon || []).map(toW); if (pts.length >= 3) addPiece(f.label, pts, f.h_ft || 2.5, "found"); });
  const span = Math.max(...poly.map(([x, z]) => Math.hypot(x, z))) * 2 + 4;
  cam.position.set(span * 0.55, span * 0.75, span * 0.75); ctl.target.set(0, 1, 0); ctl.update();
  status(null); built = model; toWorld = toW; fitPiece = null; fitKey = undefined;
  setFit(fit);
}
let toWorld = null, fitPiece = null, fitKey, hoverPiece = null, wallGroups = [], curGroup = null;
function setFit(fit) {   // the solver's placement from "Will it fit?" as a draggable piece; replaced, never duplicated
  const key = fit?.poly ? JSON.stringify(fit.poly) + fit.fits : null;
  if (key === fitKey) return; fitKey = key;
  if (fitPiece) { if (sel === fitPiece) select(null); const i = pieces.indexOf(fitPiece); if (i >= 0) pieces.splice(i, 1); dispose(fitPiece); fitPiece = null; }
  if (fit?.poly && toWorld) { fitPiece = addPiece(fit.label || "your furniture", fit.poly.map(toWorld), 2, fit.fits ? "fit" : "nofit"); select(fitPiece); }
}

function floor() {
  const shape = new THREE.Shape(poly.map(([x, z]) => new THREE.Vector2(x, -z)));
  const f = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color: 0xeadfcd, roughness: 0.9 }));
  f.rotation.x = -Math.PI / 2; f.receiveShadow = true; scene.add(f);
  const xs = poly.map((p) => p[0]), zs = poly.map((p) => p[1]), g = [];
  for (let x = Math.ceil(Math.min(...xs)); x <= Math.max(...xs); x++) g.push(x, 0.01, Math.min(...zs), x, 0.01, Math.max(...zs));
  for (let z = Math.ceil(Math.min(...zs)); z <= Math.max(...zs); z++) g.push(Math.min(...xs), 0.01, z, Math.max(...xs), 0.01, z);
  const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.Float32BufferAttribute(g, 3));
  scene.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0xc9bfae, transparent: true, opacity: 0.6 })));
}

function edges() {   // [{a, b, u (unit along), n (inward normal), len}]
  return poly.map((a, i) => {
    const b = poly[(i + 1) % poly.length], len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1e-6;
    const u = [(b[0] - a[0]) / len, (b[1] - a[1]) / len]; let n = [-u[1], u[0]];
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; if (-mid[0] * n[0] - mid[1] * n[1] < 0) n = [-n[0], -n[1]];
    return { a, b, u, n, len };
  });
}

function walls(model, toW) {
  const E = edges();
  wallInfo = E.map((e, i) => ({ id: i, length_ft: +e.len.toFixed(1), door: false, window: false })); doorZones = []; wallGroups = [];
  const openings = [...(model.doors || []).map((d) => ({ ...d, kind: "door", w: d.width_ft || 2.67 })),
                    ...(model.windows || []).map((w) => ({ ...w, kind: "window", w: w.width_ft || 3 }))];
  E.forEach((e, i) => {
    const ops = [];
    for (const o of openings) {
      if (!o.center) continue;
      const p = toW(o.center), t = (p[0] - e.a[0]) * e.u[0] + (p[1] - e.a[1]) * e.u[1];
      const d = Math.abs((p[0] - e.a[0]) * e.n[0] + (p[1] - e.a[1]) * e.n[1]);
      if (d < 1.2 && t > -0.5 && t < e.len + 0.5) { ops.push({ ...o, t0: Math.max(0, t - o.w / 2), t1: Math.min(e.len, t + o.w / 2) }); wallInfo[i][o.kind] = true; }
    }
    ops.sort((x, y) => x.t0 - y.t0);
    curGroup = new THREE.Group(); scene.add(curGroup);
    wallGroups.push({ g: curGroup, out: [-e.n[0], -e.n[1]], mid: [(e.a[0] + e.b[0]) / 2, (e.a[1] + e.b[1]) / 2] });
    let t = 0;
    for (const o of ops) {
      if (o.t0 > t) wallPiece(e, t, o.t0, 0, H);
      if (o.kind === "door") { wallPiece(e, o.t0, o.t1, Math.min(DOOR_H, H - 0.2), H); doorSwing(e, o); }
      else { wallPiece(e, o.t0, o.t1, 0, SILL); wallPiece(e, o.t0, o.t1, Math.min(HEAD, H - 0.3), H); glass(e, o); }
      t = Math.max(t, o.t1);
    }
    if (t < e.len) wallPiece(e, t, e.len, 0, H);
  });
}
const wallMat = new THREE.MeshStandardMaterial({ color: 0xe8e4dc, roughness: 0.95 });   // solid; cutaway() hides the near walls
function wallPiece(e, t0, t1, y0, y1) {
  if (t1 - t0 < 0.02 || y1 - y0 < 0.02) return;
  const len = t1 - t0, mid = (t0 + t1) / 2;
  const m = new THREE.Mesh(new THREE.BoxGeometry(len, y1 - y0, WALL_T), wallMat);
  m.position.set(e.a[0] + e.u[0] * mid - e.n[0] * WALL_T / 2, (y0 + y1) / 2, e.a[1] + e.u[1] * mid - e.n[1] * WALL_T / 2);
  m.rotation.y = -Math.atan2(e.u[1], e.u[0]); m.receiveShadow = true;
  (curGroup || scene).add(m);
  const ed = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: 0x334155 }));
  ed.position.copy(m.position); ed.rotation.copy(m.rotation); (curGroup || scene).add(ed);
}
function doorSwing(e, o) {
  const r = o.t1 - o.t0, hinge = [e.a[0] + e.u[0] * o.t0, e.a[1] + e.u[1] * o.t0], pts = [];
  const zx = [hinge[0], hinge[0] + e.u[0] * r, hinge[0] + e.n[0] * r, hinge[0] + e.u[0] * r + e.n[0] * r];
  const zz = [hinge[1], hinge[1] + e.u[1] * r, hinge[1] + e.n[1] * r, hinge[1] + e.u[1] * r + e.n[1] * r];
  doorZones.push([Math.min(...zx), Math.max(...zx), Math.min(...zz), Math.max(...zz)]);   // square the door sweeps through
  for (let k = 0; k <= 16; k++) { const t = (k / 16) * Math.PI / 2;
    pts.push(new THREE.Vector3(hinge[0] + r * (Math.cos(t) * e.u[0] + Math.sin(t) * e.n[0]), 0.02, hinge[1] + r * (Math.cos(t) * e.u[1] + Math.sin(t) * e.n[1]))); }
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: 0x2563eb })));
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(hinge[0], 0.02, hinge[1]),
    new THREE.Vector3(hinge[0] + r * e.n[0], 0.02, hinge[1] + r * e.n[1])]), new THREE.LineBasicMaterial({ color: 0x2563eb })));
  label(`door ${r.toFixed(1)} ft`, [hinge[0] + e.u[0] * r / 2, DOOR_H + 0.4, hinge[1] + e.u[1] * r / 2], "tag");
}
function glass(e, o) {
  const len = o.t1 - o.t0, mid = (o.t0 + o.t1) / 2, top = Math.min(HEAD, H - 0.3);
  const g = new THREE.Mesh(new THREE.BoxGeometry(len, top - SILL, 0.06), new THREE.MeshStandardMaterial({ color: 0x9cc9f5, transparent: true, opacity: 0.45 }));
  g.position.set(e.a[0] + e.u[0] * mid - e.n[0] * WALL_T / 2, (SILL + top) / 2, e.a[1] + e.u[1] * mid - e.n[1] * WALL_T / 2);
  g.rotation.y = -Math.atan2(e.u[1], e.u[0]); (curGroup || scene).add(g);
  label(`window ${len.toFixed(1)} ft`, [g.position.x + e.n[0] * 0.4, top + 0.4, g.position.z + e.n[1] * 0.4], "tag");
}

// ---------- dimensions ----------
function dimLine(p0, p1, text, cls = "dim", parent = scene) {
  const grp = new THREE.Group(), v0 = new THREE.Vector3(...p0), v1 = new THREE.Vector3(...p1);
  const dir = v1.clone().sub(v0).normalize(), up = Math.abs(dir.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const tick = new THREE.Vector3().crossVectors(dir, up).normalize().multiplyScalar(0.18);
  const color = cls === "bad" ? 0xdc2626 : cls === "ok" ? 0x059669 : 0x1d4ed8;
  const pts = [v0, v1, v0.clone().add(tick), v0.clone().sub(tick), v1.clone().add(tick), v1.clone().sub(tick)];
  const geo = new THREE.BufferGeometry().setFromPoints(pts); geo.setIndex([0, 1, 2, 3, 4, 5]);
  grp.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, depthTest: false })));
  grp.add(label(text, v0.clone().add(v1).multiplyScalar(0.5).toArray(), cls, null));
  parent.add(grp); return grp;
}
function label(text, pos, cls = "dim", parent = scene) {
  const div = document.createElement("div"); div.className = `dim3d ${cls}`; div.textContent = text;
  const o = new CSS2DObject(div); o.position.set(...pos); if (parent) parent.add(o); return o;
}
function dimensions() {
  for (const e of edges()) {
    const off = 0.75, a = [e.a[0] + e.n[0] * off, 0.05, e.a[1] + e.n[1] * off], b = [e.b[0] + e.n[0] * off, 0.05, e.b[1] + e.n[1] * off];
    dimLine(a, b, `${e.len.toFixed(1)} ft`);
  }
  const e = edges()[0], c = [e.a[0] + (e.n[0] + e.u[0]) * 0.4, 0, e.a[1] + (e.n[1] + e.u[1]) * 0.4];
  dimLine(c, [c[0], H, c[2]], `ceiling ${H.toFixed(1)} ft`);
}

// ---------- furniture ----------
const COLORS = { bed: 0x9db4d8, sofa: 0x9fc29a, desk: 0xc9a77c, table: 0xc9a77c, chair: 0x7f8ea3, found: 0xcfc6b8, fit: 0x34d399, nofit: 0xf87171 };
function colorFor(label, kind) { const k = Object.keys(COLORS).find((c) => label.toLowerCase().includes(c)); return COLORS[kind === "found" ? (k || "found") : kind] ?? COLORS[k] ?? COLORS.found; }
function addPiece(labelText, pts, h, kind) {
  const xs = pts.map((p) => p[0]), zs = pts.map((p) => p[1]);
  const w = Math.max(...xs) - Math.min(...xs), d = Math.max(...zs) - Math.min(...zs);
  const m = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: colorFor(labelText, kind), roughness: 0.7 }));
  m.scale.set(w, h, d); m.position.set((Math.max(...xs) + Math.min(...xs)) / 2, h / 2, (Math.max(...zs) + Math.min(...zs)) / 2);
  m.castShadow = true; m.userData = { label: labelText, w, d, h };
  const ed = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: 0x334155 }));
  m.add(ed); m.userData.edges = ed;
  m.userData.tag = label(`${labelText} ${w.toFixed(1)}×${d.toFixed(1)}`, [0, 0.5 + 0.35 / h, 0], "piece", m);
  m.userData.tag.visible = false;            // shown only when selected / hovered: no pile-up of labels
  scene.add(m); pieces.push(m); return m;
}
function add(spec) {
  if (!scene) return;
  const m = addPiece(spec.label, [[-spec.w / 2, -spec.d / 2], [spec.w / 2, spec.d / 2]], spec.h || 2, "found");
  m.material.color.set(colorFor(spec.label, "x")); select(m);
}
function footprint(m) { const { w, d } = m.userData; return [m.position.x - w / 2, m.position.x + w / 2, m.position.z - d / 2, m.position.z + d / 2]; }
function inside([x, z]) {   // point in room polygon
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, zi] = poly[i], [xj, zj] = poly[j];
    if ((zi > z) !== (zj > z) && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) c = !c;
  }
  return c;
}
function rayDist(o, dir, self) {   // nearest wall or other piece along an axis-aligned ray
  let best = Infinity;
  const hitSeg = (a, b) => {
    const ex = b[0] - a[0], ez = b[1] - a[1], den = dir[0] * ez - dir[1] * ex; if (Math.abs(den) < 1e-9) return;
    const t = ((a[0] - o[0]) * ez - (a[1] - o[1]) * ex) / den, s = ((a[0] - o[0]) * dir[1] - (a[1] - o[1]) * dir[0]) / den;
    if (t > 1e-6 && s >= -1e-6 && s <= 1 + 1e-6) best = Math.min(best, t);
  };
  poly.forEach((a, i) => hitSeg(a, poly[(i + 1) % poly.length]));
  for (const p of pieces) { if (p === self) continue; const [x0, x1, z0, z1] = footprint(p);
    [[[x0, z0], [x1, z0]], [[x1, z0], [x1, z1]], [[x1, z1], [x0, z1]], [[x0, z1], [x0, z0]]].forEach(([a, b]) => hitSeg(a, b)); }
  return best;
}
function update(m) {
  dispose(liveDims);
  liveDims = new THREE.Group(); scene.add(liveDims);
  const [x0, x1, z0, z1] = footprint(m), y = 0.08;
  const out = [[x0, z0], [x1, z0], [x1, z1], [x0, z1]].some((p) => !inside(p));
  const overlap = pieces.find((p) => p !== m && (() => { const [a0, a1, b0, b1] = footprint(p); return x0 < a1 - 0.02 && x1 > a0 + 0.02 && z0 < b1 - 0.02 && z1 > b0 + 0.02; })());
  const gaps = [];
  for (const [o, dir] of [[[x0, (z0 + z1) / 2], [-1, 0]], [[x1, (z0 + z1) / 2], [1, 0]], [[(x0 + x1) / 2, z0], [0, -1]], [[(x0 + x1) / 2, z1], [0, 1]]]) {
    const t = rayDist(o, dir, m); if (!isFinite(t)) continue; gaps.push(t);
    dimLine([o[0], y, o[1]], [o[0] + dir[0] * t, y, o[1] + dir[1] * t], `${t.toFixed(1)} ft`, t < 2 ? "bad" : "ok", liveDims);
  }
  const { label: name, w, d } = m.userData, tight = gaps.filter((g) => g < 2).length;
  const state = out ? ["bad", "outside the room"] : overlap ? ["bad", `overlaps ${overlap.userData.label}`]
              : tight > 1 ? ["warn", "fits, but less than 2 ft of walkway on " + tight + " sides"] : ["ok", "fits with walkway space"];
  m.material.emissive?.set(state[0] === "bad" ? 0x7f1d1d : 0x000000);
  status(`${name} · ${w.toFixed(1)} × ${d.toFixed(1)} ft — ${state[1]}`, state[0]);
}
function select(m) {
  if (sel) { sel.userData.edges.material.color.set(0x334155); sel.userData.tag.visible = false; }
  sel = m; if (!m) { dispose(liveDims); liveDims = null; return status(null); }
  m.userData.edges.material.color.set(0x2563eb); m.userData.tag.visible = true; update(m);
}
function status(text, cls) {
  hud.className = `cad-hud ${cls || ""}`;
  hud.textContent = text || "Drag furniture to move it · double-click to rotate · gaps to walls update live";
}

// ---------- pointer ----------
function pick(ev) {
  const r = R.domElement.getBoundingClientRect();
  ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ptr, cam);
  return ray.intersectObjects(pieces, false)[0]?.object || null;
}
function floorHit() { const p = new THREE.Vector3(); return ray.ray.intersectPlane(floorPlane, p) ? p : null; }
function onDown(ev) {
  const m = pick(ev); if (!m) return;
  select(m); const p = floorHit(); if (!p) return;
  drag = { m, dx: m.position.x - p.x, dz: m.position.z - p.z }; ctl.enabled = false;
}
function onMove(ev) {
  if (!drag) {
    const h = pick(ev); R.domElement.style.cursor = h ? "grab" : "";
    if (hoverPiece && hoverPiece !== sel && hoverPiece !== h) hoverPiece.userData.tag.visible = false;
    hoverPiece = h; if (h) h.userData.tag.visible = true;
    return;
  }
  pick(ev); const p = floorHit(); if (!p) return;
  drag.m.position.x = Math.round((p.x + drag.dx) / SNAP) * SNAP; drag.m.position.z = Math.round((p.z + drag.dz) / SNAP) * SNAP;
  update(drag.m);
}
function onUp() { if (drag) { drag = null; ctl.enabled = true; } }
function onDbl(ev) {
  const m = pick(ev); if (!m) return;
  const { w, d } = m.userData; m.userData.w = d; m.userData.d = w; m.scale.set(d, m.scale.y, w);
  m.userData.tag.element.textContent = `${m.userData.label} ${d.toFixed(1)}×${w.toFixed(1)}`; select(m);
}

// ---------- agent actions (from POST /cad_agent: wall + position + gap, never coordinates) ----------
function setTag(m) { m.userData.tag.element.textContent = `${m.userData.label} ${m.userData.w.toFixed(1)}×${m.userData.d.toFixed(1)}`; }
function placeOnWall(m, wallId, position, gap, along, depth) {
  const e = edges()[wallId];
  if (!e) { m.position.x = 0; m.position.z = 0; return; }                 // unknown wall: middle of the room
  const alongX = Math.abs(e.u[0]) > 0.7, sx = alongX ? along : depth, sz = alongX ? depth : along;
  m.userData.w = sx; m.userData.d = sz; m.scale.set(sx, m.scale.y, sz);
  const a = Math.min(along, e.len);
  const t = position === "start" ? a / 2 + 0.1 : position === "end" ? e.len - a / 2 - 0.1 : e.len / 2;
  const n = Math.max(0, gap || 0) + depth / 2 + 0.02;
  m.position.x = e.a[0] + e.u[0] * t + e.n[0] * n; m.position.z = e.a[1] + e.u[1] * t + e.n[1] * n;
  setTag(m);
}
function find(name) {
  const q = String(name || "").toLowerCase().trim(); if (!q) return null;
  return pieces.find((p) => p.userData.label.toLowerCase() === q)
      || pieces.find((p) => p.userData.label.toLowerCase().includes(q) || q.includes(p.userData.label.toLowerCase())) || null;
}
function apply(actions) {
  if (!scene) return null;
  let last = null;
  for (const a of actions || []) {
    if (a.op === "add") {
      const w = a.w_ft || 3, d = a.d_ft || 3, h = a.h_ft || 2.5;
      const m = addPiece(a.label || "item", [[-w / 2, -d / 2], [w / 2, d / 2]], h, "x");
      m.material.color.set(colorFor(a.label || "", "x"));
      if (a.wall != null) placeOnWall(m, a.wall, a.position, a.gap_ft, w, d);
      last = m; continue;
    }
    const m = find(a.target || a.label); if (!m) continue;
    if (a.op === "remove") { if (sel === m) select(null); pieces.splice(pieces.indexOf(m), 1); dispose(m); continue; }
    if (a.op === "rotate") { const { w, d } = m.userData; m.userData.w = d; m.userData.d = w; m.scale.set(d, m.scale.y, w); setTag(m); }
    if (a.op === "move") {
      const e = edges()[a.wall], alongX = e ? Math.abs(e.u[0]) > 0.7 : true;   // keep the piece's current orientation
      if (a.wall != null) placeOnWall(m, a.wall, a.position, a.gap_ft, alongX ? m.userData.w : m.userData.d, alongX ? m.userData.d : m.userData.w);
      else { m.position.x = 0; m.position.z = 0; }
    }
    last = m;
  }
  if (last) select(last);
  return last;
}
// ---------- checker, snapshots, screenshot (agent support) ----------
function sideGaps(m) {
  const [x0, x1, z0, z1] = footprint(m);
  return [[[x0, (z0 + z1) / 2], [-1, 0]], [[x1, (z0 + z1) / 2], [1, 0]], [[(x0 + x1) / 2, z0], [0, -1]], [[(x0 + x1) / 2, z1], [0, 1]]]
    .map(([o, dir]) => rayDist(o, dir, m)).filter(isFinite);
}
function check() {      // deterministic problems in the current layout (what "cramped" means here)
  const out = [];
  pieces.forEach((m, i) => {
    const [x0, x1, z0, z1] = footprint(m), name = m.userData.label;
    if ([[x0, z0], [x1, z0], [x1, z1], [x0, z1]].some((p) => !inside(p))) out.push(`${name} sticks out of the room`);
    if (doorZones.some((z) => x0 < z[1] - 0.05 && x1 > z[0] + 0.05 && z0 < z[3] - 0.05 && z1 > z[2] + 0.05)) out.push(`${name} blocks the door swing`);
    const tight = sideGaps(m).filter((g) => g > 0.1 && g < 2);   // 0 = flush against a wall or piece, which is fine
    if (tight.length >= 2) out.push(`${name} has less than 2 ft of walkway on ${tight.length} sides (${tight.map((g) => g.toFixed(1)).join(", ")} ft)`);
    pieces.slice(i + 1).forEach((p) => { const [a0, a1, b0, b1] = footprint(p);
      if (x0 < a1 - 0.05 && x1 > a0 + 0.05 && z0 < b1 - 0.05 && z1 > b0 + 0.05) out.push(`${name} overlaps ${p.userData.label}`); });
  });
  return out;
}
function snapshot() {
  return pieces.map((p) => ({ label: p.userData.label, w: p.userData.w, d: p.userData.d, h: p.scale.y, x: p.position.x, z: p.position.z, color: p.material.color.getHex() }));
}
function restore(snap) {
  if (!scene || !snap) return;
  select(null); pieces.forEach(dispose); pieces = []; fitPiece = null; hoverPiece = null;
  for (const s of snap) {
    const m = addPiece(s.label, [[s.x - s.w / 2, s.z - s.d / 2], [s.x + s.w / 2, s.z + s.d / 2]], s.h, "x");
    m.material.color.setHex(s.color);
  }
}
function screenshot(maxW = 900) {   // JPEG (base64, no prefix) of the current view, for the agent to look at
  if (!scene) return null;
  R.render(scene, cam);
  const src = R.domElement, k = Math.min(1, maxW / src.width), c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(src.width * k)); c.height = Math.max(1, Math.round(src.height * k));
  c.getContext("2d").drawImage(src, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.8).split(",")[1];
}
function describe() {
  return { walls: wallInfo, furniture: pieces.map((p) => ({ label: p.userData.label, w_ft: +p.userData.w.toFixed(1), d_ft: +p.userData.d.toFixed(1) })) };
}

init();
window.FootprintCAD = { show, add, apply, describe, check, snapshot, restore, screenshot, built: () => built };
