// Footprint 3D — dollhouse view of model.json (Three.js from CDN, no build step).
// Exposes window.Footprint3D = { build(model), focusRoom(id), overview(), showFurniture(f), clearFurniture() }.
// World units are feet: x = px / scale, z = py / scale, y up. Origin at the plan centre.
//
// Camera language:
//   overview      — high 3/4 view, walls cut down to LOW_H so every room is readable
//   focusRoom     — swoop to eye level inside a room, walls rise to full height (the signature shot)
//   furniture     — elevated 3/4 view into one room, walls low, object + clearance zone visible
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const WALL_H = 8.5, LOW_H = 3, WALL_T = 0.35, EYE_H = 5.2, SWOOP_MS = 3000;
const FOV_OUT = 50, FOV_IN = 72; // wider lens at eye level so a small room still shows its floor
const FALLBACK_SCALE = 20; // px/ft, display only when scale is undetermined (a NOT TO SCALE badge is shown)
const WALL_COLOR = 0xf4f1ea, WIN_UNCONFIRMED = 0xc4bcae;

let renderer, labelRenderer, scene, camera, controls, container, caption, notice;
let model = null, scale = FALLBACK_SCALE, center = [0, 0];
let roomMeshes = [], walls = [], panes = [], labels = {}, furniture = [], glow = [], tween = null;
let wallH = LOW_H, wallTarget = LOW_H, sigTimer = null;
const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
const toWorld = ([x, y]) => [(x - center[0]) / scale, (y - center[1]) / scale];

function init() {
  container = document.getElementById("view3d");
  caption = document.getElementById("view3d-caption");
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = "labels3d";
  container.appendChild(labelRenderer.domElement);

  notice = document.createElement("div");
  notice.className = "view3d-notice";
  notice.hidden = true;
  container.appendChild(notice);

  camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI / 2 - 0.02; // never go under the floor

  new ResizeObserver(resize).observe(container);
  renderer.domElement.addEventListener("click", onClick);
  document.getElementById("view3d-reset").addEventListener("click", () => overview(true));
  renderer.setAnimationLoop(loop);
}

function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return; // hidden tab
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

// ---------- scene ----------
function legalOf(id) { return (model.answers?.legal_bedrooms || []).find((b) => b.room === id); }

function roomStatus(r) {
  const k = model.answers?.king_bed_fits;
  if (legalOf(r.id)?.legal === false || (k?.fits === false && k.room === r.id)) return "fail";
  return r.verified ? "ok" : "unverified";
}

function build(m) {
  model = m;
  const noScale = m.scale_status === "undetermined" || !m.scale_px_per_ft;
  scale = noScale ? FALLBACK_SCALE : m.scale_px_per_ft;
  center = [m.image.w / 2, m.image.h / 2];
  roomMeshes = []; walls = []; panes = []; labels = {}; furniture = []; glow = []; tween = null;
  wallH = wallTarget = LOW_H;
  labelRenderer.domElement.replaceChildren(); // drop labels from a previous build
  notice.hidden = !noScale;
  notice.textContent = "NOT TO SCALE · no dimensions on this plan";

  scene = new THREE.Scene();
  scene.background = new THREE.Color(css("--bg"));
  scene.add(new THREE.HemisphereLight(0xffffff, 0xa69e8f, 1.3));
  scene.add(new THREE.AmbientLight(0xffffff, 0.9)); // vertical wall faces read as off-white, not grey
  const sun = new THREE.DirectionalLight(0xffffff, 1.3);
  sun.position.set(25, 45, 18);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0005;
  Object.assign(sun.shadow.camera, { left: -60, right: 60, top: 60, bottom: -60 });
  scene.add(sun);

  const span = Math.max(m.image.w, m.image.h) / scale;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(span * 3, span * 3), new THREE.MeshStandardMaterial({ color: css("--line") }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.02;
  ground.receiveShadow = true;
  scene.add(ground);
  const grid = new THREE.GridHelper(span * 3, Math.round(span * 3), 0xcbc3b2, 0xe3ded2); // 1 ft grid
  grid.position.y = -0.01;
  scene.add(grid);

  const wallMat = new THREE.MeshStandardMaterial({ color: WALL_COLOR, roughness: 0.9 });
  m.rooms.forEach((r) => r.polygon?.length >= 3 && addRoom(r, wallMat));
  (m.windows || []).forEach(addWindow);
  addPaths();
  if (m.entrance) addEntrance(m.entrance);
  addKingBed();
  applyWallHeight();

  overview(false);
  const target = signatureRoom();
  clearTimeout(sigTimer);
  if (target) sigTimer = setTimeout(() => focusRoom(target.id), 600);
}

function addRoom(r, wallMat) {
  const status = roomStatus(r);
  const pts = r.polygon.map(toWorld);

  // floor: status colour; failing rooms pulse
  const floorMat = new THREE.MeshStandardMaterial({
    color: { ok: css("--ok"), fail: css("--bad"), unverified: 0xfbf9f4 }[status],
    transparent: true, opacity: status === "ok" ? 0.35 : status === "fail" ? 0.6 : 0.9,
    emissive: status === "fail" ? new THREE.Color(css("--bad")) : new THREE.Color(0x000000),
    emissiveIntensity: status === "fail" ? 0.35 : 0,
  });
  if (status === "fail") glow.push(floorMat);
  const floor = new THREE.Mesh(new THREE.ShapeGeometry(shapeOf(pts)), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0.01;
  floor.receiveShadow = true;
  floor.userData.roomId = r.id;
  scene.add(floor);
  roomMeshes.push(floor);

  // outline on the floor: solid red for failing rooms, dashed amber for unverified ones
  if (status !== "ok") {
    const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts.map(([x, z]) => new THREE.Vector3(x, 0.06, z))),
      status === "fail" ? new THREE.LineBasicMaterial({ color: css("--bad") })
                        : new THREE.LineDashedMaterial({ color: css("--warn"), dashSize: 0.8, gapSize: 0.5 }));
    line.computeLineDistances();
    scene.add(line);
  }

  // walls: one neutral material everywhere (status lives on the floor), gaps at doors
  pts.forEach((a, i) => {
    const b = pts[(i + 1) % pts.length];
    for (const [s, e] of splitForDoors(a, b)) addWall(s, e, wallMat, r.id);
  });

  const div = document.createElement("div");
  div.className = `label3d ${status}`;
  const tag = status === "unverified" ? "unverified" : r.area_sqft == null ? "" : `${Math.round(r.area_sqft)} sq ft`;
  div.innerHTML = `<b>${esc(r.label)}</b><span>${tag}</span>`;
  div.addEventListener("click", () => focusRoom(r.id));
  const label = new CSS2DObject(div);
  const [cx, cz] = centroid(pts);
  label.position.set(cx, LOW_H + 1, cz);
  scene.add(label);
  labels[r.id] = label;
}

// Cut a wall segment where a door centre sits on it (within 1.2 ft).
function splitForDoors(a, b) {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (len < 1e-6) return [];
  const dir = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
  const gaps = [];
  (model.doors || []).forEach((d) => {
    if (!d.center) return;
    const [px, pz] = toWorld(d.center);
    const t = (px - a[0]) * dir[0] + (pz - a[1]) * dir[1];
    const off = Math.abs((px - a[0]) * dir[1] - (pz - a[1]) * dir[0]);
    if (off < 1.2 && t > -0.5 && t < len + 0.5) {
      const half = (d.width_ft || 2.8) / 2;
      gaps.push([Math.max(0, t - half), Math.min(len, t + half)]);
    }
  });
  gaps.sort((g, h) => g[0] - h[0]);
  const out = [];
  let cur = 0;
  for (const [g0, g1] of gaps) {
    if (g0 > cur) out.push([cur, g0]);
    cur = Math.max(cur, g1);
  }
  if (cur < len) out.push([cur, len]);
  return out.map(([t0, t1]) => [[a[0] + dir[0] * t0, a[1] + dir[1] * t0], [a[0] + dir[0] * t1, a[1] + dir[1] * t1]]);
}

// Unit-height box, scaled to the current wall height every frame (low in overview, full inside a room).
function addWall(a, b, mat, roomId) {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (len < 0.2) return;
  const wall = new THREE.Mesh(new THREE.BoxGeometry(len, 1, WALL_T), mat);
  wall.position.set((a[0] + b[0]) / 2, 0.5, (a[1] + b[1]) / 2);
  wall.rotation.y = -Math.atan2(b[1] - a[1], b[0] - a[0]);
  wall.castShadow = wall.receiveShadow = true;
  wall.userData.roomId = roomId;
  scene.add(wall);
  walls.push(wall);
  roomMeshes.push(wall);
}

// Windows on the nearest room edge: blue if confirmed on an exterior wall, grey if not confirmed.
function addWindow(w) {
  if (!w.center) return;
  const [px, pz] = toWorld(w.center);
  let best = null;
  model.rooms.forEach((r) => {
    const pts = (r.polygon || []).map(toWorld);
    pts.forEach((a, i) => {
      const b = pts[(i + 1) % pts.length];
      const q = nearestOnSeg([px, pz], a, b);
      const d = Math.hypot(q[0] - px, q[1] - pz);
      if (!best || d < best.d) best = { d, q, ang: Math.atan2(b[1] - a[1], b[0] - a[0]) };
    });
  });
  if (!best || best.d > 3) return;
  const ext = w.on_exterior_wall === true;
  const pane = new THREE.Mesh(new THREE.BoxGeometry(3, 1, WALL_T + 0.12),
    new THREE.MeshStandardMaterial({ color: ext ? 0xa9bccd : WIN_UNCONFIRMED, transparent: true, opacity: 0.9,
                                     emissive: ext ? 0x27466b : 0x000000, emissiveIntensity: ext ? 0.35 : 0 }));
  pane.position.set(best.q[0], 0, best.q[1]);
  pane.rotation.y = -best.ang;
  scene.add(pane);
  panes.push(pane);
}

// Walls, windows and labels follow wallH; it eases toward wallTarget in loop().
function applyWallHeight() {
  walls.forEach((w) => { w.scale.y = wallH; w.position.y = wallH / 2; });
  panes.forEach((p) => { const h = wallH * 0.38; p.scale.y = h; p.position.y = wallH * 0.42 + h / 2; });
  Object.values(labels).forEach((l) => (l.position.y = wallH + 1));
}

// One floor strip per bedroom route to the entrance (server: clear_path.per_room).
function addPaths() {
  const c = model.answers?.clear_path;
  const paths = Array.isArray(c) ? c : c?.per_room?.length ? c.per_room : c?.path ? [c] : [];
  paths.forEach((p) => {
    if (!p.path || p.path.length < 2) return;
    const color = p.ok === true ? css("--accent") : p.ok === false ? css("--bad") : 0xa69e8f;
    addPolylineStrip(p.path.map(toWorld), new THREE.MeshBasicMaterial({ color }), 0.04, scene);
  });
}

function addPolylineStrip(pts, mat, y, parent) {
  for (let i = 0; i < pts.length - 1; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const seg = new THREE.Mesh(new THREE.BoxGeometry(len + 0.4, 0.04, 0.4), mat);
    seg.position.set((a[0] + b[0]) / 2, y, (a[1] + b[1]) / 2);
    seg.rotation.y = -Math.atan2(b[1] - a[1], b[0] - a[0]);
    parent.add(seg);
  }
}

// Solver's king-bed placement (answers.king_bed_fits.placement, px corners) as a bed.
function addKingBed() {
  const k = model.answers?.king_bed_fits;
  if (!k?.placement || k.placement.length < 3 || !k.room) return;
  const b = bboxWorld(k.placement);
  const r = model.rooms.find((x) => x.id === k.room);
  scene.add(makeFurniture("bed", b, backSide(b, r), k.fits !== false, 2));
}

function addEntrance(e) {
  const [x, z] = toWorld(e);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(1.2, 0.12, 8, 32), new THREE.MeshBasicMaterial({ color: css("--accent") }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(x, 0.08, z);
  scene.add(ring);
  const div = document.createElement("div");
  div.className = "label3d entrance";
  div.textContent = "Entrance";
  const label = new CSS2DObject(div);
  label.position.set(x, 1.5, z);
  scene.add(label);
}

// ---------- furniture ----------
// bbox in world feet {x0, z0, x1, z1}; back = "x0" | "x1" | "z0" | "z1" (side against the wall).
function makeFurniture(kind, b, back, fits, h) {
  const g = new THREE.Group();
  const w = b.x1 - b.x0, d = b.z1 - b.z0, cx = (b.x0 + b.x1) / 2, cz = (b.z0 + b.z1) / 2;
  const main = new THREE.MeshStandardMaterial({ color: fits ? 0xe8eee6 : 0xf4e3dd, roughness: 0.8 });
  const accent = new THREE.MeshStandardMaterial({ color: fits ? css("--sys") || css("--ok") : css("--bad"), roughness: 0.7 });
  const box = (bw, bh, bd, x, y, z, mat) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(bw, bh, bd), mat);
    m.position.set(x, y, z); m.castShadow = m.receiveShadow = true; g.add(m);
  };
  // "back slab" along the wall side (headboard / sofa back), thickness t, height bh
  const backSlab = (t, bh, mat) => {
    if (back === "x0") box(t, bh, d, b.x0 + t / 2, bh / 2, cz, mat);
    else if (back === "x1") box(t, bh, d, b.x1 - t / 2, bh / 2, cz, mat);
    else if (back === "z0") box(w, bh, t, cx, bh / 2, b.z0 + t / 2, mat);
    else box(w, bh, t, cx, bh / 2, b.z1 - t / 2, mat);
  };
  if (kind === "bed") {
    box(w, 1.0, d, cx, 0.5, cz, accent);          // base
    box(w - 0.2, 0.7, d - 0.2, cx, 1.35, cz, main); // mattress
    backSlab(0.35, 3.5, accent);                  // headboard
  } else if (kind === "sofa") {
    box(w, 1.4, d, cx, 0.7, cz, accent);          // seat
    backSlab(Math.min(0.9, Math.min(w, d) * 0.3), h || 2.8, accent);
  } else {
    box(w, h || 2.5, d, cx, (h || 2.5) / 2, cz, accent);
  }
  return g;
}

// Ghost of the requested size (doesn't fit): red wireframe + faint fill, so you can see it poke through walls.
function makeGhost(b, h) {
  const g = new THREE.Group();
  const geo = new THREE.BoxGeometry(b.x1 - b.x0, h, b.z1 - b.z0);
  const fill = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: css("--bad"), transparent: true, opacity: 0.12, depthWrite: false }));
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: css("--bad") }));
  [fill, edges].forEach((o) => { o.position.set((b.x0 + b.x1) / 2, h / 2, (b.z0 + b.z1) / 2); g.add(o); });
  return g;
}

// Clearance zone on the floor: the object's footprint grown by c on every side except its back.
function makeClearance(b, back, c, ok) {
  const z = { x0: b.x0 - c, x1: b.x1 + c, z0: b.z0 - c, z1: b.z1 + c };
  if (back) z[back] = b[back];
  const geo = new THREE.PlaneGeometry(z.x1 - z.x0, z.z1 - z.z0);
  const color = ok ? css("--accent") : css("--bad");
  const g = new THREE.Group();
  const plane = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18, depthWrite: false }));
  plane.rotation.x = -Math.PI / 2;
  plane.position.set((z.x0 + z.x1) / 2, 0.05, (z.z0 + z.z1) / 2);
  g.add(plane);
  const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
    [[z.x0, z.z0], [z.x1, z.z0], [z.x1, z.z1], [z.x0, z.z1]].map(([x, zz]) => new THREE.Vector3(x, 0.07, zz))),
    new THREE.LineDashedMaterial({ color, dashSize: 0.5, gapSize: 0.35 }));
  outline.computeLineDistances();
  g.add(outline);
  return g;
}

// Accepts {room, w_ft, d_ft, h_ft?, fits, center_px?, clearance_ft?, kind?: "bed"|"sofa"|other}.
// center_px + w/d come from the solver's placement when it fits; otherwise the requested size at the room centre.
function showFurniture(f) {
  clearTimeout(sigTimer); // the user asked for this view; don't let the load-time swoop override it
  clearFurniture();
  const r = model.rooms.find((x) => x.id === f.room || x.label === f.room);
  if (!r) return;
  const [cx, cz] = f.center_px ? toWorld(f.center_px) : centroid(r.polygon.map(toWorld));
  const b = { x0: cx - f.w_ft / 2, x1: cx + f.w_ft / 2, z0: cz - f.d_ft / 2, z1: cz + f.d_ft / 2 };
  const back = f.fits ? backSide(b, r) : null;
  const objs = [];
  if (f.fits) objs.push(makeFurniture(f.kind || "box", b, back, true, f.h_ft));
  else objs.push(makeGhost(b, f.h_ft || 2.5));
  if (f.clearance_ft > 0) objs.push(makeClearance(b, back, f.clearance_ft, f.fits !== false));
  objs.forEach((o) => { scene.add(o); furniture.push(o); });
  viewRoomFromAbove(r, f);
}

function clearFurniture() {
  furniture.forEach((o) => scene.remove(o));
  furniture = [];
}

// The bbox side closest to the room's walls = the side the solver put against a wall.
function backSide(b, r) {
  if (!r?.polygon?.length) return "z0";
  const pts = r.polygon.map(toWorld);
  const mids = { x0: [b.x0, (b.z0 + b.z1) / 2], x1: [b.x1, (b.z0 + b.z1) / 2], z0: [(b.x0 + b.x1) / 2, b.z0], z1: [(b.x0 + b.x1) / 2, b.z1] };
  let best = "z0", bestD = Infinity;
  for (const [side, m] of Object.entries(mids)) {
    pts.forEach((a, i) => {
      const q = nearestOnSeg(m, a, pts[(i + 1) % pts.length]);
      const d = Math.hypot(q[0] - m[0], q[1] - m[1]);
      if (d < bestD) { bestD = d; best = side; }
    });
  }
  return best;
}

// ---------- camera ----------
function bounds() {
  const box = new THREE.Box3();
  model.rooms.forEach((r) => (r.polygon || []).map(toWorld).forEach(([x, z]) => box.expandByPoint(new THREE.Vector3(x, 0, z))));
  return box;
}

function overview(animate) {
  if (animate) clearTimeout(sigTimer);
  const box = bounds();
  const c = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const d = Math.max(size.x, size.z) * 1.1 + 8;
  const pos = new THREE.Vector3(c.x + d * 0.3, d * 0.95, c.z + d * 0.75);
  wallTarget = LOW_H;
  showLabels(null);
  if (animate) swoop(pos, c, FOV_OUT);
  else { camera.position.copy(pos); controls.target.copy(c); camera.fov = FOV_OUT; camera.updateProjectionMatrix(); controls.update(); }
  setCaption(null);
}

// Signature shot: eye level inside the room, walls rise to full height.
function focusRoom(id) {
  clearTimeout(sigTimer);
  const r = model.rooms.find((x) => x.id === id);
  if (!r?.polygon?.length) return;
  const pts = r.polygon.map(toWorld);
  const [cx, cz] = centroid(pts);
  // stand near the corner farthest from the centroid, pulled 30% back toward it, look across the room
  let far = pts[0];
  pts.forEach((p) => { if (Math.hypot(p[0] - cx, p[1] - cz) > Math.hypot(far[0] - cx, far[1] - cz)) far = p; });
  const sx = far[0] + (cx - far[0]) * 0.3, sz = far[1] + (cz - far[1]) * 0.3;
  wallTarget = WALL_H;
  showLabels(id);
  swoop(new THREE.Vector3(sx, EYE_H, sz), new THREE.Vector3(cx - (sx - cx) * 0.3, 1.2, cz - (sz - cz) * 0.3), FOV_IN);
  setCaption(r);
}

// Furniture shot: elevated 3/4 view into the room, walls low so the object and its clearance read clearly.
function viewRoomFromAbove(r, f) {
  const pts = r.polygon.map(toWorld);
  const [cx, cz] = centroid(pts);
  const xs = pts.map((p) => p[0]), zs = pts.map((p) => p[1]);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs));
  const d = span * 1.1 + 6;
  wallTarget = LOW_H;
  showLabels(r.id);
  swoop(new THREE.Vector3(cx + d * 0.35, d * 0.9, cz + d * 0.6), new THREE.Vector3(cx, 0, cz), FOV_OUT);
  caption.textContent = `${r.label.toUpperCase()} · ${f.w_ft.toFixed(1)}×${f.d_ft.toFixed(1)} ft · ${f.fits ? "fits" : f.fits === false ? "doesn't fit" : "can't tell"}`;
  caption.className = `view3d-caption ${f.fits === false ? "fail" : ""}`;
  caption.hidden = false;
}

function showLabels(focusId) {
  Object.entries(labels).forEach(([id, l]) => (l.visible = !focusId || id === focusId));
}

function swoop(toPos, toTarget, toFov = camera.fov) {
  tween = { t0: performance.now(), fromPos: camera.position.clone(), fromTarget: controls.target.clone(), toPos, toTarget,
            fromFov: camera.fov, toFov };
  controls.enabled = false;
}

const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function loop(now) {
  if (!scene) return;
  if (tween) {
    const t = Math.min(1, (now - tween.t0) / SWOOP_MS), k = ease(t);
    // arc up through a high point so the swoop reads as a fly-in, not a slide
    const lift = Math.sin(Math.PI * k) * 8;
    camera.position.lerpVectors(tween.fromPos, tween.toPos, k).y += lift * (1 - k);
    controls.target.lerpVectors(tween.fromTarget, tween.toTarget, k);
    camera.fov = tween.fromFov + (tween.toFov - tween.fromFov) * k;
    camera.updateProjectionMatrix();
    camera.lookAt(controls.target);
    if (t >= 1) { tween = null; controls.enabled = true; }
  } else {
    controls.update();
  }
  if (Math.abs(wallTarget - wallH) > 0.005) {
    wallH += (wallTarget - wallH) * 0.06;
    applyWallHeight();
  }
  const pulse = 0.25 + 0.25 * (1 + Math.sin(now / 350)) / 2;
  glow.forEach((m) => (m.emissiveIntensity = pulse));
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}

function onClick(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(roomMeshes)[0];
  if (hit) focusRoom(hit.object.userData.roomId);
}

// ---------- caption ----------
function captionText(r) {
  const legal = legalOf(r.id);
  const parts = [r.label.toUpperCase()];
  parts.push(r.area_sqft == null ? "size unknown" : `${Math.round(r.area_sqft)} sq ft`);
  if (legal) {
    if (legal.checks?.area_ge_80 === false) parts.push("under 80 sq ft");
    if (legal.checks?.min_width_ge_8ft === false) parts.push("under 8 ft wide");
    parts.push(legal.legal === false ? "not a legal bedroom" : legal.legal === true ? "legal bedroom" : "legal status unconfirmed");
  } else {
    parts.push(r.verified ? "verified" : "unverified");
  }
  return parts.join(" · ");
}

function setCaption(r) {
  if (!r) { caption.hidden = true; return; }
  caption.textContent = captionText(r);
  caption.className = `view3d-caption ${roomStatus(r)}`;
  caption.hidden = false;
}

// ---------- helpers ----------
function shapeOf(pts) { return new THREE.Shape(pts.map(([x, z]) => new THREE.Vector2(x, -z))); }

function centroid(pts) { // area-weighted, so L-shaped rooms get a point inside them
  let a = 0, cx = 0, cy = 0;
  pts.forEach(([x1, y1], i) => {
    const [x2, y2] = pts[(i + 1) % pts.length];
    const f = x1 * y2 - x2 * y1;
    a += f; cx += (x1 + x2) * f; cy += (y1 + y2) * f;
  });
  if (Math.abs(a) < 1e-9) {
    let x = 0, z = 0; pts.forEach(([p, q]) => { x += p; z += q; });
    return [x / pts.length, z / pts.length];
  }
  return [cx / (3 * a), cy / (3 * a)];
}

function nearestOnSeg(p, a, b) {
  const dx = b[0] - a[0], dz = b[1] - a[1], L = dx * dx + dz * dz;
  if (L < 1e-12) return a;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / L));
  return [a[0] + dx * t, a[1] + dz * t];
}

function bboxWorld(pxPts) {
  const w = pxPts.map(toWorld), xs = w.map((p) => p[0]), zs = w.map((p) => p[1]);
  return { x0: Math.min(...xs), x1: Math.max(...xs), z0: Math.min(...zs), z1: Math.max(...zs) };
}

// Signature room: a bedroom that fails the legal check, else the room where the king bed doesn't fit,
// else the first bedroom (caption then says "legal status unconfirmed", never a verdict we don't have).
function signatureRoom() {
  const byId = (id) => model.rooms.find((r) => r.id === id);
  const fail = (model.answers?.legal_bedrooms || []).find((b) => b.legal === false);
  if (fail) return byId(fail.room);
  const k = model.answers?.king_bed_fits;
  if (k?.fits === false && k.room) return byId(k.room);
  return model.rooms.find((r) => /bed|makuuhuone|^\s*(m?br|mh)\s*\d*\s*$/i.test(r.label || "")) || null;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

init();
window.Footprint3D = { build, focusRoom, overview: () => overview(true), showFurniture, clearFurniture };
if (window.__pending3d) build(window.__pending3d);
