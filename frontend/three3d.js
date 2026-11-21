// Footprint 3D — dollhouse view of model.json (Three.js from CDN, no build step).
// Exposes window.Footprint3D = { build(model), focusRoom(id), overview(), showFurniture(f), clearFurniture() }.
// World units are feet: x = px / scale, z = py / scale, y up. Origin at the plan centre.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const WALL_H = 8.5, WALL_T = 0.35, EYE_H = 5.2, SWOOP_MS = 3000;
const FALLBACK_SCALE = 20; // px/ft, display only when scale is undetermined

let renderer, labelRenderer, scene, camera, controls, container, caption;
let model = null, scale = FALLBACK_SCALE, center = [0, 0];
let roomMeshes = [], furniture = [], tween = null, glow = [];
const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
const toWorld = ([x, y]) => [(x - center[0]) / scale, (y - center[1]) / scale];

function init() {
  container = document.getElementById("view3d");
  caption = document.getElementById("view3d-caption");
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  container.appendChild(renderer.domElement);

  labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = "labels3d";
  container.appendChild(labelRenderer.domElement);

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
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

// ---------- scene ----------
function roomStatus(r) {
  const a = model.answers || {};
  const illegal = (a.legal_bedrooms || []).some((b) => b.room === r.id && b.legal === false);
  const bedFail = a.king_bed_fits?.fits === false && a.king_bed_fits.room === r.id;
  if (!r.verified) return "unverified";
  return illegal || bedFail ? "fail" : "ok";
}

function build(m) {
  model = m;
  scale = m.scale_status !== "undetermined" && m.scale_px_per_ft ? m.scale_px_per_ft : FALLBACK_SCALE;
  center = [m.image.w / 2, m.image.h / 2];
  roomMeshes = []; furniture = []; glow = []; tween = null;
  labelRenderer.domElement.replaceChildren(); // drop labels from a previous build (Replay / new upload)

  scene = new THREE.Scene();
  scene.background = new THREE.Color(css("--panel"));
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(20, 40, 15);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -60, right: 60, top: 60, bottom: -60 });
  scene.add(sun);

  const span = Math.max(m.image.w, m.image.h) / scale;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(span * 3, span * 3),
    new THREE.MeshStandardMaterial({ color: css("--line") }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.02;
  ground.receiveShadow = true;
  scene.add(ground);

  m.rooms.forEach((r) => r.polygon?.length >= 3 && addRoom(r));
  (m.windows || []).forEach(addWindow);
  addPaths();
  if (m.entrance) addEntrance(m.entrance);
  addKingBed();

  overview(false);
  const target = firstFailingBedroom();
  if (target) setTimeout(() => focusRoom(target.id), 600);
}

function addRoom(r) {
  const status = roomStatus(r);
  const color = { ok: css("--ok"), fail: css("--bad"), unverified: css("--warn") }[status];
  const pts = r.polygon.map(toWorld);

  // floor
  const shape = new THREE.Shape(pts.map(([x, z]) => new THREE.Vector2(x, -z)));
  const floorMat = new THREE.MeshStandardMaterial({
    color, transparent: true, opacity: status === "unverified" ? 0.25 : 0.55,
    emissive: status === "fail" ? new THREE.Color(css("--bad")) : new THREE.Color(0x000000),
    emissiveIntensity: status === "fail" ? 0.4 : 0,
  });
  const floor = new THREE.Mesh(new THREE.ShapeGeometry(shape), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0.01;
  floor.receiveShadow = true;
  floor.userData.roomId = r.id;
  scene.add(floor);
  roomMeshes.push(floor);

  // walls along each edge, with gaps at doors
  const wallMat = new THREE.MeshStandardMaterial({
    color: status === "fail" ? css("--bad") : 0xf2efe8,
    transparent: status === "unverified", opacity: status === "unverified" ? 0.35 : 1,
    emissive: status === "fail" ? new THREE.Color(css("--bad")) : new THREE.Color(0x000000),
    emissiveIntensity: status === "fail" ? 0.25 : 0,
  });
  if (status === "fail") glow.push(floorMat, wallMat);
  pts.forEach((a, i) => {
    const b = pts[(i + 1) % pts.length];
    for (const [s, e] of splitForDoors(a, b)) addWall(s, e, wallMat, r.id);
  });

  // label
  const div = document.createElement("div");
  div.className = `label3d ${status}`;
  div.innerHTML = `<b>${esc(r.label)}</b><span>${r.area_sqft == null ? "" : Math.round(r.area_sqft) + " sq ft"}</span>`;
  div.addEventListener("click", () => focusRoom(r.id));
  const label = new CSS2DObject(div);
  const [cx, cz] = centroid(pts);
  label.position.set(cx, WALL_H + 1, cz);
  scene.add(label);
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

function addWall(a, b, mat, roomId) {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (len < 0.2) return;
  const wall = new THREE.Mesh(new THREE.BoxGeometry(len, WALL_H, WALL_T), mat);
  wall.position.set((a[0] + b[0]) / 2, WALL_H / 2, (a[1] + b[1]) / 2);
  wall.rotation.y = -Math.atan2(b[1] - a[1], b[0] - a[0]);
  wall.castShadow = wall.receiveShadow = true;
  wall.userData.roomId = roomId;
  scene.add(wall);
  roomMeshes.push(wall);
}

// Windows: a pale blue pane on the nearest room edge.
function addWindow(w) {
  if (!w.center) return;
  const [px, pz] = toWorld(w.center);
  let best = null;
  model.rooms.forEach((r) => {
    const pts = (r.polygon || []).map(toWorld);
    pts.forEach((a, i) => {
      const b = pts[(i + 1) % pts.length];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (len < 1e-6) return;
      const t = Math.max(0, Math.min(len, ((px - a[0]) * (b[0] - a[0]) + (pz - a[1]) * (b[1] - a[1])) / len));
      const q = [a[0] + (b[0] - a[0]) * t / len, a[1] + (b[1] - a[1]) * t / len];
      const d = Math.hypot(q[0] - px, q[1] - pz);
      if (!best || d < best.d) best = { d, q, ang: Math.atan2(b[1] - a[1], b[0] - a[0]) };
    });
  });
  if (!best || best.d > 3) return;
  const pane = new THREE.Mesh(new THREE.BoxGeometry(3, 3.5, WALL_T + 0.1),
    new THREE.MeshStandardMaterial({ color: 0x8ec5ff, transparent: true, opacity: 0.85, emissive: 0x3d7bd9, emissiveIntensity: 0.3 }));
  pane.position.set(best.q[0], 4.5, best.q[1]);
  pane.rotation.y = -best.ang;
  scene.add(pane);
}

// Clear path(s) to the entrance as a strip on the floor.
function addPaths() {
  let paths = model.answers?.clear_path;
  if (!paths) return;
  if (!Array.isArray(paths)) paths = [paths];
  paths.forEach((p) => {
    if (!p.path || p.path.length < 2) return;
    const mat = new THREE.MeshBasicMaterial({ color: p.ok ? css("--accent") : css("--bad") });
    const pts = p.path.map(toWorld);
    for (let i = 0; i < pts.length - 1; i++) {
      const [a, b] = [pts[i], pts[i + 1]];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const seg = new THREE.Mesh(new THREE.BoxGeometry(len + 0.4, 0.04, 0.4), mat);
      seg.position.set((a[0] + b[0]) / 2, 0.04, (a[1] + b[1]) / 2);
      seg.rotation.y = -Math.atan2(b[1] - a[1], b[0] - a[0]);
      scene.add(seg);
    }
  });
}

// Solver's king-bed placement (answers.king_bed_fits.placement = 4 px corners) as a low green box.
function addKingBed() {
  const k = model.answers?.king_bed_fits;
  if (!k?.placement || k.placement.length < 3) return;
  const pts = k.placement.map(toWorld);
  const shape = new THREE.Shape(pts.map(([x, z]) => new THREE.Vector2(x, -z)));
  const geo = new THREE.ExtrudeGeometry(shape, { depth: 2, bevelEnabled: false });
  const bed = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: k.fits ? css("--ok") : css("--bad") }));
  bed.rotation.x = -Math.PI / 2; // extrude goes +z in shape space → +y in world
  bed.castShadow = true;
  scene.add(bed);
}

function addEntrance(e) {
  const [x, z] = toWorld(e);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(1.2, 0.12, 8, 32),
    new THREE.MeshBasicMaterial({ color: css("--bad") }));
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

// ---------- camera ----------
function bounds() {
  const box = new THREE.Box3();
  model.rooms.forEach((r) => (r.polygon || []).map(toWorld).forEach(([x, z]) => box.expandByPoint(new THREE.Vector3(x, 0, z))));
  return box;
}

function overview(animate) {
  const box = bounds();
  const c = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const d = Math.max(size.x, size.z) * 1.15 + 10;
  const pos = new THREE.Vector3(c.x + d * 0.35, d * 0.95, c.z + d * 0.75);
  if (animate) swoop(pos, c);
  else { camera.position.copy(pos); controls.target.copy(c); controls.update(); }
  setCaption(null);
}

function focusRoom(id) {
  const r = model.rooms.find((x) => x.id === id);
  if (!r?.polygon?.length) return;
  const pts = r.polygon.map(toWorld);
  const [cx, cz] = centroid(pts);
  // stand near the corner farthest from the centroid, pulled 30% back toward it, look across the room
  let far = pts[0];
  pts.forEach((p) => { if (Math.hypot(p[0] - cx, p[1] - cz) > Math.hypot(far[0] - cx, far[1] - cz)) far = p; });
  const sx = far[0] + (cx - far[0]) * 0.3, sz = far[1] + (cz - far[1]) * 0.3;
  swoop(new THREE.Vector3(sx, EYE_H, sz), new THREE.Vector3(cx - (sx - cx) * 0.5, EYE_H * 0.55, cz - (sz - cz) * 0.5));
  setCaption(r);
}

function swoop(toPos, toTarget) {
  tween = { t0: performance.now(), fromPos: camera.position.clone(), fromTarget: controls.target.clone(), toPos, toTarget };
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
    camera.lookAt(controls.target);
    if (t >= 1) { tween = null; controls.enabled = true; }
  } else {
    controls.update();
  }
  const pulse = 0.25 + 0.2 * (1 + Math.sin(now / 350)) / 2;
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
  const legal = (model.answers?.legal_bedrooms || []).find((b) => b.room === r.id);
  const parts = [r.label.toUpperCase()];
  parts.push(r.area_sqft == null ? "size unknown" : `${Math.round(r.area_sqft)} sq ft`);
  if (legal) {
    if (legal.checks?.exterior_window === false) parts.push("no exterior window");
    if (legal.checks?.min_width_ge_8ft === false && legal.checks?.exterior_window !== false) parts.push("under 8 ft wide");
    if (legal.legal === false) parts.push("not a legal bedroom");
    else if (legal.legal === true) parts.push("legal bedroom");
  } else {
    parts.push(r.verified ? "verified" : "unverified");
  }
  return parts.join(" · ");
}

function setCaption(r) {
  if (!r) { caption.hidden = true; return; }
  const status = roomStatus(r);
  caption.textContent = captionText(r);
  caption.className = `view3d-caption ${status}`;
  caption.hidden = false;
}

// ---------- furniture (from /query) ----------
// ⚠️ uncertain — /query response shape not final. Accepts {room, w_ft, d_ft, h_ft?, fits, center_px?, rotation_deg?}.
function showFurniture(f) {
  clearFurniture();
  const r = model.rooms.find((x) => x.id === f.room || x.label === f.room);
  if (!r) return;
  const [cx, cz] = f.center_px ? toWorld(f.center_px) : centroid(r.polygon.map(toWorld));
  const h = f.h_ft || 2;
  const box = new THREE.Mesh(new THREE.BoxGeometry(f.w_ft, h, f.d_ft),
    new THREE.MeshStandardMaterial({ color: f.fits ? css("--ok") : css("--bad"), transparent: true, opacity: 0.85 }));
  box.position.set(cx, h / 2, cz);
  box.rotation.y = -((f.rotation_deg || 0) * Math.PI) / 180;
  box.castShadow = true;
  scene.add(box);
  furniture.push(box);
  focusRoom(r.id);
}

function clearFurniture() {
  furniture.forEach((b) => scene.remove(b));
  furniture = [];
}

// ---------- helpers ----------
function centroid(pts) {
  let x = 0, z = 0;
  pts.forEach(([a, b]) => { x += a; z += b; });
  return [x / pts.length, z / pts.length];
}

function firstFailingBedroom() {
  const fail = (model.answers?.legal_bedrooms || []).find((b) => b.legal === false);
  return fail ? model.rooms.find((r) => r.id === fail.room) : null;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

init();
window.Footprint3D = { build, focusRoom, overview: () => overview(true), showFurniture, clearFurniture };
if (window.__pending3d) build(window.__pending3d);
