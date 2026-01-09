import * as THREE from "three";
import { XRButton } from "three/examples/jsm/webxr/XRButton.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { PhysicsWorld } from "./physics/PhysicsWorld.js";
import { ThermalGrid } from "./physics/ThermalGrid.js";
import { makeScene } from "./render/makeScene.js";
import { makeHud } from "./render/makeHud.js";
import { computeDtSeconds } from "./util/time.js";
import { randRange } from "./util/math.js";

const container = document.getElementById("container");
const hudEl = document.getElementById("hud");
const hud = makeHud(hudEl);

// --- Three.js camera + renderer ---
const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.05, 200);
camera.position.set(0, 1.6, 4);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
container.appendChild(renderer.domElement);

// WebXR button (VR)
document.body.appendChild(XRButton.createButton(renderer));

// Desktop orbit controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.2, 0);
controls.update();

// --- Scene ---
const { scene } = makeScene();

// --- Physics world (rigid-ish spheres) ---
const physics = new PhysicsWorld({
  gravity: [0, -9.81, 0],
  floorY: 0,
  restitution: 0.55,
  linearDamping: 0.02,
  worldBounds: { x: 12, z: 12 }
});

// --- Thermal digital twin tile (heatmap) ---
const thermal = new ThermalGrid(48);
const thermalTex = new THREE.DataTexture(encodeThermalToRGBA(thermal.data, thermal.n), thermal.n, thermal.n);
thermalTex.needsUpdate = true;

const chipGeo = new THREE.PlaneGeometry(1.6, 1.6);
const chipMat = new THREE.MeshBasicMaterial({ map: thermalTex, transparent: false });
const chip = new THREE.Mesh(chipGeo, chipMat);
chip.rotation.x = -Math.PI / 2;
chip.position.set(-2.2, 0.01, -1.6);
scene.add(chip);

// Label for chip tile (simple 3D text substitute: a small plate)
const label = makeLabel("Thermal Tile (Toy)");
label.position.set(-2.2, 0.02, -0.65);
scene.add(label);

// --- Spawn spheres + sync list ---
let nextId = 1;
const simObjects = []; // { id, mesh }

const sphereGeo = new THREE.SphereGeometry(1, 20, 20);

function spawnSphere(position, velocity, radius = 0.10) {
  const id = `ball_${nextId++}`;
  const mat = new THREE.MeshStandardMaterial({ color: 0x7aa6ff, roughness: 0.25, metalness: 0.05 });
  const mesh = new THREE.Mesh(sphereGeo, mat);
  mesh.scale.setScalar(radius);
  mesh.position.copy(position);
  scene.add(mesh);

  physics.addBody({
    id,
    radius,
    mass: 1,
    pos: [position.x, position.y, position.z],
    vel: [velocity.x, velocity.y, velocity.z]
  });

  simObjects.push({ id, mesh });

  if (simObjects.length > 90) {
    const old = simObjects.shift();
    scene.remove(old.mesh);
    physics.removeBody(old.id);
    old.mesh.geometry.dispose();
    old.mesh.material.dispose();
  }
}

// Seed a few spheres at start
for (let i = 0; i < 10; i++) {
  const p = new THREE.Vector3(randRange(-1, 1), randRange(1.2, 3.2), randRange(-1, 1));
  const v = new THREE.Vector3(randRange(-0.6, 0.6), randRange(0.2, 1.6), randRange(-0.6, 0.6));
  spawnSphere(p, v, 0.12);
}

// --- WebXR controllers ---
const controller1 = renderer.xr.getController(0);
const controller2 = renderer.xr.getController(1);
scene.add(controller1, controller2);

controller1.add(makeRay());
controller2.add(makeRay());

const tmpMat4 = new THREE.Matrix4();
const forward = new THREE.Vector3(0, 0, -1);

function shootFrom(controller) {
  tmpMat4.identity().extractRotation(controller.matrixWorld);
  const dir = forward.clone().applyMatrix4(tmpMat4).normalize();
  const origin = new THREE.Vector3().setFromMatrixPosition(controller.matrixWorld);
  const start = origin.clone().add(dir.clone().multiplyScalar(0.15));
  spawnSphere(start, dir.multiplyScalar(6.0), 0.09);

  // add a thermal "pulse" to show coupling (toy)
  thermal.addHotspot(Math.random(), Math.random(), 0.35);
}

controller1.addEventListener("selectstart", () => shootFrom(controller1));
controller2.addEventListener("selectstart", () => shootFrom(controller2));

// Desktop click spawns from camera
window.addEventListener("pointerdown", () => {
  const dir = new THREE.Vector3();
  camera.getWorldDirection(dir);
  const start = camera.position.clone().add(dir.clone().multiplyScalar(0.5));
  spawnSphere(start, dir.multiplyScalar(6.0), 0.09);

  thermal.addHotspot(Math.random(), Math.random(), 0.35);
});

// --- THE CORE PATTERN: XR animation loop ---
let lastMs = 0;
let accum = 0;
let frames = 0;
let fps = 0;
let fpsTimer = 0;

renderer.setAnimationLoop((tMs) => {
  const dt = computeDtSeconds(tMs, lastMs);
  lastMs = tMs;

  // 1) Step physics
  physics.step(dt);

  // 2) Step "digital twin" thermal model
  thermal.step(dt);

  // update thermal texture
  thermalTex.image.data = encodeThermalToRGBA(thermal.data, thermal.n, thermalTex.image.data);
  thermalTex.needsUpdate = true;

  // 3) Sync physics -> Three.js meshes
  for (const obj of simObjects) {
    const b = physics.getBody(obj.id);
    if (!b) continue;
    obj.mesh.position.set(b.pos[0], b.pos[1], b.pos[2]);
  }

  // 4) Render
  renderer.render(scene, camera);

  // HUD stats
  frames++;
  fpsTimer += dt;
  if (fpsTimer >= 0.5) {
    fps = Math.round(frames / fpsTimer);
    frames = 0;
    fpsTimer = 0;
  }
  hud.setText(`FPS: ${fps} • bodies: ${simObjects.length} • XR: ${renderer.xr.isPresenting ? "yes" : "no"}`);
});

// --- resize ---
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------- helpers ----------

function makeRay() {
  const geo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, -1)
  ]);
  const mat = new THREE.LineBasicMaterial({ color: 0xffffff });
  const line = new THREE.Line(geo, mat);
  line.scale.z = 2.5;
  return line;
}

function makeLabel(text) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#e7eef7";
  ctx.font = "24px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);

  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true });
  const geo = new THREE.PlaneGeometry(1.8, 0.45);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

/**
 * Encodes scalar thermal field to RGBA bytes for DataTexture.
 * If `reuse` is provided, writes into it to reduce allocations.
 */
function encodeThermalToRGBA(field, n, reuse) {
  const out = reuse && reuse.length === n * n * 4 ? reuse : new Uint8Array(n * n * 4);

  // find min/max for normalization (cheap)
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < field.length; i++) {
    const v = field[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const span = Math.max(1e-6, mx - mn);

  for (let i = 0; i < field.length; i++) {
    const t = (field[i] - mn) / span; // 0..1
    // simple blue->cyan->yellow->red ramp (manual, no libs)
    const r = Math.min(255, Math.max(0, Math.floor(255 * Math.max(0, (t - 0.5) * 2))));
    const b = Math.min(255, Math.max(0, Math.floor(255 * Math.max(0, (0.5 - t) * 2))));
    const g = Math.min(255, Math.max(0, Math.floor(255 * (1 - Math.abs(t - 0.5) * 2))));

    const o = i * 4;
    out[o + 0] = r;
    out[o + 1] = g;
    out[o + 2] = b;
    out[o + 3] = 255;
  }

  return out;
}
