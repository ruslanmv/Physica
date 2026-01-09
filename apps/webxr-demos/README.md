# Physica WebXR Demos (Local)

This folder is **additive**: it does not modify the main Physica project.

## What this demo shows

- Three.js rendering + WebXR (VR)
- A minimal **physics step loop** inside `renderer.setAnimationLoop(...)`
- Controller trigger spawns spheres with initial velocity
- A simple "digital twin" thermal tile (heat diffusion grid) updates in real time

## Run

```bash
cd apps/webxr-demos
npm install
npm run dev
```

Open the printed URL (usually [http://localhost:5173](http://localhost:5173)).

## Notes

* WebXR works best in a WebXR-enabled browser + headset.
* Desktop mode works everywhere (Orbit controls + click to spawn spheres).
* Physics is intentionally minimal and readable (not a full rigid body engine).
* Later you can swap PhysicsWorld with Rapier WASM without changing the Three.js loop structure.
