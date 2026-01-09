import * as THREE from "three";

export function makeScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0f14);

  // Lights
  scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 0.9));

  const dir = new THREE.DirectionalLight(0xffffff, 1.0);
  dir.position.set(3, 6, 2);
  scene.add(dir);

  // Floor
  const floorGeo = new THREE.PlaneGeometry(60, 60);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x111822,
    roughness: 1.0,
    metalness: 0.0
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0;
  scene.add(floor);

  // Grid helper
  const grid = new THREE.GridHelper(24, 24, 0x2a3644, 0x1a2330);
  grid.position.y = 0.001;
  scene.add(grid);

  return { scene };
}
