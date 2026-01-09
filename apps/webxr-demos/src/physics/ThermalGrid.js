/**
 * Tiny thermal "digital twin" grid:
 * - N x N scalar field
 * - Diffusion step (explicit stencil)
 * - A few hot spots
 *
 * This is a demo module: it visualizes "physics state evolving"
 * and is useful for semiconductor thermal-twin style prototypes.
 */

export class ThermalGrid {
  constructor(n = 48) {
    this.n = n;
    this.a = new Float32Array(n * n); // temperature field
    this.b = new Float32Array(n * n); // scratch
    this.diffusivity = 0.18; // stable-ish for dt ~ 1/60 with our clamp
    this.decay = 0.999;

    // seed hot spots
    this.addHotspot(0.30, 0.35, 1.0);
    this.addHotspot(0.70, 0.55, 0.9);
    this.addHotspot(0.50, 0.75, 0.7);
  }

  idx(x, y) {
    return y * this.n + x;
  }

  addHotspot(nx, ny, strength = 1.0) {
    const x = Math.floor(nx * (this.n - 1));
    const y = Math.floor(ny * (this.n - 1));
    const r = Math.max(2, Math.floor(this.n * 0.04));

    for (let j = -r; j <= r; j++) {
      for (let i = -r; i <= r; i++) {
        const xx = x + i, yy = y + j;
        if (xx < 0 || yy < 0 || xx >= this.n || yy >= this.n) continue;
        const d2 = i * i + j * j;
        if (d2 > r * r) continue;
        this.a[this.idx(xx, yy)] += strength * Math.exp(-d2 / (r * r));
      }
    }
  }

  step(dt) {
    const n = this.n;
    const a = this.a, b = this.b;
    const k = this.diffusivity;

    // explicit diffusion
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const c = a[this.idx(x, y)];
        const l = a[this.idx(Math.max(x - 1, 0), y)];
        const r = a[this.idx(Math.min(x + 1, n - 1), y)];
        const u = a[this.idx(x, Math.max(y - 1, 0))];
        const d = a[this.idx(x, Math.min(y + 1, n - 1))];

        // Laplacian stencil
        const lap = (l + r + u + d - 4 * c);

        // integrate
        b[this.idx(x, y)] = (c + k * lap * dt) * this.decay;
      }
    }

    // swap
    this.a = b;
    this.b = a;
  }

  get data() {
    return this.a;
  }
}
