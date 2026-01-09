/**
 * Minimal 3D physics world (readable, not complete).
 * - Semi-implicit Euler integration
 * - Gravity
 * - Sphere-floor collision (y=0) with restitution
 * - Light damping + floor friction
 * - Soft world bounds (x/z)
 */

export class PhysicsWorld {
  constructor({
    gravity = [0, -9.81, 0],
    floorY = 0,
    restitution = 0.55,
    linearDamping = 0.02,
    worldBounds = { x: 12, z: 12 }
  } = {}) {
    this.g = gravity;
    this.floorY = floorY;
    this.restitution = restitution;
    this.linearDamping = linearDamping;
    this.bounds = worldBounds;
    this.bodies = new Map(); // id -> body
  }

  addBody(body) {
    // body: { id, radius, pos:[x,y,z], vel:[x,y,z], mass }
    this.bodies.set(body.id, body);
  }

  removeBody(id) {
    this.bodies.delete(id);
  }

  step(dt) {
    if (!Number.isFinite(dt) || dt <= 0) return;

    for (const body of this.bodies.values()) {
      // acceleration = gravity
      body.vel[0] += this.g[0] * dt;
      body.vel[1] += this.g[1] * dt;
      body.vel[2] += this.g[2] * dt;

      // damping (simple)
      const damp = Math.max(0, 1 - this.linearDamping);
      body.vel[0] *= damp;
      body.vel[1] *= damp;
      body.vel[2] *= damp;

      // integrate position
      body.pos[0] += body.vel[0] * dt;
      body.pos[1] += body.vel[1] * dt;
      body.pos[2] += body.vel[2] * dt;

      // floor collision
      const minY = this.floorY + body.radius;
      if (body.pos[1] < minY) {
        body.pos[1] = minY;
        if (body.vel[1] < 0) body.vel[1] = -body.vel[1] * this.restitution;

        // simple floor friction
        body.vel[0] *= 0.92;
        body.vel[2] *= 0.92;
      }

      // soft bounds
      const bx = this.bounds.x, bz = this.bounds.z;
      if (body.pos[0] > bx) { body.pos[0] = bx; body.vel[0] = -Math.abs(body.vel[0]) * this.restitution; }
      if (body.pos[0] < -bx){ body.pos[0] = -bx; body.vel[0] =  Math.abs(body.vel[0]) * this.restitution; }
      if (body.pos[2] > bz) { body.pos[2] = bz; body.vel[2] = -Math.abs(body.vel[2]) * this.restitution; }
      if (body.pos[2] < -bz){ body.pos[2] = -bz; body.vel[2] =  Math.abs(body.vel[2]) * this.restitution; }
    }
  }

  getBody(id) {
    return this.bodies.get(id);
  }
}
