export function randRange(a, b) {
  return a + Math.random() * (b - a);
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}
