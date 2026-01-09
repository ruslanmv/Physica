export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

export function computeDtSeconds(nowMs, lastMs) {
  const now = nowMs * 0.001;
  const last = lastMs * 0.001;
  const dt = now - last;
  return clamp(dt, 0, 0.033); // cap to ~30 FPS step to keep stability
}
