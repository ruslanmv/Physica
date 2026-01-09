export function makeHud(el) {
  el.innerHTML = `
    <div>
      <div class="title">Physica • WebXR Digital Twin Loop (Local Demo)</div>
      <div class="hint">
        <span class="badge">Desktop</span> drag to orbit, click to shoot
        <span class="badge">XR</span> ENTER VR, trigger shoots spheres
      </div>
    </div>
    <div class="stats" id="stats"></div>
  `;

  const stats = el.querySelector("#stats");

  return {
    setText(text) {
      if (stats) stats.textContent = text;
    }
  };
}
