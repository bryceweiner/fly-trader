// The fly's brain in 3D: the neurons it runs on, coloured by population and lit by each minute's scoring, with the
// KC→MBON synapses it has changed most drawn as lines. Standalone on purpose: it depends only on {data, parentElement}
// (Streamlit's Custom Component v2 calls the default export with them; a web page can call it the same way with a
// fetched payload). The data contract is documented in ui/brain3d/__init__.py. Version 2 payloads carry ``flies``: two
// flies drawn into the whole brain, each fly's activity and pathway indices mapped through its ``index`` (sub → full);
// neurons both flies share (the central brain) show ``central_owner``'s activity.
const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";
const EMPHASIS_SIZE = 2.4, BASE_SIZE = 1.1, DIM = 0.35, REST = 0.7;
const WARM = [1.0, 0.55, 0.15], COOL = [0.25, 0.6, 1.0], GREEN = [0.2, 0.9, 0.35], RED = [1.0, 0.25, 0.2];
const FLY_WARM = { memecoin: WARM, kalshi: [1.0, 0.35, 0.75] }, FLY_COOL = { memecoin: COOL, kalshi: [0.3, 0.9, 0.85] };   // the Kalshi fly lights magenta/teal
const PALETTE_EXTRA = { VISUAL: [0.55, 0.55, 0.75] };
const PALETTE = {
  KC: [1.0, 0.8, 0.2], MBON_APP: [0.3, 0.95, 0.4], MBON_AV: [1.0, 0.3, 0.3], MBON_OTHER: [1.0, 0.65, 0.2],
  DAN_PAM: [0.95, 0.3, 0.9], DAN_PPL1: [0.8, 0.25, 0.8], DAN_OTHER: [0.7, 0.3, 0.7],
  ORN_FOOD: [0.3, 0.7, 1.0], ORN_DANGER: [0.2, 0.5, 0.9], ALPN: [0.35, 0.8, 0.9], ALLN: [0.3, 0.6, 0.8], LH: [0.4, 0.75, 0.75],
  GRN_SWEET: [0.35, 0.85, 0.8], GRN_BITTER: [0.25, 0.65, 0.75], GRN_OTHER: [0.3, 0.7, 0.7],
  MECH_JO: [0.4, 0.6, 0.9], MECH_BRISTLE: [0.45, 0.55, 0.85], MECH_OTHER: [0.4, 0.5, 0.8],
  THERMO_WARM: [0.9, 0.5, 0.4], THERMO_COOL: [0.5, 0.6, 0.95], THERMO_OTHER: [0.6, 0.55, 0.8],
  CX: [0.65, 0.45, 0.95], HUNGER: [0.9, 0.6, 0.6], DESCENDING: [0.92, 0.92, 0.95], OTHER: [0.45, 0.45, 0.5], VISUAL: [0.55, 0.55, 0.75],
};
const POINT_VS = `
attribute float psize; attribute float alpha; attribute vec3 col;
varying vec3 vColor; varying float vAlpha; uniform float uScale;
void main() { vColor = col; vAlpha = alpha; vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = psize * uScale * (320.0 / -mv.z); gl_Position = projectionMatrix * mv; }`;
const POINT_FS = `
varying vec3 vColor; varying float vAlpha;
void main() { if (vAlpha < 0.05) discard; vec2 d = gl_PointCoord - 0.5; float r = dot(d, d); if (r > 0.25) discard;
  gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.25, 0.1, r)); }`;
// The glow: a second point layer for the neurons that are active this minute -- a soft gaussian halo in the activity's
// hue, sized and brightened by its magnitude, blended additively so neighbouring halos build into a haze. It shares the
// base layer's positions, sizes and visibility (the legend hides both at once) and only the vertices with hstr > 0 draw.
const GLOW_MIN = 0.12, GLOW_SIZE = [2.6, 4.0], GLOW_ALPHA = [0.16, 0.42];
const HALO_VS = `
attribute float psize; attribute float alpha; attribute vec3 hcol; attribute float hstr;
varying vec3 vColor; varying float vA; uniform float uScale;
void main() { vColor = hcol; vA = alpha * (hstr > 0.0 ? ${GLOW_ALPHA[0]} + ${GLOW_ALPHA[1]} * hstr : 0.0);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = psize * (${GLOW_SIZE[0]} + ${GLOW_SIZE[1]} * hstr) * uScale * (320.0 / -mv.z); gl_Position = projectionMatrix * mv; }`;
const HALO_FS = `
varying vec3 vColor; varying float vA;
void main() { if (vA < 0.01) discard; vec2 d = gl_PointCoord - 0.5; float r2 = dot(d, d) * 4.0;
  float a = exp(-3.0 * r2) - exp(-3.0); if (a <= 0.0) discard; gl_FragColor = vec4(vColor, vA * a); }`;

let threeP = null;
const geomCache = new Map();          // sha → {pos, meta}
const instances = new WeakMap();      // parentElement → instance

function three() { return threeP || (threeP = import(THREE_URL)); }

async function fetchGeometry(g) {
  if (geomCache.has(g.sha)) return geomCache.get(g.sha);
  const [bin, meta] = await Promise.all([fetch(g.url).then(r => { if (!r.ok) throw new Error(`geometry ${r.status}`); return r.arrayBuffer(); }),
                                         fetch(g.meta_url).then(r => { if (!r.ok) throw new Error(`geometry.json ${r.status}`); return r.json(); })]);
  const geom = { pos: new Float32Array(bin), meta };
  if (geom.pos.length !== meta.n * 3) throw new Error("geometry size mismatch");
  geomCache.set(g.sha, geom);
  return geom;
}

const indexCache = new Map();          // url → Int32Array (sub-graph index → full-graph index)
async function fetchIndex(ix) {
  if (!ix) return null;
  if (indexCache.has(ix.url)) return indexCache.get(ix.url);
  const buf = await fetch(ix.url).then(r => { if (!r.ok) throw new Error(`index ${r.status}`); return r.arrayBuffer(); });
  const arr = new Int32Array(buf); if (arr.length !== ix.n) throw new Error("index size mismatch");
  indexCache.set(ix.url, arr); return arr;
}

function el(tag, cls, parent) { const e = document.createElement(tag); if (cls) e.className = cls; parent.appendChild(e); return e; }

function ensureDom(parent) {
  let root = parent.querySelector(".brain3d");
  if (!root) root = el("div", "brain3d", parent);
  const get = (cls, tag) => root.querySelector("." + cls) || el(tag, cls, root);
  return { root, canvas: get("brain3d-canvas", "canvas"), legend: get("brain3d-legend", "div"), tip: get("brain3d-tip", "div"),
           msg: get("brain3d-msg", "div"), hint: get("brain3d-hint", "div") };
}

function cssVar(root, name, fallback) { const v = getComputedStyle(root).getPropertyValue(name).trim(); return v || fallback; }
function hexToRgb(s) { const m = /^#?([0-9a-f]{6})$/i.exec(s.trim()); if (!m) return null; const n = parseInt(m[1], 16); return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; }
function isDark(root) {              // additive glow reads on a dark ground; on a light theme it would wash to white
  const c = hexToRgb(cssVar(root, "--st-secondary-background-color", "#0e1117")) || [0.06, 0.07, 0.09];
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] < 0.5;
}
function popOf(meta) {                // per-vertex population index, computed once per geometry
  const idx = new Uint8Array(meta.n);
  meta.pop_order.forEach((name, i) => { const [lo, hi] = meta.pop_ranges[name]; idx.fill(i, lo, hi); });
  return idx;
}

function buildScene(inst, THREE, geom) {
  const { dom } = inst; const meta = geom.meta; const n = meta.n;
  const renderer = new THREE.WebGLRenderer({ canvas: dom.canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 1, 5000);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(geom.pos, 3));
  const col = new Float32Array(n * 3), psize = new Float32Array(n), alpha = new Float32Array(n).fill(1);
  const pop = popOf(meta); const emph = new Set(inst.emphasis || []);
  meta.pop_order.forEach((name, i) => { const s = emph.has(name) ? EMPHASIS_SIZE : BASE_SIZE; const [lo, hi] = meta.pop_ranges[name]; psize.fill(s, lo, hi); });
  g.setAttribute("col", new THREE.BufferAttribute(col, 3)); g.setAttribute("psize", new THREE.BufferAttribute(psize, 1)); g.setAttribute("alpha", new THREE.BufferAttribute(alpha, 1));
  g.computeBoundingSphere();
  const mat = new THREE.ShaderMaterial({ vertexShader: POINT_VS, fragmentShader: POINT_FS, uniforms: { uScale: { value: renderer.getPixelRatio() } },
                                        transparent: true, depthWrite: false, depthTest: true });
  const points = new THREE.Points(g, mat); scene.add(points);
  const hg = new THREE.BufferGeometry();                       // the glow layer shares position, size and visibility with the points
  for (const name of ["position", "psize", "alpha"]) hg.setAttribute(name, g.getAttribute(name));
  hg.setAttribute("hcol", new THREE.BufferAttribute(new Float32Array(n * 3), 3)); hg.setAttribute("hstr", new THREE.BufferAttribute(new Float32Array(n), 1));
  hg.boundingSphere = g.boundingSphere;
  const hmat = new THREE.ShaderMaterial({ vertexShader: HALO_VS, fragmentShader: HALO_FS, uniforms: mat.uniforms, transparent: true, depthWrite: false, depthTest: false,
                                         blending: isDark(dom.root) ? THREE.AdditiveBlending : THREE.NormalBlending });
  const halo = new THREE.Points(hg, hmat); halo.renderOrder = 1; halo.frustumCulled = false; scene.add(halo);
  const r = g.boundingSphere ? g.boundingSphere.radius : 300;
  Object.assign(inst, { THREE, renderer, scene, camera, points, halo, geom, pop, colAttr: g.getAttribute("col"), alphaAttr: g.getAttribute("alpha"),
                        hcolAttr: hg.getAttribute("hcol"), hstrAttr: hg.getAttribute("hstr"),
                        lines: new Map(), orbitState: { theta: 0.6, phi: 1.15, dist: r / Math.sin(Math.PI / 8) * 1.15, target: new THREE.Vector3(), minDist: r * 0.15, maxDist: r * 8 },
                        raf: 0, hidden: new Set(), hiddenStrategies: new Set(), dragging: false, listeners: [] });
  placeCamera(inst); orbit(inst); hover(inst); resize(inst);
}

function placeCamera(inst) {
  const s = inst.orbitState, c = inst.camera;
  const sp = Math.sin(s.phi);
  c.position.set(s.target.x + s.dist * sp * Math.sin(s.theta), s.target.y + s.dist * Math.cos(s.phi), s.target.z + s.dist * sp * Math.cos(s.theta));
  c.lookAt(s.target);
}

function requestRender(inst) {
  if (inst.raf || inst.disposed) return;
  inst.raf = requestAnimationFrame(() => { inst.raf = 0; if (!inst.disposed) inst.renderer.render(inst.scene, inst.camera); });
}

function on(inst, target, type, fn, opts) { target.addEventListener(type, fn, opts); inst.listeners.push([target, type, fn, opts]); }

function orbit(inst) {                // ~40 lines instead of OrbitControls: bare "three" imports need an import map a component cannot add
  const s = inst.orbitState, cv = inst.dom.canvas; let last = null, mode = null;
  on(inst, cv, "pointerdown", e => { last = [e.clientX, e.clientY]; mode = (e.button === 2 || e.shiftKey) ? "pan" : "rot"; inst.dragging = true; cv.classList.add("dragging"); cv.setPointerCapture(e.pointerId); inst.dom.tip.style.display = "none"; });
  on(inst, cv, "pointermove", e => {
    if (!last) return;
    const dx = e.clientX - last[0], dy = e.clientY - last[1]; last = [e.clientX, e.clientY];
    if (mode === "rot") { s.theta -= dx * 0.006; s.phi = Math.min(Math.PI - 0.05, Math.max(0.05, s.phi - dy * 0.006)); }
    else { const c = inst.camera, k = s.dist * 0.0015; const right = new inst.THREE.Vector3().setFromMatrixColumn(c.matrix, 0), up = new inst.THREE.Vector3().setFromMatrixColumn(c.matrix, 1);
           s.target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k); }
    placeCamera(inst); requestRender(inst);
  });
  const end = e => { if (!last) return; last = null; inst.dragging = false; cv.classList.remove("dragging"); try { cv.releasePointerCapture(e.pointerId); } catch (_) {} };
  on(inst, cv, "pointerup", end); on(inst, cv, "pointercancel", end);
  on(inst, cv, "wheel", e => { e.preventDefault(); s.dist = Math.min(s.maxDist, Math.max(s.minDist, s.dist * Math.exp(e.deltaY * 0.0012))); placeCamera(inst); requestRender(inst); }, { passive: false });
  on(inst, cv, "contextmenu", e => e.preventDefault());
  on(inst, cv, "dblclick", () => { s.target.set(0, 0, 0); placeCamera(inst); requestRender(inst); });
}

function resize(inst) {
  const fit = () => {
    const w = inst.dom.root.clientWidth || 800, h = inst.dom.root.clientHeight || 420;
    inst.renderer.setSize(w, h, false); inst.camera.aspect = w / h; inst.camera.updateProjectionMatrix(); requestRender(inst);
  };
  inst.ro = new ResizeObserver(fit); inst.ro.observe(inst.dom.root); fit();
}

function hover(inst) {
  const { THREE, dom } = inst; const ray = new THREE.Raycaster(); ray.params.Points.threshold = 1.5; const ndc = new THREE.Vector2(); let pending = null;
  const names = inst.geom.meta.cell_type, pops = inst.geom.meta.pop_order, alpha = inst.alphaAttr.array;
  on(inst, dom.canvas, "pointermove", e => {
    if (inst.dragging) return;
    pending = e;
    requestAnimationFrame(() => {
      if (!pending || inst.disposed) return; const ev = pending; pending = null;
      const rect = dom.canvas.getBoundingClientRect();
      ndc.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
      ray.setFromCamera(ndc, inst.camera);
      const hit = ray.intersectObject(inst.points).find(h => alpha[h.index] >= 0.05);
      if (!hit) { dom.tip.style.display = "none"; return; }
      const i = hit.index; const a = inst.activity ? (inst.activity[i] / 127).toFixed(2) : null;
      dom.tip.textContent = `${names.names[names.index[i]] || "untyped"} · ${pops[inst.pop[i]]}` + (a !== null ? ` · ${a}` : "");
      dom.tip.style.display = "block"; dom.tip.style.left = `${ev.clientX - rect.left + 12}px`; dom.tip.style.top = `${ev.clientY - rect.top + 12}px`;
    });
  });
  on(inst, dom.canvas, "pointerleave", () => { dom.tip.style.display = "none"; });
}

function decode(b64) { const s = atob(b64); const u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Int8Array(u.buffer); }

function recolor(inst, data) {
  const meta = inst.geom.meta, n = meta.n, col = inst.colAttr.array, pop = inst.pop;
  const base = meta.pop_order.map(p => PALETTE[p] || PALETTE_EXTRA[p] || PALETTE.OTHER);
  let act = null, owner = new Uint8Array(0);           // owner[i]: 0 none, 1 memecoin, 2 kalshi — which fly lit neuron i
  if (data.flies) {
    act = new Int8Array(n); owner = new Uint8Array(n);
    const flies = data.flies.slice().sort((a, b) => (a.name === data.central_owner ? 1 : 0) - (b.name === data.central_owner ? 1 : 0));   // the owner writes last
    for (const f of flies) {
      const ix = inst.indices ? inst.indices.get(f.name) : null; if (!ix || !f.activity_b64) continue;
      const a = decode(f.activity_b64); if (a.length !== ix.length) continue;
      const tag = f.name === "kalshi" ? 2 : 1;
      for (let k = 0; k < ix.length; k++) { const i = ix[k]; act[i] = a[k]; owner[i] = tag; }
    }
    inst.activity = owner.some(v => v) ? act : null; inst.owner = owner;
  } else { const a = data.activity_b64 ? decode(data.activity_b64) : null; inst.activity = a && a.length === n ? a : null; inst.owner = null; }
  const hcol = inst.hcolAttr.array, hstr = inst.hstrAttr.array; let glowing = 0;
  for (let i = 0; i < n; i++) {
    const b = base[pop[i]]; let r, g, bl, s = 0;
    if (!inst.activity || (inst.owner && !inst.owner[i])) { r = b[0] * REST; g = b[1] * REST; bl = b[2] * REST; }
    else { const v = inst.activity[i] / 127, mag = Math.abs(v), m = Math.pow(mag, 0.7); const fly = inst.owner && inst.owner[i] === 2 ? "kalshi" : "memecoin";
           const h = v >= 0 ? FLY_WARM[fly] : FLY_COOL[fly];
           r = b[0] * DIM * (1 - m) + h[0] * m; g = b[1] * DIM * (1 - m) + h[1] * m; bl = b[2] * DIM * (1 - m) + h[2] * m;
           if (mag > GLOW_MIN) {                                  // the glow: strength 0..1 above the threshold, in that fly's hue, a hot core toward white
             s = Math.pow((mag - GLOW_MIN) / (1 - GLOW_MIN), 0.8); const w = 0.35 * s * s;
             r = r * (1 - w) + w; g = g * (1 - w) + w; bl = bl * (1 - w) + w;
             hcol[i * 3] = h[0] * 0.8 + 0.2 * s; hcol[i * 3 + 1] = h[1] * 0.8 + 0.2 * s; hcol[i * 3 + 2] = h[2] * 0.8 + 0.2 * s; glowing++;
           } }
    col[i * 3] = r; col[i * 3 + 1] = g; col[i * 3 + 2] = bl; hstr[i] = s;
  }
  inst.colAttr.needsUpdate = true; inst.hcolAttr.needsUpdate = true; inst.hstrAttr.needsUpdate = true;
  inst.halo.visible = glowing > 0; inst.minute = minuteKey(data);
}

function minuteKey(data) { return data.flies ? JSON.stringify([data.central_owner, data.flies.map(f => f.minute)]) : data.minute; }

function pathwayItems(inst, data) {      // [{...item, kc, mbon in scene indices, strategy}] over every fly (version 2) or the one payload
  if (!data.flies) return ((data.pathways && data.pathways.items) || []).map(it => ({ ...it }));
  const out = [];
  for (const f of data.flies) {
    const ix = inst.indices ? inst.indices.get(f.name) : null; if (!ix) continue;
    for (const it of ((f.pathways && f.pathways.items) || [])) out.push({ ...it, kc: ix[it.kc], mbon: ix[it.mbon], strategy: `${f.name}:${it.strategy}` });
  }
  return out;
}

function strategyList(data) { return data.flies ? data.flies.flatMap(f => (f.strategies || []).map(s => `${f.name}:${s}`)) : (data.strategies || []); }
function pathwayKey(data) { return data.flies ? JSON.stringify(data.flies.map(f => f.pathways ? f.pathways.key : null)) : (data.pathways ? data.pathways.key : null); }

function strategyTints(inst, strategies) {
  const raw = cssVar(inst.dom.root, "--st-chart-categorical-colors", ""); const list = raw.split(",").map(hexToRgb).filter(Boolean);
  const out = {}; strategies.forEach((s, i) => { out[s] = list.length ? list[i % list.length] : [0.8, 0.8, 0.8]; }); out.shared = [0.8, 0.8, 0.8]; return out;
}

function rebuildLines(inst, data) {
  const { THREE, scene } = inst; const pos = inst.geom.pos;
  for (const l of inst.lines.values()) { scene.remove(l); l.geometry.dispose(); l.material.dispose(); } inst.lines.clear();
  const items = pathwayItems(inst, data); const tints = strategyTints(inst, strategyList(data));
  const groups = new Map(); for (const it of items) { if (!groups.has(it.strategy)) groups.set(it.strategy, []); groups.get(it.strategy).push(it); }
  const maxR = Math.max(1e-9, ...items.map(it => Math.abs(it.ratio)));
  for (const [strategy, its] of groups) {
    const p = new Float32Array(its.length * 6), c = new Float32Array(its.length * 6); const tint = tints[strategy] || tints.shared;
    its.forEach((it, k) => {
      for (let e = 0; e < 2; e++) { const idx = e === 0 ? it.kc : it.mbon; p.set([pos[idx * 3], pos[idx * 3 + 1], pos[idx * 3 + 2]], k * 6 + e * 3); }
      const sign = it.delta > 0 ? GREEN : RED, w = 0.35 + 0.65 * Math.abs(it.ratio) / maxR;
      const rgb = [0, 1, 2].map(j => (sign[j] * 0.7 + tint[j] * 0.3) * w); c.set(rgb, k * 6); c.set(rgb, k * 6 + 3);
    });
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(p, 3)); g.setAttribute("color", new THREE.BufferAttribute(c, 3));
    const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 }));
    l.visible = !inst.hiddenStrategies.has(strategy); scene.add(l); inst.lines.set(strategy, l);
  }
  inst.pkey = pathwayKey(data);
}

function legend(inst, data) {
  const meta = inst.geom.meta, box = inst.dom.legend; box.replaceChildren();
  const row = (parent, label, swatchCss, cls, checked, onChange) => {
    const lb = el("label", checked ? "" : "off", parent); const cb = el("input", "", lb); cb.type = "checkbox"; cb.checked = checked;
    const sw = el("span", "sw " + cls, lb); sw.style.background = swatchCss; lb.appendChild(document.createTextNode(label));
    cb.onchange = () => { lb.classList.toggle("off", !cb.checked); onChange(cb.checked); requestRender(inst); };
  };
  const rgb = c => `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
  el("div", "hd", box).textContent = "Populations";
  for (const name of meta.pop_order) {
    const [lo, hi] = meta.pop_ranges[name];
    row(box, `${name} · ${hi - lo}`, rgb(PALETTE[name] || PALETTE.OTHER), "", !inst.hidden.has(name), on => {
      if (on) inst.hidden.delete(name); else inst.hidden.add(name);
      inst.alphaAttr.array.fill(on ? 1 : 0, lo, hi); inst.alphaAttr.needsUpdate = true;
    });
  }
  const strategies = strategyList(data);
  if (strategies.length) {
    el("div", "hd", box).textContent = "Learned pathways (green +, red −)"; const tints = strategyTints(inst, strategies);
    for (const s of [...strategies, "shared"]) {
      row(box, s, rgb(tints[s] || tints.shared), "line", !inst.hiddenStrategies.has(s), on => {
        if (on) inst.hiddenStrategies.delete(s); else inst.hiddenStrategies.add(s);
        const l = inst.lines.get(s); if (l) l.visible = on;
      });
    }
  }
  inst.legendKey = JSON.stringify([meta.connectome_sha256, strategies]);
}

function message(inst, text) { const m = inst.dom.msg; m.textContent = text || ""; m.style.display = text ? "block" : "none"; }

function apply(inst, data) {
  if (!inst.ready || !data) return;
  const lk = JSON.stringify([inst.geom.meta.connectome_sha256, strategyList(data)]); if (inst.legendKey !== lk) legend(inst, data);
  if (minuteKey(data) !== inst.minute || inst.minute === undefined) recolor(inst, data);
  const pk = pathwayKey(data); if (pk !== inst.pkey) rebuildLines(inst, data);
  message(inst, data.message); inst.dom.hint.textContent = "drag to orbit · wheel to zoom · shift-drag to pan · double-click to recentre";
  requestRender(inst);
}

async function boot(inst, data) {
  try {
    if (!data || !data.geometry) { message(inst, "no geometry in the payload"); return; }
    inst.emphasis = data.emphasis || [];
    const THREE = await three();
    let geom; try { geom = await fetchGeometry(data.geometry); } catch (e) { message(inst, `the neuron geometry could not be fetched (${e.message}) — is static serving on?`); return; }
    if (data.flies) {                       // version 2: each fly's sub-graph → scene index map
      inst.indices = new Map();
      for (const f of data.flies) { try { inst.indices.set(f.name, await fetchIndex(f.index)); } catch (e) { message(inst, `${f.name}: index map could not be fetched (${e.message})`); } }
    }
    if (inst.disposed) return;
    buildScene(inst, THREE, geom); inst.ready = true; apply(inst, inst.pending || data); inst.pending = null;
  } catch (e) {
    message(inst, `three.js could not be loaded (${e && e.message ? e.message : e}) — is the console offline?`);
  }
}

function dispose(inst) {
  inst.disposed = true; if (inst.raf) cancelAnimationFrame(inst.raf);
  for (const [t, ty, fn, o] of inst.listeners || []) t.removeEventListener(ty, fn, o);
  if (inst.ro) inst.ro.disconnect();
  if (inst.lines) for (const l of inst.lines.values()) { l.geometry.dispose(); l.material.dispose(); }
  if (inst.halo) { inst.halo.geometry.dispose(); inst.halo.material.dispose(); }
  if (inst.points) { inst.points.geometry.dispose(); inst.points.material.dispose(); }
  if (inst.renderer) { inst.renderer.dispose(); inst.renderer.forceContextLoss(); }
}

export default function (component) {
  const { data, parentElement } = component;
  let inst = instances.get(parentElement);
  if (!inst) {
    inst = { dom: ensureDom(parentElement), ready: false, disposed: false, pending: null, minute: undefined, pkey: undefined, legendKey: null };
    instances.set(parentElement, inst); message(inst, "loading the brain…"); boot(inst, data);
  } else if (inst.ready) apply(inst, data); else inst.pending = data;
  return () => { dispose(inst); instances.delete(parentElement); };
}
