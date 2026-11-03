// The fly's brain in 3D: the neurons it runs on, coloured by population and lit by each minute's scoring, with the
// KC→MBON synapses it has changed most drawn as lines. Ported from the Streamlit component (ui/brain3d/brain.js) onto
// the bundled three.js. Each fly's activity and pathway indices are over its own sub-graph; a fly with an ``index``
// (sub → full) is placed into the whole brain through it (version 2), and neurons both flies share (the central brain)
// show the central owner's activity. The scene renders on demand only: a frame is drawn when something changed.
import * as THREE from 'three'
import type { FlyName, Geometry } from './data'

const EMPHASIS_SIZE = 2.4, BASE_SIZE = 1.1, DIM = 0.35, REST = 0.7
type RGB = [number, number, number]
const WARM: RGB = [1.0, 0.55, 0.15], COOL: RGB = [0.25, 0.6, 1.0], GREEN: RGB = [0.2, 0.9, 0.35], RED: RGB = [1.0, 0.25, 0.2]
const FLY_WARM: Record<FlyName, RGB> = { memecoin: WARM, kalshi: [1.0, 0.35, 0.75] }     // the Kalshi fly lights magenta/teal
const FLY_COOL: Record<FlyName, RGB> = { memecoin: COOL, kalshi: [0.3, 0.9, 0.85] }
export const POP_COLORS: Record<string, RGB> = {
  KC: [1.0, 0.8, 0.2], MBON_APP: [0.3, 0.95, 0.4], MBON_AV: [1.0, 0.3, 0.3], MBON_OTHER: [1.0, 0.65, 0.2],
  DAN_PAM: [0.95, 0.3, 0.9], DAN_PPL1: [0.8, 0.25, 0.8], DAN_OTHER: [0.7, 0.3, 0.7],
  ORN_FOOD: [0.3, 0.7, 1.0], ORN_DANGER: [0.2, 0.5, 0.9], ALPN: [0.35, 0.8, 0.9], ALLN: [0.3, 0.6, 0.8], LH: [0.4, 0.75, 0.75],
  GRN_SWEET: [0.35, 0.85, 0.8], GRN_BITTER: [0.25, 0.65, 0.75], GRN_OTHER: [0.3, 0.7, 0.7],
  MECH_JO: [0.4, 0.6, 0.9], MECH_BRISTLE: [0.45, 0.55, 0.85], MECH_OTHER: [0.4, 0.5, 0.8],
  THERMO_WARM: [0.9, 0.5, 0.4], THERMO_COOL: [0.5, 0.6, 0.95], THERMO_OTHER: [0.6, 0.55, 0.8],
  CX: [0.65, 0.45, 0.95], HUNGER: [0.9, 0.6, 0.6], DESCENDING: [0.92, 0.92, 0.95], OTHER: [0.45, 0.45, 0.5], VISUAL: [0.55, 0.55, 0.75],
}
export const popColor = (name: string): RGB => POP_COLORS[name] || POP_COLORS.OTHER
export const rgbCss = (c: RGB) => `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`

const POINT_VS = `
attribute float psize; attribute float alpha; attribute vec3 col;
varying vec3 vColor; varying float vAlpha; uniform float uScale;
void main() { vColor = col; vAlpha = alpha; vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = psize * uScale * (320.0 / -mv.z); gl_Position = projectionMatrix * mv; }`
const POINT_FS = `
varying vec3 vColor; varying float vAlpha;
void main() { if (vAlpha < 0.05) discard; vec2 d = gl_PointCoord - 0.5; float r = dot(d, d); if (r > 0.25) discard;
  gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.25, 0.1, r)); }`
// The glow: a second point layer for the neurons that are active this minute -- a soft gaussian halo in the activity's
// hue, sized and brightened by its magnitude, blended additively so neighbouring halos build into a haze. It shares the
// base layer's positions, sizes and visibility (the legend hides both at once) and only the vertices with hstr > 0 draw.
// (GLSL has no int → float promotion: the constants go in as float literals — "4.0", never "4", which kept the
// Streamlit version's halo shader from compiling.)
const GLOW_MIN = 0.12, GLOW_SIZE = [2.6, 4.0], GLOW_ALPHA = [0.16, 0.42]
const fl = (x: number) => x.toFixed(3)
const HALO_VS = `
attribute float psize; attribute float alpha; attribute vec3 hcol; attribute float hstr;
varying vec3 vColor; varying float vA; uniform float uScale;
void main() { vColor = hcol; vA = alpha * (hstr > 0.0 ? ${fl(GLOW_ALPHA[0])} + ${fl(GLOW_ALPHA[1])} * hstr : 0.0);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = psize * (${fl(GLOW_SIZE[0])} + ${fl(GLOW_SIZE[1])} * hstr) * uScale * (320.0 / -mv.z); gl_Position = projectionMatrix * mv; }`
const HALO_FS = `
varying vec3 vColor; varying float vA;
void main() { if (vA < 0.01) discard; vec2 d = gl_PointCoord - 0.5; float r2 = dot(d, d) * 4.0;
  float a = exp(-3.0 * r2) - exp(-3.0); if (a <= 0.0) discard; gl_FragColor = vec4(vColor, vA * a); }`
// The pinned neuron: a ring of constant screen size drawn over everything.
const RING_VS = `uniform float uSize; void main() { gl_PointSize = uSize; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`
const RING_FS = `uniform vec3 uColor; void main() { float d = length(gl_PointCoord - 0.5); if (d > 0.5 || d < 0.34) discard;
  gl_FragColor = vec4(uColor, smoothstep(0.5, 0.44, d) * smoothstep(0.34, 0.4, d)); }`

/** One fly's activity for the scene: ``index`` null places it 1:1 (the memecoin sub-graph geometry). */
export type Layer = { name: FlyName; activity: Int8Array | null; index: Int32Array | null }
/** A pathway in scene indices, grouped by ``group`` (the strategy, prefixed with the fly in the whole brain). */
export type SceneLine = { kc: number; mbon: number; delta: number; ratio: number; group: string }

function cssVar(el: Element, name: string, fallback: string): string {
  return getComputedStyle(el).getPropertyValue(name).trim() || fallback
}
function hexToRgb(s: string): RGB | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(s.trim())
  if (!m) return null
  const n = parseInt(m[1], 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

/** The strategy tints: the brand's chart colours (magenta, violet, lime, blue, orange, …) in order; "shared" is grey. */
export function strategyTints(el: Element, groups: string[]): Record<string, RGB> {
  const vars: [string, string][] = [['--fly-magenta', '#ff2fb9'], ['--fly-violet', '#8b5cf6'], ['--fly-lime', '#b6ff3b'], ['--fly-blue', '#4cc3ff'],
    ['--fly-orange', '#ffb020'], ['--fly-red', '#ff4d6d']]
  const list = vars.map(([v, fb]) => hexToRgb(cssVar(el, v, fb)) || hexToRgb(fb)!).concat([hexToRgb('#2dd4bf')!, hexToRgb('#f472b6')!])
  const out: Record<string, RGB> = {}
  let i = 0
  for (const g of groups) out[g] = g === 'shared' || g.endsWith(':shared') ? [0.8, 0.8, 0.8] : list[i++ % list.length]
  out.shared = [0.8, 0.8, 0.8]
  return out
}

export type HoverInfo = { index: number; x: number; y: number }

export class BrainScene {
  readonly canvas: HTMLCanvasElement
  readonly geom: Geometry
  private root: HTMLElement
  private tip: HTMLDivElement
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(45, 1, 1, 5000)
  private points: THREE.Points
  private halo: THREE.Points
  private ring: THREE.Points
  private colAttr: THREE.BufferAttribute
  private alphaAttr: THREE.BufferAttribute
  private hcolAttr: THREE.BufferAttribute
  private hstrAttr: THREE.BufferAttribute
  private lines = new Map<string, THREE.LineSegments>()
  private orbit: { theta: number; phi: number; dist: number; target: THREE.Vector3; minDist: number; maxDist: number; radius: number; zoomed: boolean }
  private raf = 0
  private dragging = false
  private disposed = false
  private listeners: [EventTarget, string, EventListener, AddEventListenerOptions | undefined][] = []
  private ro: ResizeObserver
  private hiddenGroups = new Set<string>()
  private activity: Int8Array | null = null          // per scene neuron
  private owner: Uint8Array | null = null            // 0 none, 1 memecoin, 2 kalshi: which fly lit the neuron (whole brain only)
  /** Called with the scene index of a clicked neuron (a click, not a drag). */
  onPick: ((index: number) => void) | null = null

  constructor(root: HTMLElement, geom: Geometry, emphasis: string[]) {
    this.root = root; this.geom = geom
    const meta = geom.meta, n = meta.n
    this.canvas = document.createElement('canvas'); this.canvas.className = 'brain3d-canvas'
    this.tip = document.createElement('div'); this.tip.className = 'brain3d-tip'
    root.prepend(this.canvas); root.appendChild(this.tip)
    try {
      this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true })
    } catch (e) {
      this.canvas.remove(); this.tip.remove()
      throw e
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.setClearColor(0x000000, 0)
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(geom.pos, 3))
    const col = new Float32Array(n * 3), psize = new Float32Array(n), alpha = new Float32Array(n).fill(1)
    const emph = new Set(emphasis)
    meta.pop_order.forEach((name) => { const [lo, hi] = meta.pop_ranges[name]; psize.fill(emph.has(name) ? EMPHASIS_SIZE : BASE_SIZE, lo, hi) })
    this.colAttr = new THREE.BufferAttribute(col, 3); this.alphaAttr = new THREE.BufferAttribute(alpha, 1)
    g.setAttribute('col', this.colAttr); g.setAttribute('psize', new THREE.BufferAttribute(psize, 1)); g.setAttribute('alpha', this.alphaAttr)
    g.computeBoundingSphere()
    const uniforms = { uScale: { value: this.renderer.getPixelRatio() } }
    const mat = new THREE.ShaderMaterial({ vertexShader: POINT_VS, fragmentShader: POINT_FS, uniforms, transparent: true, depthWrite: false, depthTest: true })
    this.points = new THREE.Points(g, mat); this.scene.add(this.points)
    const hg = new THREE.BufferGeometry()                         // the glow layer shares position, size and visibility with the points
    for (const name of ['position', 'psize', 'alpha']) hg.setAttribute(name, g.getAttribute(name))
    this.hcolAttr = new THREE.BufferAttribute(new Float32Array(n * 3), 3); this.hstrAttr = new THREE.BufferAttribute(new Float32Array(n), 1)
    hg.setAttribute('hcol', this.hcolAttr); hg.setAttribute('hstr', this.hstrAttr)
    hg.boundingSphere = g.boundingSphere
    const hmat = new THREE.ShaderMaterial({ vertexShader: HALO_VS, fragmentShader: HALO_FS, uniforms, transparent: true, depthWrite: false, depthTest: false,
      blending: this.isDark() ? THREE.AdditiveBlending : THREE.NormalBlending })
    this.halo = new THREE.Points(hg, hmat); this.halo.renderOrder = 1; this.halo.frustumCulled = false; this.halo.visible = false; this.scene.add(this.halo)
    const rg = new THREE.BufferGeometry(); rg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3))
    const ringColor = hexToRgb(cssVar(root, '--fly-lime', '#b6ff3b')) || [0.71, 1, 0.23]
    const rmat = new THREE.ShaderMaterial({ vertexShader: RING_VS, fragmentShader: RING_FS, transparent: true, depthWrite: false, depthTest: false,
      uniforms: { uSize: { value: 22 * this.renderer.getPixelRatio() }, uColor: { value: new THREE.Color(...ringColor) } } })
    this.ring = new THREE.Points(rg, rmat); this.ring.renderOrder = 3; this.ring.frustumCulled = false; this.ring.visible = false; this.scene.add(this.ring)
    const r = g.boundingSphere ? g.boundingSphere.radius : 300
    this.orbit = { theta: 0.6, phi: 1.15, dist: (r / Math.sin(Math.PI / 8)) * 1.15, target: new THREE.Vector3(), minDist: r * 0.15, maxDist: r * 8, radius: r, zoomed: false }
    this.recolor([], 'kalshi')
    this.placeCamera(); this.controls(); this.hover()
    this.ro = new ResizeObserver(() => this.fit()); this.ro.observe(root); this.fit()
  }

  private isDark(): boolean {             // additive glow reads on a dark ground; on a light theme it would wash to white
    const c = hexToRgb(cssVar(this.root, '--fly-bg', '#121016')) || [0.07, 0.06, 0.09]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] < 0.5
  }

  private on(target: EventTarget, type: string, fn: EventListener, opts?: AddEventListenerOptions) {
    target.addEventListener(type, fn, opts); this.listeners.push([target, type, fn, opts])
  }

  private placeCamera() {
    const s = this.orbit, c = this.camera, sp = Math.sin(s.phi)
    c.position.set(s.target.x + s.dist * sp * Math.sin(s.theta), s.target.y + s.dist * Math.cos(s.phi), s.target.z + s.dist * sp * Math.cos(s.theta))
    c.lookAt(s.target)
  }

  requestRender() {
    if (this.raf || this.disposed) return
    this.raf = requestAnimationFrame(() => { this.raf = 0; if (!this.disposed) this.renderer.render(this.scene, this.camera) })
  }

  private fit() {
    const w = this.root.clientWidth || 800, h = this.root.clientHeight || 420, s = this.orbit
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix()
    if (!s.zoomed) {        // frame the brain in the narrower field of view (a flat brain fills less than its bounding sphere: 0.8)
      const half = Math.min(Math.PI / 8, Math.atan(Math.tan(Math.PI / 8) * (w / h)))
      s.dist = (s.radius / Math.sin(half)) * 0.8; this.placeCamera()
    }
    this.requestRender()
  }

  // Hand-written orbit controls: drag rotates, right- or shift-drag pans, the wheel zooms, a double click recentres.
  // A press that barely moves is a click: it pins the neuron under the pointer.
  private controls() {
    const s = this.orbit, cv = this.canvas
    let last: [number, number] | null = null, mode: 'pan' | 'rot' | null = null, down: [number, number, number] | null = null
    this.on(cv, 'pointerdown', ((e: PointerEvent) => {
      last = [e.clientX, e.clientY]; down = [e.clientX, e.clientY, performance.now()]; mode = e.button === 2 || e.shiftKey ? 'pan' : 'rot'
      this.dragging = true; cv.classList.add('dragging'); cv.setPointerCapture(e.pointerId); this.tip.style.display = 'none'
    }) as EventListener)
    this.on(cv, 'pointermove', ((e: PointerEvent) => {
      if (!last) return
      const dx = e.clientX - last[0], dy = e.clientY - last[1]; last = [e.clientX, e.clientY]
      if (mode === 'rot') { s.theta -= dx * 0.006; s.phi = Math.min(Math.PI - 0.05, Math.max(0.05, s.phi - dy * 0.006)) }
      else {
        const c = this.camera, k = s.dist * 0.0015
        const right = new THREE.Vector3().setFromMatrixColumn(c.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(c.matrix, 1)
        s.target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k)
      }
      this.placeCamera(); this.requestRender()
    }) as EventListener)
    const end = ((e: PointerEvent) => {
      if (!last) return
      last = null; this.dragging = false; cv.classList.remove('dragging')
      try { cv.releasePointerCapture(e.pointerId) } catch { /* already released */ }
      if (down && e.type === 'pointerup' && e.button === 0 && !e.shiftKey && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5 && performance.now() - down[2] < 500) {
        const i = this.pick(e.clientX, e.clientY)
        if (i !== null) this.onPick?.(i)
      }
      down = null
    }) as EventListener
    this.on(cv, 'pointerup', end); this.on(cv, 'pointercancel', end)
    this.on(cv, 'wheel', ((e: WheelEvent) => {
      e.preventDefault(); s.zoomed = true; s.dist = Math.min(s.maxDist, Math.max(s.minDist, s.dist * Math.exp(e.deltaY * 0.0012))); this.placeCamera(); this.requestRender()
    }) as EventListener, { passive: false })
    this.on(cv, 'contextmenu', (e) => e.preventDefault())
    this.on(cv, 'dblclick', () => { s.target.set(0, 0, 0); this.placeCamera(); this.requestRender() })
  }

  private ray = new THREE.Raycaster()
  private ndc = new THREE.Vector2()

  /** The visible neuron under a viewport point, or null. */
  pick(clientX: number, clientY: number): number | null {
    const rect = this.canvas.getBoundingClientRect(), alpha = this.alphaAttr.array as Float32Array
    this.ray.params.Points = { threshold: 1.5 }
    this.ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1)
    this.ray.setFromCamera(this.ndc, this.camera)
    const hit = this.ray.intersectObject(this.points).find((h) => h.index !== undefined && alpha[h.index] >= 0.05)
    return hit && hit.index !== undefined ? hit.index : null
  }

  /** "cell type · population" (and the activity when the neuron is lit). */
  describe(i: number): string {
    const a = this.valueAt(i)
    return `${this.cellType(i)} · ${this.population(i)}` + (a !== null ? ` · ${a.toFixed(2)}` : '')
  }

  cellType(i: number): string {
    const ct = this.geom.meta.cell_type
    return ct.names[ct.index[i]] || 'untyped'
  }

  population(i: number): string {
    return this.geom.meta.pop_order[this.geom.pop[i]]
  }

  /** The neuron's activity this minute in −1..1, or null when no fly lit it. */
  valueAt(i: number): number | null {
    if (!this.activity || (this.owner && !this.owner[i])) return null
    return this.activity[i] / 127
  }

  private hover() {
    const cv = this.canvas
    let pending: PointerEvent | null = null
    this.on(cv, 'pointermove', ((e: PointerEvent) => {
      if (this.dragging) return
      const first = !pending
      pending = e
      if (!first) return
      requestAnimationFrame(() => {
        if (!pending || this.disposed) return
        const ev = pending; pending = null
        const i = this.pick(ev.clientX, ev.clientY)
        if (i === null) { this.tip.style.display = 'none'; cv.style.cursor = ''; return }
        const rect = cv.getBoundingClientRect()
        this.tip.textContent = this.describe(i)
        this.tip.style.display = 'block'; this.tip.style.left = `${ev.clientX - rect.left + 12}px`; this.tip.style.top = `${ev.clientY - rect.top + 12}px`
        cv.style.cursor = 'pointer'
      })
    }) as EventListener)
    this.on(cv, 'pointerleave', () => { this.tip.style.display = 'none' })
  }

  /** Colour every neuron: population colour at rest, the lighting fly's warm/cool hue by the activity's sign and size. */
  recolor(layers: Layer[], centralOwner: FlyName) {
    const meta = this.geom.meta, n = meta.n, col = this.colAttr.array as Float32Array, pop = this.geom.pop
    const base = meta.pop_order.map(popColor)
    const whole = layers.some((l) => l.index)
    if (whole) {
      const act = new Int8Array(n), owner = new Uint8Array(n)
      const order = layers.slice().sort((a, b) => (a.name === centralOwner ? 1 : 0) - (b.name === centralOwner ? 1 : 0))   // the owner writes last
      for (const l of order) {
        const ix = l.index, a = l.activity
        if (!ix || !a || a.length !== ix.length) continue
        const tag = l.name === 'kalshi' ? 2 : 1
        for (let k = 0; k < ix.length; k++) { const i = ix[k]; act[i] = a[k]; owner[i] = tag }
      }
      this.activity = owner.some((v) => v) ? act : null; this.owner = owner
    } else {
      const a = layers[0]?.activity ?? null
      this.activity = a && a.length === n ? a : null; this.owner = null
    }
    const hcol = this.hcolAttr.array as Float32Array, hstr = this.hstrAttr.array as Float32Array
    let glowing = 0
    for (let i = 0; i < n; i++) {
      const b = base[pop[i]]
      let r: number, g: number, bl: number, s = 0
      if (!this.activity || (this.owner && !this.owner[i])) { r = b[0] * REST; g = b[1] * REST; bl = b[2] * REST }
      else {
        const v = this.activity[i] / 127, mag = Math.abs(v), m = Math.pow(mag, 0.7)
        const fly: FlyName = this.owner && this.owner[i] === 2 ? 'kalshi' : 'memecoin'
        const h = v >= 0 ? FLY_WARM[fly] : FLY_COOL[fly]
        r = b[0] * DIM * (1 - m) + h[0] * m; g = b[1] * DIM * (1 - m) + h[1] * m; bl = b[2] * DIM * (1 - m) + h[2] * m
        if (mag > GLOW_MIN) {                                     // the glow: strength 0..1 above the threshold, in that fly's hue, a hot core toward white
          s = Math.pow((mag - GLOW_MIN) / (1 - GLOW_MIN), 0.8); const w = 0.35 * s * s
          r = r * (1 - w) + w; g = g * (1 - w) + w; bl = bl * (1 - w) + w
          hcol[i * 3] = h[0] * 0.8 + 0.2 * s; hcol[i * 3 + 1] = h[1] * 0.8 + 0.2 * s; hcol[i * 3 + 2] = h[2] * 0.8 + 0.2 * s; glowing++
        }
      }
      col[i * 3] = r; col[i * 3 + 1] = g; col[i * 3 + 2] = bl; hstr[i] = s
    }
    this.colAttr.needsUpdate = true; this.hcolAttr.needsUpdate = true; this.hstrAttr.needsUpdate = true
    this.halo.visible = glowing > 0
    this.requestRender()
  }

  /** Rebuild the pathway lines: one LineSegments per group, green (+) or red (−) mixed with the group's tint, brighter by |ratio|. */
  setLines(items: SceneLine[], tints: Record<string, RGB>) {
    const pos = this.geom.pos, n = this.geom.meta.n
    for (const l of this.lines.values()) { this.scene.remove(l); l.geometry.dispose(); (l.material as THREE.Material).dispose() }
    this.lines.clear()
    const ok = items.filter((it) => it.kc >= 0 && it.kc < n && it.mbon >= 0 && it.mbon < n)
    const groups = new Map<string, SceneLine[]>()
    for (const it of ok) { if (!groups.has(it.group)) groups.set(it.group, []); groups.get(it.group)!.push(it) }
    const maxR = Math.max(1e-9, ...ok.map((it) => Math.abs(it.ratio)))
    for (const [group, its] of groups) {
      const p = new Float32Array(its.length * 6), c = new Float32Array(its.length * 6), tint = tints[group] || tints.shared || [0.8, 0.8, 0.8]
      its.forEach((it, k) => {
        for (let e = 0; e < 2; e++) { const idx = e === 0 ? it.kc : it.mbon; p.set([pos[idx * 3], pos[idx * 3 + 1], pos[idx * 3 + 2]], k * 6 + e * 3) }
        const sign = it.delta > 0 ? GREEN : RED, w = 0.35 + (0.65 * Math.abs(it.ratio)) / maxR
        const rgb = [0, 1, 2].map((j) => (sign[j] * 0.7 + tint[j] * 0.3) * w)
        c.set(rgb, k * 6); c.set(rgb, k * 6 + 3)
      })
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p, 3)); g.setAttribute('color', new THREE.BufferAttribute(c, 3))
      const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 }))
      l.visible = !this.hiddenGroups.has(group); this.scene.add(l); this.lines.set(group, l)
    }
    this.requestRender()
  }

  /** The legend: populations hidden through their alpha (both layers and the raycaster honour it), line groups by visibility. */
  setHidden(pops: Set<string>, groups: Set<string>) {
    const meta = this.geom.meta, alpha = this.alphaAttr.array as Float32Array
    for (const name of meta.pop_order) { const [lo, hi] = meta.pop_ranges[name]; alpha.fill(pops.has(name) ? 0 : 1, lo, hi) }
    this.alphaAttr.needsUpdate = true
    this.hiddenGroups = new Set(groups)
    for (const [g, l] of this.lines) l.visible = !groups.has(g)
    this.requestRender()
  }

  setPinned(i: number | null) {
    if (i === null || i < 0 || i >= this.geom.meta.n) { this.ring.visible = false }
    else {
      const a = this.ring.geometry.getAttribute('position') as THREE.BufferAttribute, pos = this.geom.pos
      a.setXYZ(0, pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]); a.needsUpdate = true; this.ring.visible = true
    }
    this.requestRender()
  }

  dispose() {
    this.disposed = true
    if (this.raf) cancelAnimationFrame(this.raf)
    for (const [t, ty, fn, o] of this.listeners) t.removeEventListener(ty, fn, o)
    this.ro.disconnect()
    for (const l of this.lines.values()) { l.geometry.dispose(); (l.material as THREE.Material).dispose() }
    for (const p of [this.halo, this.points, this.ring]) { p.geometry.dispose(); (p.material as THREE.Material).dispose() }
    this.renderer.dispose(); this.renderer.forceContextLoss()
    this.canvas.remove(); this.tip.remove()
  }
}
