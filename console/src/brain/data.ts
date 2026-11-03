// What the brain view fetches besides its topic: the neuron geometry and the per-fly index maps (static files, cached
// per connectome for the life of the tab), older minutes' activity as raw int8 (an LRU bounded by bytes, so playback
// can prefetch ahead), and a pinned neuron's day.

export type FlyName = 'memecoin' | 'kalshi'
export type Mode = 'bootstrap' | '24h'

export type GeometryRef = { url: string; meta_url: string; sha: string; n: number }
export type IndexRef = { url: string; n: number }
export type PathwayItem = { kc: number; mbon: number; delta: number; ratio: number; strategy: string; valence: string }
export type Pathways = { mode: string; key: string; reference: string | null; items: PathwayItem[]; note: string | null; changed: number }

export type GeoMeta = {
  n: number
  pop_order: string[]
  pop_ranges: Record<string, [number, number]>
  cell_type: { names: string[]; index: number[] }
  connectome_sha256: string
  connectome_file: string
  coverage?: { centroid?: number }
  flies?: Record<string, { n: number; index: string }>
}
export type Geometry = { pos: Float32Array; meta: GeoMeta; pop: Uint8Array }

/** One fly's part of a payload (version 1 is normalised to a single memecoin fly without an index map). */
export type FlyData = {
  name: FlyName
  minute: string | null
  stamp: string | null
  activity_b64: string | null
  n: number
  rows: number
  candidates: number
  pathways: Pathways
  strategies: string[]
  index: IndexRef | null
  note: string | null
  minutes: string[]
  running: boolean
}

export type BrainPayload = {
  version: 1 | 2
  unavailable?: string
  geometry?: GeometryRef
  emphasis?: string[]
  message?: string | null
  caption?: string
  caption_rest?: string
  flies?: FlyData[]
  // version 1 carries the memecoin fly's fields at the top level
  minute?: string | null
  stamp?: string | null
  activity_b64?: string | null
  rows?: number
  candidates?: number
  pathways?: Pathways
  strategies?: string[]
  note?: string | null
  minutes?: string[]
  running?: boolean
}

/** A minute's activity for one fly: int8 per neuron of its sub-graph (value / 127 = the rate, −1..1). */
export type Frame = { fly: FlyName; stamp: string; minute: string | null; rows: number; candidates: number; connectome: string; activity: Int8Array }

export function normalise(p: BrainPayload): FlyData[] {
  if (p.flies) return p.flies
  return [{
    name: 'memecoin', minute: p.minute ?? null, stamp: p.stamp ?? null, activity_b64: p.activity_b64 ?? null, n: p.geometry?.n ?? 0,
    rows: p.rows ?? 0, candidates: p.candidates ?? 0, pathways: p.pathways ?? { mode: 'bootstrap', key: 'none', reference: null, items: [], note: null, changed: 0 },
    strategies: p.strategies ?? [], index: null, note: null, minutes: p.minutes ?? [], running: !!p.running,
  }]
}

export function decodeB64(b64: string): Int8Array {
  const s = atob(b64)
  const u = new Uint8Array(s.length)
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i)
  return new Int8Array(u.buffer)
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const pad = (v: number) => String(v).padStart(2, '0')

/** "20261016T1453" → epoch ms (UTC). */
export function stampTime(s: string): number {
  return Date.UTC(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8), +s.slice(9, 11), +s.slice(11, 13))
}

/** "20261016T1453" → "16 Oct 14:53" (the Streamlit slider's %d %b %H:%M). */
export function stampLabel(s: string): string {
  return `${s.slice(6, 8)} ${MONTHS[+s.slice(4, 6) - 1]} ${s.slice(9, 11)}:${s.slice(11, 13)}`
}

/** An ISO minute → "16 Oct 14:53 UTC" (the caption's %d %b %H:%M UTC). */
export function minuteLabel(iso: string): string {
  const d = new Date(iso)
  return `${pad(d.getUTCDate())} ${MONTHS[d.getUTCMonth()]} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`
}

// ---------------------------------------------------------------- static files, cached per connectome
const geomCache = new Map<string, Promise<Geometry>>()
const indexCache = new Map<string, Promise<Int32Array>>()
const inverseCache = new WeakMap<Int32Array, Int32Array>()

async function fetchOk(url: string, what: string): Promise<Response> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${what} ${r.status}`)
  return r
}

function popOf(meta: GeoMeta): Uint8Array {
  const idx = new Uint8Array(meta.n)
  meta.pop_order.forEach((name, i) => { const [lo, hi] = meta.pop_ranges[name]; idx.fill(i, lo, hi) })
  return idx
}

export function loadGeometry(g: GeometryRef): Promise<Geometry> {
  let p = geomCache.get(g.sha)
  if (!p) {
    p = (async () => {
      const [bin, meta] = await Promise.all([fetchOk(g.url, 'geometry').then((r) => r.arrayBuffer()), fetchOk(g.meta_url, 'geometry.json').then((r) => r.json() as Promise<GeoMeta>)])
      const pos = new Float32Array(bin)
      if (pos.length !== meta.n * 3) throw new Error('geometry size mismatch')
      return { pos, meta, pop: popOf(meta) }
    })()
    p.catch(() => geomCache.delete(g.sha))
    geomCache.set(g.sha, p)
  }
  return p
}

export function loadIndex(ix: IndexRef): Promise<Int32Array> {
  let p = indexCache.get(ix.url)
  if (!p) {
    p = (async () => {
      const arr = new Int32Array(await (await fetchOk(ix.url, 'index')).arrayBuffer())
      if (arr.length !== ix.n) throw new Error('index size mismatch')
      return arr
    })()
    p.catch(() => indexCache.delete(ix.url))
    indexCache.set(ix.url, p)
  }
  return p
}

/** full-graph index → the fly's sub-graph index (−1 where the fly has no such neuron). */
export function inverseIndex(ix: Int32Array, n: number): Int32Array {
  let inv = inverseCache.get(ix)
  if (!inv) {
    inv = new Int32Array(n).fill(-1)
    for (let k = 0; k < ix.length; k++) if (ix[k] >= 0 && ix[k] < n) inv[ix[k]] = k
    inverseCache.set(ix, inv)
  }
  return inv
}

// ---------------------------------------------------------------- older minutes (playback, the slider)
const BUDGET = 64 * 1024 * 1024          // a day of memecoin minutes is ~40 MB; Kalshi's sub-graph is 3x larger
const frames = new Map<string, Frame>()  // insertion order = recency
const inflight = new Map<string, Promise<Frame | null>>()
let bytes = 0

const key = (fly: FlyName, stamp: string) => `${fly}/${stamp}`

export function cachedFrame(fly: FlyName, stamp: string): Frame | undefined {
  const k = key(fly, stamp), f = frames.get(k)
  if (f) { frames.delete(k); frames.set(k, f) }
  return f
}

function remember(f: Frame): void {
  const k = key(f.fly, f.stamp)
  if (frames.has(k)) return
  frames.set(k, f); bytes += f.activity.byteLength
  for (const [ok, old] of frames) {
    if (bytes <= BUDGET) break
    frames.delete(ok); bytes -= old.activity.byteLength
  }
}

/** Fetch a minute (deduplicated; resolves null when the minute is gone or the request failed). */
export function fetchFrame(fly: FlyName, stamp: string): Promise<Frame | null> {
  const hit = cachedFrame(fly, stamp)
  if (hit) return Promise.resolve(hit)
  const k = key(fly, stamp)
  let p = inflight.get(k)
  if (!p) {
    p = (async () => {
      try {
        const r = await fetch(`/api/brain/activity/${fly}/${stamp}`)
        if (!r.ok) return null
        const f: Frame = { fly, stamp, minute: r.headers.get('X-Minute'), rows: +(r.headers.get('X-Rows') || 0), candidates: +(r.headers.get('X-Candidates') || 0),
          connectome: r.headers.get('X-Connectome') || '', activity: new Int8Array(await r.arrayBuffer()) }
        remember(f)
        return f
      } catch {
        return null
      } finally {
        inflight.delete(k)
      }
    })()
    inflight.set(k, p)
  }
  return p
}

export function inFlight(): number {
  return inflight.size
}

/** The frame the topic carried (its newest minute), remembered like a fetched one so playback never refetches it. */
export function payloadFrame(f: FlyData): Frame | null {
  if (!f.stamp || !f.activity_b64) return null
  const hit = cachedFrame(f.name, f.stamp)
  if (hit) return hit
  const fr: Frame = { fly: f.name, stamp: f.stamp, minute: f.minute, rows: f.rows, candidates: f.candidates, connectome: '', activity: decodeB64(f.activity_b64) }
  remember(fr)
  return fr
}

export type NeuronPoint = { t: string; v: number; candidates: number }

const histories = new Map<string, NeuronPoint[]>()      // fly/index/newest minute → its day (a few pins' worth)

/** One neuron's activity at every minute on disk; ``newest`` keys the cache (the server's answer changes once a minute). */
export async function neuronHistory(fly: FlyName, index: number, newest: string, signal?: AbortSignal): Promise<NeuronPoint[]> {
  const k = `${fly}/${index}/${newest}`, hit = histories.get(k)
  if (hit) return hit
  const r = await fetch(`/api/brain/neuron/${fly}/${index}`, { signal })
  if (!r.ok) throw new Error(`neuron ${r.status}`)
  const series = ((await r.json()) as { series: NeuronPoint[] }).series
  histories.set(k, series)
  while (histories.size > 16) histories.delete(histories.keys().next().value!)
  return series
}

/** epoch ms → "15 Oct 15:12" (UTC). */
export function shortLabel(ms: number): string {
  const d = new Date(ms)
  return `${pad(d.getUTCDate())} ${MONTHS[d.getUTCMonth()]} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`
}
