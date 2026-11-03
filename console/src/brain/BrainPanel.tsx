// The brain card: the 3D view (scene.ts) with the Streamlit controls — Follow live, Learned pathways, the central owner
// (whole brain), a minute slider per fly — plus playback through the last 24 hours and a pinned neuron's history.
// Variant "memecoin" is the Overview's (the Solana fly's 41,756 neurons, topic brain.memecoin); "flies" is the Kalshi
// overview's (the whole brain with both flies, topic brain.flies).
import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import { Callout, Caption, Card, Help, Icon, Segmented, Toggle } from '../components/ui'
import { hm } from '../lib/format'
import { useStored, useTopic } from '../lib/hooks'
import {
  cachedFrame, fetchFrame, inFlight, inverseIndex, loadGeometry, loadIndex, minuteLabel, neuronHistory, normalise, payloadFrame, shortLabel, stampLabel, stampTime,
  type BrainPayload, type FlyData, type FlyName, type Frame, type Geometry, type Mode, type NeuronPoint,
} from './data'
import { BrainScene, popColor, rgbCss, strategyTints, type Layer, type SceneLine } from './scene'
import './brain.css'

export type BrainVariant = 'memecoin' | 'flies'

const MODES = ['bootstrap', '24h'] as const
const MODE_LABELS: Record<Mode, string> = { bootstrap: 'since bootstrap', '24h': 'last 24 h' }
const OWNERS = ['kalshi', 'memecoin'] as const
const SPEEDS = ['1', '4', '16'] as const
type Speed = (typeof SPEEDS)[number]
const SPEED_LABELS: Record<Speed, string> = { '1': '1×', '4': '4×', '16': '16×' }
const MAX_INFLIGHT = 6
const HEAD: Record<BrainVariant, { title: string; sub: string; live: string; height: number }> = {
  memecoin: {
    title: "The fly's brain", height: 560,
    sub: "the 41,756 neurons it runs on, lit by each minute's scoring (warm = positive, cool = negative); green and red lines are the KC→MBON synapses it has changed most",
    live: 'Show the newest minute as it arrives; moving the slider turns this off.',
  },
  flies: {
    title: 'One brain, two flies', height: 620,
    sub: 'the optic lobes belong to the Kalshi fly (its inputs enter the photoreceptors), the antennal lobe and the other senses to the memecoin fly; the central brain is shared and shows the fly chosen below',
    live: "Show each fly's newest minute as it arrives; moving the slider turns this off.",
  },
}
const HINT = 'drag to orbit · wheel to zoom · shift-drag to pan · double-click to recentre · click a neuron to pin it'
const FOREIGN = "this minute's activity was recorded on another connectome"

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e))

/** The greatest stamp ≤ ``at`` in a sorted list, or null. */
function stampAt(minutes: string[], at: string): string | null {
  let lo = 0, hi = minutes.length - 1, out: string | null = null
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (minutes[mid] <= at) { out = minutes[mid]; lo = mid + 1 } else hi = mid - 1
  }
  return out
}

/** The index of the first stamp > ``at`` in a sorted list (its length when none is). */
function after(minutes: string[], at: string): number {
  let lo = 0, hi = minutes.length
  while (lo < hi) { const mid = (lo + hi) >> 1; if (minutes[mid] <= at) lo = mid + 1; else hi = mid }
  return lo
}

type Chosen = Partial<Record<FlyName, string | null>>     // a fly's picked minute; null: it had none yet (playback)

export function BrainPanel({ variant }: { variant: BrainVariant }) {
  const flies = variant === 'flies'
  const [storedMode, setMode] = useStored<Mode>(flies ? 'kbrain_pmode' : 'brain_pmode', 'bootstrap')
  const mode: Mode = MODES.includes(storedMode) ? storedMode : 'bootstrap'
  const t = useTopic<BrainPayload>(flies ? 'brain.flies' : 'brain.memecoin', { mode })
  const [last, setLast] = useState<BrainPayload | undefined>(undefined)       // the previous payload stays up while a new mode loads
  useEffect(() => { if (t.data) setLast(t.data) }, [t.data])
  const data = t.data ?? last
  const h = HEAD[variant]
  if (data && !data.unavailable && data.geometry) return <BrainView data={data} variant={variant} mode={mode} setMode={setMode} error={t.error} />
  return (
    <Card title={h.title} icon="neurology" sub={h.sub}>
      {!data ? (t.error ? <Callout kind="error">{t.error}</Callout> : <div className="skeleton" style={{ height: h.height }} />)
        : <Caption>{data.unavailable || 'No brain to draw: the payload has no geometry.'}</Caption>}
    </Card>
  )
}

function BrainView({ data, variant, mode, setMode, error }: { data: BrainPayload; variant: BrainVariant; mode: Mode; setMode: (m: Mode) => void; error?: string }) {
  const whole = variant === 'flies'
  const flies = useMemo(() => normalise(data), [data])
  const [storedOwner, setOwner] = useStored<FlyName>('kbrain_owner', 'kalshi')
  const owner: FlyName = whole ? (OWNERS.includes(storedOwner) ? storedOwner : 'kalshi') : 'memecoin'
  const [storedSpeed, setSpeed] = useStored<Speed>('brain_speed', '1')
  const speed: Speed = SPEEDS.includes(storedSpeed) ? storedSpeed : '1'
  const [legendOpen, setLegendOpen] = useStored<boolean>('brain_legend', true)
  const [live, setLive] = useState(true)
  const [chosen, setChosen] = useState<Chosen>({})
  const [playing, setPlaying] = useState(false)
  const [buffering, setBuffering] = useState(false)
  const [shown, setShown] = useState<Partial<Record<FlyName, Frame | null>>>({})
  const [pin, setPin] = useState<number | null>(null)
  const [hiddenPops, setHiddenPops] = useState<Set<string>>(() => new Set())
  const [hiddenGroups, setHiddenGroups] = useState<Set<string>>(() => new Set())
  const host = useRef<HTMLDivElement>(null)
  const [scene, setScene] = useState<BrainScene | null>(null)
  const [indices, setIndices] = useState<Partial<Record<FlyName, Int32Array>>>({})
  const [loadMsg, setLoadMsg] = useState<string | null>('loading the brain…')
  const geomRef = data.geometry!
  const emphasis = data.emphasis ?? []

  // ---- the scene: built once per connectome (geometry and index maps are cached for the tab), disposed on unmount
  const indexKey = JSON.stringify(flies.map((f) => f.index?.url ?? null))
  useEffect(() => {
    const el = host.current
    if (!el) return
    let cancelled = false, sc: BrainScene | null = null
    setLoadMsg('loading the brain…'); setScene(null)
    void (async () => {
      let geom: Geometry
      try {
        geom = await loadGeometry(geomRef)
      } catch (e) {
        if (!cancelled) setLoadMsg(`the neuron geometry could not be fetched (${errText(e)}) — is static serving on?`)
        return
      }
      const ix: Partial<Record<FlyName, Int32Array>> = {}, errs: string[] = []
      if (whole) {
        for (const f of flies) {
          if (!f.index) continue
          try { ix[f.name] = await loadIndex(f.index) } catch (e) { errs.push(`${f.name}: index map could not be fetched (${errText(e)})`) }
        }
      }
      if (cancelled) return
      try {
        sc = new BrainScene(el, geom, emphasis)
      } catch (e) {
        setLoadMsg(`WebGL is not available in this browser (${errText(e)})`)
        return
      }
      sc.onPick = (i) => setPin(i)
      setIndices(ix); setScene(sc); setLoadMsg(errs.length ? errs.join('; ') : null)
    })()
    return () => { cancelled = true; sc?.dispose() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geomRef.sha, geomRef.url, indexKey, whole])
  const meta = scene?.geom.meta

  // ---- which minute each fly shows: its newest (live), or the picked one; frames come from the topic or the activity endpoint
  const target = useMemo(() => {
    const out: Partial<Record<FlyName, string | null>> = {}
    for (const f of flies) {
      const newest = f.stamp ?? f.minutes[f.minutes.length - 1] ?? null, c = chosen[f.name]
      out[f.name] = live || c === undefined ? newest : c === null ? null : f.minutes.includes(c) ? c : newest
    }
    return out
  }, [flies, chosen, live])
  useEffect(() => {
    let cancelled = false
    for (const f of flies) {
      const fromPayload = payloadFrame(f), s = target[f.name] ?? null
      if (!s) { setShown((p) => (p[f.name] === null ? p : { ...p, [f.name]: null })); continue }
      const hit = s === f.stamp && fromPayload ? fromPayload : cachedFrame(f.name, s)
      if (hit) { setShown((p) => (p[f.name] === hit ? p : { ...p, [f.name]: hit })); continue }
      void fetchFrame(f.name, s).then((fr) => { if (!cancelled && fr) setShown((p) => ({ ...p, [f.name]: fr })) })
    }
    return () => { cancelled = true }
  }, [flies, target])

  const check = (f: FlyData, fr: Frame | null | undefined): { act: Int8Array | null; note: string | null } => {
    if (!fr) return { act: null, note: null }
    if (fr.activity.length !== f.n || (fr.connectome && meta && fr.connectome !== meta.connectome_file)) return { act: null, note: FOREIGN }
    return { act: fr.activity, note: null }
  }
  const valid = flies.map((f) => ({ f, fr: shown[f.name] ?? null, ...check(f, shown[f.name]) }))
  const layersKey = valid.map((v) => `${v.f.name}:${v.fr?.stamp ?? ''}:${v.act ? 1 : 0}`).join('|') + `|${owner}|${Object.keys(indices).join(',')}`
  useEffect(() => {
    if (!scene) return
    const layers: Layer[] = valid.map((v) => ({ name: v.f.name, activity: v.act, index: whole ? indices[v.f.name] ?? null : null }))
    scene.recolor(layers, owner)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, layersKey])

  // ---- learned pathways: one line group per strategy (prefixed with the fly in the whole brain), rebuilt when a key changes
  const pkey = JSON.stringify(flies.map((f) => f.pathways?.key ?? null))
  const lines = useMemo<SceneLine[]>(() => {
    if (!whole) return (flies[0]?.pathways?.items ?? []).map((it) => ({ kc: it.kc, mbon: it.mbon, delta: it.delta, ratio: it.ratio, group: it.strategy }))
    const out: SceneLine[] = []
    for (const f of flies) {
      const ix = indices[f.name]
      if (!ix) continue
      for (const it of f.pathways?.items ?? []) out.push({ kc: ix[it.kc] ?? -1, mbon: ix[it.mbon] ?? -1, delta: it.delta, ratio: it.ratio, group: `${f.name}:${it.strategy}` })
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pkey, indices, whole])
  const groups = useMemo(() => {
    const base = whole ? flies.flatMap((f) => (f.strategies.length ? [...f.strategies.map((s) => `${f.name}:${s}`), `${f.name}:shared`] : []))
      : flies[0]?.strategies.length ? [...flies[0].strategies, 'shared'] : []
    for (const l of lines) if (!base.includes(l.group)) base.push(l.group)
    return base
  }, [flies, lines, whole])
  const groupsKey = groups.join('|')
  const tints = useMemo(() => strategyTints(document.documentElement, groups), [groupsKey])   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { scene?.setLines(lines, tints) }, [scene, lines, tints])
  useEffect(() => { scene?.setHidden(hiddenPops, hiddenGroups) }, [scene, hiddenPops, hiddenGroups])

  // ---- playback: step through every minute of the last 24 hours (1× = 2 minutes a second), prefetching ahead
  const timeline = useMemo(() => {
    const newest = flies.map((f) => f.minutes[f.minutes.length - 1]).filter(Boolean).sort().pop()
    if (!newest) return [] as string[]
    const from = stampTime(newest) - 86400e3, all = new Set<string>()
    for (const f of flies) for (const s of f.minutes) if (stampTime(s) >= from) all.add(s)
    return [...all].sort()
  }, [flies])
  const stampsAt = (at: string): Chosen => Object.fromEntries(flies.filter((f) => f.minutes.length).map((f) => [f.name, stampAt(f.minutes, at)]))
  const cursor = useRef('')                                 // the minute playback shows (a stamp: the timeline shifts as minutes arrive)
  const live$ = useRef({ timeline, flies, stampsAt })
  live$.current = { timeline, flies, stampsAt }
  useEffect(() => {
    if (!playing) return
    const period = 500 / Number(speed), ahead = Math.max(8, Math.ceil(Number(speed) * 2 * 3))
    let raf = 0, lastT = performance.now(), acc = period, wasBuffering = false
    const ready = (c: Chosen) => Object.entries(c).every(([fly, s]) => !s || !!cachedFrame(fly as FlyName, s))
    const prefetch = (from: number) => {
      const { timeline: tl, flies: fl } = live$.current
      for (let k = from; k < Math.min(tl.length, from + ahead) && inFlight() < MAX_INFLIGHT; k++) {
        for (const f of fl) {
          const s = f.minutes.length ? stampAt(f.minutes, tl[k]) : null
          if (s && !cachedFrame(f.name, s)) void fetchFrame(f.name, s)
        }
      }
    }
    const tick = () => {
      const now = performance.now(); acc = Math.min(acc + now - lastT, period * 8); lastT = now      // a background tab does not jump ahead
      const { timeline: tl, stampsAt: at } = live$.current
      let moved: Chosen | null = null, stalled = false
      while (acc >= period) {                                     // several minutes a frame at 16× when the frames are there
        const next = after(tl, cursor.current)
        if (next >= tl.length) { setPlaying(false); setBuffering(false); setLive(true); return }
        const want = at(tl[next])
        if (!ready(want)) {
          for (const [fly, s] of Object.entries(want)) if (s) void fetchFrame(fly as FlyName, s)
          stalled = true; acc = period
          break
        }
        cursor.current = tl[next]; acc -= period; moved = want
      }
      if (moved) setChosen(moved)
      if (stalled && !wasBuffering) { wasBuffering = true; setBuffering(true) }
      else if (moved && wasBuffering) { wasBuffering = false; setBuffering(false) }
      prefetch(after(tl, cursor.current))
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(raf); setBuffering(false) }
  }, [playing, speed])

  const play = () => {
    if (playing) { setPlaying(false); return }
    if (timeline.length < 2) return
    const shownMax = flies.map((f) => target[f.name]).filter((s): s is string => !!s).sort().pop()
    let k = 0
    if (!live && shownMax && shownMax < timeline[timeline.length - 1]) {
      for (let i = 0; i < timeline.length; i++) if (timeline[i] <= shownMax) k = i
    }
    cursor.current = timeline[k]
    setChosen(stampsAt(timeline[k])); setLive(false); setPlaying(true)
  }
  const setLiveTo = (v: boolean) => {
    setPlaying(false)
    if (v) setLive(true)
    else { setChosen(Object.fromEntries(flies.map((f) => [f.name, target[f.name] ?? null]))); setLive(false) }
  }
  const pick = (f: FlyData, i: number) => {        // the other fly stays on the minute it shows
    setPlaying(false); setLive(false)
    setChosen({ ...Object.fromEntries(flies.map((x) => [x.name, target[x.name] ?? null])), [f.name]: f.minutes[i] })
  }

  // ---- the pinned neuron: its fly (where both flies have it, the central owner — unless only the other fly has a minute
  // to show, as the scene then shows that one) and that fly's sub-graph index
  useEffect(() => { scene?.setPinned(pin) }, [scene, pin])
  const lit = valid.filter((v) => v.act).map((v) => v.f.name).join(',')
  const pinned = useMemo(() => {
    if (pin === null || !meta) return null
    if (!whole) return { fly: 'memecoin' as FlyName, sub: pin }
    const other: FlyName = owner === 'kalshi' ? 'memecoin' : 'kalshi', on = lit.split(',')
    for (const name of (on.includes(owner) || !on.includes(other) ? [owner, other] : [other, owner])) {
      const ix = indices[name]
      if (!ix) continue
      const k = inverseIndex(ix, meta.n)[pin]
      if (k >= 0) return { fly: name, sub: k }
    }
    return null
  }, [pin, meta, whole, owner, indices, lit])

  // ---- what the card says
  const notes = valid.filter((v) => v.note).map((v) => (whole ? `${v.f.name}: ${v.note}` : v.note!))
  const anyLit = valid.some((v) => v.act)
  const message = loadMsg ?? (live ? data.message ?? null : notes.length ? notes.join('; ') : anyLit ? null : data.message ?? null)
  let caption: string
  if (whole) {
    caption = valid.map((v) => `${v.f.name}: ` + (v.act && v.fr ? `${hm(v.fr.minute ?? '')} UTC, ${v.fr.candidates} candidates of ${v.fr.rows} rows` : 'no minute')).join(' · ')
  } else {
    const v = valid[0], fr = v?.act ? v.fr : null
    const when = fr ? (fr.minute ? minuteLabel(fr.minute) : `${stampLabel(fr.stamp)} UTC`) : null
    const head = fr && when ? (fr.candidates ? `${when}: mean activity over ${fr.candidates} candidate rows of ${fr.rows} eligible` : `${when}: no candidates, mean over all ${fr.rows} eligible rows`) : null
    const rest = data.caption_rest ?? (data.caption ?? '').replace(/^\d{2} \w{3} \d{2}:\d{2} UTC: [^·]*(· )?/, '')      // an older server: drop its minute
    caption = [head, rest].filter(Boolean).join(' · ')
  }
  const h = HEAD[variant]
  const actions = (
    <>
      <Toggle label="Follow live" checked={live} onChange={setLiveTo} help={h.live} />
      <span className="brain-sep" />
      <button type="button" className="btn btn-sm btn-light" onClick={play} disabled={timeline.length < 2}
        title="Animate the brain through every minute of the last 24 hours (1× = 2 minutes a second). Playing turns Follow live off; it comes back on when playback reaches the newest minute.">
        <Icon name={playing ? 'pause' : 'play'} />{playing ? 'Pause' : 'Play 24 h'}
      </button>
      <Segmented options={SPEEDS} value={speed} onChange={setSpeed} labels={SPEED_LABELS} />
      <Help text="Brain playback: every minute of the last 24 hours, 1× = 2 minutes a second (4× and 16× faster), the minutes ahead fetched while it plays. Playing turns Follow live off; it comes back on at the newest minute." />
    </>
  )

  return (
    <Card title={h.title} icon="neurology" sub={h.sub} actions={actions}>
      {error && <Callout kind="error">{error} (showing the last good data)</Callout>}
      <div className="brain-controls">
        {whole && (
          <div className="field">
            <span className="lbl">Central brain shows</span>
            <Segmented options={OWNERS} value={owner} onChange={setOwner} />
          </div>
        )}
        <div className="field">
          <span className="lbl">Learned pathways</span>
          <Segmented options={MODES} value={mode} onChange={setMode} labels={MODE_LABELS} />
        </div>
        {buffering && <span className="caption m-0"><Icon name="bx-loader-alt" className="bx-spin" /> loading minutes…</span>}
      </div>
      {flies.filter((f) => f.minutes.length).map((f) => (
        <MinuteSlider key={f.name} label={whole ? `${f.name} minute (UTC)` : 'Minute (UTC)'} minutes={f.minutes} value={target[f.name] ?? null} onPick={(i) => pick(f, i)} />
      ))}
      <div className={`brain-stage ${whole ? 'tall' : ''}`}>
        <div ref={host} className="brain3d-host" style={{ position: 'absolute', inset: 0 }} />
        {meta && (
          <Legend meta={meta} groups={groups} tints={tints} hiddenPops={hiddenPops} setHiddenPops={setHiddenPops} hiddenGroups={hiddenGroups}
            setHiddenGroups={setHiddenGroups} open={legendOpen} setOpen={setLegendOpen} />
        )}
        {message && <div className="brain3d-msg">{message}</div>}
        {pin !== null && scene && (
          <PinPanel scene={scene} index={pin} pinned={pinned} whole={whole} flies={flies} valid={valid} onClose={() => setPin(null)} />
        )}
        {scene && <div className="brain3d-hint">{HINT}</div>}
      </div>
      <Caption>{caption}</Caption>
    </Card>
  )
}

function MinuteSlider({ label, minutes, value, onPick }: { label: string; minutes: string[]; value: string | null; onPick: (i: number) => void }) {
  const i = value ? minutes.indexOf(value) : -1
  return (
    <div className="brain-slider">
      <div className="head"><span>{label}</span><span className="val">{value ? stampLabel(value) : '—'}</span></div>
      <input type="range" min={0} max={Math.max(0, minutes.length - 1)} step={1} value={i < 0 ? 0 : i} aria-label={label}
        onChange={(e) => onPick(Number(e.target.value))} />
      <div className="ends"><span>{stampLabel(minutes[0])}</span><span>{stampLabel(minutes[minutes.length - 1])}</span></div>
    </div>
  )
}

function Legend({ meta, groups, tints, hiddenPops, setHiddenPops, hiddenGroups, setHiddenGroups, open, setOpen }: {
  meta: Geometry['meta']; groups: string[]; tints: Record<string, [number, number, number]>
  hiddenPops: Set<string>; setHiddenPops: Dispatch<SetStateAction<Set<string>>>; hiddenGroups: Set<string>; setHiddenGroups: Dispatch<SetStateAction<Set<string>>>
  open: boolean; setOpen: (v: boolean) => void
}) {
  const flip = (set: Dispatch<SetStateAction<Set<string>>>, name: string, on: boolean) =>
    set((s) => { const n = new Set(s); if (on) n.delete(name); else n.add(name); return n })
  return (
    <div className={`brain3d-legend ${open ? '' : 'folded'}`}>
      <div className="hd">
        Populations
        <button type="button" className="fold" title={open ? 'Fold the legend' : 'Show the legend'} aria-label={open ? 'Fold the legend' : 'Show the legend'} onClick={() => setOpen(!open)}>
          <Icon name={open ? 'bx-chevron-up' : 'bx-chevron-down'} />
        </button>
      </div>
      {open && (
        <>
          {meta.pop_order.map((name) => {
            const [lo, hi] = meta.pop_ranges[name], on = !hiddenPops.has(name)
            return (
              <label key={name} className={on ? '' : 'off'}>
                <input type="checkbox" checked={on} onChange={(e) => flip(setHiddenPops, name, e.target.checked)} />
                <span className="sw" style={{ background: rgbCss(popColor(name)) }} />{`${name} · ${hi - lo}`}
              </label>
            )
          })}
          {groups.length > 0 && <div className="hd">Learned pathways (green +, red −)</div>}
          {groups.map((g) => {
            const on = !hiddenGroups.has(g)
            return (
              <label key={g} className={on ? '' : 'off'}>
                <input type="checkbox" checked={on} onChange={(e) => flip(setHiddenGroups, g, e.target.checked)} />
                <span className="sw line" style={{ background: rgbCss(tints[g] || tints.shared) }} />{g}
              </label>
            )
          })}
        </>
      )}
    </div>
  )
}

type Valid = { f: FlyData; fr: Frame | null; act: Int8Array | null; note: string | null }

function PinPanel({ scene, index, pinned, whole, flies, valid, onClose }: {
  scene: BrainScene; index: number; pinned: { fly: FlyName; sub: number } | null; whole: boolean; flies: FlyData[]; valid: Valid[]; onClose: () => void
}) {
  const v = pinned ? valid.find((x) => x.f.name === pinned.fly) : undefined
  const raw = v?.act && pinned ? v.act[pinned.sub] : undefined
  const value = raw === undefined ? null : raw / 127
  const newest = pinned ? flies.find((f) => f.name === pinned.fly)?.minutes.slice(-1)[0] ?? '' : ''
  const [hist, setHist] = useState<{ key: string; series: NeuronPoint[] | null; error: string | null }>({ key: '', series: null, error: null })
  const nkey = pinned ? `${pinned.fly}/${pinned.sub}` : ''
  useEffect(() => {
    if (!pinned) return
    const ctl = new AbortController()
    neuronHistory(pinned.fly, pinned.sub, newest, ctl.signal).then(
      (series) => { if (!ctl.signal.aborted) setHist({ key: nkey, series, error: null }) },
      (e) => { if (!ctl.signal.aborted) setHist((h) => ({ key: nkey, series: h.key === nkey ? h.series : null, error: errText(e) })) })
    return () => ctl.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nkey, newest])
  const series = hist.key === nkey ? hist.series : null
  const at = v?.fr?.minute ? Date.parse(v.fr.minute) : null
  return (
    <div className="brain3d-pin">
      <div className="top">
        <div>
          <div className="name">{scene.cellType(index)}</div>
          <div className="sub">{scene.population(index)}{pinned ? ` · neuron ${pinned.sub.toLocaleString('en-US')}` : ''}{whole && pinned ? ` · the ${pinned.fly} fly` : ''}</div>
        </div>
        <button type="button" className="x" title="Unpin" aria-label="Unpin" onClick={onClose}>×</button>
      </div>
      <div className={`now ${value === null ? '' : value > 0 ? 'pos' : value < 0 ? 'neg' : ''}`}>
        {value === null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(2)}`}
        <span className="sub"> {value === null ? (pinned ? 'not lit this minute' : 'no fly runs this neuron') : `activity at ${hm(v?.fr?.minute ?? '')} UTC`}</span>
      </div>
      {!pinned ? null : series ? <NeuronChart series={series} at={at} /> : hist.key === nkey && hist.error ? <div className="sub">history unavailable: {hist.error}</div> : <div className="sub">loading its day…</div>}
    </div>
  )
}

function NeuronChart({ series, at }: { series: NeuronPoint[]; at: number | null }) {
  const W = 290, H = 64
  const pts = useMemo(() => series.map((p) => [Date.parse(p.t), p.v] as const), [series])
  if (pts.length < 2) return <div className="sub">Not enough minutes on disk yet for a history.</div>
  const t0 = pts[0][0], t1 = pts[pts.length - 1][0], span = t1 - t0 || 1
  const m = Math.max(1, ...pts.map((p) => Math.abs(p[1])))
  const x = (t: number) => ((t - t0) / span) * W, y = (v: number) => H / 2 - (v / m) * (H / 2 - 3)
  const line = pts.map(([t, v]) => `${x(t).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" height={H} role="img" aria-label="the neuron's activity over the day">
        <line x1={0} x2={W} y1={H / 2} y2={H / 2} stroke="rgb(255 255 255 / 18%)" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
        <polyline points={line} fill="none" stroke="var(--fly-lime)" strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
        {at !== null && at >= t0 && at <= t1 && <line x1={x(at)} x2={x(at)} y1={0} y2={H} stroke="var(--fly-magenta)" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />}
      </svg>
      <div className="axis"><span>{shortLabel(pts[0][0])}</span><span>±{(m / 127).toFixed(2)}</span><span>{shortLabel(pts[pts.length - 1][0])}</span></div>
    </>
  )
}

export default BrainPanel
