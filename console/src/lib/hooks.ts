import { useEffect, useMemo, useState } from 'react'
import { live, topicKey, type ConnState, type TopicMsg } from './live'

export type TopicState<T> = { data: T | undefined; error: string | undefined; loading: boolean }

/** Subscribe to a pushed topic; re-subscribes when the topic or its params change. ``enabled: false`` pauses it. */
export function useTopic<T = unknown>(topic: string | null, params: Record<string, unknown> = {}, enabled = true): TopicState<T> {
  const key = topic ? topicKey(topic, params) : ''
  const [msg, setMsg] = useState<{ key: string; m: TopicMsg<T> } | null>(null)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const p = useMemo(() => params, [key])
  useEffect(() => {
    if (!topic || !enabled) return
    return live.subscribe(topic, p, (m) => setMsg({ key, m: m as TopicMsg<T> }))
  }, [topic, key, p, enabled])
  const cur = msg && msg.key === key ? msg.m : undefined
  return { data: cur?.data, error: cur?.error, loading: !cur }
}

export function useConn(): ConnState {
  const [s, setS] = useState<ConnState>(live.state)
  useEffect(() => live.onState(setS), [])
  return s
}

/** Re-render every ``ms`` (for "12 s ago" texts computed in the browser). */
export function useNow(ms = 1000): number {
  const [t, setT] = useState(Date.now())
  useEffect(() => {
    const h = setInterval(() => setT(Date.now()), ms)
    return () => clearInterval(h)
  }, [ms])
  return t
}

/** A value remembered in this browser (localStorage; falls back to memory when storage is blocked). */
export function useStored<T>(key: string, initial: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(`fly:${key}`)
      return raw === null ? initial : (JSON.parse(raw) as T)
    } catch {
      return initial
    }
  })
  const set = (nv: T) => {
    setV(nv)
    try {
      localStorage.setItem(`fly:${key}`, JSON.stringify(nv))
    } catch {
      /* private window */
    }
  }
  return [v, set]
}
