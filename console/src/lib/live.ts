// One WebSocket per tab (fly_trader/ui/hub.py): components subscribe to (topic, params); identical subscriptions share
// one server subscription; the server pushes a topic only when it changed. Reconnects with backoff and resubscribes.

export type TopicMsg<T = unknown> = { data?: T; error?: string }
type Listener = (m: TopicMsg) => void
type Sub = { id: string; topic: string; params: Record<string, unknown>; listeners: Set<Listener>; last?: TopicMsg }
export type ConnState = 'connecting' | 'open' | 'closed'

export function topicKey(topic: string, params: Record<string, unknown> = {}): string {
  const keys = Object.keys(params).sort()
  return topic + '?' + keys.map((k) => `${k}=${JSON.stringify(params[k] ?? null)}`).join('&')
}

class Live {
  private ws: WebSocket | null = null
  private subs = new Map<string, Sub>()
  private byId = new Map<string, Sub>()
  private seq = 0
  private retry = 0
  private timer: ReturnType<typeof setTimeout> | null = null
  private ping: ReturnType<typeof setInterval> | null = null
  state: ConnState = 'connecting'
  private stateListeners = new Set<(s: ConnState) => void>()

  start(): void {
    if (this.ws) return
    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`
    const ws = new WebSocket(url)
    this.ws = ws
    this.setState('connecting')
    ws.onopen = () => {
      this.retry = 0
      this.setState('open')
      for (const s of this.subs.values()) this.send({ op: 'sub', id: s.id, topic: s.topic, params: s.params })
      this.ping = setInterval(() => this.send({ op: 'ping' }), 25000)
    }
    ws.onmessage = (ev) => {
      let m: { op: string; id?: string; data?: unknown; error?: string }
      try {
        m = JSON.parse(ev.data as string)
      } catch {
        return
      }
      if (!m.id) return
      const s = this.byId.get(m.id)
      if (!s) return
      const msg: TopicMsg = m.op === 'data' ? { data: m.data } : { data: s.last?.data, error: m.error }
      s.last = msg
      for (const l of s.listeners) l(msg)
    }
    ws.onclose = () => {
      if (this.ping) clearInterval(this.ping)
      this.ws = null
      this.setState('closed')
      const wait = Math.min(5000, 400 * 2 ** this.retry++)
      this.timer = setTimeout(() => {
        this.timer = null
        this.start()
      }, wait)
    }
    ws.onerror = () => ws.close()
  }

  private send(m: unknown): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(m))
  }

  private setState(s: ConnState): void {
    this.state = s
    for (const l of this.stateListeners) l(s)
  }

  onState(l: (s: ConnState) => void): () => void {
    this.stateListeners.add(l)
    return () => this.stateListeners.delete(l)
  }

  subscribe(topic: string, params: Record<string, unknown>, listener: Listener): () => void {
    this.start()
    const key = topicKey(topic, params)
    let s = this.subs.get(key)
    if (!s) {
      s = { id: `s${++this.seq}`, topic, params, listeners: new Set() }
      this.subs.set(key, s)
      this.byId.set(s.id, s)
      this.send({ op: 'sub', id: s.id, topic, params })
    }
    s.listeners.add(listener)
    if (s.last) listener(s.last)
    const sub = s
    return () => {
      sub.listeners.delete(listener)
      if (sub.listeners.size === 0) {
        this.subs.delete(key)
        this.byId.delete(sub.id)
        this.send({ op: 'unsub', id: sub.id })
      }
    }
  }

  /** Reconnect now (after the tab wakes up). */
  kick(): void {
    if (this.timer) {
      clearTimeout(this.timer)
      this.timer = null
      this.start()
    }
  }
}

export const live = new Live()
if (typeof document !== 'undefined') document.addEventListener('visibilitychange', () => document.visibilityState === 'visible' && live.kick())
