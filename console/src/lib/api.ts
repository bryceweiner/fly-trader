// Writes and one-off reads. Every state-changing request carries X-Fly-Console: 1 (fly_trader/ui/security.py refuses
// it otherwise), which a cross-site page cannot send without a CORS preflight the console never approves.

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function parse<T>(res: Response): Promise<T> {
  let body: unknown = null
  try {
    body = await res.json()
  } catch {
    /* not JSON */
  }
  if (!res.ok) {
    const msg = (body as { error?: string } | null)?.error || `${res.status} ${res.statusText}`
    throw new ApiError(msg, res.status)
  }
  return body as T
}

export type ActionResult = { ok: true; message?: string; [k: string]: unknown }

/** POST /api/action/<name>: one write. Throws ApiError with the server's refusal on failure. */
export async function action<T = ActionResult>(name: string, body: Record<string, unknown> = {}): Promise<T & ActionResult> {
  const res = await fetch(`/api/action/${name}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Fly-Console': '1' },
    body: JSON.stringify(body),
  })
  return parse<T & ActionResult>(res)
}

/** GET /api/topic/<name>: a topic read once (for things that need no live updates). */
export async function readTopic<T>(name: string, params: Record<string, unknown> = {}): Promise<T> {
  const q = Object.keys(params).length ? `?params=${encodeURIComponent(JSON.stringify(params))}` : ''
  return parse<T>(await fetch(`/api/topic/${name}${q}`))
}

export async function getJSON<T>(url: string): Promise<T> {
  return parse<T>(await fetch(url))
}
