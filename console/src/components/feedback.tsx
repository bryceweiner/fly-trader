// Toasts, the confirmation modal every state-changing action goes through, and generic modals.
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { action as postAction, type ActionResult } from '../lib/api'
import { Icon, Tip } from './ui'

type ToastKind = 'info' | 'success' | 'error' | 'warning' | 'live'
type Toast = { id: number; kind: ToastKind; text: ReactNode; ttl: number }
type Confirm = { title: string; body: ReactNode; confirmLabel?: string; danger?: boolean; resolve: (ok: boolean) => void }

type Feedback = {
  toast: (text: ReactNode, kind?: ToastKind, ttl?: number) => void
  confirm: (o: Omit<Confirm, 'resolve'>) => Promise<boolean>
}

const Ctx = createContext<Feedback>({ toast: () => {}, confirm: async () => false })
export const useFeedback = () => useContext(Ctx)

export function FeedbackProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const [confirm, setConfirm] = useState<Confirm | null>(null)
  const seq = useRef(0)
  const toast = useCallback((text: ReactNode, kind: ToastKind = 'info', ttl = 6000) => {
    const id = ++seq.current
    setToasts((t) => [...t.slice(-5), { id, kind, text, ttl }])
    if (ttl > 0) setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), ttl)
  }, [])
  const ask = useCallback((o: Omit<Confirm, 'resolve'>) => new Promise<boolean>((resolve) => setConfirm({ ...o, resolve })), [])
  const close = (ok: boolean) => {
    confirm?.resolve(ok)
    setConfirm(null)
  }
  return (
    <Ctx.Provider value={{ toast, confirm: ask }}>
      {children}
      <div className="toasts" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`ftoast ${t.kind}`}>
            <Icon name={t.kind === 'error' ? 'error' : t.kind === 'success' ? 'success' : t.kind === 'warning' ? 'warning' : t.kind === 'live' ? 'payments' : 'info'} />
            <div>{t.text}</div>
            <button type="button" className="x" aria-label="Dismiss" onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}>×</button>
          </div>
        ))}
      </div>
      {confirm && (
        <Modal title={confirm.title} icon={confirm.danger ? 'warning' : 'info'} onClose={() => close(false)}
          foot={<>
            <button type="button" className="btn btn-outline-light" onClick={() => close(false)}>Cancel</button>
            <button type="button" autoFocus className={`btn ${confirm.danger ? 'btn-danger' : 'btn-primary'}`} onClick={() => close(true)}>{confirm.confirmLabel || 'Confirm'}</button>
          </>}>
          {confirm.body}
        </Modal>
      )}
    </Ctx.Provider>
  )
}

export function Modal({ title, icon, onClose, children, foot, wide }: { title: ReactNode; icon?: string; onClose: () => void; children: ReactNode; foot?: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose])
  return (
    <div className="fmodal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`fmodal ${wide ? 'wide' : ''}`} role="dialog" aria-modal="true">
        <div className="head"><Icon name={icon} />{title}<button type="button" className="close" aria-label="Close" onClick={onClose}>×</button></div>
        <div className="body">{children}</div>
        {foot && <div className="foot">{foot}</div>}
      </div>
    </div>
  )
}

/**
 * A button for one state-changing action: it asks for confirmation (naming the effect), POSTs the action, and toasts the
 * result or the server's refusal. Every write in the console goes through this.
 */
export function ActionButton({ label, icon, name, body, confirm, variant = 'outline-light', disabled, help, size = 'sm', onDone, className = '' }: {
  label: ReactNode; icon?: string; name: string; body?: Record<string, unknown>
  confirm: { title: string; body: ReactNode; confirmLabel?: string; danger?: boolean }
  variant?: string; disabled?: boolean; help?: ReactNode; size?: 'sm' | 'md'; onDone?: (r: ActionResult) => void; className?: string
}) {
  const fb = useFeedback()
  const [busy, setBusy] = useState(false)
  const run = async () => {
    if (!(await fb.confirm(confirm))) return
    setBusy(true)
    try {
      const r = await postAction(name, body ?? {})
      fb.toast(r.message || 'Done', 'success')
      onDone?.(r)
    } catch (e) {
      fb.toast((e as Error).message, 'error', 10000)
    } finally {
      setBusy(false)
    }
  }
  const btn = (
    <button type="button" className={`btn btn-${variant} ${size === 'sm' ? 'btn-sm' : ''} ${className}`} disabled={disabled || busy} onClick={run}>
      {busy ? <span className="spinner-border spinner-border-sm" /> : <Icon name={icon} />}{label}
    </button>
  )
  return help ? <Tip text={help}>{btn}</Tip> : btn
}

