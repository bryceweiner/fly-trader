/** Small DOM helpers and the wallet bar shared by the Trading and Vault pages. All relay data goes in as text. */
import { config, explorer } from '../config'
import type { Wallets, WalletState } from './appkit'
import { short } from './format'

export function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const e = document.getElementById(id)
  if (!e) throw new Error(`#${id} missing`)
  return e as T
}

type Child = Node | string | null | undefined | false
type Attrs = Record<string, string | number | boolean | null | undefined | ((e: Event) => void)>

/** h('a', { href, class: 'x' }, 'text') — attributes starting with "on" become listeners. */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag)
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue
    if (typeof v === 'function') e.addEventListener(k.slice(2), v)
    else if (v === true) e.setAttribute(k, '')
    else e.setAttribute(k, String(v))
  }
  for (const c of children) if (c != null && c !== false) e.append(c)
  return e
}

export function setText(id: string, text: string, cls?: string) {
  const e = document.getElementById(id)
  if (!e) return
  e.textContent = text
  if (cls !== undefined) e.className = cls
}

export function link(href: string, text: string, cls = 'mono'): HTMLAnchorElement {
  return h('a', { href, target: '_blank', rel: 'noopener', class: cls }, text)
}

/** aria-live status line for transactions: status('…', 'busy' | 'ok' | 'error'). */
export function statusLine(id: string) {
  const el = $(id)
  return (msg: string | Node, kind: 'busy' | 'ok' | 'error' | 'info' = 'info') => {
    el.className = `status-line ${kind}`
    el.replaceChildren(msg)
    el.hidden = !msg
  }
}

/** Disables a button while `fn` runs and marks it busy. */
export async function busy<T>(btn: HTMLButtonElement, fn: () => Promise<T>): Promise<T | undefined> {
  if (btn.disabled) return
  btn.disabled = true
  btn.setAttribute('aria-busy', 'true')
  try {
    return await fn()
  } finally {
    btn.disabled = false
    btn.removeAttribute('aria-busy')
  }
}

export function txLink(hash: string): Node {
  return h('span', {}, 'Confirmed: ', link(explorer.evmTx(hash), short(hash, 6, 6)))
}

/**
 * Renders the wallet bar into #wallet-bar and loads AppKit (dynamic import). Resolves to null when this build
 * has no Reown project id, or AppKit fails to load; the page stays fully readable either way.
 */
export async function mountWalletBar(opts: { solana: boolean; onChange: (s: WalletState, w: Wallets) => void }): Promise<Wallets | null> {
  const bar = $('wallet-bar')
  if (!config.reownProjectId) {
    bar.replaceChildren(
      h('div', { class: 'notice' }, h('span', { class: 'micro' }, 'Wallets'), h('p', {}, 'Wallet connection is not configured on this build (no Reown project id). Everything below is still readable.')),
    )
    return null
  }
  bar.replaceChildren(h('p', { class: 'micro' }, 'Loading wallet support…'))
  let wallets: Wallets
  try {
    const mod = await import('./appkit')
    wallets = mod.initWallets()
  } catch (e) {
    console.error(e)
    bar.replaceChildren(h('div', { class: 'notice danger' }, h('span', { class: 'micro' }, 'Wallets'), h('p', {}, 'Wallet support failed to load. Reload the page to try again.')))
    return null
  }

  const row = (label: string, value: Node | string, action: HTMLButtonElement, note?: string) =>
    h('div', { class: 'wallet-row' }, h('span', { class: 'micro' }, label), h('div', { class: 'wallet-val' }, value), action, note ? h('p', { class: 'wallet-note' }, note) : null)

  const render = (s: WalletState) => {
    const evmBtn = s.evm
      ? h('button', { class: 'btn small', type: 'button', onclick: () => void wallets.disconnect('eip155') }, 'Disconnect')
      : h('button', { class: 'btn small primary', type: 'button', onclick: () => void wallets.connect('eip155') }, 'Connect')
    const wrongChain = s.evm && s.evmChainId !== config.evm.chainId
    const rows = [
      row(
        `EVM · ${config.evm.name}`,
        s.evm ? link(explorer.evmAddress(s.evm), short(s.evm)) : 'not connected',
        evmBtn,
        wrongChain ? `Your wallet is on chain ${s.evmChainId ?? '?'}. Transactions will ask to switch to ${config.evm.name}; signing a claim works on any chain.` : undefined,
      ),
    ]
    if (opts.solana) {
      const solBtn = s.sol
        ? h('button', { class: 'btn small', type: 'button', onclick: () => void wallets.disconnect('solana') }, 'Disconnect')
        : h('button', { class: 'btn small primary', type: 'button', onclick: () => void wallets.connect('solana') }, 'Connect')
      rows.push(row(`Solana · ${config.solana.cluster}`, s.sol ? link(explorer.solAccount(s.sol), short(s.sol)) : 'not connected', solBtn))
    }
    bar.replaceChildren(
      h('div', { class: 'wallet-rows' }, ...rows),
      h('p', { class: 'wallet-note' }, `Some mobile wallets cannot add ${config.evm.name} (chain ${config.evm.chainId}); they can still connect and sign claims.`),
    )
  }
  wallets.subscribe((s) => {
    render(s)
    opts.onChange(s, wallets)
  })
  return wallets
}

/** role=tablist behaviour: click or arrow keys select; `onSelect` gets the chosen tab's data-* value. */
export function tabs(list: HTMLElement, key: string, onSelect: (value: string) => void) {
  const all = () => Array.from(list.querySelectorAll<HTMLButtonElement>('[role="tab"]'))
  const select = (b: HTMLButtonElement, focus = false) => {
    for (const t of all()) {
      const on = t === b
      t.setAttribute('aria-selected', String(on))
      t.tabIndex = on ? 0 : -1
    }
    if (focus) b.focus()
    onSelect(b.dataset[key] ?? '')
  }
  for (const t of all()) {
    t.tabIndex = t.getAttribute('aria-selected') === 'true' ? 0 : -1
    t.addEventListener('click', () => select(t))
    t.addEventListener('keydown', (e) => {
      const ts = all()
      const i = ts.indexOf(t)
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault()
        select(ts[(i + (e.key === 'ArrowRight' ? 1 : ts.length - 1)) % ts.length], true)
      }
    })
  }
}
