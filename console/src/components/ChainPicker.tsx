// The memecoin chain a page shows (Solana, or Robinhood Chain when RH_ENABLED): one choice shared by the pages, shown only
// when more than one chain is enabled. PageHead is every page's top: Dashtreme's breadcrumb row with the page's tools,
// then the status widgets for the page's market.
import type { ReactNode } from 'react'
import { useLocation } from 'react-router'
import { useApp } from '../context'
import { NAV } from './Layout'
import { StatusWidgets } from './StatusStrip'
import { Breadcrumb, Segmented } from './ui'

export function ChainPicker() {
  const { meta, chain, setChain } = useApp()
  if (!meta || meta.markets.length < 2) return null
  const names = Object.fromEntries(meta.markets.map((m) => [m.chain, m.name])) as Record<string, string>
  return <Segmented options={meta.markets.map((m) => m.chain)} value={chain} onChange={setChain} labels={names} />
}

/** The page's top: breadcrumb (menu group › page), tools on the right (the chain picker unless ``chain={false}``), status widgets. */
export function PageHead({ title, icon, children, chain = true, status = true }: { title: string; icon?: string; children?: ReactNode; chain?: boolean; status?: boolean }) {
  const loc = useLocation()
  const group = NAV.find((g) => g.items.some((i) => (i.to === '/' ? loc.pathname === '/' : loc.pathname.startsWith(i.to))))?.label ?? 'Console'
  return (
    <>
      <Breadcrumb group={group} page={title} icon={icon}>{children}{chain && <ChainPicker />}</Breadcrumb>
      {status && <StatusWidgets />}
    </>
  )
}
