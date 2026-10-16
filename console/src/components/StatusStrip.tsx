// What the system is doing, as Dashtreme colour static widgets (fly_trader/ui/state.py status_rows, pushed every second):
// paper or live, trading or not, which model, the fly, training, whether the feed is current. A memecoin page shows its
// chain's row, a Kalshi page Kalshi's, a system page one widget per market.
import { useLocation } from 'react-router'
import { useApp } from '../context'
import { useTopic } from '../lib/hooks'
import { Callout, Metric, STATE_COLOR, Tip } from './ui'

export type StatusWidget = { key: string; label: string; value: string; sub: string | null; color: string; icon: string; help: string | null }
type Row = { key: string; label: string; widgets: StatusWidget[]; error: string | null }

export function useStatusRows(): { rows: Row[] | undefined; error: string | undefined } {
  const { data, error } = useTopic<{ rows: Row[] }>('status')
  return { rows: data?.rows, error }
}

function Tile({ w }: { w: StatusWidget }) {
  const tile = <Metric label={w.label} value={w.value} delta={w.sub} tone="off" icon={w.icon} color={STATE_COLOR[w.color] ?? 'dark'} />
  return w.help ? <Tip text={w.help}>{tile}</Tip> : tile
}

/** Which row a route shows: kalshi pages → kalshi, /rh → rh, memecoin pages → the chosen chain, system pages → all. */
export function rowFor(path: string, chain: string): string | 'all' {
  if (path.startsWith('/kalshi')) return 'kalshi'
  if (path.startsWith('/rh')) return 'rh'
  if (path === '/' || path.startsWith('/trades')) return chain
  if (path.startsWith('/model') || path.startsWith('/data')) return 'sol'
  return 'all'
}

export function StatusWidgets() {
  const { chain } = useApp()
  const loc = useLocation()
  const { rows, error } = useStatusRows()
  if (!rows) return error ? <Callout kind="error">Status unavailable: {error}</Callout> : <div className="status-widgets"><div className="skeleton" style={{ height: 84 }} /></div>
  const want = rowFor(loc.pathname, chain)
  if (want === 'all') {
    return (
      <div className="status-widgets">
        {rows.map((r) => {
          if (r.error) return <div key={r.key} className="status-error"><Callout kind="error">{r.error}</Callout></div>
          const eng = r.widgets.find((w) => w.key === 'engine'), money = r.widgets.find((w) => w.key === 'money'), feed = r.widgets.find((w) => w.key === 'feed')
          if (!eng) return null
          const bad = feed && feed.color === 'red' && eng.color === 'green'
          return <Tile key={r.key} w={{ ...eng, key: r.key, label: r.label, icon: 'candlestick', color: bad ? 'orange' : eng.color,
            sub: [money ? `${money.value.toLowerCase()} money` : null, feed ? `feed ${feed.value.toLowerCase()}` : null].filter(Boolean).join(' · '),
            help: [eng.help, money?.help, feed?.help].filter(Boolean).join(' ') }} />
        })}
      </div>
    )
  }
  const row = rows.find((r) => r.key === want) ?? rows[0]
  if (!row) return null
  if (row.error) return <Callout kind="error">{row.error}</Callout>
  return <div className="status-widgets">{row.widgets.map((w) => <Tile key={w.key} w={w} />)}</div>
}
