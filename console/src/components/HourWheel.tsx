// Trades by hour of day as a 24-hour wheel: Highcharts' variable-radius pie (variablepie). Every hour is an equal slice of
// the clock (00:00 at the top, clockwise) and its radius is the hour's trade count, so the busy hours stand out; each hour
// has its own colour around the spectrum (night blues, morning greens, midday yellows, evening pinks).
import Highcharts from 'highcharts'
import 'highcharts/modules/variable-pie'
import { useEffect, useRef } from 'react'

export type HourBucket = { hour: number; trades: number; closed: number; wins: number; pnl: number }

const pad = (h: number) => String(h).padStart(2, '0')

/** The hour's colour: hue walks backwards round the wheel from blue at midnight (green at 06, yellow at noon, pink at 18). */
export function hourColor(h: number): string {
  return `hsl(${(240 - h * 15 + 360) % 360}, 88%, 60%)`
}

/** ``buckets`` are by UTC hour; ``shift`` moves them to the hour shown (the browser's offset for local time). */
export function HourWheel({ buckets, unit, shift = 0, height = 440 }: { buckets: HourBucket[]; unit: string; shift?: number; height?: number }) {
  const el = useRef<HTMLDivElement>(null)
  const chart = useRef<Highcharts.Chart | null>(null)
  const total = buckets.reduce((s, b) => s + b.trades, 0)
  const data = Array.from({ length: 24 }, (_, h) => {
    const b = buckets[(((h - shift) % 24) + 24) % 24] ?? { hour: h, trades: 0, closed: 0, wins: 0, pnl: 0 }
    return {
      name: `${pad(h)}:00`, y: 1, z: b.trades, color: hourColor(h),
      custom: { trades: b.trades, share: total ? b.trades / total : 0, closed: b.closed, wins: b.wins, pnl: b.pnl, hour: h },
    }
  })
  useEffect(() => {
    if (!el.current) return
    const c = Highcharts.chart(el.current, {
      chart: { type: 'variablepie', backgroundColor: 'transparent', height, style: { fontFamily: 'Inter Variable, Inter, Roboto, system-ui, sans-serif' } },
      title: { text: undefined },
      credits: { enabled: false },
      accessibility: { enabled: false },
      tooltip: {
        useHTML: true, backgroundColor: 'rgba(28,16,38,0.96)', borderColor: 'rgba(255,255,255,0.18)', style: { color: '#fff', fontSize: '12.5px' }, headerFormat: '',
        pointFormatter: function (this: Highcharts.Point) {
          const x = (this.options as { custom?: Record<string, number> }).custom ?? {}
          const win = x.closed ? `${Math.round((x.wins / x.closed) * 100)}% winners of ${x.closed} closed` : 'none closed yet'
          return `<b>${pad(x.hour)}:00–${pad(x.hour)}:59</b><br/>${x.trades} trades · ${(x.share * 100).toFixed(1)}% of all<br/>${win}<br/>`
            + `realized ${x.pnl >= 0 ? '+' : ''}${x.pnl.toFixed(4)} ${unit}`
        },
      },
      plotOptions: {
        variablepie: {
          startAngle: 0, size: '96%', center: ['50%', '50%'], borderColor: 'rgba(10,6,16,0.55)', borderWidth: 1,
          dataLabels: {
            enabled: true, distance: 12, format: '{point.name}', connectorColor: 'rgba(255,255,255,0.35)',
            style: { color: '#fff', textOutline: 'none', fontSize: '10.5px', fontWeight: '500' },
            filter: { property: 'z', operator: '>', value: 0 },
          },
        },
      },
      series: [{ type: 'variablepie', name: 'trades', minPointSize: 12, innerSize: '20%', zMin: 0, data }],
    } as Highcharts.Options)
    chart.current = c
    return () => {
      c.destroy()
      chart.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, unit])
  useEffect(() => {
    chart.current?.series[0]?.setData(data, true, false, false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(data)])
  return <div ref={el} style={{ minHeight: height }} />
}
