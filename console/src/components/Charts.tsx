// ApexCharts in Dashtreme's chart language: white primary series and translucent white for the next one (brand colours
// after that), smooth curves, areas fading to transparent, rounded bars, a faint grid, donuts and radial gauges with the
// figure in the middle. One wrapper keeps a chart per element and updates its series in place, so a pushed topic
// redraws without flicker.
import ApexCharts, { type ApexOptions } from 'apexcharts'
import { useEffect, useRef } from 'react'

export const PALETTE = ['#ffffff', 'rgba(255,255,255,0.5)', '#ff2fb9', '#b6ff3b', '#8b5cf6', '#3d9bff', '#ffc107', '#14b8a6']
const GRID = 'rgba(221,221,221,0.08)'
const AXIS = 'rgba(255,255,255,0.75)'

type Point = [number | string, number | null]
export type Series = { name: string; data: Point[] }

function base(height: number): ApexOptions {
  return {
    chart: { height, background: 'transparent', foreColor: AXIS, fontFamily: 'Inter Variable, Inter, Roboto, system-ui, sans-serif',
      toolbar: { show: false }, zoom: { enabled: false }, animations: { enabled: false }, parentHeightOffset: 0, dropShadow: { enabled: false } },
    theme: { mode: 'dark' },
    colors: PALETTE,
    grid: { borderColor: GRID, strokeDashArray: 0, padding: { left: 8, right: 12, top: 0 }, xaxis: { lines: { show: true } } },
    legend: { position: 'top', horizontalAlign: 'right', fontSize: '12.5px', labels: { colors: '#fff' }, markers: { size: 6, shape: 'circle' } as never, itemMargin: { horizontal: 8 } },
    tooltip: { theme: 'dark', x: { format: 'MMM d HH:mm' } },
    dataLabels: { enabled: false },
  }
}

export function ApexChart({ options, series, height = 260 }: { options: ApexOptions; series: ApexOptions['series']; height?: number }) {
  const el = useRef<HTMLDivElement>(null)
  const chart = useRef<ApexCharts | null>(null)
  const sig = JSON.stringify(options)
  useEffect(() => {
    if (!el.current) return
    const c = new ApexCharts(el.current, { ...options, series })
    chart.current = c
    void c.render()
    return () => {
      c.destroy()
      chart.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig, height])
  useEffect(() => {
    void chart.current?.updateSeries(series ?? [], false)
  }, [series])
  return <div ref={el} style={{ minHeight: height }} />
}

/** Time series (x: ISO time or epoch ms) as smooth lines, or areas fading to transparent (Dashtreme's "Site traffic"). */
export function LineChart({ series, height = 260, yLabel, yFormat, area = true, curve = 'smooth', colors }: {
  series: Series[]; height?: number; yLabel?: string; yFormat?: (v: number) => string; area?: boolean; curve?: 'smooth' | 'straight' | 'stepline'; colors?: string[]
}) {
  const o = base(height)
  const options: ApexOptions = {
    ...o,
    colors: colors ?? PALETTE,
    chart: { ...o.chart, type: area ? 'area' : 'line' },
    stroke: { width: series.map((_, i) => (i === 0 ? 3 : 2)), curve },
    fill: area ? { type: 'gradient', gradient: { shadeIntensity: 0, opacityFrom: 0.45, opacityTo: 0.02, stops: [0, 95] } } : { opacity: 1 },
    xaxis: { type: 'datetime', labels: { datetimeUTC: true, style: { fontSize: '11px' } }, axisBorder: { show: false }, axisTicks: { show: false } },
    yaxis: { title: { text: yLabel, style: { fontWeight: 400, color: AXIS } }, labels: { formatter: yFormat ?? ((v: number) => fmtAxis(v)) } },
    legend: { ...o.legend, show: series.length > 1 },
    markers: { size: 0, hover: { size: 4 } },
  }
  const data = series.map((s) => ({ name: s.name, data: s.data.map(([x, y]) => ({ x: typeof x === 'string' ? new Date(x).getTime() : x, y })) }))
  return <ApexChart options={options} series={data} height={height} />
}

/** Grouped (unstacked) rounded bars by category (Dashtreme's "Visitor status"). */
export function BarChart({ categories, series, height = 240, yLabel, yFormat, horizontal = false, colors }: {
  categories: string[]; series: { name: string; data: (number | null)[] }[]; height?: number; yLabel?: string; yFormat?: (v: number) => string; horizontal?: boolean; colors?: string[]
}) {
  const o = base(height)
  const options: ApexOptions = {
    ...o,
    colors: colors ?? PALETTE,
    chart: { ...o.chart, type: 'bar', stacked: false },
    plotOptions: { bar: { horizontal, columnWidth: series.length > 1 ? '55%' : '38%', barHeight: '62%', borderRadius: 4, borderRadiusApplication: 'end' } },
    stroke: { show: true, width: 2, colors: ['transparent'] },
    xaxis: { categories, labels: { rotate: -45, hideOverlappingLabels: true, style: { fontSize: '10.5px' }, formatter: horizontal ? ((v: string) => fmtAxis(Number(v))) as never : undefined },
      axisBorder: { show: false }, axisTicks: { show: false } },
    yaxis: { title: { text: yLabel, style: { fontWeight: 400, color: AXIS } }, labels: { formatter: horizontal ? undefined : yFormat ?? ((v: number) => fmtAxis(v)) } },
    legend: { ...o.legend, show: series.length > 1 },
    tooltip: { theme: 'dark', y: { formatter: yFormat ? (v: number) => yFormat(v) : undefined } },
  }
  return <ApexChart options={options} series={series} height={height} />
}

/** A donut with its total (or any figure) in the middle (Dashtreme's "Weekly sales" / "Top categories"). */
export function DonutChart({ labels, values, height = 260, center, centerLabel, colors }: {
  labels: string[]; values: number[]; height?: number; center?: string; centerLabel?: string; colors?: string[]
}) {
  const o = base(height)
  const cols = colors ?? ['#ffffff', 'rgba(255,255,255,0.6)', 'rgba(255,255,255,0.35)', 'rgba(255,255,255,0.18)', '#ff2fb9', '#8b5cf6', '#b6ff3b', '#3d9bff']
  const options: ApexOptions = {
    ...o,
    colors: cols,
    chart: { ...o.chart, type: 'donut' },
    labels,
    stroke: { width: 0 },
    legend: { show: false },
    plotOptions: { pie: { donut: { size: '78%', labels: { show: true, name: { show: true, color: AXIS, fontSize: '13px', offsetY: 18 },
      value: { show: true, color: '#fff', fontSize: '26px', fontWeight: 600, offsetY: -14 },
      total: { show: true, showAlways: true, label: centerLabel ?? 'Total', color: AXIS, fontSize: '13px', formatter: () => center ?? String(values.reduce((a, b) => a + b, 0)) } } } } },
    tooltip: { theme: 'dark', fillSeriesColor: false },
  }
  return <ApexChart options={options} series={values} height={height} />
}

/** A radial gauge with the figure in the middle (Dashtreme's "Sales target · Reached 68%"). ``value`` is 0..1. */
export function RadialGauge({ value, label, display, height = 230, color = '#ffffff' }: { value: number | null | undefined; label: string; display?: string; height?: number; color?: string }) {
  const o = base(height)
  const v = Math.max(0, Math.min(1, Number(value ?? 0))) * 100
  const options: ApexOptions = {
    ...o,
    colors: [color],
    chart: { ...o.chart, type: 'radialBar' },
    plotOptions: { radialBar: { startAngle: -135, endAngle: 135, hollow: { size: '68%' }, track: { background: 'rgba(255,255,255,0.12)', strokeWidth: '100%' },
      dataLabels: { name: { show: true, color: AXIS, fontSize: '13px', offsetY: 24 }, value: { show: true, color: '#fff', fontSize: '28px', fontWeight: 600, offsetY: -12,
        formatter: () => display ?? `${v.toFixed(0)}%` } } } },
    stroke: { lineCap: 'round' },
    labels: [label],
  }
  return <ApexChart options={options} series={[v]} height={height} />
}

export function fmtAxis(v: number): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return ''
  const a = Math.abs(v)
  return a >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : a >= 0.01 ? v.toFixed(3) : v.toPrecision(2)
}
