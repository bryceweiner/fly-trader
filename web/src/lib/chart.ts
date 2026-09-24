/**
 * $FLY market data from GeckoTerminal and the charts drawn with TradingView Lightweight Charts™
 * (bundled; its attribution logo stays on, as its licence requires).
 */
import {
  AreaSeries,
  CandlestickSeries,
  ColorType,
  createChart,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts'
import { config } from '../config'
import { price } from './format'

export type Interval = '15m' | '1h' | '4h' | '1d'
export const INTERVALS: Record<Interval, { timeframe: 'minute' | 'hour' | 'day'; aggregate: number }> = {
  '15m': { timeframe: 'minute', aggregate: 15 },
  '1h': { timeframe: 'hour', aggregate: 1 },
  '4h': { timeframe: 'hour', aggregate: 4 },
  '1d': { timeframe: 'day', aggregate: 1 },
}

export interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Market {
  priceUsd: number
  change24h: number | null
  fdvUsd: number | null
  liquidityUsd: number | null
  volume24hUsd: number | null
}

const num = (v: unknown): number | null => {
  const n = typeof v === 'string' ? parseFloat(v) : typeof v === 'number' ? v : NaN
  return Number.isFinite(n) ? n : null
}

async function gecko<T>(path: string): Promise<T> {
  const res = await fetch(`${config.market.geckoApi}/networks/${config.market.geckoNetwork}/pools/${config.market.pool}${path}`, {
    headers: { accept: 'application/json' },
  })
  if (!res.ok) throw new Error(`GeckoTerminal ${res.status}`)
  return res.json() as Promise<T>
}

export async function fetchMarket(): Promise<Market> {
  if (import.meta.env.VITE_NETWORK === 'testnet') return (await import('../../dev/fixtures/gecko_pool.json')).default as Market
  const body = await gecko<{ data: { attributes: Record<string, unknown> } }>('')
  const a = body.data.attributes
  const change = a.price_change_percentage as Record<string, string> | undefined
  const vol = a.volume_usd as Record<string, string> | undefined
  return {
    priceUsd: num(a.base_token_price_usd) ?? 0,
    change24h: num(change?.h24),
    fdvUsd: num(a.fdv_usd),
    liquidityUsd: num(a.reserve_in_usd),
    volume24hUsd: num(vol?.h24),
  }
}

export async function fetchCandles(interval: Interval, limit = 300): Promise<Candle[]> {
  let rows: number[][]
  if (import.meta.env.VITE_NETWORK === 'testnet') {
    rows = (await import('../../dev/fixtures/gecko_ohlcv.json')).default as number[][]
  } else {
    const { timeframe, aggregate } = INTERVALS[interval]
    const body = await gecko<{ data: { attributes: { ohlcv_list: number[][] } } }>(
      `/ohlcv/${timeframe}?aggregate=${aggregate}&limit=${limit}&currency=usd&token=base`,
    )
    rows = body.data.attributes.ohlcv_list
  }
  // GeckoTerminal returns newest first; the chart needs strictly ascending, unique times.
  const seen = new Set<number>()
  return rows
    .map(([time, open, high, low, close, volume]) => ({ time, open, high, low, close, volume }))
    .filter((c) => !seen.has(c.time) && seen.add(c.time))
    .sort((a, b) => a.time - b.time)
}

const COLORS = {
  bg: '#131119',
  grid: 'rgba(139, 92, 246, 0.08)',
  text: '#a99cc7',
  border: '#2b2540',
  up: '#b6ff3b',
  down: '#ff4d6d',
  magenta: '#ff2fb9',
}

function baseChart(el: HTMLElement): IChartApi {
  return createChart(el, {
    autoSize: true,
    layout: {
      background: { type: ColorType.Solid, color: COLORS.bg },
      textColor: COLORS.text,
      fontFamily: '"IBM Plex Mono", Menlo, monospace',
      fontSize: 11,
      attributionLogo: true,
    },
    grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
    rightPriceScale: { borderColor: COLORS.border },
    timeScale: { borderColor: COLORS.border, timeVisible: true, secondsVisible: false },
    crosshair: { mode: 0 },
  })
}

export class CandleChart {
  private chart: IChartApi
  private candles: ISeriesApi<'Candlestick'>
  private volume: ISeriesApi<'Histogram'>

  constructor(el: HTMLElement) {
    this.chart = baseChart(el)
    this.candles = this.chart.addSeries(CandlestickSeries, {
      upColor: COLORS.up,
      downColor: COLORS.down,
      borderUpColor: COLORS.up,
      borderDownColor: COLORS.down,
      wickUpColor: COLORS.up,
      wickDownColor: COLORS.down,
      priceFormat: { type: 'custom', formatter: price, minMove: 1e-12 },
    })
    this.volume = this.chart.addSeries(HistogramSeries, {
      priceScaleId: '',
      priceFormat: { type: 'volume' },
      lastValueVisible: false,
      priceLineVisible: false,
    })
    this.volume.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } })
    this.candles.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0.25 } })
  }

  set(data: Candle[], fit: boolean) {
    this.candles.setData(data.map((c) => ({ time: c.time as UTCTimestamp, open: c.open, high: c.high, low: c.low, close: c.close })))
    this.volume.setData(
      data.map((c) => ({
        time: c.time as UTCTimestamp,
        value: c.volume,
        color: c.close >= c.open ? 'rgba(182, 255, 59, 0.35)' : 'rgba(255, 77, 109, 0.35)',
      })),
    )
    // Show the latest ~120 candles; the price scale then ignores older extremes such as the launch spike.
    if (fit) {
      if (data.length > 120) this.chart.timeScale().setVisibleLogicalRange({ from: data.length - 120, to: data.length + 2 })
      else this.chart.timeScale().fitContent()
    }
  }
}

export class LineChart {
  private chart: IChartApi
  private series: ISeriesApi<'Area'>

  constructor(el: HTMLElement, formatter: (v: number) => string) {
    this.chart = baseChart(el)
    this.series = this.chart.addSeries(AreaSeries, {
      lineColor: COLORS.magenta,
      topColor: 'rgba(255, 47, 185, 0.28)',
      bottomColor: 'rgba(255, 47, 185, 0.02)',
      lineWidth: 2,
      priceFormat: { type: 'custom', formatter, minMove: 1e-9 },
    })
  }

  set(points: { time: number; value: number }[], fit: boolean) {
    const seen = new Set<number>()
    const data = points
      .filter((p) => Number.isFinite(p.value) && !seen.has(p.time) && seen.add(p.time))
      .sort((a, b) => a.time - b.time)
      .map((p) => ({ time: p.time as UTCTimestamp, value: p.value }))
    this.series.setData(data)
    if (fit) this.chart.timeScale().fitContent()
  }

  remove() {
    this.chart.remove()
  }
}
