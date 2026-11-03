/** Rows as CSV (RFC 4180 quoting), downloaded as a file — the tables' "Download" button. */
export function toCsv(headers: string[], rows: unknown[][]): string {
  const cell = (v: unknown): string => {
    if (v === null || v === undefined) return ''
    const s = typeof v === 'object' ? JSON.stringify(v) : String(v)
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  return [headers.map(cell).join(','), ...rows.map((r) => r.map(cell).join(','))].join('\r\n')
}

export function download(filename: string, text: string, type = 'text/csv'): void {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
