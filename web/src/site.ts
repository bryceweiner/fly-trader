/* fly-trader.app — behaviour shared by every page (port of the old site.js). */
import { LINKS } from './config'

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

/* ---- links driven by config ---- */
document.querySelectorAll<HTMLAnchorElement>('[data-link]').forEach((el) => {
  const kind = el.dataset.link
  if (kind === 'github') el.href = LINKS.github
  if (kind === 'readme') el.href = LINKS.github + '#readme'
  if (kind === 'x') {
    if (LINKS.x) {
      el.href = LINKS.x
    } else {
      el.setAttribute('aria-disabled', 'true')
      el.title = 'X community link coming at launch'
      el.addEventListener('click', (e) => e.preventDefault())
    }
  }
})

/* ---- $FLY contract address ---- */
const caEl = document.getElementById('fly-ca')
const copyBtn = document.getElementById('copy-ca')
const buyBtn = document.getElementById('buy-fly') as HTMLAnchorElement | null
if (caEl && copyBtn) {
  caEl.textContent = LINKS.flyCa
  copyBtn.removeAttribute('aria-disabled')
  if (buyBtn) {
    buyBtn.removeAttribute('aria-disabled')
    buyBtn.href = LINKS.venue + LINKS.flyCa
  }
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(LINKS.flyCa)
      copyBtn.textContent = 'Copied'
      setTimeout(() => (copyBtn.textContent = 'Copy'), 1500)
    } catch {
      copyBtn.textContent = 'Select'
      const range = document.createRange()
      range.selectNodeContents(caEl)
      const sel = window.getSelection()
      sel?.removeAllRanges()
      sel?.addRange(range)
    }
  })
}

/* ---- mobile menu ---- */
const burger = document.querySelector<HTMLButtonElement>('.burger')
const menu = document.getElementById('mobile-menu')
if (burger && menu) {
  const close = () => {
    menu.hidden = true
    burger.setAttribute('aria-expanded', 'false')
    burger.setAttribute('aria-label', 'Open menu')
  }
  burger.addEventListener('click', () => {
    const open = menu.hidden
    menu.hidden = !open
    burger.setAttribute('aria-expanded', String(open))
    burger.setAttribute('aria-label', open ? 'Close menu' : 'Open menu')
  })
  menu.querySelectorAll('a').forEach((a) => a.addEventListener('click', close))
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !menu.hidden) {
      close()
      burger.focus()
    }
  })
}

/* ---- rotating hero headline ---- */
const headline = document.getElementById('headline')
if (headline && !reducedMotion) {
  const passages = Array.from(headline.querySelectorAll('.passage'))
  if (passages.length > 1) {
    let i = 0
    const PERIOD = 4000
    const LEAVE = 600
    setInterval(() => {
      const cur = passages[i]
      i = (i + 1) % passages.length
      const next = passages[i]
      cur.classList.remove('is-active')
      cur.classList.add('is-leaving')
      next.classList.add('is-active')
      setTimeout(() => cur.classList.remove('is-leaving'), LEAVE)
    }, PERIOD)
  }
}

/* ---- hero video: fall back to the still when playback is refused ---- */
const video = document.querySelector<HTMLVideoElement>('.hero-media video')
if (video) {
  if (reducedMotion) {
    video.pause()
    video.removeAttribute('autoplay')
  } else {
    const fail = () => document.body.classList.add('no-video')
    video.addEventListener('error', fail, true)
    const p = video.play()
    if (p && typeof p.catch === 'function') p.catch(fail)
  }
}

/* ---- footer year ---- */
const y = document.getElementById('year')
if (y) y.textContent = String(new Date().getFullYear())
