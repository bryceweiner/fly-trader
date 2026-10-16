// Dashtreme's horizontal layout: the topbar (logo, search, connection, notifications), the primary menu (three dropdown
// groups, an off-canvas panel on phones), the status strip under it on every page, and the footer.
import { useEffect, useState, type ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router'
import { useApp } from '../context'
import { InspectorDrawer } from '../fly/Inspector'
import { LiveToasts } from '../fly/Ticker'
import { useConn } from '../lib/hooks'
import { Notifications } from './Notifications'
import { SearchBox } from './SearchBox'
import { Icon } from './ui'

export type NavItem = { to: string; label: string; icon: string }
export type NavGroup = { label: string; icon: string; items: NavItem[]; when?: (m: NonNullable<ReturnType<typeof useApp>['meta']>) => boolean }

export const NAV: NavGroup[] = [
  { label: 'Memecoins', icon: 'bx-coin', items: [
    { to: '/', label: 'Overview', icon: 'dashboard' },
    { to: '/trades', label: 'Trades', icon: 'trades' },
    { to: '/model', label: 'Model & training', icon: 'psychology' },
    { to: '/data', label: 'Data pipelines', icon: 'database' },
    { to: '/rh', label: 'Robinhood Chain', icon: 'link' },
  ] },
  { label: 'Prediction markets', icon: 'bx-target-lock', when: (m) => m.kalshi, items: [
    { to: '/kalshi', label: 'Kalshi overview', icon: 'visibility' },
    { to: '/kalshi-trades', label: 'Kalshi trades', icon: 'receipt' },
    { to: '/kalshi-model', label: 'Kalshi model & training', icon: 'neurology' },
    { to: '/kalshi-data', label: 'Kalshi data', icon: 'database' },
  ] },
  { label: 'System', icon: 'bx-cog', items: [
    { to: '/processes', label: 'Processes', icon: 'settings' },
    { to: '/safety', label: 'Safety & wallet', icon: 'shield' },
    { to: '/vault', label: '$FLY vault', icon: 'savings' },
    { to: '/logs', label: 'Logs', icon: 'logs' },
  ] },
]

function Logo() {
  return (
    <NavLink to="/" className="d-flex align-items-center text-decoration-none">
      <img src="/favicon.svg" className="logo-icon" alt="" />
      <h4 className="logo-text">fly<span className="dot">·</span>trader</h4>
    </NavLink>
  )
}

function Menu({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { meta } = useApp()
  const loc = useLocation()
  const [drop, setDrop] = useState<string | null>(null)
  useEffect(() => { setDrop(null); onClose() }, [loc.pathname]) // eslint-disable-line react-hooks/exhaustive-deps
  const groups = NAV.filter((g) => !g.when || (meta && g.when(meta)))
  return (
    <div className="primary-menu">
      <nav className="navbar navbar-expand-lg align-items-center">
        <div className={`offcanvas offcanvas-start ${open ? 'show' : ''}`} tabIndex={-1}>
          <div className="offcanvas-header border-bottom">
            <Logo />
            <button type="button" className="btn-close btn-close-white" aria-label="Close" onClick={onClose} />
          </div>
          <div className="offcanvas-body">
            <ul className="navbar-nav align-items-center flex-grow-1">
              {groups.map((g) => {
                const active = g.items.some((i) => (i.to === '/' ? loc.pathname === '/' : loc.pathname.startsWith(i.to)))
                return (
                  <li key={g.label} className="nav-item dropdown" onMouseLeave={() => setDrop(null)}>
                    <a className={`nav-link dropdown-toggle dropdown-toggle-nocaret ${active ? 'active' : ''}`} role="button" tabIndex={0}
                      onClick={() => setDrop(drop === g.label ? null : g.label)} onKeyDown={(e) => e.key === 'Enter' && setDrop(drop === g.label ? null : g.label)}>
                      <div className="parent-icon"><i className={`bx ${g.icon}`} /></div>
                      <div className="menu-title d-flex align-items-center">{g.label}</div>
                      <div className="ms-auto dropy-icon"><i className="bx bx-chevron-down" /></div>
                    </a>
                    <ul className="dropdown-menu" style={drop === g.label ? { display: 'block' } : undefined}>
                      {g.items.map((i) => (
                        <li key={i.to}>
                          <NavLink to={i.to} end={i.to === '/'} className={({ isActive }) => `dropdown-item ${isActive ? 'active' : ''}`}>
                            <Icon name={i.icon} />{i.label}
                          </NavLink>
                        </li>
                      ))}
                    </ul>
                  </li>
                )
              })}
            </ul>
          </div>
        </div>
      </nav>
      {open && <div className="offcanvas-backdrop fade show" onClick={onClose} />}
    </div>
  )
}

/** Dashtreme's user box, for the operator: which console this is and the system pages. */
function OperatorBox() {
  const { meta } = useApp()
  const [open, setOpen] = useState(false)
  return (
    <div className="user-box dropdown px-3 d-none d-md-flex" onMouseLeave={() => setOpen(false)}>
      <a className="d-flex align-items-center nav-link dropdown-toggle gap-3 dropdown-toggle-nocaret" role="button" onClick={() => setOpen(!open)}>
        <img src="/favicon.svg" className="user-img" alt="" style={{ background: 'rgb(255 255 255 / 12%)', padding: 4 }} />
        <div className="user-info">
          <p className="user-name mb-0">Operator</p>
          <p className="designattion mb-0">{meta ? (meta.hosts_workers ? `console · pid ${meta.pid}` : 'observer · read only') : '…'}</p>
        </div>
      </a>
      <ul className="dropdown-menu dropdown-menu-end" style={open ? { display: 'block', right: 0, top: '100%' } : undefined}>
        <li><NavLink className="dropdown-item d-flex align-items-center gap-2" to="/processes"><i className="bx bx-cog fs-5" />Processes</NavLink></li>
        <li><NavLink className="dropdown-item d-flex align-items-center gap-2" to="/safety"><i className="bx bx-shield-quarter fs-5" />Safety & wallet</NavLink></li>
        <li><NavLink className="dropdown-item d-flex align-items-center gap-2" to="/logs"><i className="bx bx-terminal fs-5" />Logs</NavLink></li>
        <li><div className="dropdown-divider mb-0" /></li>
        <li><span className="dropdown-item-text d-block small text-muted2 px-3 py-2">{meta ? `fly-trader ${meta.version}` : ''}</span></li>
      </ul>
    </div>
  )
}

function TopBar({ onMenu }: { onMenu: () => void }) {
  const conn = useConn()
  return (
    <header>
      <div className="topbar d-flex align-items-center">
        <nav className="navbar navbar-expand gap-3">
          <div className="topbar-logo-header d-none d-lg-flex"><Logo /></div>
          <div className="mobile-toggle-menu d-block d-lg-none" role="button" aria-label="Menu" onClick={onMenu}><i className="bx bx-menu" style={{ fontSize: 26 }} /></div>
          <div className="search-bar flex-grow-1"><SearchBox /></div>
          <div className="top-menu ms-auto">
            <ul className="navbar-nav align-items-center gap-1">
              <li className="nav-item d-flex align-items-center" title={conn === 'open' ? 'Live: the console pushes changes as they happen' : 'Reconnecting to the console…'}>
                <span className={`conn-dot ${conn === 'open' ? 'ok' : 'bad'}`} />
                <span className="d-none d-md-inline" style={{ fontSize: 12.5, color: 'var(--fly-muted)' }}>{conn === 'open' ? 'live' : 'reconnecting'}</span>
              </li>
              <li className="nav-item"><Notifications /></li>
            </ul>
          </div>
          <OperatorBox />
        </nav>
      </div>
    </header>
  )
}

export function Layout({ children }: { children: ReactNode }) {
  const { meta, inspect, closeInspector } = useApp()
  const [menu, setMenu] = useState(false)
  return (
    <div className="wrapper">
      <div className="header-wrapper">
        <TopBar onMenu={() => setMenu(true)} />
        <Menu open={menu} onClose={() => setMenu(false)} />
      </div>
      <div className="page-wrapper">
        <div className="page-content">
          {children}
        </div>
      </div>
      <footer className="page-footer">
        fly-trader operator console{meta ? ` · ${meta.hosts_workers ? `pid ${meta.pid} hosts the workers` : 'observer: hosts no workers, controls are refused'}` : ''}
      </footer>
      <LiveToasts />
      {inspect && <InspectorDrawer target={inspect} onClose={closeInspector} />}
    </div>
  )
}
