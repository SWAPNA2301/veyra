import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { ArrowUpRight, CalendarDays, ChartNoAxesCombined, ChevronRight, CircleHelp, Cpu, Database, Fingerprint, LayoutDashboard, LockKeyhole, Menu, RefreshCw, ScanLine, ShieldCheck, Layers3, X } from 'lucide-react'
import { useDashboard } from './api'
import type { Page } from './types'
import { Badge, date, Note } from './components/ui'

const Overview = lazy(() => import('./pages/Overview'))
const Performance = lazy(() => import('./pages/Performance'))
const Explainability = lazy(() => import('./pages/Explainability'))
const Predictor = lazy(() => import('./pages/Predictor'))
const BatchDetection = lazy(() => import('./pages/BatchDetection'))
const Monitoring = lazy(() => import('./pages/Monitoring'))
const pages = [
  { id: 'overview' as const, label: 'Overview', icon: LayoutDashboard, title: 'Overview dashboard', description: 'See the patterns. Understand the risk. Your fraud landscape, in focus.' },
  { id: 'performance' as const, label: 'Model performance', icon: ChartNoAxesCombined, title: 'Performance, beyond accuracy.', description: 'Compare validation and final holdout results at the frozen operating point.' },
  { id: 'explainability' as const, label: 'Explainability', icon: Fingerprint, title: 'Understand the signal.', description: 'Explore the features and associations behind the saved model’s predictions.' },
  { id: 'predictor' as const, label: 'Risk predictor', icon: ScanLine, title: 'Inspect a transaction.', description: 'A private workspace for one manual scenario, powered by your saved LightGBM model.' },
  { id: 'batch' as const, label: 'Batch Detection', icon: Layers3, title: 'A wider view of the signal.', description: 'Analyze a CSV with the saved model. Inspect each prediction, with the context intact.' },
  { id: 'monitoring' as const, label: 'Drift monitoring', icon: ChartNoAxesCombined, title: 'Know when the inputs shift.', description: 'Compare batch distributions with a fixed training reference. Evidence for review, never an automatic decision.' },
]
function currentPage(): Page { const value = location.hash.slice(1); return pages.some(p => p.id === value) ? value as Page : 'overview' }
function Loading() { return <div className="loading-page" role="status" aria-label="Loading saved reports"><div className="kpi-grid">{[0, 1, 2, 3].map(i => <div className="skeleton skeleton-card" key={i} />)}</div><div className="overview-charts"><div className="skeleton skeleton-chart" /><div className="skeleton skeleton-chart" /></div><p>Connecting to your local reports…</p></div> }

export default function App() {
  const [page, setPage] = useState<Page>(currentPage)
  const [mobileOpen, setMobileOpen] = useState(false)
  const { data, loading, error, refresh } = useDashboard()
  const about = useRef<HTMLDialogElement>(null)
  const navigation = useRef<HTMLElement>(null)
  const menu = useRef<HTMLButtonElement>(null)
  const active = pages.find(p => p.id === page)!
  useEffect(() => { const change = () => { setPage(currentPage()); setMobileOpen(false); window.scrollTo({ top: 0 }) }; window.addEventListener('hashchange', change); return () => window.removeEventListener('hashchange', change) }, [])
  useEffect(() => { document.title = `Veyra · ${active.label}` }, [active.label])
  useEffect(() => {
    if (!mobileOpen) return
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    navigation.current?.querySelector<HTMLAnchorElement>('a[aria-current="page"]')?.focus()
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setMobileOpen(false); menu.current?.focus() }
      if (event.key === 'Tab' && navigation.current) {
        const focusable = [...navigation.current.querySelectorAll<HTMLElement>('a,button')].filter(e => e.getClientRects().length > 0)
        const first = focusable[0], last = focusable.at(-1)
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
      }
    }
    document.addEventListener('keydown', key)
    return () => { document.body.style.overflow = previous; document.removeEventListener('keydown', key) }
  }, [mobileOpen])
  function navigate(next: Page) { location.hash = next; setPage(next); setMobileOpen(false) }
  return <div className="app-shell">
    <a href="#main-content" className="skip-link" onClick={event => { event.preventDefault(); document.getElementById('main-content')?.focus() }}>Skip to content</a>
    {mobileOpen && <button className="sidebar-backdrop" aria-label="Close navigation" onClick={() => { setMobileOpen(false); menu.current?.focus() }} />}
    <aside ref={navigation} className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`} aria-label="Main navigation" id="navigation">
      <a className="brand" href="#overview" onClick={() => navigate('overview')} aria-label="Veyra home"><span className="brand-mark">v</span><span>veyra<span className="brand-period">.</span></span><span className="brand-tag">INTELLIGENCE</span></a>
      <button className="icon-button mobile-close" aria-label="Close navigation" onClick={() => { setMobileOpen(false); menu.current?.focus() }}><X size={18} /></button>
      <div className="workspace-switch"><span className="workspace-symbol"><ShieldCheck size={18} /></span><div><strong>Fraud intelligence</strong><small>Private workspace</small></div><LockKeyhole size={13} /></div>
      <div className="nav-section-label">INTELLIGENCE</div>
      <nav className="nav-list">{pages.slice(0, 3).map(item => <a key={item.id} href={`#${item.id}`} onClick={() => navigate(item.id)} aria-current={page === item.id ? 'page' : undefined} className={page === item.id ? 'active' : ''}><item.icon size={17} /><span>{item.label}</span>{page === item.id && <span className="nav-indicator" />}</a>)}</nav>
      <div className="nav-section-label workspace-label">WORKSPACE</div>
      <nav className="nav-list">{pages.slice(3).map(item => <a key={item.id} href={`#${item.id}`} onClick={() => navigate(item.id)} aria-current={page === item.id ? 'page' : undefined} className={page === item.id ? 'active' : ''}><item.icon size={17} /><span>{item.label}</span>{page === item.id && <span className="nav-indicator" />}</a>)}</nav>
      <div className="sidebar-bottom"><div className="engine-card"><div className="engine-heading"><Cpu size={16} /><span>MODEL ENGINE</span><span className={`engine-status ${data?.model.available ? 'available' : ''}`} /></div><strong>LightGBM</strong><p>Saved pipeline. Frozen threshold.<br />No background training.</p><button onClick={() => navigate('performance')}>View performance<ArrowUpRight size={14} /></button></div><div className="sidebar-source"><Database size={16} /><div><strong>IEEE-CIS</strong><small>Local dataset · historical reports</small></div></div><div className="sidebar-privacy"><LockKeyhole size={11} />Your data stays on this machine.</div></div>
    </aside>
    <div className="main-shell"><header className="topbar"><div className="flex items-center gap-3"><button ref={menu} className="icon-button mobile-menu" aria-label="Open navigation" aria-controls="navigation" aria-expanded={mobileOpen} onClick={() => setMobileOpen(true)}><Menu size={20} /></button><div className="breadcrumb"><span>Workspace</span><ChevronRight size={13} /><strong>{active.label}</strong></div></div><div className="topbar-right"><span className={`connection-status ${error ? 'connection-error' : ''}`}><i />{error ? 'API unavailable' : data ? 'Local API connected' : 'Connecting to API'}</span><span className="topbar-divider" /><button className="icon-button" title="About this workspace" aria-label="About this workspace" onClick={() => about.current?.showModal()}><CircleHelp size={18} /></button><span className="workspace-avatar" title="Local workspace">V</span></div></header>
      <main id="main-content" tabIndex={-1}><div className="page-heading"><div><div className="heading-eyebrow">{page === 'monitoring' ? 'INPUT DRIFT MONITORING' : page === 'batch' ? 'BATCH INTELLIGENCE' : page === 'predictor' ? 'MANUAL INFERENCE' : 'YOUR RISK. IN PERSPECTIVE.'}</div><h1>{active.title}</h1><p>{active.description}</p></div><div className="heading-actions"><span className="date-chip"><CalendarDays size={14} />{page === 'monitoring' ? 'Fixed training reference' : page === 'batch' ? 'Private batch workspace' : data ? `Snapshot · ${date(data.overview.generated_at)}` : 'Saved local reports'}</span><button aria-label={loading ? 'Refreshing reports' : 'Refresh reports'} className="button button-secondary refresh-button" onClick={() => void refresh()} disabled={loading}><RefreshCw size={14} className={loading ? 'animate-spin' : ''} /><span>{loading ? 'Refreshing' : 'Refresh'}</span></button></div></div>
        {error && <div className="mb-5" role="alert"><Note warning>{error}{data && ' Showing the last loaded snapshot.'}<button className="text-button" onClick={() => void refresh()}>Retry connection<RefreshCw size={13} /></button></Note></div>}
        {!data ? loading ? <Loading /> : <div className="api-empty"><Database size={36} /><h2>Your workspace is ready.</h2><p>Start the local FastAPI bridge to load the real reports and model availability. No demonstration data is shown.</p><code>python -m uvicorn veyra.api:app --app-dir src --host 127.0.0.1 --port 8000</code><button className="button button-primary" onClick={() => void refresh()}><RefreshCw size={15} />Reconnect</button></div> : <><Suspense fallback={<Loading />}>{page === 'overview' && <Overview data={data} navigate={navigate} />}{page === 'performance' && <Performance data={data} />}{page === 'explainability' && <Explainability data={data} />}{page === 'predictor' && <Predictor data={data} />}{page === 'batch' && <BatchDetection data={data} />}{page === 'monitoring' && <Monitoring />}</Suspense>{data.warnings.length > 0 && <details className="artifact-warnings"><summary>{data.warnings.length} local artifact notice{data.warnings.length === 1 ? '' : 's'}</summary><ul>{data.warnings.map((message, i) => <li key={i}>{message}</li>)}</ul></details>}</>}
        <footer className="footer"><span><span className="footer-mark">v</span>VEYRA <span className="footer-slash">/</span> FRAUD INTELLIGENCE</span><span><LockKeyhole size={11} />Saved reports. Local inference. No live transaction feed.</span></footer>
      </main>
    </div>
    <dialog ref={about} className="about-dialog"><form method="dialog"><div className="dialog-heading"><h2>About your workspace</h2><button className="icon-button" aria-label="Close workspace information"><X size={18} /></button></div><Badge>Local only</Badge><p>Veyra reads existing aggregate IEEE-CIS reports and saved SHAP figures. The model is loaded only when you submit a manual transaction or CSV batch.</p><p>Preprocessing was fitted on train only. The decision threshold was frozen on validation. Test metrics are the saved one-time evaluation, never a live rerun.</p><p>No transaction inputs are stored. Model scores are uncalibrated and must not be interpreted as fraud probabilities.</p><button className="button button-primary">Got it</button></form></dialog>
  </div>
}
