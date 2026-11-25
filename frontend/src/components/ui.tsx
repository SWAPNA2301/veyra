import type { ReactNode } from 'react'
import { ArrowUpRight, CircleHelp, Database, Download, Info } from 'lucide-react'

export const palette = { mint: '#62e6bf', blue: '#739eff', amber: '#e5ba75', muted: '#92a1b8', grid: '#252d3b' }
export const tooltipStyle = { backgroundColor: '#1a2434', border: '1px solid #354157', borderRadius: 10, color: '#e6edf8', fontSize: 12 }
export const number = (value: number | null | undefined, digits = 0) => value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
export const percent = (value: number | null | undefined, digits = 2) => value == null ? '—' : `${number(value, digits)}%`
export const compact = (value: number | null | undefined) => value == null ? '—' : Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(value)
export const date = (value: string | null) => value ? new Date(value).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' }) : 'Date unavailable'

export function Badge({ children, tone = 'mint' }: { children: ReactNode; tone?: 'mint' | 'amber' | 'neutral' }) {
  return <span className={`badge badge-${tone}`}><span className="status-dot" />{children}</span>
}
export function Panel({ title, subtitle, action, children, className = '' }: { title: string; subtitle?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}><div className="panel-heading"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>
}
export function Kpi({ label, value, detail, icon, series, color = palette.mint, accent = false }: { label: string; value: string; detail: string; icon: ReactNode; series?: number[]; color?: string; accent?: boolean }) {
  const id = `spark-${label.replace(/\W/g, '')}`
  const low = series?.length ? Math.min(...series) : 0
  const span = series?.length ? Math.max(...series) - low || 1 : 1
  const path = series && series.length > 1 ? series.map((value, i) => `${i ? 'L' : 'M'}${i / (series.length - 1) * 260},${40 - (value - low) / span * 31}`).join(' ') : ''
  return <article className={`kpi-card ${accent ? 'kpi-accent' : ''}`}><div className="flex items-center justify-between gap-2"><span className="kpi-label">{label}</span><span className="kpi-icon" style={{ color }}>{icon}</span></div><div className="kpi-value">{value}</div><div className="kpi-detail">{detail}</div>{path && <div className="sparkline"><svg viewBox="0 0 260 44" preserveAspectRatio="none" width="100%" height="100%" role="img" aria-label={`${label}: saved weekly trend, independently scaled`}><defs><linearGradient id={id} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity={.24} /><stop offset="100%" stopColor={color} stopOpacity={0} /></linearGradient></defs><path d={`${path} L260,44 L0,44 Z`} fill={`url(#${id})`} /><path d={path} fill="none" stroke={color} strokeWidth={1.7} vectorEffect="non-scaling-stroke" /></svg></div>}</article>
}
export function Empty({ title = 'Report unavailable', children }: { title?: string; children?: ReactNode }) {
  return <div className="empty-state"><Database size={28} /><h3>{title}</h3><p>{children ?? 'Restore the matching local report to see this view. No data is substituted.'}</p></div>
}
export function Note({ children, warning = false }: { children: ReactNode; warning?: boolean }) {
  return <div className={`note ${warning ? 'note-warning' : ''}`}><Info size={16} className="shrink-0" /><div>{children}</div></div>
}
export function Help({ text }: { text: string }) { return <span title={text} aria-label={text} tabIndex={0} className="help-icon"><CircleHelp size={14} /></span> }
export function LinkButton({ children, onClick }: { children: ReactNode; onClick: () => void }) { return <button className="text-button" onClick={onClick}>{children}<ArrowUpRight size={15} /></button> }
export function DownloadButton({ label, href, name }: { label: string; href: string; name: string }) { return <a className="button button-secondary" href={href} download={name}><Download size={15} />{label}</a> }
