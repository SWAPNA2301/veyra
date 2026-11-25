import { useEffect, useRef, useState } from 'react'
import { ChartNoAxesCombined, Download, Expand, Fingerprint, Layers3, X } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DashboardData } from '../types'
import { Badge, Empty, Kpi, Note, number, palette, Panel, tooltipStyle } from '../components/ui'

const descriptions = {
  global: { title: 'The features behind the signal', detail: 'Signed one-hot contributions are grouped by original feature, then ranked by mean absolute magnitude.', label: 'Global importance' },
  beeswarm: { title: 'The spread behind the average', detail: 'Each point is one sampled validation row. Positive contributions raise the model score; they do not establish causality.', label: 'SHAP distribution' },
  local: { title: 'One prediction, explained', detail: 'The saved waterfall explains sample position 0. Contributions are raw log-odds, not probability percentage points.', label: 'Local explanation' },
}

export default function Explainability({ data }: { data: DashboardData }) {
  const e = data.explainability
  const [tab, setTab] = useState<'global' | 'beeswarm' | 'local'>('global')
  const [all, setAll] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [broken, setBroken] = useState<Record<string, boolean>>({})
  const dialog = useRef<HTMLDialogElement>(null)
  const image = e.images.find(i => i.key === tab)
  const imageAvailable = image?.available && !broken[tab]
  const info = descriptions[tab]
  useEffect(() => { if (expanded) dialog.current?.showModal(); else dialog.current?.close() }, [expanded])
  const download = imageAvailable && image ? <a className="button button-secondary" href={image.url} download={`veyra-shap-${tab}.png`}><Download size={14} />Original figure</a> : undefined
  return <div className="page-content">
    <div className="explain-kpis"><Kpi label="Validation sample" value={number(e.sample_rows)} detail={`Reproducible sample · seed ${number(e.seed)}`} icon={<Fingerprint size={17} />} accent /><Kpi label="Original features" value={number(e.original_features)} detail={`Grouped from ${number(e.encoded_columns)} encoded columns`} icon={<Layers3 size={17} />} /><Kpi label="Local example score" value={number(e.local_score, 4)} detail="One fixed case · uncalibrated model score" icon={<ChartNoAxesCombined size={17} />} /></div>
    <div className="tabs" role="tablist" aria-label="Explanation view">{(Object.keys(descriptions) as (keyof typeof descriptions)[]).map(key => <button role="tab" aria-selected={tab === key} aria-controls={`shap-${key}`} id={`tab-${key}`} key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>{descriptions[key].label}</button>)}</div>
    <div className="explain-layout" role="tabpanel" id={`shap-${tab}`} aria-labelledby={`tab-${tab}`}>
      <Panel title={info.title} subtitle={tab === 'global' ? 'Mean absolute grouped SHAP · raw log-odds' : 'Original saved figure · no computation on this page'} action={tab === 'global' ? <label className="check-label"><input type="checkbox" checked={all} onChange={event => setAll(event.target.checked)} />All features</label> : <button className="icon-button" onClick={() => setExpanded(true)} disabled={!imageAvailable} title="Expand original figure" aria-label="Expand original figure"><Expand size={16} /></button>}>
        {tab === 'global' ? e.importance.length ? <div className="chart-frame shap-chart" style={{ height: all ? Math.max(350, e.importance.length * 25) : 350 }}><ResponsiveContainer width="100%" height="100%"><BarChart layout="vertical" data={all ? e.importance : e.importance.slice(0, 10)} margin={{ top: 0, left: 0, right: 18, bottom: 0 }}><CartesianGrid stroke={palette.grid} strokeDasharray="3 5" horizontal={false} /><XAxis type="number" tick={{ fill: palette.muted, fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis dataKey="feature" type="category" tick={{ fill: '#bdc9dc', fontSize: 11 }} width={110} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} formatter={v => [number(Number(v), 4), 'Mean |grouped SHAP|']} cursor={{ fill: '#ffffff04' }} /><Bar dataKey="mean_abs_grouped_shap" fill={palette.mint} radius={[0, 4, 4, 0]} maxBarSize={13} isAnimationActive={false} /></BarChart></ResponsiveContainer></div> : <Empty /> : imageAvailable && image ? <div className="shap-paper"><img src={image.url} alt={info.title} onError={() => setBroken(current => ({ ...current, [tab]: true }))} /></div> : <Empty title="Saved SHAP figure unavailable">Restore the original local image. The dashboard will not recompute SHAP values.</Empty>}
        <div className="figure-footer"><span>{tab === 'global' ? `${all ? e.importance.length : Math.min(10, e.importance.length)} of ${e.importance.length} original features` : 'Original resolution available to download'}</span><div className="flex gap-2">{tab === 'global' && imageAvailable && <button className="icon-button" onClick={() => setExpanded(true)} aria-label="View original global SHAP figure" title="View original global SHAP figure"><Expand size={16} /></button>}{download}</div></div>
      </Panel>
      <aside className="explain-guide"><Panel title="How to read this" subtitle="Context before conclusions"><span className="insight-number">{tab === 'global' ? '01' : tab === 'beeswarm' ? '02' : '03'}</span><h3>{tab === 'global' ? 'A global view of influence.' : tab === 'beeswarm' ? 'Patterns, not just averages.' : 'A single case, in context.'}</h3><p className="guide-copy">{info.detail}</p><div className="guide-divider" /><span className="eyebrow">STRONGEST GLOBAL SIGNALS</span><div className="ranking-list">{e.importance.slice(0, 5).map((row, i) => <div key={row.feature}><span className="rank-index">0{i + 1}</span><span>{row.feature}</span><strong>{number(row.mean_abs_grouped_shap, 3)}</strong></div>)}</div><p className="panel-footnote">Mean absolute grouped contributions, not causal effects.</p></Panel><div className="explain-scope"><Badge>Validation only</Badge><p>Saved LightGBM pipeline. Existing fitted preprocessing. No access to test data.</p>{e.max_additivity_error != null && <span>Saved reconstruction error: {e.max_additivity_error.toExponential(2)}</span>}</div></aside>
    </div>
    <Note>These are saved explanations from a reproducible validation sample. Scores are uncalibrated; SHAP contributions should not be interpreted as fraud probabilities.</Note>
    <dialog ref={dialog} className="image-dialog" onClose={() => setExpanded(false)} onClick={event => { if (event.target === event.currentTarget) setExpanded(false) }}><div className="dialog-heading"><h2>{info.label} · original figure</h2><button className="icon-button" onClick={() => setExpanded(false)} aria-label="Close expanded figure"><X size={18} /></button></div>{imageAvailable && image && <img src={image.url} alt={info.title} />}</dialog>
  </div>
}
