import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, ArrowRight, BarChart3, Check, Database, FileSpreadsheet, LoaderCircle, LockKeyhole, Search, ShieldAlert, UploadCloud, X } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { request } from '../api'
import { fileError } from '../batch'
import { compareDrift } from '../drift'
import type { DriftFeature, DriftResult, DriftSeverity, ReferenceStatus } from '../drift'
import { Badge, Empty, Kpi, Note, number, palette, Panel, percent, tooltipStyle } from '../components/ui'
import './batch.css'
import './monitoring.css'

const severityLabel = (severity: DriftSeverity | null) => severity === 'high' ? 'High shift' : severity === 'watch' ? 'Watch' : severity === 'stable' ? 'Low shift' : 'Insufficient data'
function Severity({ value }: { value: DriftSeverity | null }) { return <span className={`drift-severity drift-${value ?? 'unknown'}`}><i />{severityLabel(value)}</span> }
const signed = (value: number) => `${value > 0 ? '+' : ''}${number(value, 2)} pp`

function FeatureComparison({ feature, result }: { feature: DriftFeature; result: DriftResult }) {
  const chart = feature.bins.map((bin, index) => ({ ...bin, key: index, reference: bin.reference_share * 100, batch: bin.batch_share * 100 }))
  return <Panel title={feature.name} subtitle={`${feature.kind.toUpperCase()} / FIXED TRAINING BINS`} action={<Severity value={feature.severity} />}>
    <div className="drift-feature-stats"><div><span>Population stability index</span><strong>{number(feature.psi, 4)}</strong><small>{feature.psi === null ? 'PSI withheld · too few observed values' : 'Includes the explicit missing-value bin'}</small></div><div><span>Missingness change</span><strong>{signed(feature.missing_delta_pp)}</strong><small>{percent(feature.reference_missing_rate * 100)} train → {percent(feature.missing_rate * 100)} batch</small></div></div>
    {!feature.provided && <Note warning>This column was omitted. Its batch values are all missing; the missingness alert may indicate incomplete input coverage rather than a population change.</Note>}
    {feature.psi === null && <p className="drift-sample-note">PSI requires {number(result.reference.policy.min_batch_rows)} total batch rows and {number(result.reference.policy.min_observed_rows)} nonmissing values in each population. Observed: {number(feature.reference_observed_rows)} train / {number(feature.observed_rows)} batch. Missingness proportions remain descriptive.</p>}
    <div className="drift-chart" style={{ height: Math.max(260, chart.length * 29 + 35) }} role="img" aria-label={`${feature.name}: training versus uploaded batch shares across ${chart.length} fixed bins`}><ResponsiveContainer width="100%" height="100%"><BarChart data={chart} layout="vertical" margin={{ left: 0, right: 18, top: 10, bottom: 8 }}><CartesianGrid stroke={palette.grid} strokeDasharray="3 5" horizontal={false} /><XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} axisLine={false} tickLine={false} tick={{ fill: palette.muted, fontSize: 10 }} /><YAxis type="category" dataKey="key" width={130} tickFormatter={v => { const label = chart[Number(v)]?.label.replace('Category: ', '') ?? ''; return label.length > 19 ? label.slice(0, 18) + '…' : label }} tick={{ fill: '#b3c5da', fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} labelFormatter={key => chart[Number(key)]?.label ?? ''} formatter={(value, name) => [percent(Number(value)), name]} cursor={{ fill: '#ffffff04' }} /><Bar dataKey="reference" name="Training reference" fill={palette.blue} radius={[0, 2, 2, 0]} maxBarSize={8} isAnimationActive={false} /><Bar dataKey="batch" name="Uploaded batch" fill={palette.mint} radius={[0, 2, 2, 0]} maxBarSize={8} isAnimationActive={false} /></BarChart></ResponsiveContainer></div>
    <div className="chart-legend"><span><i style={{ background: palette.blue }} />Training reference</span><span><i style={{ background: palette.mint }} />Uploaded batch</span></div>
    <details className="drift-bin-details"><summary>Exact bin labels & aggregate counts</summary><div className="table-scroll"><table><thead><tr><th>Bin</th><th>Train</th><th>Batch</th></tr></thead><tbody>{feature.bins.map((bin, index) => <tr key={index}><td>{bin.label}</td><td>{number(bin.reference_count)}</td><td>{number(bin.batch_count)}</td></tr>)}</tbody></table></div></details>
    <p className="panel-footnote">Unsmoothed observed shares are charted. PSI uses fixed training bins and 10⁻⁶ proportion smoothing. Other/unseen categories share one bucket; changes within that bucket are not resolved.</p>
  </Panel>
}

export default function Monitoring() {
  const [metadata, setMetadata] = useState<ReferenceStatus | null>(null)
  const [referenceError, setReferenceError] = useState('')
  const [reload, setReload] = useState(0)
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<DriftResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [selected, setSelected] = useState('')
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('all')
  const input = useRef<HTMLInputElement>(null)
  const pending = useRef<AbortController | null>(null)
  const generation = useRef(0)
  useEffect(() => () => { generation.current++; pending.current?.abort() }, [])
  useEffect(() => {
    const controller = new AbortController(); setMetadata(null); setReferenceError('')
    request<ReferenceStatus>('/drift/reference', { signal: controller.signal }).then(value => { if (!controller.signal.aborted) setMetadata(value) }).catch(e => { if (!controller.signal.aborted) setReferenceError(e instanceof Error ? e.message : 'Reference unavailable.') })
    return () => controller.abort()
  }, [reload])
  function reset() { generation.current++; pending.current?.abort(); pending.current = null; setResult(null); setSelected(''); setQuery(''); setFilter('all'); setError(''); setDragging(false); setBusy(false) }
  function choose(files: File[]) { if (busy) return; reset(); const issue = fileError(files); setError(issue ?? ''); setFile(issue ? null : files[0]) }
  function clear() { reset(); setFile(null); if (input.current) input.current.value = '' }
  async function compare() {
    if (!file || busy || metadata?.status !== 'ready') return
    reset(); const attempt = ++generation.current; const controller = new AbortController(); pending.current = controller; setBusy(true)
    try {
      const response = await compareDrift(file, controller.signal)
      if (response.reference.profile_id !== metadata.reference.profile_id) throw new Error('The reference changed outside this session. Reload the page and review the reference before comparing again.')
      if (attempt === generation.current && !controller.signal.aborted) { setResult(response); setSelected(response.features[0]?.name ?? ''); setFile(null) }
    } catch (e) { if (attempt === generation.current && !controller.signal.aborted) setError(e instanceof TypeError ? 'Could not reach the local API. Restart the updated backend and retry.' : e instanceof Error ? e.message : 'The batch could not be compared.') }
    finally { if (attempt === generation.current) { setBusy(false); pending.current = null } }
  }
  const reference = result?.reference ?? (metadata?.status === 'ready' ? metadata.reference : null)
  const features = useMemo(() => result?.features.filter(f => f.name.toLowerCase().includes(query.trim().toLowerCase()) && (filter === 'all' || (filter === 'insufficient' ? f.psi === null : f.severity === filter))) ?? [], [result, query, filter])
  const current = result?.features.find(f => f.name === selected)
  const summary = result?.summary
  return <div className="page-content monitoring-page">
    <div className="drift-reference-strip"><span className="drift-reference-icon"><LockKeyhole size={21} /></span><div><strong>A fixed point of comparison.</strong><p>{reference ? `${number(reference.reference_rows)} training rows · ${reference.model_run_id}` : 'Only the frozen training window can define this reference.'}</p></div><Badge tone={reference ? 'mint' : 'neutral'}>{reference ? 'Training only' : 'Awaiting reference'}</Badge></div>
    <div className="drift-intake">
      <Panel title="Compare a batch, not a guess." subtitle="01 / INPUT DISTRIBUTIONS · no model inference required" action={<Badge tone="neutral">5 MB / 5,000 rows</Badge>}>
        <input ref={input} type="file" className="sr-only" accept=".csv,text/csv" aria-label="Choose monitoring CSV" disabled={busy} onChange={e => { const files = Array.from(e.currentTarget.files ?? []); if (files.length) choose(files); e.currentTarget.value = '' }} />
        <div className={`batch-dropzone drift-dropzone ${dragging ? 'is-dragging' : ''}`} onDragOver={e => { e.preventDefault(); if (!busy) setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); if (!busy) choose(Array.from(e.dataTransfer.files)) }} aria-busy={busy}>
          <span className="batch-upload-icon"><UploadCloud size={24} /></span><h3>{file ? file.name : result ? 'Compare another batch' : 'Drop a CSV to check for drift'}</h3><p>{file ? `${number(file.size / 1000, 1)} KB · ready for server validation` : 'The same reviewed input columns as Batch Detection.'}</p><button className="button button-secondary" disabled={busy} onClick={() => input.current?.click()}><FileSpreadsheet size={14} />{file ? 'Replace CSV' : 'Choose CSV'}</button>
        </div>
        <div className="batch-action-row"><button className="button button-primary" disabled={!file || busy || metadata?.status !== 'ready'} onClick={() => void compare()}>{busy ? <LoaderCircle size={15} className="animate-spin" /> : <Activity size={15} />}{busy ? 'Comparing distributions…' : 'Check drift'}{!busy && <ArrowRight size={15} />}</button><button className="button button-quiet" onClick={clear} disabled={!file && !result && !error && !busy}><X size={14} />{busy ? 'Cancel & clear' : 'Clear comparison'}</button></div>
        {error && <p role="alert" className="form-error">{error}</p>}
        <p className="batch-live" role="status">{busy ? 'Validating and comparing in memory. No input rows or history are saved.' : result ? 'Comparison complete. The uploaded file has been released; only aggregate results remain in this page.' : 'No upload until you choose Check drift. TransactionAmt is required; optional omitted fields count as missing.'}</p>
      </Panel>
      <Panel title="Evidence before escalation" subtitle="FIXED MONITORING POLICY">
        <div className="drift-policy-item"><span>01</span><div><strong>Enough data to compare</strong><p>{reference?.policy.min_batch_rows ?? 200} batch rows for severity; {reference?.policy.min_observed_rows ?? 100} nonmissing values per population for feature PSI. Smaller samples are descriptive only.</p></div></div>
        <div className="drift-policy-item"><span>02</span><div><strong>Distribution + missingness</strong><p>Numeric training quantiles; frequent training categories plus other/unseen. Missingness is checked before any imputation.</p></div></div>
        <div className="drift-policy-item"><span>03</span><div><strong>No automatic response</strong><p>No retraining, score adjustment or baseline updates. Inspect input coverage and collection changes before drawing conclusions.</p></div></div>
        <div className="drift-policy-bands"><span><Severity value="stable" />PSI &lt; 0.10</span><span><Severity value="watch" />0.10 ≤ PSI &lt; 0.25</span><span><Severity value="high" />PSI ≥ 0.25</span></div><p className="panel-footnote">Absolute missingness changes: watch ≥ 5 pp; high ≥ 10 pp. Overall feature severity takes the higher eligible signal. These are review heuristics, not significance tests.</p>
      </Panel>
    </div>
    {(referenceError || metadata?.status === 'unavailable') && <Note warning><strong>Training reference unavailable.</strong> {referenceError || (metadata?.status === 'unavailable' ? metadata.detail : '')}<p>Create once from the original local training sources:</p><code className="drift-command">$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m veyra.drift_reference</code><button className="text-button" onClick={() => setReload(value => value + 1)}>Check reference again</button></Note>}
    <div className="kpi-grid drift-kpis"><Kpi label="Batch rows" value={number(result?.rows)} detail={result ? `${number(summary?.provided_features)} input columns supplied` : 'Awaiting your uploaded batch'} icon={<Database size={17} />} /><Kpi label="High-shift features" value={number(summary?.high)} detail={result?.status === 'insufficient_data' ? 'Not assessed · sample too small' : 'PSI or missingness needs review'} icon={<ShieldAlert size={17} />} accent /><Kpi label="Highest eligible PSI" value={number(summary?.max_psi, 3)} detail={result ? `${number(summary?.psi_assessed)} / ${number(summary?.features)} features eligible` : 'No fabricated baseline comparisons'} icon={<BarChart3 size={17} />} /><Kpi label="Largest missingness shift" value={summary?.max_missing_change_pp == null ? '—' : `${number(summary.max_missing_change_pp, 1)} pp`} detail="Absolute change from training" icon={<Activity size={17} />} /></div>
    {result?.status === 'insufficient_data' && <Note warning><strong>Insufficient data — no severity assessment.</strong> This batch has {number(result.rows)} rows; at least {number(result.reference.policy.min_batch_rows)} are required. PSI and summary severity metrics are withheld. The charts below show observed proportions only.</Note>}
    <Note>{result?.notice ?? 'Feature drift is not concept drift, measured performance loss or confirmed fraud. This view compares one uploaded batch with a fixed training reference; it does not provide a historical monitoring feed.'}</Note>
    {result ? <div className="drift-results-grid"><Panel title="Feature watchlist" subtitle="02 / RANKED BY SEVERITY, PSI, THEN MISSINGNESS" action={<Badge tone="neutral">{number(summary?.watch)} watch</Badge>}>
      <div className="table-toolbar drift-table-toolbar"><label className="search-field"><Search size={14} /><input aria-label="Search drift features" placeholder="Find a feature…" value={query} onChange={e => setQuery(e.target.value)} /></label><select aria-label="Filter drift severity" value={filter} onChange={e => setFilter(e.target.value)}><option value="all">All features</option><option value="high">High shift</option><option value="watch">Watch</option><option value="stable">Low shift</option><option value="insufficient">PSI insufficient</option></select></div>
      <div className="table-scroll drift-ranking"><table><thead><tr><th>Feature</th><th>PSI</th><th>Missing Δ</th><th>Severity</th></tr></thead><tbody>{features.map(feature => <tr key={feature.name} className={selected === feature.name ? 'selected' : ''} onClick={() => setSelected(feature.name)}><td><button className="drift-feature-button" onClick={() => setSelected(feature.name)} aria-pressed={selected === feature.name}>{feature.name}</button><small>{feature.provided ? feature.kind : 'omitted column'}</small></td><td className="mono">{number(feature.psi, 3)}</td><td className="mono">{signed(feature.missing_delta_pp)}</td><td><Severity value={feature.severity} /></td></tr>)}</tbody></table>{!features.length && <p className="table-empty">No features match these filters.</p>}</div><p className="panel-footnote">Select a feature for its fixed-bin comparison. A missing PSI is not a zero or an assurance of stability. Low-observation features can still raise a missingness alert.</p>
    </Panel>{current && <FeatureComparison feature={current} result={result} />}</div> : <Panel title="A clear view of distribution change" subtitle="02 / COMPARISON"><Empty title={busy ? 'Comparing against training…' : 'Your first comparison starts here.'}>{busy ? 'Metrics appear only after a complete, validated response.' : 'Upload a supported CSV to see feature rankings, PSI and missingness changes. No sample results are substituted.'}</Empty></Panel>}
    {reference && <details className="drift-provenance"><summary><Check size={14} />Fixed reference provenance & method</summary><p>Created {new Date(reference.created_at_utc).toLocaleString()} · {number(reference.reference_rows)} training rows. TransactionDT window: {number(reference.train_window.TransactionDT_min)}–{number(reference.train_window.TransactionDT_max)}.</p><p>{reference.method}</p><p>The shared source CSVs are scanned, but only training-window feature rows are materialized. No validation/test feature distributions or labels are used.</p><code>{reference.profile_id}</code></details>}
    <p className="batch-retention"><LockKeyhole size={12} />Uploads are processed in memory and released after comparison. Clearing or leaving this page discards the aggregate result. The fixed reference contains training aggregates only.</p>
  </div>
}
