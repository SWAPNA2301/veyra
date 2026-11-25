import { useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { ArrowRight, ChartNoAxesCombined, Check, ChevronLeft, ChevronRight, Download, FileCheck2, FileSpreadsheet, Layers3, LoaderCircle, LockKeyhole, Search, ShieldAlert, UploadCloud, X } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { analyzeBatch, exportResults, fileError, scoreDistribution } from '../batch'
import type { BatchPrediction, BatchRow, DashboardData } from '../types'
import TransactionDrawer from '../components/TransactionDrawer'
import { Badge, Empty, Kpi, Note, number, palette, Panel, percent, tooltipStyle } from '../components/ui'
import './batch.css'

const PAGE_SIZE = 50

export default function BatchDetection({ data }: { data: DashboardData }) {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<BatchPrediction | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [dragging, setDragging] = useState(false)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('all')
  const [page, setPage] = useState(0)
  const [investigating, setInvestigating] = useState<BatchRow | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const pending = useRef<AbortController | null>(null)
  const generation = useRef(0)
  const dragDepth = useRef(0)
  useEffect(() => () => { generation.current++; pending.current?.abort() }, [])

  function resetResults() { setInvestigating(null); setResult(null); setQuery(''); setFilter('all'); setPage(0) }
  function select(files: File[]) {
    if (busy) return
    generation.current++; pending.current?.abort(); resetResults(); setMessage(''); setDragging(false); dragDepth.current = 0
    const problem = fileError(files)
    setError(problem ?? ''); setFile(problem ? null : files[0])
  }
  function clear() {
    generation.current++; pending.current?.abort(); pending.current = null
    setMessage(busy ? 'Cleared from this page. In-flight server processing may finish in memory; no results will be retained here.' : 'File and results cleared from this page.')
    setBusy(false); setFile(null); setError(''); resetResults(); setDragging(false); dragDepth.current = 0
    if (input.current) input.current.value = ''
  }
  async function analyze() {
    if (!file || busy || !data.model.available) return
    pending.current?.abort()
    const controller = new AbortController()
    pending.current = controller
    const attempt = ++generation.current
    resetResults(); setBusy(true); setError(''); setMessage('')
    try {
      const scored = await analyzeBatch(file, controller.signal)
      if (generation.current === attempt && !controller.signal.aborted) setResult(scored)
    } catch (e) {
      if (generation.current === attempt && !controller.signal.aborted) setError(e instanceof Error ? e.message : 'The batch could not be analyzed. Check the local API and try again.')
    } finally {
      if (generation.current === attempt) { setBusy(false); pending.current = null }
    }
  }
  function drag(event: DragEvent, entering: boolean) {
    event.preventDefault(); event.stopPropagation()
    if (busy || !event.dataTransfer.types.includes('Files')) return
    dragDepth.current = Math.max(0, dragDepth.current + (entering ? 1 : -1))
    setDragging(dragDepth.current > 0)
  }

  const filtered = useMemo(() => result?.rows.filter(row => {
    const text = query.trim().toLowerCase()
    return (filter === 'all' || (filter === 'flagged' ? row.flagged : !row.flagged))
      && (!text || String(row.row_number).includes(text) || row.score.toFixed(6).includes(text))
  }) ?? [], [result, query, filter])
  const bins = useMemo(() => scoreDistribution(result?.rows ?? []), [result])
  const lastPage = Math.max(0, Math.ceil(filtered.length / PAGE_SIZE) - 1)
  const current = Math.min(page, lastPage)
  const visible = filtered.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE)
  const threshold = result?.threshold ?? data.model.threshold
  const busyLabel = busy ? 'Validating & scoring…' : result ? 'Analyze Batch again' : 'Analyze Batch'

  return <div className="page-content batch-page">
    <div className="batch-topline"><Badge tone="neutral"><LockKeyhole size={12} />Session only</Badge><span>Upload → validate → inspect. Nothing saved automatically.</span></div>
    <div className="batch-intake-grid">
      <Panel title="Bring the batch. Find the signal." subtitle="01 / UPLOAD · one CSV, one frozen operating point" action={<Badge tone="neutral">5 MB / 5,000 rows</Badge>}>
        <input ref={input} id="batch-file" className="sr-only" type="file" accept=".csv,text/csv" aria-label="Choose batch CSV" disabled={busy} onChange={e => { const files = Array.from(e.currentTarget.files ?? []); if (files.length) select(files); e.currentTarget.value = '' }} />
        <div className={`batch-dropzone ${dragging ? 'is-dragging' : ''} ${file ? 'has-file' : ''}`} aria-busy={busy}
          onDragEnter={e => drag(e, true)} onDragLeave={e => drag(e, false)}
          onDragOver={e => { e.preventDefault(); e.stopPropagation(); e.dataTransfer.dropEffect = busy ? 'none' : 'copy' }}
          onDrop={e => { e.preventDefault(); e.stopPropagation(); if (!busy) select(Array.from(e.dataTransfer.files)) }}>
          <span className="batch-upload-icon">{file ? <FileCheck2 size={26} /> : <UploadCloud size={26} />}</span>
          <h3>{dragging ? 'Drop your CSV here' : file ? file.name : 'Drop your CSV into focus'}</h3>
          <p>{file ? `${number(file.size / 1000, 1)} KB · ${result ? 'Analysis complete' : 'File selected — CSV validation runs on Analyze'}` : 'Drag and drop a file, or select one from your computer.'}</p>
          <button className="button button-secondary" disabled={busy} onClick={() => input.current?.click()}><FileSpreadsheet size={15} />{file ? 'Replace CSV' : 'Choose CSV'}</button>
          <span className="batch-format">UTF-8 · comma-separated · header required</span>
        </div>
        {error && <div className="form-error batch-feedback" role="alert"><ShieldAlert size={17} /><div><strong>Batch not analyzed</strong><p>{error}</p><span>Correct the file and choose it again, or retry if the local API was unavailable.</span></div></div>}
        <div className="batch-action-row"><button className="button button-primary" onClick={() => void analyze()} disabled={!file || busy || !data.model.available}>{busy ? <LoaderCircle size={16} className="animate-spin" /> : <Layers3 size={16} />}{busyLabel}{!busy && <ArrowRight size={15} />}</button><button className="button button-quiet" onClick={clear} disabled={!file && !result && !error}><X size={14} />{busy ? 'Cancel & clear' : 'Clear batch'}</button></div>
        <div className="batch-live" role="status" aria-live="polite">{busy ? 'Validating the entire CSV, then scoring with the saved pipeline. No partial results; the first model load may take a moment.' : message || (result ? `${number(result.counts.total)} rows analyzed in original order. Results exist only on this page until you export.` : 'No file is sent until you choose Analyze Batch.')}</div>
      </Panel>
      <aside className="batch-guidance">
        <Panel title="The input contract" subtitle="Designed for deliberate, bounded analysis">
          <div className="batch-contract"><span>01</span><div><strong>One required column</strong><p><code>TransactionAmt</code> must be finite and nonnegative on every row.</p></div></div>
          <div className="batch-contract"><span>02</span><div><strong>Only supported features</strong><p>Blank optional fields use fitted preprocessing. Unique headers; no IDs, labels or timestamps.</p></div></div>
          <div className="batch-contract"><span>03</span><div><strong>All rows, or a clear error</strong><p>Invalid encoding, types, duplicate headers or excess rows reject the whole batch.</p></div></div>
          <details className="batch-schema"><summary>View optional columns</summary><p><code>dist1, dist2</code>: blank or finite, nonnegative numbers. Numeric values must fit float32.</p><p>Text categories (up to 128 characters):</p><code>ProductCD, card4, card6, P_emaildomain, R_emaildomain, M1–M9, DeviceType, DeviceInfo, id_30, id_31, id_33</code></details>
        </Panel>
        <div className="batch-policy"><LockKeyhole size={17} /><div><strong>Frozen, not adjustable</strong><p>Validation-selected threshold <b>{number(threshold, 10)}</b></p><span>{result ? 'Threshold returned with this batch.' : 'Saved model and fitted preprocessing only.'}</span></div></div>
      </aside>
    </div>
    {!data.model.available && <Note warning>{data.model.reason ?? 'The matching local model is unavailable. Restore it and refresh to enable analysis.'}</Note>}
    <div className="explain-kpis batch-kpis" aria-label="Batch summary">
      <Kpi label="Total analyzed" value={number(result?.counts.total)} detail={result ? 'All accepted rows · original order' : 'Awaiting a completed analysis'} icon={<Layers3 size={17} />} />
      <Kpi label="Flagged predictions" value={number(result?.counts.flagged)} detail="At or above the frozen threshold" icon={<ShieldAlert size={17} />} accent />
      <Kpi label="Flag rate" value={percent(result ? result.counts.flagged / result.counts.total * 100 : null)} detail="Model flags / analyzed rows · not a fraud rate" icon={<ChartNoAxesCombined size={17} />} />
    </div>
    <Note warning><strong>Predictions, not confirmed fraud.</strong> Scores are uncalibrated model outputs, not fraud probabilities. A flag means score ≥ the frozen threshold; an unflagged row is not a guarantee of safety.</Note>
    <Panel title="How the batch scores" subtitle="02 / DISTRIBUTION · full analyzed batch, independent of table filters" action={result ? <Badge tone="neutral">{number(result.counts.total)} rows</Badge> : undefined}>
      {result ? <><div className="chart-frame batch-chart" role="img" aria-label={`Uncalibrated score distribution for ${result.counts.total} rows in 20 equal-width bins. ${result.counts.flagged} flagged predictions. Frozen threshold ${result.threshold}.`}><ResponsiveContainer width="100%" height="100%"><BarChart data={bins} margin={{ top: 20, right: 15, left: 0, bottom: 10 }}><CartesianGrid stroke={palette.grid} strokeDasharray="3 5" vertical={false} /><XAxis dataKey="mid" type="number" domain={[0, 1]} ticks={[0, .25, .5, .75, 1]} tickFormatter={v => number(v, 2)} tick={{ fill: palette.muted, fontSize: 11 }} axisLine={false} tickLine={false} /><YAxis allowDecimals={false} tick={{ fill: palette.muted, fontSize: 11 }} tickFormatter={v => number(v)} axisLine={false} tickLine={false} width={45} /><Tooltip contentStyle={tooltipStyle} labelFormatter={v => `Score bin ${bins.find(b => Math.abs(b.mid - Number(v)) < .00001)?.label ?? number(Number(v), 3)}`} formatter={(value, name) => [number(Number(value)), name]} cursor={{ fill: '#ffffff04' }} /><Bar dataKey="below" name="Below threshold" fill={palette.blue} stackId="scores" maxBarSize={30} isAnimationActive={false} /><Bar dataKey="flagged" name="Flagged prediction" fill={palette.amber} stackId="scores" maxBarSize={30} radius={[3, 3, 0, 0]} isAnimationActive={false} /><ReferenceLine x={result.threshold} stroke={palette.mint} strokeDasharray="4 4" label={{ value: 'Frozen threshold', fill: palette.mint, fontSize: 11, position: result.threshold > .75 ? 'insideTopLeft' : 'insideTopRight' }} /></BarChart></ResponsiveContainer></div><div className="batch-chart-footer"><span>Uncalibrated score · 0 to 1 · count per bin</span><div className="chart-legend"><span><i style={{ background: palette.blue }} />Below threshold</span><span><i style={{ background: palette.amber }} />Flagged prediction</span></div></div></> : <Empty title={busy ? 'Reading the batch signal…' : 'A distribution, once you analyze.'}>{busy ? 'The chart will appear after the complete response arrives.' : 'No sample scores or placeholder transactions. Your completed batch will populate this chart.'}</Empty>}
    </Panel>
    <Panel title="Every row, in context" subtitle="03 / RESULTS · one-based CSV data-row number, excluding the header" action={<button className="button button-secondary" disabled={!result || !filtered.length} onClick={() => result && exportResults(filtered, result.threshold)}><Download size={14} />Export {result ? `matching (${number(filtered.length)})` : 'results'}</button>}>
      {result ? <><div className="table-toolbar batch-table-toolbar"><label className="search-field"><Search size={15} /><input aria-label="Search batch row or score" placeholder="Search row number or score…" value={query} autoComplete="off" onChange={e => { setQuery(e.target.value); setPage(0) }} /></label><label className="sort-field"><span>Status</span><select aria-label="Filter batch predictions" value={filter} onChange={e => { setFilter(e.target.value); setPage(0) }}><option value="all">All predictions</option><option value="flagged">Flagged predictions</option><option value="below">Below threshold</option></select></label></div>
        <p className="panel-footnote">Select a row to investigate its original inputs and on-demand SHAP contributions.</p>
        <div className="table-scroll batch-table"><table><caption className="sr-only">Batch predictions in original CSV row order. Scores are uncalibrated and flags are not confirmed fraud. Use each row's investigation button for its SHAP explanation.</caption><thead><tr><th scope="col">CSV row</th><th scope="col">Uncalibrated score</th><th scope="col">Prediction status</th></tr></thead><tbody>{visible.map(row => <tr key={row.row_number} className="batch-investigate-row" onClick={() => setInvestigating(row)}><td className="mono"><button className="batch-row-button" aria-label={`Investigate row ${row.row_number}`} aria-haspopup="dialog" onClick={event => { event.stopPropagation(); setInvestigating(row) }}><span className="batch-row-prefix">#</span>{number(row.row_number)} ↗</button></td><td><div className="batch-score-cell"><span className="mono">{number(row.score, 6)}</span><span className="batch-score-track" aria-hidden="true"><i style={{ width: `${row.score * 100}%`, background: row.flagged ? palette.amber : palette.blue }} /></span></div></td><td><Badge tone={row.flagged ? 'amber' : 'neutral'}>{row.flagged ? 'Flagged prediction' : 'Below threshold'}</Badge></td></tr>)}</tbody></table>{!filtered.length && <div className="table-empty"><p>No matching results. The full batch summary is unchanged.</p><button className="text-button" onClick={() => { setQuery(''); setFilter('all'); setPage(0) }}>Reset filters</button></div>}</div>
        <div className="batch-pagination"><span>{filtered.length ? `${number(current * PAGE_SIZE + 1)}–${number(Math.min((current + 1) * PAGE_SIZE, filtered.length))}` : '0'} of {number(filtered.length)} matching rows <span className="text-muted">/ {number(result.counts.total)} total</span></span><div><button className="icon-button" aria-label="Previous results page" disabled={current === 0} onClick={() => setPage(current - 1)}><ChevronLeft size={17} /></button><span>Page {current + 1} of {lastPage + 1}</span><button className="icon-button" aria-label="Next results page" disabled={current === lastPage} onClick={() => setPage(current + 1)}><ChevronRight size={17} /></button></div></div>
        <p className="panel-footnote">Export includes all matching predictions across pages, in original row order—not uploaded transaction fields. Downloading is the only action that saves results to your device.</p>
        <div className="batch-provenance"><Check size={13} /><span>Completed with model run <code>{result.model_run_id}</code></span></div></> : <Empty title={busy ? 'Analysis in progress' : 'Your results belong here.'}>{busy ? 'Validation must pass for every row before scores are shown.' : 'Choose a CSV and analyze it to inspect, filter, and export real model predictions.'}</Empty>}
    </Panel>
    <p className="batch-retention"><LockKeyhole size={12} />File and results are held in memory only. Leaving this page, reloading, replacing the file, or clearing the batch removes them from this page.</p>
    {investigating && file && result && <TransactionDrawer key={investigating.row_number} file={file} row={investigating} batch={result} onClose={() => setInvestigating(null)} />}
  </div>
}
