import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ArrowDownRight, ArrowUpRight, Check, LoaderCircle, ScanSearch, X } from 'lucide-react'
import type { BatchPrediction, BatchRow, TransactionExplanation } from '../types'
import { csvRow, explainRow } from '../investigation'
import { Badge, Note, number, Panel } from './ui'
import './investigation.css'

function Waterfall({ explanation }: { explanation: TransactionExplanation }) {
  const [all, setAll] = useState(false)
  const ordered = [...explanation.contributions].sort((a, b) => Math.abs(b.value) - Math.abs(a.value) || a.feature.localeCompare(b.feature))
  const shown = all ? ordered : ordered.slice(0, 10)
  const remainder = ordered.slice(shown.length)
  const parts = remainder.length ? [...shown, { feature: `${remainder.length} other features`, value: remainder.reduce((sum, item) => sum + item.value, 0) }] : shown
  let running = explanation.base_value
  const steps = parts.map(part => { const start = running; running += part.value; return { ...part, start, end: running } })
  const rows = [{ feature: 'Base value', start: 0, end: explanation.base_value, value: explanation.base_value, total: true },
    ...steps.map(step => ({ ...step, total: false })),
    { feature: 'Model output', start: 0, end: explanation.raw_margin, value: explanation.raw_margin, total: true }]
  const points = rows.flatMap(row => [row.start, row.end])
  const lo = Math.min(...points), hi = Math.max(...points), span = Math.max(hi - lo, 1)
  const x = (value: number) => 165 + (value - lo + span * .08) / (span * 1.16) * 355
  const height = rows.length * 32 + 65
  return <>
    <div className="investigation-legend"><span><ArrowUpRight size={14} />Increases output</span><span><ArrowDownRight size={14} />Decreases output</span></div>
    <div className="waterfall-scroll"><svg className="shap-waterfall" viewBox={`0 0 620 ${height}`} role="img" aria-label="SHAP waterfall: grouped feature contributions in raw log-odds from base value to model output">
      <title>Grouped TreeSHAP waterfall — model contributions, not causal effects</title>
      <desc>Base value {explanation.base_value}. Model output {explanation.raw_margin}. Each signed contribution advances the running total. Full numeric values are available below.</desc>
      <line x1={x(0)} x2={x(0)} y1={5} y2={rows.length * 32 + 5} stroke="#53667c" strokeDasharray="3 4" />
      {rows.map((row, index) => <g key={row.feature}>
        <text x={151} y={index * 32 + 22} textAnchor="end" fill={row.total ? '#e2edf7' : '#adc2d8'} fontSize={11}>{row.feature}</text>
        {index > 0 && <line x1={x(rows[index - 1].end)} x2={x(rows[index - 1].end)} y1={(index - 1) * 32 + 28} y2={index * 32 + 8} stroke="#53667c" strokeDasharray="2 3" />}
        <rect x={Math.min(x(row.start), x(row.end))} y={index * 32 + 8} width={Math.max(1, Math.abs(x(row.end) - x(row.start)))} height={20} rx={3} fill={row.total ? '#7ea8ee' : row.value >= 0 ? '#e9b967' : '#7bdfbd'}>
          <title>{row.feature}: {row.total ? '' : row.value >= 0 ? '+' : ''}{row.value} raw log-odds</title>
        </rect>
        <text x={545} y={index * 32 + 22} fill={row.total ? '#b8d1f6' : row.value >= 0 ? '#e9c58b' : '#90e5c8'} fontSize={11}>{!row.total && row.value >= 0 ? '+' : ''}{number(row.value, 4)}</text>
      </g>)}
      {[0, .25, .5, .75, 1].map(fraction => { const value = lo + fraction * (hi - lo); return <text key={fraction} x={x(value)} y={rows.length * 32 + 25} fill="#9caec5" textAnchor="middle" fontSize={10}>{number(value, 2)}</text> })}
      <text x={340} y={height - 5} textAnchor="middle" fill="#a9bdd3" fontSize={11}>Raw model output / log-odds — not probability points</text>
    </svg></div>
    {ordered.length > 10 && <button className="text-button" onClick={() => setAll(!all)}>{all ? 'Show top 10 + grouped remainder' : `Show all ${ordered.length} original features`}</button>}
    <details className="contribution-details"><summary>Full contribution values & method</summary><p>{explanation.method}. {explanation.grouping}</p><table><thead><tr><th>Original feature</th><th>Contribution</th><th>Encoded columns</th></tr></thead><tbody>{ordered.map(part => <tr key={part.feature}><td>{part.feature}</td><td className="mono">{part.value.toPrecision(10)}</td><td>{part.encoded_columns}</td></tr>)}</tbody></table></details>
  </>
}

export default function TransactionDrawer({ file, row, batch, onClose }: { file: File; row: BatchRow; batch: BatchPrediction; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [values, setValues] = useState<Record<string, string> | null>(null)
  const [explanation, setExplanation] = useState<TransactionExplanation | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    const element = dialog.current!, focus = document.activeElement as HTMLElement | null
    const overflow = document.body.style.overflow
    element.showModal(); document.body.style.overflow = 'hidden'
    return () => { element.close(); document.body.style.overflow = overflow; focus?.focus() }
  }, [])
  useEffect(() => {
    const controller = new AbortController()
    setValues(null); setExplanation(null); setError('')
    async function load() {
      try {
        const original = csvRow(await file.text(), row.row_number)
        if (controller.signal.aborted) return
        setValues(original)
        const response = await explainRow(original, row, batch, controller.signal)
        if (!controller.signal.aborted) setExplanation(response)
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof TypeError
          ? 'Could not reach the local explanation API. Check that the updated backend is running, then retry.'
          : e instanceof Error ? e.message : 'The explanation is unavailable. Please retry.')
      }
    }
    void load()
    return () => controller.abort()
  }, [file, row, batch, retry])
  return createPortal(<dialog ref={dialog} className="investigation-drawer" aria-labelledby="investigation-title" onCancel={e => { e.preventDefault(); onClose() }} onClick={e => { if (e.target === e.currentTarget) { const rect = e.currentTarget.getBoundingClientRect(); if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) onClose() } }}>
    <header className="investigation-header"><div><span className="heading-eyebrow"><ScanSearch size={14} /> TRANSACTION INVESTIGATION</span><h2 id="investigation-title">Inside row #{number(row.row_number)}</h2><p>Original CSV data-row number · on-demand explanation</p></div><button className="icon-button" aria-label="Close investigation" onClick={onClose} autoFocus><X size={20} /></button></header>
    <div className="investigation-body">
      <div className="investigation-score"><div><span>Uncalibrated model score</span><strong>{number(row.score, 6)}</strong></div><div><Badge tone={row.flagged ? 'amber' : 'neutral'}>{row.flagged ? 'Flagged prediction' : 'Below threshold'}</Badge><p>Frozen threshold <b>{number(batch.threshold, 10)}</b></p><small>From the scored batch · not a confirmed fraud decision</small></div></div>
      <Note warning><strong>Model contributions, not causal explanations.</strong> SHAP describes this model’s output, not why an event happened. Scores are uncalibrated, not fraud probabilities; below threshold does not mean safe.</Note>
      <Panel title="What moved this score" subtitle="GROUPED TREESHAP / signed contributions in raw log-odds">
        {error ? <div role="alert" className="investigation-error"><h3>Explanation unavailable</h3><p>{error}</p><button className="button button-secondary" onClick={() => setRetry(value => value + 1)}>Retry explanation</button></div>
          : !explanation ? <div className="investigation-loading" role="status"><LoaderCircle size={22} className="animate-spin" /><p>Explaining this row with the saved model…</p><span>Fitted preprocessing only. No training or dataset lookup.</span></div>
          : <><Waterfall explanation={explanation} /><div className="investigation-check"><Check size={14} /><span>Reconstruction verified · error {explanation.additivity_error.toExponential(2)}<br />sigmoid({number(explanation.raw_margin, 6)}) = {number(explanation.score, 6)} uncalibrated score</span></div></>}
      </Panel>
      <Panel title="Original submitted values" subtitle="Only this uploaded row · never a record retrieved from the dataset">
        <p className="panel-footnote">Cell text is preserved below. The server uses the same trimming, numeric conversion and fitted missing-value handling as batch scoring. Omitted fields are missing inputs, not observed zeros.</p>
        {values ? <dl className="investigation-inputs">{Object.entries(values).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value.trim() ? value : <span className="text-muted">Blank / missing</span>}</dd></div>)}</dl> : <p className="panel-footnote">Reading the selected row from the in-memory file…</p>}
      </Panel>
      <p className="investigation-retention">Only the selected row is sent for explanation. No upload or explanation is saved automatically. Closing discards this drawer’s values and explanation.</p>
      <p className="investigation-run">Model run <code>{batch.model_run_id}</code></p>
    </div>
  </dialog>, document.body)
}
