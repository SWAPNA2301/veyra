import { useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { ArrowRight, Check, ChevronDown, Fingerprint, LoaderCircle, LockKeyhole, ScanLine, ShieldAlert } from 'lucide-react'
import type { DashboardData, Prediction } from '../types'
import { request } from '../api'
import { Badge, Note, number, Panel } from '../components/ui'

type FieldSpec = { name: string; label: string; options?: string[]; numeric?: boolean; placeholder?: string }
const deviceFields: FieldSpec[] = [
  { name: 'DeviceType', label: 'Device type', options: ['desktop', 'mobile'] }, { name: 'DeviceInfo', label: 'Device information' },
  { name: 'id_30', label: 'Operating system' }, { name: 'id_31', label: 'Browser' }, { name: 'id_33', label: 'Screen resolution' },
]
const signalFields: FieldSpec[] = [{ name: 'dist1', label: 'Distance signal 1', numeric: true }, { name: 'dist2', label: 'Distance signal 2', numeric: true }, ...Array.from({ length: 9 }, (_, i) => ({ name: `M${i + 1}`, label: `Match signal M${i + 1}`, options: i === 3 ? ['M0', 'M1', 'M2'] : ['T', 'F'] }))]

function RiskGauge({ score, threshold }: { score: number; threshold: number }) {
  const x = 150 - 112 * Math.cos(threshold * Math.PI)
  const y = 142 - 112 * Math.sin(threshold * Math.PI)
  const color = score >= threshold ? '#e5ba75' : '#62e6bf'
  return <div className="risk-gauge"><svg viewBox="0 0 300 190" role="img" aria-label={`Uncalibrated model score ${score.toFixed(4)}. Frozen threshold ${threshold}.`}><path d="M38 142 A112 112 0 0 1 262 142" fill="none" stroke="#253040" strokeWidth="15" strokeLinecap="round" /><path d="M38 142 A112 112 0 0 1 262 142" pathLength="1" fill="none" stroke={color} strokeWidth="15" strokeLinecap="round" strokeDasharray={`${Math.max(score, .000001)} 1`} /><circle cx={x} cy={y} r="6" fill="#eaf2ff" stroke="#121b26" strokeWidth="3" /><text x="150" y="122" textAnchor="middle" className="gauge-value">{number(score, 4)}</text><text x="150" y="147" textAnchor="middle" className="gauge-label">UNCALIBRATED RISK SCORE</text><text x="38" y="178" textAnchor="middle" className="gauge-tick">0</text><text x="262" y="178" textAnchor="middle" className="gauge-tick">1</text></svg></div>
}

export default function Predictor({ data }: { data: DashboardData }) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<Prediction | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dirty, setDirty] = useState(false)
  const sequence = useRef(0)
  const model = data.model
  function change(name: string, value: string) { setValues(current => ({ ...current, [name]: value })); setDirty(true) }
  function field(spec: FieldSpec) { return <label className="form-field" key={spec.name} htmlFor={spec.name}><span>{spec.label}{spec.name === 'TransactionAmt' && <i>Required</i>}</span>{spec.options ? <select id={spec.name} value={values[spec.name] ?? ''} onChange={e => change(spec.name, e.target.value)} disabled={busy}><option value="">Unknown / not provided</option>{spec.options.map(option => <option key={option} value={option}>{option}</option>)}</select> : <input id={spec.name} type={spec.numeric ? 'number' : 'text'} min={spec.numeric ? 0 : undefined} step={spec.numeric ? 'any' : undefined} maxLength={spec.numeric ? undefined : 128} required={spec.name === 'TransactionAmt'} value={values[spec.name] ?? ''} onChange={e => change(spec.name, e.target.value)} disabled={busy} placeholder={spec.placeholder ?? 'Unknown / not provided'} autoComplete="off" />}</label> }
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (busy || !model.available) return
    const attempt = ++sequence.current
    setBusy(true); setResult(null); setError('')
    const body: Record<string, number | string | null> = {}
    for (const name of model.features) {
      const value = values[name]?.trim()
      body[name] = !value ? null : ['TransactionAmt', 'dist1', 'dist2'].includes(name) ? Number(value) : value
    }
    try {
      const scored = await request<Prediction>('/predict', { method: 'POST', body: JSON.stringify(body) })
      if (sequence.current === attempt) { setResult(scored); setDirty(false) }
    } catch (e) { if (sequence.current === attempt) setError(e instanceof Error ? e.message : 'Scoring failed.') }
    finally { if (sequence.current === attempt) setBusy(false) }
  }
  const shown = result && model.available && result.run_id === model.run_id && result.model_sha === model.sha256 && result.threshold === model.threshold ? result : null
  return <div className="page-content">
    <div className="predictor-banner"><span className="predictor-banner-icon"><LockKeyhole size={18} /></span><div><strong>Private by design</strong><p>One manual scenario. Existing fitted preprocessing. Inputs are never saved to disk.</p></div><Badge tone={model.available ? 'mint' : 'amber'}>{model.available ? 'Local model found' : 'Scoring unavailable'}</Badge></div>
    {!model.available && <Note warning>{model.reason ?? 'Restore the matching saved LightGBM artifact to enable scoring. Report pages remain available.'}</Note>}
    <div className="predictor-layout"><form onSubmit={event => void submit(event)} className="panel prediction-form"><div className="panel-heading"><div><span className="eyebrow">01 / INPUT</span><h2>Transaction details</h2><p>Enter what you know. Leave unknown attributes blank.</p></div><ScanLine size={19} className="text-muted" /></div><div className="form-grid">{field({ name: 'TransactionAmt', label: 'Transaction amount', numeric: true, placeholder: 'Enter amount in dataset units' })}{field({ name: 'ProductCD', label: 'Product code', options: data.overview.products.length ? data.overview.products.map(p => p.product).sort() : undefined })}{field({ name: 'card4', label: 'Card network', options: ['visa', 'mastercard', 'american express', 'discover'] })}{field({ name: 'card6', label: 'Card type', options: ['credit', 'debit', 'charge card', 'debit or credit'] })}{field({ name: 'P_emaildomain', label: 'Purchaser email domain', placeholder: 'Domain only, e.g. gmail.com' })}{field({ name: 'R_emaildomain', label: 'Recipient email domain', placeholder: 'Domain only, not an email address' })}</div><p className="form-hint">Amounts use original recorded units; no currency conversion is applied.</p><details className="form-section"><summary><span><Fingerprint size={16} />Device context <small>Optional</small></span><ChevronDown size={16} /></summary><div className="form-grid">{deviceFields.map(field)}</div></details><details className="form-section"><summary><span><ActivityIcon />Distance & match signals <small>Optional</small></span><ChevronDown size={16} /></summary><p className="form-hint">Use original dataset codes and distance units. Do not guess unknown attributes.</p><div className="form-grid">{signalFields.map(field)}</div></details>{error && <div role="alert" className="form-error">{error}</div>}<div className="form-actions"><button className="button button-primary" type="submit" disabled={!model.available || busy}>{busy ? <LoaderCircle size={16} className="animate-spin" /> : <ScanLine size={16} />}{busy ? 'Analyzing scenario…' : 'Analyze transaction'}{!busy && <ArrowRight size={16} />}</button><button type="button" className="button button-quiet" disabled={busy} onClick={() => { setValues({}); setResult(null); setError(''); setDirty(false) }}>Reset fields</button></div><span className="form-footnote"><LockKeyhole size={12} />Only this submitted row is scored. No training or holdout evaluation.</span></form>
      <div className="assessment-column"><Panel title="Risk assessment" subtitle="02 / OUTPUT · last submitted scenario" action={<Badge tone="neutral">LightGBM</Badge>}>
        {busy ? <div className="prediction-empty"><LoaderCircle size={32} className="animate-spin text-mint" /><h3>Reading the signal</h3><p>The saved model is loading and scoring your manual input.</p></div> : shown ? <><RiskGauge score={shown.score} threshold={shown.threshold} /><div className={`decision ${shown.above ? 'decision-review' : ''}`}>{shown.above ? <ShieldAlert size={19} /> : <Check size={19} />}<div><strong>{shown.above ? 'Above threshold · review suggested' : 'Below the frozen threshold'}</strong><p>Not a verified fraud decision</p></div></div>{dirty && <Note warning>Inputs changed. This result belongs to the last submission; analyze again to update it.</Note>}<dl className="result-details"><div><dt>Frozen threshold</dt><dd>{number(shown.threshold, 10)}</dd></div><div><dt>Provided fields</dt><dd>{shown.provided} / {shown.expected}</dd></div><div><dt>Transaction amount</dt><dd>{number(shown.amount, 3)}</dd></div><div><dt>Submitted</dt><dd>{shown.submitted_at}</dd></div></dl><button className="button button-secondary w-full" onClick={() => setResult(null)}>Clear result</button></> : <div className="prediction-empty"><div className="scan-orbit"><ScanLine size={31} /></div><h3>Ready when you are.</h3><p>Enter an amount and any known context, then analyze to see the model’s signal.</p><span>No score until you submit</span></div>}
      </Panel><div className="score-disclaimer"><LockKeyhole size={18} /><h3>A score, not certainty.</h3><p>This class-weighted model is not probability-calibrated. The 0–1 score is <strong>not a fraud probability</strong>. Unknown fields use training-fitted rules; unseen categories use the existing encoder.</p><div>Threshold policy<span>Max precision at recall constraint</span></div><strong className="frozen-value">{number(model.threshold, 10)}</strong><p className="text-xs">Frozen on validation. Never adjusted from these inputs. No automatic transaction action is taken.</p></div></div>
    </div>
  </div>
}

function ActivityIcon() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M2 12h5l3-8 4 16 3-8h5" /></svg> }
