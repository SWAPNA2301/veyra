import { useEffect, useRef, useState } from 'react'
import { ArrowUpRight, BookOpen, Bot, Check, Cpu, Database, LoaderCircle, LockKeyhole, RefreshCw, Send, Square, Trash2 } from 'lucide-react'
import { Badge, Note, Panel } from '../components/ui'
import { askCopilot, getCopilotStatus, HISTORY_LIMIT, QUESTION_LIMIT, TURN_LIMIT } from '../copilot'
import type { CopilotAnswer, CopilotStatus } from '../types'
import './copilot.css'

interface Turn {
  id: number
  question: string
  history: string[]
  state: 'pending' | 'complete' | 'error' | 'cancelled'
  response?: CopilotAnswer
  error?: string
}
const suggestions = [
  { label: 'Understand performance', question: 'What do the saved reports say about precision and recall at the frozen threshold?' },
  { label: 'Explore the signals', question: 'Which features stand out in the saved SHAP report, and what are its limitations?' },
  { label: 'Understand the evaluation', question: 'How were the chronological train, validation and test splits used in the saved evaluation?' },
]
const answerLabels: Record<CopilotAnswer['status'], string> = {
  answered: 'Report-grounded answer', insufficient_evidence: 'Insufficient evidence',
  evidence_only: 'Evidence only', welcome: 'Getting started',
}
const cancelledMessage = 'Stopped waiting for this answer. The backend may still finish its bounded local inference.'

export default function Copilot() {
  const [status, setStatus] = useState<CopilotStatus | null>(null)
  const [statusError, setStatusError] = useState('')
  const [checking, setChecking] = useState(true)
  const [checkVersion, setCheckVersion] = useState(0)
  const [draft, setDraft] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const [busy, setBusy] = useState(false)
  const [announcement, setAnnouncement] = useState('')
  const nextId = useRef(0)
  const pending = useRef<{ controller: AbortController; id: number } | null>(null)
  const textarea = useRef<HTMLTextAreaElement>(null)
  const composing = useRef(false)
  const latestTurn = useRef<HTMLElement>(null)

  useEffect(() => {
    const controller = new AbortController()
    setChecking(true)
    setStatusError('')
    const timeout = window.setTimeout(() => {
      controller.abort()
      setStatus(null)
      setStatusError('The local status check timed out. Check the backend and recheck status.')
      setChecking(false)
    }, 15000)
    getCopilotStatus(controller.signal).then(value => {
      if (!controller.signal.aborted) setStatus(value)
    }).catch(error => {
      if (!controller.signal.aborted) {
        setStatus(null)
        setStatusError(error instanceof Error ? error.message : 'Could not check local Copilot status.')
      }
    }).finally(() => {
      window.clearTimeout(timeout)
      if (!controller.signal.aborted) setChecking(false)
    })
    return () => { window.clearTimeout(timeout); controller.abort() }
  }, [checkVersion])

  useEffect(() => () => { pending.current?.controller.abort(); pending.current = null }, [])
  useEffect(() => { latestTurn.current?.scrollIntoView({ block: 'nearest' }) }, [turns.length])

  const canAsk = !checking && Boolean(status?.index_ready)
  const full = turns.length >= TURN_LIMIT

  async function run(turn: Turn) {
    // Ref guard also prevents rapid clicks or repeated Enter before React renders.
    if (pending.current || !canAsk) return
    const controller = new AbortController()
    pending.current = { controller, id: turn.id }
    setBusy(true)
    setAnnouncement('Reading local report evidence. The first request may take several minutes while retrieval and the local model warm up.')
    setTurns(current => current.some(item => item.id === turn.id)
      ? current.map(item => item.id === turn.id ? { ...turn, state: 'pending', error: undefined } : item)
      : [...current, turn])
    let timedOut = false
    const timeout = window.setTimeout(() => { timedOut = true; controller.abort() }, 360000)
    try {
      const response = await askCopilot(turn.question, turn.history, controller.signal)
      if (pending.current?.controller !== controller || controller.signal.aborted) return
      setTurns(current => current.map(item => item.id === turn.id ? { ...item, state: 'complete', response } : item))
      setAnnouncement(`${answerLabels[response.status]} received. ${response.sources.length} source passages available below the answer.`)
    } catch (error) {
      if (pending.current?.controller !== controller) return
      const detail = timedOut
        ? 'This request took too long. The backend may still finish its bounded inference. You can retry this question.'
        : error instanceof Error ? error.message : 'The question could not be answered. Please retry.'
      setTurns(current => current.map(item => item.id === turn.id ? { ...item, state: 'error', error: detail } : item))
      setAnnouncement('The request failed. Retry is available on the question.')
    } finally {
      window.clearTimeout(timeout)
      if (pending.current?.controller === controller) { pending.current = null; setBusy(false) }
    }
  }

  function send() {
    const question = draft.trim()
    if (!question || draft.length > QUESTION_LIMIT || pending.current || !canAsk || full || composing.current) return
    const history = turns.filter(turn => turn.state === 'complete').slice(-HISTORY_LIMIT).map(turn => turn.question)
    void run({ id: ++nextId.current, question, history, state: 'pending' })
    setDraft('')
  }
  function cancel() {
    const current = pending.current
    if (!current) return
    pending.current = null
    current.controller.abort()
    setBusy(false)
    setTurns(items => items.map(turn => turn.id === current.id ? { ...turn, state: 'cancelled' } : turn))
    setAnnouncement(cancelledMessage)
    textarea.current?.focus()
  }
  function clear() {
    const wasPending = Boolean(pending.current)
    pending.current?.controller.abort()
    pending.current = null
    setBusy(false)
    setTurns([])
    setDraft('')
    setAnnouncement(wasPending ? `Chat cleared. ${cancelledMessage}` : 'Chat cleared. All page conversation memory has been discarded.')
    textarea.current?.focus()
  }

  return <div className="page-content copilot-page">
    <div className="copilot-status-strip">
      <span className="copilot-symbol"><Bot size={23} aria-hidden="true" /></span>
      <div className="copilot-status-copy"><strong>Your reports. A local conversation.</strong><p>Ollama (local) · Report evidence only · No raw transaction access</p></div>
      <div className="copilot-badges">
        <Badge tone={checking ? 'neutral' : status?.model_ready ? 'mint' : 'amber'}><Cpu size={12} aria-hidden="true" />{checking ? 'Checking model' : status ? `${status.model} · ${status.model_ready ? 'ready' : 'unavailable'}` : 'Model unknown'}</Badge>
        <Badge tone={checking ? 'neutral' : status?.index_ready ? 'mint' : 'amber'}><Database size={12} aria-hidden="true" />{checking ? 'Checking index' : status ? status.index_ready ? 'Index ready' : 'Index missing' : 'Index unknown'}</Badge>
      </div>
      <button type="button" className="button button-secondary" disabled={checking} onClick={() => setCheckVersion(value => value + 1)}><RefreshCw size={14} className={checking ? 'animate-spin' : ''} aria-hidden="true" />{checking ? 'Checking' : 'Recheck status'}</button>
    </div>

    {statusError && <div role="alert"><Note warning><strong>Copilot status unavailable.</strong> {statusError} Use Recheck status above to try again.</Note></div>}
    {!checking && status?.status === 'setup_required' && <Note warning><strong>Local setup needs attention.</strong><p className="copilot-plain">{status.detail}</p>
      {!status.model_ready && <p>Start Ollama locally and make the configured model ({status.model}) available, then recheck status. {status.index_ready && 'You can still ask for report evidence while the model is unavailable.'}</p>}
      {!status.index_ready && <p>Build or restore the local report index through the backend setup workflow, then recheck status.</p>}
    </Note>}

    <div className="copilot-layout">
      <Panel title="Veyra Copilot" subtitle="ASK / READ / VERIFY" className="copilot-chat" action={<button type="button" className="button button-quiet" onClick={clear} disabled={!turns.length && !draft}><Trash2 size={14} aria-hidden="true" />Clear chat</button>}>
        {!turns.length ? <div className="copilot-empty">
          <div className="copilot-empty-mark"><BookOpen size={29} aria-hidden="true" /></div>
          <span className="copilot-eyebrow">START WITH A QUESTION</span>
          <h2>Make sense of the evidence.</h2>
          <p>Explore saved metrics, SHAP explanations and chronological evaluation splits. Open the cited passages to check what each answer is based on.</p>
          <div className="copilot-suggestions">{suggestions.map(suggestion => <button type="button" key={suggestion.label} onClick={() => { setDraft(suggestion.question); textarea.current?.focus() }}><span><strong>{suggestion.label}</strong><span>{suggestion.question}</span></span><ArrowUpRight size={16} aria-hidden="true" /></button>)}</div>
          <p className="copilot-empty-hint">Suggestions fill the question box. Nothing is sent until you choose Send.</p>
          {status?.model === 'llama3.2:1b' && <p className="copilot-empty-hint">Known metrics come directly from verified aggregate reports. Other answers use concise, relevant report sentences with citations; unavailable metrics are never inferred.</p>}
        </div> : <div className="copilot-transcript" role="region" aria-label="Copilot conversation" tabIndex={0}>
          {turns.map((turn, index) => <article className="copilot-turn" key={turn.id} ref={index === turns.length - 1 ? latestTurn : undefined} aria-label={`Question ${index + 1}`}>
            <div className="copilot-question"><span className="copilot-speaker">YOU <span>{String(index + 1).padStart(2, '0')}</span></span><p className="copilot-plain">{turn.question}</p></div>
            <div className="copilot-answer"><span className="copilot-speaker"><Bot size={14} aria-hidden="true" />VEYRA COPILOT</span>
              {turn.state === 'pending' && <div className="copilot-pending"><LoaderCircle size={17} className="animate-spin" aria-hidden="true" /><div><strong>Reading the local evidence…</strong><p>The first request may take several minutes while retrieval and the local model warm up. You can cancel waiting below.</p></div></div>}
              {turn.state === 'complete' && turn.response && <>
                <div className="copilot-answer-meta"><Badge tone={turn.response.status === 'insufficient_evidence' ? 'amber' : 'neutral'}>{answerLabels[turn.response.status]}</Badge><span>{turn.response.model ?? 'No model generation'}</span></div>
                <p className="copilot-plain copilot-answer-text">{turn.response.answer}</p>
                {turn.response.notice && <p className="copilot-notice copilot-plain">{turn.response.notice}</p>}
                {!!turn.response.sources.length && <div className="copilot-sources"><h3><BookOpen size={14} aria-hidden="true" />Report evidence <span>{turn.response.sources.length} passages</span></h3>
                  {turn.response.sources.map((source, sourceIndex) => <details className="copilot-source" key={`${source.id}-${sourceIndex}`}><summary><span className="copilot-citation">[{source.id}]</span><span><strong>{source.citation.source}</strong><span>{source.citation.section} · {source.citation.locator}</span></span><span className="copilot-source-open">Read passage</span></summary><div className="copilot-source-body"><p className="copilot-plain">{source.passage}</p><dl><div><dt>Evidence</dt><dd>{source.similarity === null ? 'Verified aggregate lookup · not vector scored' : `${source.similarity.toFixed(4)} · retrieval similarity, not confidence`}</dd></div><div><dt>Source ID</dt><dd>{source.citation.chunk_id}</dd></div></dl></div></details>)}
                </div>}
              </>}
              {(turn.state === 'error' || turn.state === 'cancelled') && <div className="copilot-failure"><p className="copilot-plain" role={turn.state === 'error' ? 'alert' : undefined}>{turn.state === 'error' ? turn.error : cancelledMessage}</p>{index === turns.length - 1 && <button type="button" className="button button-secondary" disabled={busy || !canAsk} onClick={() => void run(turn)}><RefreshCw size={14} aria-hidden="true" />Retry this question</button>}</div>}
            </div>
          </article>)}
        </div>}

        <form className="copilot-composer" onSubmit={event => { event.preventDefault(); send() }}>
          <label htmlFor="copilot-question">Ask about your saved reports</label>
          <textarea ref={textarea} id="copilot-question" rows={3} maxLength={QUESTION_LIMIT} value={draft} aria-describedby="copilot-input-help copilot-input-limit" placeholder="What would you like to understand?" disabled={full}
            onChange={event => setDraft(event.target.value.slice(0, QUESTION_LIMIT))}
            onCompositionStart={() => { composing.current = true }} onCompositionEnd={() => { composing.current = false }}
            onKeyDown={event => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && !composing.current && event.keyCode !== 229) { event.preventDefault(); send() }
            }} />
          <div className="copilot-composer-footer"><div><p id="copilot-input-help">Enter to send · Shift + Enter for a new line</p><p id="copilot-input-limit">{draft.length}/{QUESTION_LIMIT} characters · {turns.length}/{TURN_LIMIT} turns</p></div>
            {busy ? <button type="button" className="button button-secondary" onClick={cancel}><Square size={13} aria-hidden="true" />Cancel waiting</button> : <button type="submit" className="button button-primary" disabled={!draft.trim() || !canAsk || full}><Send size={14} aria-hidden="true" />Send question</button>}
          </div>
          <p className="copilot-composer-note">{full ? 'This chat has reached 20 turns. Clear chat to start a fresh conversation.' : checking ? 'Checking local model and index availability…' : !canAsk ? 'Questions become available when the local report index is ready. Recheck status above.' : 'Only your last 4 successful questions are sent as context. Assistant text is never sent back.'}</p>
          {busy && <p className="copilot-composer-note">Cancel stops waiting here; the backend may still finish its bounded inference.</p>}
        </form>
      </Panel>

      <aside className="copilot-guidance" aria-label="How to use Copilot">
        <Panel title="A reading companion" subtitle="EVIDENCE BEFORE CONCLUSIONS">
          <div className="copilot-guide-item"><BookOpen size={17} aria-hidden="true" /><div><h3>Follow the citations</h3><p>Expand [n] to read the actual report filename, section, locator and supporting passage.</p></div></div>
          <div className="copilot-guide-item"><Check size={17} aria-hidden="true" /><div><h3>Check the interpretation</h3><p>AI can err. Citations are report evidence, not proof of causation or fraud. Confirm conclusions against the source.</p></div></div>
          <div className="copilot-guide-item"><Cpu size={17} aria-hidden="true" /><div><h3>Keep scores in context</h3><p>Model scores are uncalibrated, not fraud probabilities. Retrieval similarity measures passage matching, not confidence or truth.</p></div></div>
        </Panel>
        <div className="copilot-privacy"><LockKeyhole size={18} aria-hidden="true" /><h3>Only here, only for now.</h3><p>This page holds up to 20 turns in component memory. Clear chat or leave Copilot to discard the conversation. No browser storage is used.</p><p>Questions go to the local API and Ollama. Copilot has no access to raw transactions or your other page inputs.</p></div>
      </aside>
    </div>
    <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">{announcement}</div>
  </div>
}
