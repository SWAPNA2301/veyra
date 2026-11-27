import type { CopilotAnswer, CopilotSource, CopilotStatus } from './types'

export const QUESTION_LIMIT = 600
export const HISTORY_LIMIT = 4
export const TURN_LIMIT = 20

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
const text = (value: unknown): value is string => typeof value === 'string'
function source(value: unknown): value is CopilotSource {
  if (!record(value) || !record(value.citation)) return false
  const citation = value.citation
  return Number.isInteger(value.id) && Number(value.id) > 0 && text(value.passage)
    && (value.similarity === null || typeof value.similarity === 'number' && Number.isFinite(value.similarity))
    && ['source', 'section', 'locator', 'chunk_id'].every(key => text(citation[key]))
}

async function localRequest(path: string, signal: AbortSignal, body?: object): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(`/api/copilot/${path}`, {
      method: body ? 'POST' : 'GET', signal, cache: 'no-store',
      ...(body ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}),
    })
  } catch {
    throw new Error('Could not reach the local Copilot API. Check the local backend, then retry.')
  }
  const value: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    // The API provides sanitized detail; never render raw response bodies or exception stacks.
    const detail = record(value) && text(value.detail)
      ? value.detail.replace(/[\u0000-\u001f\u007f]/g, ' ').trim().slice(0, 600) : ''
    throw new Error(detail || 'The local request could not be completed. Recheck status and retry.')
  }
  return value
}

export async function getCopilotStatus(signal: AbortSignal): Promise<CopilotStatus> {
  const value = await localRequest('status', signal)
  if (!record(value) || !['ready', 'setup_required'].includes(String(value.status))
    || !text(value.model) || value.provider !== 'Ollama (local)'
    || typeof value.index_ready !== 'boolean' || typeof value.model_ready !== 'boolean' || !text(value.detail)) {
    throw new Error('The local API returned an unreadable Copilot status. Recheck status to try again.')
  }
  return value as unknown as CopilotStatus
}

export async function askCopilot(message: string, history: string[], signal: AbortSignal): Promise<CopilotAnswer> {
  if (!message.trim() || message.length > QUESTION_LIMIT || history.length > HISTORY_LIMIT
    || history.some(question => question.length > QUESTION_LIMIT)) throw new Error('Keep each question within 600 characters.')
  const value = await localRequest('chat', signal, { message, history })
  if (!record(value) || !['answered', 'insufficient_evidence', 'evidence_only', 'welcome'].includes(String(value.status))
    || !text(value.answer) || !text(value.notice) || !(value.model === null || text(value.model))
    || !Array.isArray(value.sources) || !value.sources.every(source)) {
    throw new Error('The local API returned an unreadable answer. Retry this question to try again.')
  }
  return value as unknown as CopilotAnswer
}
