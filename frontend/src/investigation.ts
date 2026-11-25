import { request } from './api'
import type { BatchPrediction, BatchRow, TransactionExplanation } from './types'

// Only extract a record from a CSV that the batch endpoint has already accepted.
// Keep CSV quoting, escaped quotes, multiline fields and original cell text intact.
// This is not a substitute for the server's schema/type validation.
export function csvRow(text: string, target: number): Record<string, string> {
  if (!Number.isInteger(target) || target < 1) throw new Error('Invalid CSV row reference.')
  text = text.replace(/^\uFEFF/, '')
  let header: string[] | null = null
  let cells: string[] = [], cell = '', quoted = false, record = 0
  for (let i = 0; i <= text.length; i++) {
    const char = text[i]
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { cell += '"'; i++ }
      else if (char === '"') quoted = false
      else if (char === undefined) break
      else cell += char
    } else if (char === '"' && cell.length === 0) quoted = true
    else if (char === ',') { cells.push(cell); cell = '' }
    else if (char === '\r' || char === '\n' || char === undefined) {
      cells.push(cell); cell = ''
      if (!header) header = cells.map(name => name.trim())
      else if (++record === target) {
        if (cells.length !== header.length || new Set(header).size !== header.length) break
        return Object.fromEntries(header.map((name, index) => [name, cells[index]]))
      }
      cells = []
      if (char === '\r' && text[i + 1] === '\n') i++
    } else cell += char
  }
  throw new Error('The selected CSV row could not be read. Choose the file and analyze the batch again.')
}

export async function explainRow(values: Record<string, string>, row: BatchRow, batch: BatchPrediction, signal: AbortSignal) {
  const result = await request<TransactionExplanation>('/transaction-explanation', {
    method: 'POST', signal,
    body: JSON.stringify({ values, score: row.score, threshold: batch.threshold, model_run_id: batch.model_run_id }),
  })
  const parts = result?.contributions
  if (!result || result.model_run_id !== batch.model_run_id || result.threshold !== batch.threshold
      || Math.abs(result.score - row.score) > 1e-12 || result.flagged !== row.flagged
      || result.score_type !== 'uncalibrated' || result.units !== 'raw_log_odds'
      || ![result.score, result.base_value, result.raw_margin, result.additivity_error].every(Number.isFinite)
      || !Array.isArray(parts) || !parts.length || parts.length > 22
      || parts.some(p => !p || typeof p.feature !== 'string' || !Number.isFinite(p.value)
        || !Number.isInteger(p.encoded_columns) || p.encoded_columns < 1)
      || new Set(parts.map(p => p.feature)).size !== parts.length
      || Math.abs(result.base_value + parts.reduce((sum, p) => sum + p.value, 0) - result.raw_margin) > 1e-6
      || Math.abs(1 / (1 + Math.exp(-result.raw_margin)) - result.score) > 1e-7) {
    throw new Error('The explanation did not match this batch score. No contributions are shown; analyze the batch again.')
  }
  return result
}
