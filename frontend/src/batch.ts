import { request } from './api'
import type { BatchPrediction, BatchRow } from './types'

export const MAX_BATCH_BYTES = 5_000_000
export const MAX_BATCH_ROWS = 5000

export function fileError(files: File[]): string | null {
  if (files.length !== 1) return 'Choose one CSV file at a time.'
  if (!files[0].name.toLowerCase().endsWith('.csv')) return 'Choose a .csv file. Spreadsheets and other file formats are not supported.'
  if (files[0].size === 0) return 'This file is empty. Include a header and at least one data row.'
  if (files[0].size > MAX_BATCH_BYTES) return 'This CSV exceeds 5 MB (5,000,000 bytes). Reduce its size and try again.'
  return null
}

export async function analyzeBatch(file: File, signal: AbortSignal): Promise<BatchPrediction> {
  const body = new FormData()
  // The local filename is not needed by the server and is not sent.
  body.append('file', file, 'batch.csv')
  const result = await request<BatchPrediction>('/batch-predict', { method: 'POST', body, signal })
  const rows = result?.rows
  const counts = result?.counts
  if (!Array.isArray(rows) || !rows.length || rows.length > MAX_BATCH_ROWS
      || !Number.isFinite(result.threshold) || result.threshold < 0 || result.threshold > 1
      || result.score_type !== 'uncalibrated' || typeof result.model_run_id !== 'string'
      || typeof result.notice !== 'string' || !counts
      || rows.some((r, i) => !r || r.row_number !== i + 1 || !Number.isFinite(r.score)
        || r.score < 0 || r.score > 1 || typeof r.flagged !== 'boolean' || r.flagged !== (r.score >= result.threshold))
      || counts.total !== rows.length || counts.flagged !== rows.filter(r => r.flagged).length
      || counts.not_flagged !== counts.total - counts.flagged) {
    throw new Error('The API returned inconsistent batch results. Nothing is displayed or exported. Please retry.')
  }
  return result
}

export function scoreDistribution(rows: BatchRow[]) {
  const bins = Array.from({ length: 20 }, (_, i) => ({ mid: (i + .5) / 20, label: `${(i / 20).toFixed(2)}–${((i + 1) / 20).toFixed(2)}`, flagged: 0, below: 0 }))
  rows.forEach(row => { const bin = bins[Math.min(19, Math.floor(row.score * 20))]; bin[row.flagged ? 'flagged' : 'below']++ })
  return bins
}

export function resultsCsv(rows: BatchRow[], threshold: number) {
  // Only validated numbers, booleans and fixed headers: no uploaded cells or formulas.
  return ['row_number,uncalibrated_score,flagged_prediction,frozen_threshold',
    ...rows.map(r => [r.row_number, r.score, r.flagged, threshold].join(','))].join('\r\n') + '\r\n'
}

export function exportResults(rows: BatchRow[], threshold: number) {
  const url = URL.createObjectURL(new Blob([resultsCsv(rows, threshold)], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url; link.download = 'veyra-batch-predictions.csv'; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
