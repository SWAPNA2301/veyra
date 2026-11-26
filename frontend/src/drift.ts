import { request } from './api'

export type DriftSeverity = 'stable' | 'watch' | 'high'
export interface DriftReference {
  profile_id: string; created_at_utc: string; source_split: 'train'; reference_rows: number
  model_run_id: string; train_window: { rows: number; TransactionDT_min: number; TransactionDT_max: number }
  policy: { min_batch_rows: number; min_observed_rows: number; psi_watch: number; psi_high: number; missing_watch_pp: number; missing_high_pp: number }
  method: string
}
export type ReferenceStatus = { status: 'ready'; reference: DriftReference; features: number } | { status: 'unavailable'; detail: string }
export interface DriftFeature {
  name: string; kind: 'numeric' | 'categorical'; provided: boolean
  psi: number | null; psi_status: 'ready' | 'insufficient_data'
  psi_severity: DriftSeverity | null; missing_severity: DriftSeverity | null; severity: DriftSeverity | null
  observed_rows: number; reference_observed_rows: number
  missing_rate: number; reference_missing_rate: number; missing_delta_pp: number
  bins: { label: string; reference_count: number; batch_count: number; reference_share: number; batch_share: number }[]
}
export interface DriftResult {
  status: 'ready' | 'insufficient_data'; rows: number; reference: DriftReference; features: DriftFeature[]; notice: string
  summary: { features: number; provided_features: number; psi_assessed: number; high: number | null; watch: number | null; max_psi: number | null; max_missing_change_pp: number | null }
}

export async function compareDrift(file: File, signal: AbortSignal): Promise<DriftResult> {
  const body = new FormData(); body.append('file', file, 'batch.csv')
  const result = await request<DriftResult>('/drift/compare', { method: 'POST', body, signal })
  const validSeverity = (value: unknown) => value === null || ['stable', 'watch', 'high'].includes(String(value))
  if (!result || !['ready', 'insufficient_data'].includes(result.status)
    || !Number.isInteger(result.rows) || result.rows < 1 || result.rows > 5000
    || result.reference?.source_split !== 'train' || !Number.isInteger(result.reference.reference_rows) || result.reference.reference_rows < 1000
    || !Array.isArray(result.features) || !result.features.length || result.features.length > 22
    || new Set(result.features.map(f => f.name)).size !== result.features.length
    || result.summary?.features !== result.features.length
    || result.features.some(f => !f || typeof f.name !== 'string' || !['numeric', 'categorical'].includes(f.kind)
      || (f.psi !== null && (!Number.isFinite(f.psi) || f.psi < 0))
      || ![f.severity, f.psi_severity, f.missing_severity].every(validSeverity)
      || ![f.missing_rate, f.reference_missing_rate, f.missing_delta_pp].every(Number.isFinite)
      || !Array.isArray(f.bins) || f.bins.some(b => typeof b.label !== 'string'
        || !Number.isInteger(b.batch_count) || !Number.isInteger(b.reference_count) || b.batch_count < 0 || b.reference_count < 0
        || !Number.isFinite(b.batch_share) || !Number.isFinite(b.reference_share)
        || Math.abs(b.batch_share - b.batch_count / result.rows) > 1e-8
        || Math.abs(b.reference_share - b.reference_count / result.reference.reference_rows) > 1e-8)
      || f.bins.reduce((sum, b) => sum + b.batch_count, 0) !== result.rows
      || f.bins.reduce((sum, b) => sum + b.reference_count, 0) !== result.reference.reference_rows
      || (result.status === 'insufficient_data' && (f.psi !== null || f.severity !== null)))) {
    throw new Error('The API returned an inconsistent drift report. No metrics are displayed. Please retry.')
  }
  return result
}
