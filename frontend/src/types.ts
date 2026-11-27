export type Page = 'overview' | 'performance' | 'explainability' | 'predictor' | 'batch' | 'monitoring' | 'copilot'

export interface CopilotStatus {
  status: 'ready' | 'setup_required'
  model: string
  provider: 'Ollama (local)'
  index_ready: boolean
  model_ready: boolean
  detail: string
}
export interface CopilotSource {
  id: number
  passage: string
  similarity: number | null
  citation: { source: string; section: string; locator: string; chunk_id: string }
}
export interface CopilotAnswer {
  status: 'answered' | 'insufficient_evidence' | 'evidence_only' | 'welcome'
  answer: string
  sources: CopilotSource[]
  model: string | null
  notice: string
}
export interface Product {
  product: string
  transactions: number
  fraud_transactions: number
  fraud_rate_pct: number
  total_amount: number
  fraud_amount: number
  fraud_amount_share_pct: number
}
export interface Trend {
  elapsed_week: number
  transactions: number
  fraud_transactions: number
  fraud_rate_pct: number
  mean_amount: number
  identity_coverage_pct: number
}
export interface Metrics {
  pr_auc: number; roc_auc: number; precision: number; recall: number; f1: number
  rows: number; fraud_rows: number; confusion_matrix: number[][]
}
export interface Baseline extends Pick<Metrics, 'pr_auc' | 'roc_auc' | 'precision' | 'recall' | 'f1'> { model: string; threshold: number }
export interface OperatingPoint extends Pick<Metrics, 'precision' | 'recall' | 'f1' | 'confusion_matrix'> { policy: string; threshold: number }
export interface DashboardData {
  overview: {
    status: string | null; generated_at: string | null
    transactions: number | null; fraud_transactions: number | null
    fraud_rate_pct: number | null; transaction_value: number | null
    identity_coverage_pct: number | null; identity_matches: number | null
    missing_columns: number | null; duplicate_rows: number | null
    products: Product[]; trends: Trend[]
  }
  performance: {
    validation: Metrics | null; test: Metrics | null; threshold: number | null
    recall_floor: number | null; completed_at: string | null
    baselines: Baseline[]; operating_points: OperatingPoint[]
  }
  explainability: {
    sample_rows: number | null; seed: number | null; original_features: number | null
    encoded_columns: number | null; local_score: number | null; max_additivity_error: number | null
    importance: { feature: string; mean_abs_grouped_shap: number }[]
    images: { key: 'global' | 'beeswarm' | 'local'; url: string; available: boolean }[]
  }
  model: { available: boolean; reason: string | null; threshold: number | null; run_id: string | null; sha256: string | null; features: string[] }
  warnings: string[]
}
export interface Prediction {
  score: number; threshold: number; above: boolean; provided: number; expected: number
  amount: number; submitted_at: string; run_id: string; model_sha: string
}
export interface PrecisionRecall {
  points: { precision: number; recall: number; threshold: number }[]
  operating_points: OperatingPoint[]
}

export interface BatchRow { row_number: number; score: number; flagged: boolean }
export interface BatchPrediction {
  rows: BatchRow[]
  counts: { total: number; flagged: number; not_flagged: number }
  threshold: number
  model_run_id: string
  score_type: 'uncalibrated'
  notice: string
}

export interface TransactionExplanation {
  score: number; flagged: boolean; threshold: number; model_run_id: string
  score_type: 'uncalibrated'; units: 'raw_log_odds'
  base_value: number; raw_margin: number; additivity_error: number
  contributions: { feature: string; value: number; encoded_columns: number }[]
  method: string; grouping: string; notice: string
}
