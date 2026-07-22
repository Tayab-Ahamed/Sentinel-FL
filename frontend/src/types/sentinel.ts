// SENTINEL-FL TypeScript types — mirrors SCHEMAS.md

export interface EvaluationResult {
  experiment_id: string
  clean_accuracy?: number | null
  attack_success_rate?: number | null
  robust_accuracy?: number | null
  false_acceptance_rate?: number | null
  false_rejection_rate?: number | null
  detection_latency_ms?: number | null
  communication_cost_bytes?: number | null
  precision?: number | null
  recall?: number | null
  f1_score?: number | null
  false_positive_rate?: number | null
  runtime_seconds?: number | null
  peak_memory_mb?: number | null
  warnings: string[]
}

export interface AttackReport {
  attack_id?: string
  attack_type: string
  malicious_client_ids: string[]
  target_class: number
  poison_fraction: number
  rounds_active: number[]
}

export interface Experiment {
  experiment_id: string
  config_ref: string
  dataset_phase: 'phase0_synthetic' | 'phase1_official'
  layers_enabled: string[]
  attack_config: AttackReport
  result?: EvaluationResult | null
  seeds?: Record<string, number>
  _raw?: Record<string, unknown>
}

export interface TrainingRound {
  round_num: number
  participating_clients: string[]
  excluded_clients: string[]
  flagged_clusters: string[][]
  global_model_id?: string
  clean_accuracy?: number | null
  attack_success_rate?: number | null
}

export interface MetricPoint {
  metric_name: string
  round_num: number
  value: number
}

export interface MetricSeries {
  series: Record<string, MetricPoint[]>
  display_names?: Record<string, string>
}

export interface HeatmapData {
  client_ids: string[]
  rounds: number[]
  scores: number[][]
}

export interface AlertEvent {
  id: string
  round_num?: number | null
  layer_id: string
  severity: 'high' | 'medium' | 'low'
  event_type: string
  subject_id: string
  message: string
  timestamp?: string
}

export interface ClientStat {
  client_id: string
  trust_score: number
  flag_count: number
  layers_flagged: string[]
  last_round: number
  status: string
  is_suspicious: boolean
}

export interface DemoRaw {
  fedavg?: { clean_accuracy: number; attack_success_rate: number }
  multikrum?: { clean_accuracy: number; attack_success_rate: number }
  'multikrum+guard'?: {
    clean_accuracy: number
    attack_success_rate: number
    rounds_with_any_cluster_flag?: number
    strip_frr_on_clean?: number
    strip_detection_rate_on_triggered?: number
    strip_boundary?: number
  }
}
