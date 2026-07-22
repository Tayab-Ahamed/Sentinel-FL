import React from 'react'
import { useExperiment, useRounds, useAlerts } from '../hooks/useExperiment'
import StatCard from '../components/Cards/StatCard'
import AccuracyChart from '../components/Charts/AccuracyChart'
import AlertTimeline from '../components/Charts/AlertTimeline'
import { pct, fmt, bytes } from '../api/client'
import type { Experiment } from '../types/sentinel'
import '../components/Charts/Charts.css'
import './Pages.css'

interface DashboardProps { experimentId: string | null }

export default function Dashboard({ experimentId }: DashboardProps) {
  const { data: exp, loading, error } = useExperiment(experimentId)
  const { data: rounds } = useRounds(experimentId)
  const { data: alerts } = useAlerts(experimentId)

  if (!experimentId) return <EmptyState msg="Select an experiment from the sidebar." />
  if (loading) return <Spinner />
  if (error) return <ErrorState msg={error} />
  if (!exp) return <EmptyState msg="Experiment not found." />

  const r = exp.result
  const raw = exp._raw as Record<string, Record<string, number>> | undefined

  return (
    <div className="page fade-in">
      {/* KPI Grid */}
      <div className="kpi-grid">
        <StatCard label="Clean Accuracy"       value={pct(r?.clean_accuracy)}       color="teal"   icon="✓" trend={r?.clean_accuracy && r.clean_accuracy > 0.8 ? 'up' : 'down'} />
        <StatCard label="Attack Success Rate"  value={pct(r?.attack_success_rate)}  color="red"    icon="☠" sub="lower = better" />
        <StatCard label="Robust Accuracy"      value={pct(r?.robust_accuracy)}      color="blue"   icon="◈" />
        <StatCard label="Detection F1"         value={fmt(r?.f1_score)}             color="purple" icon="⊕" />
        <StatCard label="False Accept Rate"    value={pct(r?.false_acceptance_rate)} color="amber" icon="⚠" sub="L3 missed triggers" />
        <StatCard label="Comm. Cost"           value={bytes(r?.communication_cost_bytes)} color="teal" icon="⇆" />
      </div>

      {/* Charts row */}
      <div className="charts-row">
        <div className="card chart-card">
          <div className="section-title">Accuracy &amp; ASR Over Rounds</div>
          <AccuracyChart rounds={rounds ?? []} />
        </div>

        <div className="card chart-card">
          <div className="section-title">Recent Alerts</div>
          <AlertTimeline alerts={alerts ?? []} maxItems={8} />
        </div>
      </div>

      {/* Attack config summary */}
      <div className="card">
        <div className="section-title">Attack Configuration</div>
        <div className="info-grid">
          <InfoRow label="Type"          value={exp.attack_config.attack_type} />
          <InfoRow label="Target Class"  value={String(exp.attack_config.target_class)} />
          <InfoRow label="Poison Frac."  value={pct(exp.attack_config.poison_fraction)} />
          <InfoRow label="Malicious IDs" value={exp.attack_config.malicious_client_ids.join(', ') || '—'} />
          <InfoRow label="Layers"        value={exp.layers_enabled.join(', ')} />
          <InfoRow label="Phase"         value={exp.dataset_phase} />
        </div>
      </div>

      {/* Defense warnings */}
      {r?.warnings && r.warnings.length > 0 && (
        <div className="card card--warn">
          <div className="section-title">⚠ Evaluation Warnings</div>
          {r.warnings.map((w, i) => (
            <div key={i} className="warn-row">{w}</div>
          ))}
        </div>
      )}
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-row">
      <span className="info-label">{label}</span>
      <span className="info-value">{value}</span>
    </div>
  )
}

export function Spinner() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200 }}>
      <div className="spinner" />
    </div>
  )
}
export function EmptyState({ msg }: { msg: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200, color: 'var(--text-muted)', fontSize: 14 }}>
      {msg}
    </div>
  )
}
export function ErrorState({ msg }: { msg: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200, color: 'var(--red)', fontSize: 13 }}>
      ⚠ {msg}
    </div>
  )
}
