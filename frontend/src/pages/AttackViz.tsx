import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts'
import { useExperiment } from '../hooks/useExperiment'
import { Spinner, EmptyState, ErrorState } from './Dashboard'
import type { DemoRaw } from '../types/sentinel'
import './Pages.css'
import '../components/Charts/Charts.css'

interface AttackVizProps { experimentId: string | null }

export default function AttackViz({ experimentId }: AttackVizProps) {
  const { data: exp, loading, error } = useExperiment(experimentId)

  if (!experimentId) return <EmptyState msg="Select an experiment." />
  if (loading) return <Spinner />
  if (error) return <ErrorState msg={error} />

  const raw = exp?._raw as DemoRaw | undefined
  const hasRaw = raw && (raw.fedavg || raw.multikrum || raw['multikrum+guard'])

  const barData = hasRaw ? [
    {
      strategy: 'FedAvg (No Defense)',
      'Clean Accuracy': (raw.fedavg?.clean_accuracy ?? 0) * 100,
      'Attack Success Rate': (raw.fedavg?.attack_success_rate ?? 0) * 100,
    },
    {
      strategy: 'Multi-Krum (L1)',
      'Clean Accuracy': (raw.multikrum?.clean_accuracy ?? 0) * 100,
      'Attack Success Rate': (raw.multikrum?.attack_success_rate ?? 0) * 100,
    },
    {
      strategy: 'SENTINEL (L1+L3)',
      'Clean Accuracy': (raw['multikrum+guard']?.clean_accuracy ?? 0) * 100,
      'Attack Success Rate': (raw['multikrum+guard']?.attack_success_rate ?? 0) * 100,
    },
  ] : []

  const guardData = raw?.['multikrum+guard']

  return (
    <div className="page fade-in">
      <div className="card">
        <div className="section-title">Defense Strategy Comparison</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
          Clean Accuracy vs Attack Success Rate across defense strategies.
          Lower ASR = better defense. High C-Acc = preserved utility.
        </p>
        {barData.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" vertical={false} />
              <XAxis
                dataKey="strategy"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                tickLine={false}
                axisLine={{ stroke: '#1e2d4a' }}
              />
              <YAxis
                tickFormatter={v => `${v}%`}
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #1e2d4a', borderRadius: 8, fontSize: 12 }}
                formatter={(v: number) => `${v.toFixed(1)}%`}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="Clean Accuracy"      fill="#00d4aa" radius={[4,4,0,0]} />
              <Bar dataKey="Attack Success Rate" fill="#ef4444" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <EmptyState msg="No multi-strategy data available (demo_results.json not found)." />
        )}
      </div>

      {guardData && (
        <div className="card">
          <div className="section-title">SENTINEL L1+L3 Detail</div>
          <div className="info-grid">
            <InfoRow label="Rounds w/ Cluster Flags" value={String(guardData.rounds_with_any_cluster_flag ?? '—')} />
            <InfoRow label="STRIP Detection Rate"    value={guardData.strip_detection_rate_on_triggered != null ? `${(guardData.strip_detection_rate_on_triggered * 100).toFixed(1)}%` : '—'} />
            <InfoRow label="STRIP FRR (Clean)"       value={guardData.strip_frr_on_clean != null ? `${(guardData.strip_frr_on_clean * 100).toFixed(1)}%` : '—'} />
            <InfoRow label="STRIP Boundary"          value={guardData.strip_boundary?.toFixed(4) ?? '—'} />
          </div>
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
