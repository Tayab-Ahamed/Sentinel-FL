import React from 'react'
import { useHeatmap } from '../hooks/useExperiment'
import TrustHeatmap from '../components/Charts/TrustHeatmap'
import { Spinner, EmptyState, ErrorState } from './Dashboard'
import '../components/Charts/Charts.css'
import './Pages.css'

interface HeatmapPageProps { experimentId: string | null }

export default function HeatmapPage({ experimentId }: HeatmapPageProps) {
  const { data: heatmap, loading, error } = useHeatmap(experimentId)

  if (!experimentId) return <EmptyState msg="Select an experiment." />
  if (loading) return <Spinner />
  if (error) return <ErrorState msg={error} />

  return (
    <div className="page fade-in">
      <div className="card">
        <div className="section-title">Client × Round Trust Score Heatmap</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
          Each cell shows the peak trust score (anomaly level) for a client in a given round.
          Teal = safe, amber = suspicious, red = malicious.
        </p>
        {heatmap ? <TrustHeatmap data={heatmap} /> : <EmptyState msg="No heatmap data." />}
      </div>
    </div>
  )
}
