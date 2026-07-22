import React, { useState } from 'react'
import { useAlerts } from '../hooks/useExperiment'
import AlertTimeline from '../components/Charts/AlertTimeline'
import { Spinner, EmptyState, ErrorState } from './Dashboard'
import '../components/Charts/Charts.css'
import './Pages.css'

interface AlertsPageProps { experimentId: string | null }

type SevFilter = 'all' | 'high' | 'medium' | 'low'

export default function AlertsPage({ experimentId }: AlertsPageProps) {
  const { data: alerts, loading, error } = useAlerts(experimentId)
  const [sev, setSev] = useState<SevFilter>('all')
  const [search, setSearch] = useState('')

  if (!experimentId) return <EmptyState msg="Select an experiment." />
  if (loading) return <Spinner />
  if (error) return <ErrorState msg={error} />

  const allAlerts = alerts ?? []
  const filtered = allAlerts.filter(a =>
    (sev === 'all' || a.severity === sev) &&
    (a.message.toLowerCase().includes(search.toLowerCase()) ||
     a.subject_id.toLowerCase().includes(search.toLowerCase()))
  )
  const counts = {
    high:   allAlerts.filter(a => a.severity === 'high').length,
    medium: allAlerts.filter(a => a.severity === 'medium').length,
    low:    allAlerts.filter(a => a.severity === 'low').length,
  }

  return (
    <div className="page fade-in">
      <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <div className="card" style={{ textAlign: 'center', borderColor: 'var(--red-dim)' }}>
          <div className="section-title">High Severity</div>
          <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--red)' }}>{counts.high}</div>
        </div>
        <div className="card" style={{ textAlign: 'center', borderColor: 'var(--amber-dim)' }}>
          <div className="section-title">Medium Severity</div>
          <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--amber)' }}>{counts.medium}</div>
        </div>
        <div className="card" style={{ textAlign: 'center' }}>
          <div className="section-title">Low Severity</div>
          <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--blue)' }}>{counts.low}</div>
        </div>
      </div>

      <div className="card">
        <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap' }}>
          {(['all','high','medium','low'] as SevFilter[]).map(s => (
            <button
              key={s}
              onClick={() => setSev(s)}
              style={{
                padding: '5px 14px',
                borderRadius: 999,
                border: `1px solid ${sev === s ? 'var(--teal)' : 'var(--border)'}`,
                background: sev === s ? 'var(--teal-subtle)' : 'transparent',
                color: sev === s ? 'var(--teal)' : 'var(--text-muted)',
                fontSize: 12, fontWeight: 600, cursor: 'pointer', textTransform: 'capitalize',
              }}
            >{s}</button>
          ))}
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search alerts..."
            style={{
              marginLeft: 'auto',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              color: 'var(--text-primary)',
              padding: '5px 12px',
              fontSize: 12, outline: 'none',
            }}
          />
        </div>
        <AlertTimeline alerts={filtered} maxItems={200} />
        {!filtered.length && <EmptyState msg="No alerts match the current filter." />}
      </div>
    </div>
  )
}
