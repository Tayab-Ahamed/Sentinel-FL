import React from 'react'
import { useRounds } from '../hooks/useExperiment'
import AccuracyChart from '../components/Charts/AccuracyChart'
import { Spinner, EmptyState, ErrorState } from './Dashboard'
import { pct } from '../api/client'
import './Pages.css'

interface RoundsPageProps { experimentId: string | null }

export default function RoundsPage({ experimentId }: RoundsPageProps) {
  const { data: rounds, loading, error } = useRounds(experimentId)

  if (!experimentId) return <EmptyState msg="Select an experiment." />
  if (loading) return <Spinner />
  if (error) return <ErrorState msg={error} />

  const r = rounds ?? []

  return (
    <div className="page fade-in">
      <div className="card">
        <div className="section-title">Accuracy &amp; ASR — Round Timeline</div>
        <AccuracyChart rounds={r} />
      </div>

      <div className="card">
        <div className="section-title">Per-Round Statistics</div>
        <div className="rounds-table-wrap">
          <table className="rounds-table">
            <thead>
              <tr>
                <th className="rt-head">Round</th>
                <th className="rt-head">Participants</th>
                <th className="rt-head">Excluded</th>
                <th className="rt-head">Flagged Clusters</th>
                <th className="rt-head">Clean Acc.</th>
                <th className="rt-head">ASR</th>
              </tr>
            </thead>
            <tbody>
              {r.map(round => (
                <tr key={round.round_num}>
                  <td className="rt-cell" style={{ fontWeight: 700, color: 'var(--teal)' }}>
                    #{round.round_num}
                  </td>
                  <td className="rt-cell">{round.participating_clients?.length ?? '—'}</td>
                  <td className="rt-cell" style={{ color: round.excluded_clients?.length ? 'var(--red)' : 'inherit' }}>
                    {round.excluded_clients?.length ?? '—'}
                  </td>
                  <td className="rt-cell" style={{ color: round.flagged_clusters?.length ? 'var(--amber)' : 'var(--text-muted)' }}>
                    {round.flagged_clusters?.length ?? 0}
                  </td>
                  <td className="rt-cell" style={{ color: 'var(--teal)' }}>
                    {pct(round.clean_accuracy)}
                  </td>
                  <td className="rt-cell" style={{ color: 'var(--red)' }}>
                    {pct(round.attack_success_rate)}
                  </td>
                </tr>
              ))}
              {!r.length && (
                <tr><td colSpan={6} style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>
                  No round data found.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
