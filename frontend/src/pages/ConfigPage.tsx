import React from 'react'
import { useConfig } from '../hooks/useExperiment'
import { Spinner, EmptyState, ErrorState } from './Dashboard'
import './Pages.css'

interface ConfigPageProps { experimentId: string | null }

function renderValue(v: unknown): React.ReactNode {
  if (v === null || v === undefined) return <span style={{ color: 'var(--text-muted)' }}>null</span>
  if (typeof v === 'boolean') return <span style={{ color: v ? 'var(--teal)' : 'var(--red)' }}>{String(v)}</span>
  if (typeof v === 'number') return <span style={{ color: 'var(--blue)' }}>{String(v)}</span>
  if (typeof v === 'string') return <span style={{ color: 'var(--amber)' }}>"{v}"</span>
  if (Array.isArray(v)) {
    return <span>[{v.map((item, i) => <span key={i}>{i > 0 && ', '}{renderValue(item)}</span>)}]</span>
  }
  if (typeof v === 'object') {
    return (
      <div style={{ paddingLeft: 16 }}>
        {Object.entries(v as Record<string, unknown>).map(([k, val]) => (
          <div key={k} style={{ display: 'flex', gap: 8, alignItems: 'flex-start', padding: '3px 0' }}>
            <span style={{ color: 'var(--purple)', minWidth: 160, fontWeight: 500 }}>{k}:</span>
            <span>{renderValue(val)}</span>
          </div>
        ))}
      </div>
    )
  }
  return <span>{String(v)}</span>
}

export default function ConfigPage({ experimentId }: ConfigPageProps) {
  const { data: config, loading, error } = useConfig(experimentId)
  const [view, setView] = React.useState<'pretty' | 'raw'>('pretty')

  if (!experimentId) return <EmptyState msg="Select an experiment." />
  if (loading) return <Spinner />
  if (error) return <ErrorState msg={error} />

  const cfg = config ?? {}
  const isEmpty = Object.keys(cfg).length === 0

  return (
    <div className="page fade-in">
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div className="section-title" style={{ margin: 0 }}>Experiment Configuration</div>
          <div style={{ display: 'flex', gap: 8 }}>
            {(['pretty', 'raw'] as const).map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                style={{
                  padding: '4px 12px',
                  borderRadius: 6,
                  border: `1px solid ${view === v ? 'var(--teal)' : 'var(--border)'}`,
                  background: view === v ? 'var(--teal-subtle)' : 'transparent',
                  color: view === v ? 'var(--teal)' : 'var(--text-muted)',
                  fontSize: 12, fontWeight: 600, cursor: 'pointer',
                }}
              >{v === 'pretty' ? 'Pretty' : 'Raw JSON'}</button>
            ))}
          </div>
        </div>

        {isEmpty ? (
          <EmptyState msg="No configuration data found. Ensure experiment.json exists." />
        ) : view === 'raw' ? (
          <pre className="config-pre">{JSON.stringify(cfg, null, 2)}</pre>
        ) : (
          <div style={{ fontFamily: 'monospace', fontSize: 13, lineHeight: 1.7 }}>
            {Object.entries(cfg).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', gap: 8, padding: '4px 0', borderBottom: '1px solid var(--border)' }}>
                <span style={{ color: 'var(--purple)', minWidth: 200, fontWeight: 600, flexShrink: 0 }}>{k}</span>
                <span>{renderValue(v)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
