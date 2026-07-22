import React from 'react'
import type { AlertEvent } from '../../types/sentinel'

interface AlertTimelineProps {
  alerts: AlertEvent[]
  maxItems?: number
}

const SEVERITY_COLOR: Record<string, string> = {
  high:   'var(--red)',
  medium: 'var(--amber)',
  low:    'var(--blue)',
}

export default function AlertTimeline({ alerts, maxItems = 20 }: AlertTimelineProps) {
  const visible = alerts.slice(0, maxItems)
  if (!visible.length) return <div className="chart-empty">No alerts detected.</div>

  return (
    <div className="alert-timeline">
      {visible.map(alert => (
        <div key={alert.id} className="alert-row">
          <div
            className="alert-dot"
            style={{ background: SEVERITY_COLOR[alert.severity] ?? 'var(--text-muted)' }}
          />
          <div className="alert-content">
            <div className="alert-message">{alert.message}</div>
            <div className="alert-meta">
              <span className={`badge badge-${alert.severity}`}>{alert.severity}</span>
              <span className="text-muted">{alert.layer_id}</span>
              {alert.round_num != null && <span className="text-muted">Round {alert.round_num}</span>}
              {alert.timestamp && (
                <span className="text-muted">{alert.timestamp.slice(0, 19).replace('T', ' ')}</span>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
