import React from 'react'
import { useHealth } from '../../hooks/useExperiment'
import './TopBar.css'

interface TopBarProps {
  title: string
  experimentId: string | null
}

export default function TopBar({ title, experimentId }: TopBarProps) {
  const { data: health } = useHealth()
  const online = health?.status === 'ok'

  return (
    <header className="topbar">
      <div className="topbar-title">{title}</div>
      <div className="topbar-right">
        {experimentId && (
          <span className="topbar-exp">
            <span className="topbar-exp-label">Experiment</span>
            <code className="topbar-exp-id">{experimentId}</code>
          </span>
        )}
        <div className={`topbar-status ${online ? 'topbar-status--online' : 'topbar-status--offline'}`}>
          <span className="status-dot" />
          {online ? 'API Online' : 'API Offline'}
        </div>
      </div>
    </header>
  )
}
