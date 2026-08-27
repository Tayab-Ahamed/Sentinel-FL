import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useExperimentList } from '../../hooks/useExperiment'
import './Sidebar.css'

const NAV = [
  { path: '/',              icon: '⊞', label: 'Dashboard'    },
  { path: '/clients',       icon: '◈', label: 'Clients'       },
  { path: '/heatmap',       icon: '▦', label: 'Trust Heatmap' },
  { path: '/alerts',        icon: '⚡', label: 'Alerts'        },
  { path: '/rounds',        icon: '◷', label: 'Rounds'        },
  { path: '/attack',        icon: '☠', label: 'Attack Viz'    },
  { path: '/model',         icon: '⬇', label: 'Model Download'},
  { path: '/config',        icon: '⚙', label: 'Config'        },
]

interface SidebarProps {
  experimentId: string | null
  onExperimentChange: (id: string) => void
}

export default function Sidebar({ experimentId, onExperimentChange }: SidebarProps) {
  const { data: experiments } = useExperimentList()

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="logo-icon">◉</span>
        <div>
          <div className="logo-title">SENTINEL-FL</div>
          <div className="logo-sub">Backdoor Immune System</div>
        </div>
      </div>

      {experiments && experiments.length > 0 && (
        <div className="sidebar-experiment-picker">
          <label className="picker-label">Experiment</label>
          <select
            className="picker-select"
            value={experimentId ?? ''}
            onChange={e => onExperimentChange(e.target.value)}
          >
            {experiments.map(exp => (
              <option key={exp.experiment_id} value={exp.experiment_id}>
                {exp.experiment_id}
              </option>
            ))}
          </select>
        </div>
      )}

      <nav className="sidebar-nav">
        {NAV.map(({ path, icon, label }) => (
          <NavLink
            key={path}
            to={path}
            className={({ isActive }) => `nav-item${isActive ? ' nav-item--active' : ''}`}
            end={path === '/'}
          >
            <span className="nav-icon">{icon}</span>
            <span className="nav-label">{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className="sidebar-version">v1.0.0 · Open Source</span>
      </div>
    </aside>
  )
}
