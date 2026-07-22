import React from 'react'
import './StatCard.css'

interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  color?: 'teal' | 'amber' | 'red' | 'blue' | 'purple'
  icon?: string
  trend?: 'up' | 'down' | 'neutral'
}

export default function StatCard({ label, value, sub, color = 'teal', icon, trend }: StatCardProps) {
  return (
    <div className={`stat-card stat-card--${color}`}>
      <div className="stat-card-top">
        <span className="stat-label">{label}</span>
        {icon && <span className="stat-icon">{icon}</span>}
      </div>
      <div className="stat-value">{value}</div>
      {sub && (
        <div className="stat-sub">
          {trend === 'up'   && <span className="trend-up">↑</span>}
          {trend === 'down' && <span className="trend-down">↓</span>}
          {sub}
        </div>
      )}
    </div>
  )
}
