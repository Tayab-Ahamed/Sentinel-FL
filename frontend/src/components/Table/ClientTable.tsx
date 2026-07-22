import React, { useState } from 'react'
import type { ClientStat } from '../../types/sentinel'
import './ClientTable.css'

interface ClientTableProps {
  clients: ClientStat[]
}

function TrustBar({ score }: { score: number }) {
  const color = score >= 0.6 ? '#ef4444' : score >= 0.3 ? '#f59e0b' : '#00d4aa'
  return (
    <div className="trust-bar-bg">
      <div
        className="trust-bar-fill"
        style={{ width: `${Math.min(score * 100, 100)}%`, background: color }}
      />
    </div>
  )
}

type SortKey = 'client_id' | 'trust_score' | 'flag_count'

export default function ClientTable({ clients }: ClientTableProps) {
  const [sort, setSort] = useState<SortKey>('trust_score')
  const [asc, setAsc] = useState(false)
  const [filter, setFilter] = useState('')

  const toggle = (key: SortKey) => {
    if (sort === key) setAsc(a => !a)
    else { setSort(key); setAsc(false) }
  }

  const filtered = clients
    .filter(c => c.client_id.includes(filter))
    .sort((a, b) => {
      const va = a[sort] as string | number
      const vb = b[sort] as string | number
      if (va < vb) return asc ? -1 : 1
      if (va > vb) return asc ? 1 : -1
      return 0
    })

  return (
    <div className="client-table-wrap">
      <div className="client-table-toolbar">
        <input
          className="client-filter"
          placeholder="Filter by client ID..."
          value={filter}
          onChange={e => setFilter(e.target.value)}
        />
        <span className="text-muted" style={{ fontSize: 12 }}>{filtered.length} clients</span>
      </div>
      <div className="client-table-scroll">
        <table className="client-table">
          <thead>
            <tr>
              <Th label="Client ID"    col="client_id"   sort={sort} asc={asc} onClick={toggle} />
              <th className="ct-head">Trust Score</th>
              <Th label="Flags"        col="flag_count"  sort={sort} asc={asc} onClick={toggle} />
              <th className="ct-head">Layers</th>
              <th className="ct-head">Status</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(c => (
              <tr key={c.client_id} className={`ct-row ${c.is_suspicious ? 'ct-row--sus' : ''}`}>
                <td className="ct-cell ct-cell--id">{c.client_id}</td>
                <td className="ct-cell">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <TrustBar score={c.trust_score} />
                    <span style={{ fontSize: 11, minWidth: 36, color: 'var(--text-secondary)' }}>
                      {(c.trust_score * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td className="ct-cell ct-cell--num">{c.flag_count}</td>
                <td className="ct-cell">
                  {c.layers_flagged.map(l => (
                    <span key={l} className="badge badge-teal" style={{ marginRight: 4 }}>{l}</span>
                  ))}
                </td>
                <td className="ct-cell">
                  <span className={`badge ${c.is_suspicious ? 'badge-high' : 'badge-teal'}`}>
                    {c.is_suspicious ? 'Suspicious' : 'Active'}
                  </span>
                </td>
              </tr>
            ))}
            {!filtered.length && (
              <tr><td colSpan={5} className="ct-empty">No clients found.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Th({ label, col, sort, asc, onClick }: {
  label: string; col: SortKey; sort: SortKey; asc: boolean; onClick: (k: SortKey) => void
}) {
  return (
    <th className="ct-head ct-head--sort" onClick={() => onClick(col)}>
      {label}
      {sort === col && <span style={{ marginLeft: 4 }}>{asc ? '↑' : '↓'}</span>}
    </th>
  )
}
