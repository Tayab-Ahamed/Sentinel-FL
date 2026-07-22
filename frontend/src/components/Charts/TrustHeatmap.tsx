import React, { useMemo } from 'react'
import type { HeatmapData } from '../../types/sentinel'

interface TrustHeatmapProps {
  data: HeatmapData
}

function scoreToColor(score: number): string {
  // 0 = safe (teal), 1 = suspicious (red)
  if (score < 0.3) return '#00d4aa'
  if (score < 0.6) return '#f59e0b'
  return '#ef4444'
}

export default function TrustHeatmap({ data }: TrustHeatmapProps) {
  const { client_ids, rounds, scores } = data

  if (!client_ids.length) return (
    <div className="chart-empty">No heatmap data — Trust Ledger may be empty.</div>
  )

  const CELL_W = Math.max(20, Math.min(40, Math.floor(600 / Math.max(rounds.length, 1))))
  const CELL_H = 28

  return (
    <div className="heatmap-wrapper">
      <div className="heatmap-scroll">
        <table className="heatmap-table">
          <thead>
            <tr>
              <th className="heatmap-corner">Client</th>
              {rounds.map(r => (
                <th key={r} className="heatmap-round-head" style={{ width: CELL_W }}>
                  {r}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {client_ids.map((cid, ci) => (
              <tr key={cid}>
                <td className="heatmap-client-label">{cid}</td>
                {(scores[ci] || []).map((score, ri) => (
                  <td
                    key={ri}
                    className="heatmap-cell"
                    style={{ background: scoreToColor(score), height: CELL_H }}
                    title={`${cid} | Round ${rounds[ri]} | Score: ${score.toFixed(2)}`}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="heatmap-legend">
        <span style={{ color: '#00d4aa' }}>■ Safe (0.0)</span>
        <span style={{ color: '#f59e0b' }}>■ Suspicious (0.3–0.6)</span>
        <span style={{ color: '#ef4444' }}>■ Malicious (0.6+)</span>
      </div>
    </div>
  )
}
