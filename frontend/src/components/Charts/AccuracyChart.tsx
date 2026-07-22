import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts'

interface Point { round_num: number; clean_accuracy?: number | null; attack_success_rate?: number | null }

interface AccuracyChartProps {
  rounds: Point[]
  title?: string
}

const fmt = (v: number) => `${(v * 100).toFixed(1)}%`

export default function AccuracyChart({ rounds, title }: AccuracyChartProps) {
  if (!rounds.length) return (
    <div className="chart-empty">No round data available</div>
  )
  return (
    <div className="chart-wrap">
      {title && <div className="chart-title">{title}</div>}
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={rounds} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
          <XAxis
            dataKey="round_num"
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: '#1e2d4a' }}
            label={{ value: 'Round', position: 'insideBottomRight', offset: -4, fill: '#64748b', fontSize: 11 }}
          />
          <YAxis
            tickFormatter={fmt}
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            domain={[0, 1]}
          />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #1e2d4a', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#94a3b8' }}
            formatter={(val: number) => fmt(val)}
          />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
          <Line
            type="monotone"
            dataKey="clean_accuracy"
            name="Clean Accuracy"
            stroke="#00d4aa"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#00d4aa' }}
          />
          <Line
            type="monotone"
            dataKey="attack_success_rate"
            name="Attack Success Rate"
            stroke="#ef4444"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#ef4444' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
