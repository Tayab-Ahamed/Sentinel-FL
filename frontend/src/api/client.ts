// api/client.ts — fetch helpers for the SENTINEL-FL backend

const BASE = '/api/v1'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err?.error || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  experiments: {
    list: ()                       => get<{ experiments: import('../types/sentinel').Experiment[] }>('/experiments'),
    get:  (id: string)             => get<import('../types/sentinel').Experiment>(`/experiments/${id}`),
    rounds: (id: string)           => get<{ rounds: import('../types/sentinel').TrainingRound[] }>(`/experiments/${id}/rounds`),
    heatmap: (id: string)          => get<import('../types/sentinel').HeatmapData>(`/experiments/${id}/reputation-heatmap`),
    metrics: (id: string, names: string[]) =>
      get<import('../types/sentinel').MetricSeries>(`/experiments/${id}/metrics?names=${names.join(',')}`),
    alerts: (id: string)           => get<{ alerts: import('../types/sentinel').AlertEvent[]; count: number }>(`/experiments/${id}/alerts`),
    clients: (id: string)          => get<{ clients: import('../types/sentinel').ClientStat[]; count: number }>(`/experiments/${id}/clients`),
    config: (id: string)           => get<{ config: Record<string, unknown> }>(`/experiments/${id}/config`),
    audit: (id: string, round: number) => get<unknown>(`/experiments/${id}/audits/${round}`),
  },
  health: ()                       => get<{ status: string; version: string }>('/health').catch(() => ({ status: 'offline', version: '?' })),
}

// Helper: format a number as a percentage string
export function pct(v: number | null | undefined, decimals = 1): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(decimals)}%`
}

// Helper: format a float
export function fmt(v: number | null | undefined, decimals = 3): string {
  if (v == null) return '—'
  return v.toFixed(decimals)
}

// Helper: format bytes
export function bytes(v: number | null | undefined): string {
  if (v == null) return '—'
  if (v < 1024) return `${v} B`
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(1)} KB`
  return `${(v / 1024 / 1024).toFixed(1)} MB`
}
