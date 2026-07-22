import { useState, useEffect } from 'react'
import { api } from '../api/client'
import type { Experiment, TrainingRound, HeatmapData, AlertEvent, ClientStat, MetricSeries } from '../types/sentinel'

// Generic hook factory
function useFetch<T>(fetcher: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetcher()
      .then(d => { if (!cancelled) { setData(d); setLoading(false) } })
      .catch(e => { if (!cancelled) { setError(String(e?.message ?? e)); setLoading(false) } })
    return () => { cancelled = true }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, loading, error }
}

export function useExperimentList() {
  return useFetch(() => api.experiments.list().then(r => r.experiments), [])
}

export function useExperiment(id: string | null) {
  return useFetch(() => id ? api.experiments.get(id) : Promise.resolve(null as unknown as Experiment), [id])
}

export function useRounds(id: string | null) {
  return useFetch(
    () => id ? api.experiments.rounds(id).then(r => r.rounds) : Promise.resolve([] as TrainingRound[]),
    [id]
  )
}

export function useHeatmap(id: string | null) {
  return useFetch(
    () => id ? api.experiments.heatmap(id) : Promise.resolve(null as unknown as HeatmapData),
    [id]
  )
}

export function useAlerts(id: string | null) {
  return useFetch(
    () => id ? api.experiments.alerts(id).then(r => r.alerts) : Promise.resolve([] as AlertEvent[]),
    [id]
  )
}

export function useClients(id: string | null) {
  return useFetch(
    () => id ? api.experiments.clients(id).then(r => r.clients) : Promise.resolve([] as ClientStat[]),
    [id]
  )
}

export function useMetrics(id: string | null, names: string[]) {
  return useFetch(
    () => id && names.length ? api.experiments.metrics(id, names) : Promise.resolve(null as unknown as MetricSeries),
    [id, names.join(',')]
  )
}

export function useConfig(id: string | null) {
  return useFetch(
    () => id ? api.experiments.config(id).then(r => r.config) : Promise.resolve({} as Record<string, unknown>),
    [id]
  )
}

export function useHealth() {
  return useFetch(() => api.health(), [])
}
