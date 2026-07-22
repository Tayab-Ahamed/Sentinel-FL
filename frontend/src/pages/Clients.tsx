import React from 'react'
import { useClients } from '../hooks/useExperiment'
import ClientTable from '../components/Table/ClientTable'
import { Spinner, EmptyState, ErrorState } from './Dashboard'
import './Pages.css'

interface ClientsProps { experimentId: string | null }

export default function Clients({ experimentId }: ClientsProps) {
  const { data: clients, loading, error } = useClients(experimentId)

  if (!experimentId) return <EmptyState msg="Select an experiment from the sidebar." />
  if (loading) return <Spinner />
  if (error) return <ErrorState msg={error} />

  const suspicious = (clients ?? []).filter(c => c.is_suspicious)
  const clean = (clients ?? []).filter(c => !c.is_suspicious)

  return (
    <div className="page fade-in">
      <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <div className="card" style={{ textAlign: 'center' }}>
          <div className="section-title">Total Clients</div>
          <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--teal)' }}>{(clients ?? []).length}</div>
        </div>
        <div className="card" style={{ textAlign: 'center' }}>
          <div className="section-title">Suspicious</div>
          <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--red)' }}>{suspicious.length}</div>
        </div>
        <div className="card" style={{ textAlign: 'center' }}>
          <div className="section-title">Active / Clean</div>
          <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--teal)' }}>{clean.length}</div>
        </div>
      </div>

      <div className="card">
        <div className="section-title">Client Trust Scores</div>
        <ClientTable clients={clients ?? []} />
      </div>
    </div>
  )
}
