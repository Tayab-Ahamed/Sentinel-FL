import React from 'react'
import { useExperimentList } from '../hooks/useExperiment'
import { Spinner, EmptyState } from './Dashboard'
import './Pages.css'

export default function ModelDownload() {
  const { data: experiments, loading } = useExperimentList()

  if (loading) return <Spinner />
  if (!experiments?.length) return <EmptyState msg="No experiments found." />

  return (
    <div className="page fade-in">
      <div className="card">
        <div className="section-title">Experiment Checkpoints</div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 20 }}>
          Trained model checkpoints are stored under <code style={{ color: 'var(--teal)' }}>experiments/checkpoints/</code>.
          Download links point to the experiment artefact JSON.
        </p>
        <div className="model-grid">
          {experiments.map(exp => (
            <div className="card model-card" key={exp.experiment_id}>
              <div className="model-card-title">{exp.experiment_id}</div>
              <div className="model-card-meta">
                <span>Phase: {exp.dataset_phase}</span>
                <span>Layers: {exp.layers_enabled.join(', ')}</span>
                {exp.result?.clean_accuracy != null && (
                  <span style={{ color: 'var(--teal)' }}>
                    C-Acc: {(exp.result.clean_accuracy * 100).toFixed(1)}%
                  </span>
                )}
                {exp.result?.attack_success_rate != null && (
                  <span style={{ color: 'var(--red)' }}>
                    ASR: {(exp.result.attack_success_rate * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              <a
                className="dl-btn"
                href={`/api/v1/experiments/${exp.experiment_id}`}
                target="_blank"
                rel="noreferrer"
              >
                ⬇ Download Artefact JSON
              </a>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
