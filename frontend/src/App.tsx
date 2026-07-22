import React, { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import Sidebar from './components/Layout/Sidebar'
import TopBar from './components/Layout/TopBar'
import Dashboard from './pages/Dashboard'
import Clients from './pages/Clients'
import HeatmapPage from './pages/HeatmapPage'
import AlertsPage from './pages/AlertsPage'
import RoundsPage from './pages/RoundsPage'
import AttackViz from './pages/AttackViz'
import ModelDownload from './pages/ModelDownload'
import ConfigPage from './pages/ConfigPage'
import { api } from './api/client'
import './App.css'

const PAGE_TITLES: Record<string, string> = {
  '/':        'Dashboard',
  '/clients': 'Clients',
  '/heatmap': 'Trust Heatmap',
  '/alerts':  'Alerts',
  '/rounds':  'Round Statistics',
  '/attack':  'Attack Visualization',
  '/model':   'Model Download',
  '/config':  'Configuration',
}

function AppShell() {
  const [experimentId, setExperimentId] = useState<string | null>(null)
  const location = useLocation()
  const title = PAGE_TITLES[location.pathname] ?? 'SENTINEL-FL'

  // Auto-select first experiment on load
  useEffect(() => {
    api.experiments.list().then(r => {
      if (r.experiments.length > 0 && !experimentId) {
        setExperimentId(r.experiments[0].experiment_id)
      }
    }).catch(() => {})
  }, [])

  return (
    <div className="app-layout">
      <Sidebar experimentId={experimentId} onExperimentChange={setExperimentId} />
      <div className="app-main">
        <TopBar title={title} experimentId={experimentId} />
        <main className="app-content">
          <Routes>
            <Route path="/"        element={<Dashboard    experimentId={experimentId} />} />
            <Route path="/clients" element={<Clients      experimentId={experimentId} />} />
            <Route path="/heatmap" element={<HeatmapPage  experimentId={experimentId} />} />
            <Route path="/alerts"  element={<AlertsPage   experimentId={experimentId} />} />
            <Route path="/rounds"  element={<RoundsPage   experimentId={experimentId} />} />
            <Route path="/attack"  element={<AttackViz    experimentId={experimentId} />} />
            <Route path="/model"   element={<ModelDownload />} />
            <Route path="/config"  element={<ConfigPage   experimentId={experimentId} />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  )
}
