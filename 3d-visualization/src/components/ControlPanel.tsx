import React from 'react'
import { type ColorMode } from '../types'

interface ControlPanelProps {
  minDegree: number
  maxDegree: number
  onMinDegreeChange: (value: number) => void
  showLinkLabels: boolean
  onShowLinkLabelsChange: (value: boolean) => void
  colorMode: ColorMode
  onColorModeChange: (value: ColorMode) => void
  separateByType: boolean
  onSeparateByTypeChange: (value: boolean) => void
  nodeCount: number
  linkCount: number
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  minDegree,
  maxDegree,
  onMinDegreeChange,
  showLinkLabels,
  onShowLinkLabelsChange,
  colorMode,
  onColorModeChange,
  separateByType,
  onSeparateByTypeChange,
  nodeCount,
  linkCount
}) => {
  return (
    <div style={{
      position: 'absolute',
      top: 20,
      left: 20,
      zIndex: 1000,
      background: 'rgba(0, 0, 0, 0.8)',
      padding: '15px 20px',
      borderRadius: 8,
      color: '#fff',
      minWidth: 250
    }}>
      <h3 style={{ margin: '0 0 15px 0', fontSize: 14 }}>Controls</h3>

      <div style={{ marginBottom: 10 }}>
        <label style={{ display: 'block', marginBottom: 5, fontSize: 12 }}>
          Min Node Degree: {minDegree}
        </label>
        <input
          type="range"
          min={0}
          max={maxDegree}
          value={minDegree}
          onChange={(e) => onMinDegreeChange(Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      <div style={{ marginBottom: 10 }}>
        <label style={{ display: 'flex', alignItems: 'center', fontSize: 12, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showLinkLabels}
            onChange={(e) => onShowLinkLabelsChange(e.target.checked)}
            style={{ marginRight: 8 }}
          />
          Show Link Weight Labels
        </label>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label style={{ display: 'block', marginBottom: 5, fontSize: 12 }}>
          Color By:
        </label>
        <select
          value={colorMode}
          onChange={(e) => onColorModeChange(e.target.value as ColorMode)}
          style={{
            width: '100%',
            padding: '4px 8px',
            borderRadius: 4,
            border: 'none',
            background: '#333',
            color: '#fff',
            fontSize: 12
          }}
        >
          <option value="degree">Degree (Viridis)</option>
          <option value="type">Type (Actor/Director)</option>
        </select>
      </div>

      <div style={{ marginBottom: 10 }}>
        <label style={{ display: 'flex', alignItems: 'center', fontSize: 12, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={separateByType}
            onChange={(e) => onSeparateByTypeChange(e.target.checked)}
            style={{ marginRight: 8 }}
          />
          Separate Actors/Directors
        </label>
      </div>

      <div style={{ fontSize: 11, color: '#888' }}>
        Nodes: {nodeCount.toLocaleString()} |
        Links: {linkCount.toLocaleString()}
      </div>
    </div>
  )
}