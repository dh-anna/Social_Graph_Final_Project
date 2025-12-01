import React from 'react'
import { type ColorMode } from '../types'
import { type ClusterNames } from '../hooks/useClusterNames'
import { getClusterColor, getViridisColor } from '../utils/colors'

const TYPE_COLORS: Record<string, string> = {
  actor: '#4CAF50',
  director: '#FF9800'
}

interface LegendPanelProps {
  colorMode: ColorMode
  clusterNames: ClusterNames
  maxDegree: number
}

export const LegendPanel: React.FC<LegendPanelProps> = ({
  colorMode,
  clusterNames,
  maxDegree
}) => {
  return (
    <div style={{
      position: 'absolute',
      top: 20,
      right: 20,
      zIndex: 1000,
      background: 'rgba(0, 0, 0, 0.8)',
      padding: '15px 20px',
      borderRadius: 8,
      color: '#fff',
      minWidth: 200,
      maxHeight: 400,
      overflowY: 'auto'
    }}>
      <h3 style={{ margin: '0 0 15px 0', fontSize: 14 }}>Legend</h3>

      {colorMode === 'cluster' && (
        <ClusterLegend clusterNames={clusterNames} />
      )}

      {colorMode === 'type' && (
        <TypeLegend />
      )}

      {colorMode === 'degree' && (
        <DegreeLegend maxDegree={maxDegree} />
      )}
    </div>
  )
}

const ClusterLegend: React.FC<{ clusterNames: ClusterNames }> = ({ clusterNames }) => {
  const clusterIds = Object.keys(clusterNames).sort((a, b) => Number(a) - Number(b))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {clusterIds.map((id) => (
        <div key={id} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div
            style={{
              width: 12,
              height: 12,
              borderRadius: 2,
              backgroundColor: getClusterColor(Number(id)),
              flexShrink: 0
            }}
          />
          <span style={{ fontSize: 11, lineHeight: 1.2 }}>
            Cluster_{id}: {clusterNames[id]}
          </span>
        </div>
      ))}
    </div>
  )
}

const TypeLegend: React.FC = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div
          style={{
            width: 12,
            height: 12,
            borderRadius: 2,
            backgroundColor: TYPE_COLORS.actor
          }}
        />
        <span style={{ fontSize: 12 }}>Actor</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div
          style={{
            width: 12,
            height: 12,
            borderRadius: 2,
            backgroundColor: TYPE_COLORS.director
          }}
        />
        <span style={{ fontSize: 12 }}>Director</span>
      </div>
    </div>
  )
}

const DegreeLegend: React.FC<{ maxDegree: number }> = ({ maxDegree }) => {
  // Generate gradient stops for Viridis
  const gradientStops = Array.from({ length: 10 }, (_, i) => {
    const t = i / 9
    return `${getViridisColor(t)} ${t * 100}%`
  }).join(', ')

  return (
    <div>
      <div
        style={{
          width: '100%',
          height: 16,
          borderRadius: 4,
          background: `linear-gradient(to right, ${gradientStops})`
        }}
      />
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        marginTop: 4,
        fontSize: 11,
        color: '#888'
      }}>
        <span>0</span>
        <span>{maxDegree}</span>
      </div>
      <div style={{ fontSize: 11, color: '#888', marginTop: 4, textAlign: 'center' }}>
        Node Degree
      </div>
    </div>
  )
}
