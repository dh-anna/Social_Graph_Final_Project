import React, {useEffect, useRef} from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import SpriteText from 'three-spritetext'
// @ts-expect-error d3-force-3d don't have types
import {forceX, forceY} from 'd3-force-3d'
import {type ColorMode, type GraphData} from '../types'
import {getViridisColor} from '../utils/colors'

interface GraphViewerProps {
  graphData: GraphData
  maxDegree: number
  width: number
  height: number
  showLinkLabels: boolean
  colorMode: ColorMode
  separateByType: boolean
}

// Colors for actor/director types
const TYPE_COLORS = {
  actor: '#4CAF50',    // Green
  director: '#FF9800'  // Orange
}

export const GraphViewer: React.FC<GraphViewerProps> = ({
  graphData,
  maxDegree,
  width,
  height,
  showLinkLabels,
  colorMode,
  separateByType
}) => {
  // @ts-expect-error We allow any
  const fgRef = useRef<any>()

  // Apply separation force when enabled
  useEffect(() => {
    // Wait for graph to be fully initialized
    const timeoutId = setTimeout(() => {
      if (!fgRef.current) return

      const fg = fgRef.current

      try {
        if (separateByType) {
          // Push actors and directors to opposite sides on X axis
          fg.d3Force('x', forceX((node: any) => {
            return node.type === 'actor' ? -200 : node.type === 'director' ? 200 : 0
          }).strength(0.3))

          // Optional: also separate on Y axis for better visibility
          fg.d3Force('y', forceY((node: any) => {
            return node.type === 'actor' ? -100 : node.type === 'director' ? 100 : 0
          }).strength(0.1))
        } else {
          // Use neutral forces instead of null to avoid breaking simulation
          fg.d3Force('x', forceX(0).strength(0))
          fg.d3Force('y', forceY(0).strength(0))
        }

        // Reheat the simulation to apply changes
        fg.d3ReheatSimulation()
      } catch (e) {
        console.warn('Could not apply separation force:', e)
      }
    }, 100)

    return () => clearTimeout(timeoutId)
  }, [separateByType])

  return (
    <ForceGraph3D
      ref={fgRef}
      graphData={graphData}
      width={width}
      height={height}
      // Performance optimizations
      enableNodeDrag={true}
      // Node styling based on degree
      nodeVal={(node: any) => Math.min((node.degree || 1) / 10, 50)}
      nodeColor={(node: any) => {
        if (colorMode === 'type') {
          return TYPE_COLORS[node.type as keyof typeof TYPE_COLORS] || '#999999'
        }
        const degree = node.degree || 0
        const t = Math.min(degree / maxDegree, 1)
        return getViridisColor(t)
      }}
      nodeOpacity={0.9}
      nodeLabel={(node: any) => `${node.name}<br>Type: ${node.type || 'unknown'}<br>Connections: ${node.degree || 0}`}
      // Link styling
      linkWidth={(link: any) => Math.min((link.degree || 1) * 0.5, 5)}
      linkOpacity={0.5}
      linkColor={() => 'rgba(125,125,125,0.5)'}
      // Link labels
      linkThreeObjectExtend={true}
      linkThreeObject={showLinkLabels ? (link: any) => {
        const sprite = new SpriteText(`${link.degree}`)
        sprite.color = '#ffffff'
        sprite.textHeight = 3
        return sprite
      } : undefined}
      linkPositionUpdate={showLinkLabels ? (sprite: any, { start, end }: any) => {
        if (!sprite || !sprite.position) return
        const middlePos = {
          x: start.x + (end.x - start.x) / 2,
          y: start.y + (end.y - start.y) / 2,
          z: start.z + (end.z - start.z) / 2
        }
        Object.assign(sprite.position, middlePos)
      } : undefined}
    />
  )
}
