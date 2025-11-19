import React, { useRef } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import SpriteText from 'three-spritetext'
import { type GraphData } from '../types'
import { getViridisColor } from '../utils/colors'

interface GraphViewerProps {
  graphData: GraphData
  maxDegree: number
  width: number
  height: number
  showLinkLabels: boolean
}

export const GraphViewer: React.FC<GraphViewerProps> = ({
  graphData,
  maxDegree,
  width,
  height,
  showLinkLabels
}) => {
  // @ts-expect-error We allow any
  const fgRef = useRef<any>()

  return (
    <ForceGraph3D
      key={showLinkLabels ? 'with-labels' : 'without-labels'}
      ref={fgRef}
      graphData={graphData}
      width={width}
      height={height}
      // Performance optimizations
      enableNodeDrag={true}
      // Node styling based on degree
      nodeVal={(node: any) => Math.min((node.degree || 1) / 10, 50)}
      nodeColor={(node: any) => {
        const degree = node.degree || 0
        const t = Math.min(degree / maxDegree, 1)
        return getViridisColor(t)
      }}
      nodeOpacity={0.9}
      nodeLabel={(node: any) => `${node.name}<br>Connections: ${node.degree || 0}`}
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
