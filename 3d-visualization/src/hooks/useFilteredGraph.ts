import { useMemo } from 'react'
import { type GraphData } from '../types'

export function useFilteredGraph(graphData: GraphData | null, minDegree: number): GraphData | null {
  return useMemo(() => {
    if (!graphData) return null

    const filteredNodes = graphData.nodes
      .filter(node => (node.degree || 0) >= minDegree)
      .map(node => ({ ...node })) // Create fresh copies

    const nodeIds = new Set(filteredNodes.map(n => n.id))

    const filteredLinks = graphData.links
      .filter(link => {
        // Handle both string IDs and object references
        const sourceId = typeof link.source === 'object' ? (link.source as any).id : link.source
        const targetId = typeof link.target === 'object' ? (link.target as any).id : link.target
        return nodeIds.has(sourceId) && nodeIds.has(targetId)
      })
      .map(link => ({
        // Always return string IDs for new filtered data
        source: typeof link.source === 'object' ? (link.source as any).id : link.source,
        target: typeof link.target === 'object' ? (link.target as any).id : link.target,
        degree: link.degree
      }))

    return { nodes: filteredNodes, links: filteredLinks }
  }, [graphData, minDegree])
}
