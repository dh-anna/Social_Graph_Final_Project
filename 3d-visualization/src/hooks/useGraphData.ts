import { useEffect, useState } from 'react'
import { type GraphData } from '../types'

interface UseGraphDataReturn {
  graphData: GraphData | null
  maxDegree: number
  isLoading: boolean
  error: string | null
}

async function fetchAndDecompress(url: string): Promise<any> {
  const response = await fetch(url)

  if (!response.ok) {
    throw new Error(`Failed to fetch: ${response.status}`)
  }

  const contentType = response.headers.get('Content-Type') || ''

  // If server says it's gzip, decompress it
  if (contentType.includes('gzip') || contentType.includes('octet-stream')) {
    const arrayBuffer = await response.arrayBuffer()
    const decompressedStream = new Response(arrayBuffer).body!
      .pipeThrough(new DecompressionStream('gzip'))
    const decompressedResponse = new Response(decompressedStream)
    return decompressedResponse.json()
  }

  // Otherwise treat as regular JSON
  return response.json()
}

export function useGraphData(url: string): UseGraphDataReturn {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [maxDegree, setMaxDegree] = useState(100)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setIsLoading(true)
    setError(null)

    fetchAndDecompress(url)
      .then((data) => {
        // Calculate node degrees from links
        const nodeDegrees: Record<string, number> = {}
        data.links.forEach((link: any) => {
          nodeDegrees[link.source] = (nodeDegrees[link.source] || 0) + 1
          nodeDegrees[link.target] = (nodeDegrees[link.target] || 0) + 1
        })

        const transformedData: GraphData = {
          nodes: data.nodes.map((node: any) => ({
            id: node.id,
            name: node.name || node.id,
            degree: nodeDegrees[node.id] || 0,
            ...node
          })),
          links: data.links.map((link: any) => ({
            source: link.source,
            target: link.target,
            degree: link.degree || link.weight || 1
          }))
        }

        // Set max degree for slider
        const max = Math.max(...Object.values(nodeDegrees))
        setMaxDegree(max)
        setGraphData(transformedData)
        setIsLoading(false)
      })
      .catch(err => {
        console.error('Failed to load graph:', err)
        setError(err.message)
        setIsLoading(false)
      })
  }, [url])

  return { graphData, maxDegree, isLoading, error }
}
