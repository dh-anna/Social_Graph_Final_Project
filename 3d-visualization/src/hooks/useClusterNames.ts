import { useState, useEffect } from 'react'

export type ClusterNames = Record<string, string>

export function useClusterNames(url: string) {
  const [clusterNames, setClusterNames] = useState<ClusterNames>({})
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadClusterNames = async () => {
      try {
        setIsLoading(true)
        const response = await fetch(url)
        if (!response.ok) {
          throw new Error(`Failed to load cluster names: ${response.statusText}`)
        }
        const data = await response.json()
        setClusterNames(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setIsLoading(false)
      }
    }

    loadClusterNames()
  }, [url])

  return { clusterNames, isLoading, error }
}
