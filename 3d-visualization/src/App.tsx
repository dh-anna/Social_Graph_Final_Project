import { useState, useEffect, useCallback, useRef } from 'react'
import { ControlPanel } from './components/ControlPanel'
import { GraphViewer } from './components/GraphViewer'
import { LegendPanel } from './components/LegendPanel'
import { useGraphData } from './hooks/useGraphData'
import { useWindowSize } from './hooks/useWindowSize'
import { useFilteredGraph } from './hooks/useFilteredGraph'
import { useClusterNames } from './hooks/useClusterNames'
import { type ColorMode } from './types'
import './App.css'

function App() {
  const [minDegree, setMinDegree] = useState(0)
  const [showLinkLabels, setShowLinkLabels] = useState(false)
  const [colorMode, setColorMode] = useState<ColorMode>('degree')
  const [separateByType, setSeparateByType] = useState(false)
  const [separateByCluster, setSeparateByCluster] = useState(false)
  const [enabledClusters, setEnabledClusters] = useState<Set<number>>(new Set())
  const dimensions = useWindowSize()

  const { graphData, maxDegree, isLoading, error } = useGraphData(
    import.meta.env.BASE_URL + 'graph.json.gz'
  )
  const { clusterNames } = useClusterNames(
    import.meta.env.BASE_URL + 'cluster_names.json'
  )

  const clustersInitialized = useRef(false)

  useEffect(() => {
    const clusterIds = Object.keys(clusterNames)
    if (clusterIds.length > 0 && !clustersInitialized.current) {
      clustersInitialized.current = true
      setEnabledClusters(new Set(clusterIds.map(Number)))
    }
  }, [clusterNames])

  const handleClusterToggle = useCallback((clusterId: number) => {
    setEnabledClusters(prev => {
      const next = new Set(prev)
      if (next.has(clusterId)) {
        next.delete(clusterId)
      } else {
        next.add(clusterId)
      }
      return next
    })
  }, [])

  const handleSelectAllClusters = useCallback(() => {
    setEnabledClusters(new Set(Object.keys(clusterNames).map(Number)))
  }, [clusterNames])

  const handleDeselectAllClusters = useCallback(() => {
    setEnabledClusters(new Set())
  }, [])

  const filteredData = useFilteredGraph(graphData, minDegree, enabledClusters)

  if (isLoading) return <div>Loading...</div>
  if (error) return <div>Error: {error}</div>
  if (!filteredData) return <div>No data</div>

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ControlPanel
        minDegree={minDegree}
        maxDegree={maxDegree}
        onMinDegreeChange={setMinDegree}
        showLinkLabels={showLinkLabels}
        onShowLinkLabelsChange={setShowLinkLabels}
        colorMode={colorMode}
        onColorModeChange={setColorMode}
        separateByType={separateByType}
        onSeparateByTypeChange={setSeparateByType}
        separateByCluster={separateByCluster}
        onSeparateByClusterChange={setSeparateByCluster}
        nodeCount={filteredData.nodes.length}
        linkCount={filteredData.links.length}
      />

      <LegendPanel
        colorMode={colorMode}
        clusterNames={clusterNames}
        maxDegree={maxDegree}
        enabledClusters={enabledClusters}
        onClusterToggle={handleClusterToggle}
        onSelectAll={handleSelectAllClusters}
        onDeselectAll={handleDeselectAllClusters}
      />

      <GraphViewer
        graphData={filteredData}
        maxDegree={maxDegree}
        width={dimensions.width}
        height={dimensions.height}
        showLinkLabels={showLinkLabels}
        colorMode={colorMode}
        separateByType={separateByType}
        separateByCluster={separateByCluster}
      />
    </div>
  )
}

export default App