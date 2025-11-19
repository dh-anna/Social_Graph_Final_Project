import { useState } from 'react'
import { ControlPanel } from './components/ControlPanel'
import { GraphViewer } from './components/GraphViewer'
import { useGraphData } from './hooks/useGraphData'
import { useWindowSize } from './hooks/useWindowSize'
import { useFilteredGraph } from './hooks/useFilteredGraph'
import { type ColorMode } from './types'
import './App.css'

function App() {
  const [minDegree, setMinDegree] = useState(0)
  const [showLinkLabels, setShowLinkLabels] = useState(false)
  const [colorMode, setColorMode] = useState<ColorMode>('degree')
  const [separateByType, setSeparateByType] = useState(false)
  const dimensions = useWindowSize()

  const { graphData, maxDegree, isLoading, error } = useGraphData(
    import.meta.env.BASE_URL + 'graph.json.gz'
  )

  const filteredData = useFilteredGraph(graphData, minDegree)

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
        nodeCount={filteredData.nodes.length}
        linkCount={filteredData.links.length}
      />

      <GraphViewer
        graphData={filteredData}
        maxDegree={maxDegree}
        width={dimensions.width}
        height={dimensions.height}
        showLinkLabels={showLinkLabels}
        colorMode={colorMode}
        separateByType={separateByType}
      />
    </div>
  )
}

export default App