import { useEffect, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import SpriteText from 'three-spritetext'
import * as THREE from 'three'
import './App.css'

interface GraphData {
  nodes: { id: string; name: string }[]
  links: { source: string; target: string; degree: number }[]
}

function App() {
  const [graphData, setGraphData] = useState<GraphData | null>(null)

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + 'graph-data.json')
      .then(res => res.json())
      .then(data => setGraphData(data))
      .catch(err => console.error('Failed to load graph:', err))
  }, [])

  if (!graphData) return <div>Loading...</div>

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ForceGraph3D
        graphData={graphData}
        nodeAutoColorBy="id"
        linkWidth={(link: any) => link.degree}
        nodeThreeObject={(node: any) => {
          const group = new THREE.Group()

          // Sphere
          const geometry = new THREE.SphereGeometry(5)
          const material = new THREE.MeshLambertMaterial({
            color: node.color || '#ffffff',
            transparent: true,
            opacity: 0.75
          })
          const sphere = new THREE.Mesh(geometry, material)
          group.add(sphere)

          // Label above sphere
          const sprite = new SpriteText(node.name)
          sprite.color = node.color || '#ffffff'
          sprite.textHeight = 6
          sprite.position.y = 10
          group.add(sprite)

          return group
        }}
        linkThreeObjectExtend={true}
        linkThreeObject={(link: any) => {
          const sprite = new SpriteText(`${link.degree}`)
          sprite.color = '#ffffff'
          sprite.textHeight = 4
          return sprite
        }}
        linkPositionUpdate={(sprite: any, { start, end }: any) => {
          const middlePos = {
            x: start.x + (end.x - start.x) / 2,
            y: start.y + (end.y - start.y) / 2,
            z: start.z + (end.z - start.z) / 2
          }
          Object.assign(sprite.position, middlePos)
        }}
      />
    </div>
  )
}

export default App
