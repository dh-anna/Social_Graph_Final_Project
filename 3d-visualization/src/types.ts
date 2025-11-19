export interface Node {
  id: string
  name: string
  degree?: number
  color?: string
}

export interface Link {
  source: string
  target: string
  degree: number
}

export interface GraphData {
  nodes: Node[]
  links: Link[]
}