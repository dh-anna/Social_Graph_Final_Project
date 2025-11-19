export interface Node {
  id: string
  name: string
  degree?: number
  color?: string
  type?: 'actor' | 'director'
}

export type ColorMode = 'degree' | 'type'

export interface GraphSettings {
  separateByType: boolean
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