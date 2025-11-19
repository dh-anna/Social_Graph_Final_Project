// Viridis-like color scale
export function getViridisColor(t: number): string {
  // t should be between 0 and 1
  const colors = [
    [68, 1, 84],
    [72, 40, 120],
    [62, 74, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [110, 206, 88],
    [181, 222, 43],
    [253, 231, 37]
  ]

  const idx = Math.min(Math.floor(t * (colors.length - 1)), colors.length - 2)
  const frac = t * (colors.length - 1) - idx

  const r = Math.round(colors[idx][0] + frac * (colors[idx + 1][0] - colors[idx][0]))
  const g = Math.round(colors[idx][1] + frac * (colors[idx + 1][1] - colors[idx][1]))
  const b = Math.round(colors[idx][2] + frac * (colors[idx + 1][2] - colors[idx][2]))

  return `rgb(${r},${g},${b})`
}