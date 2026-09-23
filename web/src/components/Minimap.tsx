import {useEffect, useMemo, useState} from 'react';

type MapData = {
  map?: string;
  bounds: [number, number, number, number];
  lines: [number, number, number, number][];
  exit: [number, number] | null;
};

let cache: MapData | null = null;
let pending: Promise<MapData | null> | null = null;

function loadMap(): Promise<MapData | null> {
  if (cache) return Promise.resolve(cache);
  if (!pending) {
    pending = fetch(new URL('./e1m1-map.json', window.location.href))
      .then(response => response.ok ? response.json() as Promise<MapData> : null)
      .then(map => { cache = map; return map; })
      .catch(() => null);
  }
  return pending;
}

const WIDTH = 300;
const PAD = 10;

export default function Minimap({x, y, goalDist}: {x?: number; y?: number; goalDist?: number}) {
  const [map, setMap] = useState<MapData | null>(cache);
  useEffect(() => { if (!map) void loadMap().then(setMap); }, [map]);

  const projection = useMemo(() => {
    if (!map) return null;
    const [minX, minY, maxX, maxY] = map.bounds;
    const scale = (WIDTH - PAD * 2) / Math.max(1, maxX - minX);
    const height = (maxY - minY) * scale + PAD * 2;
    return {minX, minY, maxY, scale, height};
  }, [map]);

  if (!map || !projection) return null;
  const px = (worldX: number) => PAD + (worldX - projection.minX) * projection.scale;
  const py = (worldY: number) => PAD + (projection.maxY - worldY) * projection.scale;

  return <aside className="minimap" aria-label="E1M1 automap">
    <div className="minimap-head"><b>E1M1 AUTOMAP</b><span>EXIT {goalDist != null ? `${goalDist.toFixed(0)}u` : '—'}</span></div>
    <svg viewBox={`0 0 ${WIDTH} ${projection.height}`} role="img" aria-label="E1M1 walls, exit and current player position" preserveAspectRatio="xMidYMid meet">
      <g className="minimap-walls">
        {map.lines.map((line, index) => <line key={index} x1={px(line[0])} y1={py(line[1])} x2={px(line[2])} y2={py(line[3])}/>)}
      </g>
      {map.exit && <g className="minimap-exit">
        <circle cx={px(map.exit[0])} cy={py(map.exit[1])} r={6}/>
        <circle cx={px(map.exit[0])} cy={py(map.exit[1])} r={2}/>
      </g>}
      {x != null && y != null && <g className="minimap-player">
        <circle cx={px(x)} cy={py(y)} r={4.3}/>
        <circle cx={px(x)} cy={py(y)} r={1.6}/>
      </g>}
    </svg>
  </aside>;
}
