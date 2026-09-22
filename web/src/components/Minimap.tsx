import {useEffect, useMemo, useState} from 'react';

type MapData = {bounds: [number, number, number, number]; lines: [number, number, number, number][]; exit: [number, number] | null};

let cache: MapData | null = null;
let pending: Promise<MapData | null> | null = null;

function loadMap(): Promise<MapData | null> {
  if (cache) return Promise.resolve(cache);
  if (!pending) {
    pending = fetch('/e1m1-map.json')
      .then(r => (r.ok ? r.json() as Promise<MapData> : null))
      .then(m => { cache = m; return m; })
      .catch(() => null);
  }
  return pending;
}

const W = 220, H = 170, PAD = 8;

export default function Minimap({x, y, goalDist}: {x?: number; y?: number; goalDist?: number}) {
  const [map, setMap] = useState<MapData | null>(cache);
  useEffect(() => { if (!map) void loadMap().then(setMap); }, [map]);

  const project = useMemo(() => {
    if (!map) return null;
    const [minx, miny, maxx, maxy] = map.bounds;
    const scale = Math.min((W - 2 * PAD) / (maxx - minx), (H - 2 * PAD) / (maxy - miny));
    return {minx, miny, scale};
  }, [map]);

  if (!map || !project) return null;
  const px = (gx: number) => PAD + (gx - project.minx) * project.scale;
  const py = (gy: number) => H - PAD - (gy - project.miny) * project.scale; // DOOM y is north-up

  return <div className="minimap">
    <small>E1M1 · EXIT {goalDist != null ? `${goalDist.toFixed(0)}u` : '—'}</small>
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="auto" aria-label="E1M1 automap with player position">
      {map.lines.map((l, i) => <line key={i} x1={px(l[0])} y1={py(l[1])} x2={px(l[2])} y2={py(l[3])} stroke="#2f4a5e" strokeWidth={0.7}/>)}
      {map.exit && <circle cx={px(map.exit[0])} cy={py(map.exit[1])} r={4} fill="none" stroke="#e8b23a" strokeWidth={1.4}/>}
      {x != null && y != null && <circle cx={px(x)} cy={py(y)} r={2.6} fill="#7ec8ff"/>}
    </svg>
  </div>;
}
