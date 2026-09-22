"""Extract an E1M1 automap (wall polylines + exit location) from a WAD.

Parses the classic DOOM map lumps (VERTEXES / LINEDEFS) for E1M1 of the
bundled freedoom1.wad and writes e1m1_map.json next to this file:
{"bounds": [minx, miny, maxx, maxy], "lines": [[x1,y1,x2,y2], ...],
 "exit": [x, y]} — the exit is the midpoint of the first exit-special
linedef (11/52/51/97). Used by the live server (goal/victory geometry) and
copied to web/public/ for the minimap. Not biology; level geometry.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

EXIT_SPECIALS = {11, 52, 51, 97, 124, 197, 198}
SECRET_TAGS = {9}  # secret-sector tag on lines is ignored; specials above only


def extract(wad_path: Path, mapname: str = "E1M1") -> dict:
    data = wad_path.read_bytes()
    n_lumps, dir_off = struct.unpack_from("<ii", data, 4)
    lumps = {}
    order = []
    for i in range(n_lumps):
        off, size, name = struct.unpack_from("<ii8s", data, dir_off + i * 16)
        name = name.rstrip(b"\0").decode("ascii", "replace")
        lumps.setdefault(name, []).append((off, size))
        order.append(name)
    mi = order.index(mapname)
    map_lumps = {}
    for name in order[mi + 1:]:
        if name in ("THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS",
                    "SSECTORS", "NODES", "SECTORS", "REJECT", "BLOCKMAP"):
            map_lumps[name] = lumps[name].pop(0)
        else:
            break
    voff, vsize = map_lumps["VERTEXES"]
    verts = [struct.unpack_from("<hh", data, voff + i * 4)
             for i in range(vsize // 4)]
    loff, lsize = map_lumps["LINEDEFS"]
    lines, exit_pt = [], None
    for i in range(lsize // 14):
        v1, v2, _flags, special, _tag, _s0, _s1 = struct.unpack_from(
            "<hhhhhhh", data, loff + i * 14)
        x1, y1 = verts[v1]
        x2, y2 = verts[v2]
        lines.append([x1, y1, x2, y2])
        if exit_pt is None and special in EXIT_SPECIALS:
            exit_pt = [(x1 + x2) // 2, (y1 + y2) // 2]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    return {"map": mapname,
            "bounds": [min(xs), min(ys), max(xs), max(ys)],
            "lines": lines,
            "exit": exit_pt}


def main() -> None:
    import vizdoom as vzd
    wad = Path(vzd.__file__).parent / "freedoom1.wad"
    out = extract(wad)
    dest = Path(__file__).parent / "e1m1_map.json"
    dest.write_text(json.dumps(out))
    web = Path(__file__).parents[3] / "web" / "public" / "e1m1-map.json"
    if web.parent.is_dir():
        web.write_text(json.dumps(out))
    print(f"{dest}: {len(out['lines'])} linedefs, exit={out['exit']}, "
          f"bounds={out['bounds']}" + (f" (+web copy)" if web.exists() else ""))


if __name__ == "__main__":
    sys.exit(main())
