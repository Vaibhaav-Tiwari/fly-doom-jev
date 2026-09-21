"""Generate the fly_arena scenario WAD (derived from ViZDoom's defend_the_line.wad).

Why this exists (documented deviation from stock scenarios): stock ViZDoom
combat scenarios are unsurvivable past ~30 s even for a perfect-aim oracle at
doom_skill 1 (measured in docs/BENCHMARKS.md): defend_the_center dies in
8-12 s, defend_the_line in ~20-36 s, deadly_corridor in ~10-19 s. The demo
needs >=60 s of visible fighting, so we generate a gentler custom arena:

  * same room geometry as defend_the_line (UDMF TEXTMAP, kept verbatim)
  * a small, hand-placed set of monsters (imps = ranged, demons = melee)
    instead of the ACS wave script (BEHAVIOR/SCRIPTS lumps dropped)
  * MAPINFO with AllowMonsterRespawn so kills keep coming all episode

Provenance: derived from defend_the_line.wad, part of ViZDoom
(https://github.com/Farama-Foundation/ViZDoom, MIT license) — MIT permits
derivatives with notice; see PROVENANCE.md / THIRD_PARTY.md.

The generated WAD is tiny (~2 KB) and committed to data/scenarios/ so runs
are reproducible without network access. Regenerate with:

    python -m flydoom.scenarios.make_fly_arena
"""

from __future__ import annotations

import os
import struct

import vizdoom as vzd

HERE = os.path.dirname(__file__)
SRC_WAD = os.path.join(os.path.dirname(vzd.__file__), "scenarios",
                       "defend_the_line.wad")
OUT_WAD = os.path.join(HERE, "fly_arena.wad")
OUT_CFG = os.path.join(HERE, "fly_arena.cfg")

# Monster placement: (type_id, x, y, angle_deg, ambush)
# Type ids are the custom DECORATE classes below (weak variants of imp/demon,
# mirroring stock dtl's scripted hp=1 design so kills come in ones and twos).
# Room spans x in [-512, 128], y in [64, 512]; player starts at (-480, 288)
# facing east (angle 0). Ranged imps along the east wall like the original,
# melee demons beside them so melee pressure arrives gradually. Counts are
# deliberately modest (skill-1 halved damage + respawn).
MONSTERS = [
    (30101,  64,  96, 180, False),   # FlyImp row (east wall), like stock dtl
    (30101,  64, 200, 180, False),
    (30101,  64, 288, 180, False),
    (30101,  64, 376, 180, False),
    (30101,  64, 480, 180, False),
    (30102,  32, 128, 180, False),   # FlyDemons at the east wall
    (30102,  32, 448, 180, False),
]

# Weak monster variants so a slow neural controller gets kills (stock
# defend_the_line scripted hp=1; DECORATE is the script-free equivalent).
# doomednums 30101/30102 are unused by Doom II / the stock scenarios.
DECORATE = """actor FlyImp : DoomImp 30101
{
    Health 20
}
actor FlyDemon : Demon 30102
{
    Health 50
    Speed 7
}
"""

MAPINFO = """map MAP01 "fly_arena"
{
    allowmonsterrespawn = true
}
"""


def _read_lumps(path: str) -> dict[str, bytes]:
    data = open(path, "rb").read()
    numlumps, diroff = struct.unpack("<II", data[4:12])
    lumps: dict[str, bytes] = {}
    for i in range(numlumps):
        off, size, name = struct.unpack("<II8s", data[diroff + i * 16:
                                                    diroff + (i + 1) * 16])
        lumps[name.rstrip(b"\0").decode()] = data[off:off + size]
    return lumps


def _thing_block(idx: int, type_id: int, x: float, y: float,
                 angle: int, tid: int = 0) -> str:
    return f"""thing // {idx}
{{
x = {x:.3f};
y = {y:.3f};
angle = {angle};
type = {type_id};
skill1 = true;
skill2 = true;
skill3 = true;
skill4 = true;
skill5 = true;
skill6 = true;
skill7 = true;
skill8 = true;
single = true;
dm = true;
coop = true;
class1 = true;
class2 = true;
class3 = true;
class4 = true;
class5 = true;
class6 = true;
class7 = true;
class8 = true;
}}
"""


def build_textmap(orig: bytes) -> bytes:
    text = orig.decode()
    out = [_thing_block(0, 1, -480.0, 288.0, 0)]  # player start (as stock)
    for i, (typ, x, y, ang, _ambush) in enumerate(MONSTERS, start=1):
        out.append(_thing_block(i, typ, float(x), float(y), ang))
    # insert monster things right after the player-start thing block
    player_end = text.index("vertex // 0")
    return (text[:player_end] + "\n".join(out) + "\n"
            + text[player_end:]).encode()


def _write_wad(path: str, lumps: list[tuple[str, bytes]]) -> None:
    header = struct.pack("<4sII", b"PWAD", len(lumps), 0)  # diroff patched later
    offset = 12
    body = b""
    directory = []
    for name, data in lumps:
        body += data
        directory.append((offset, len(data), name))
        offset += len(data)
    diroff = offset
    dir_bytes = b"".join(
        struct.pack("<II8s", off, size, name.encode().ljust(8, b"\0"))
        for off, size, name in directory)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"PWAD", len(lumps), diroff))
        f.write(body)
        f.write(dir_bytes)


def main() -> str:
    lumps = _read_lumps(SRC_WAD)
    textmap = build_textmap(lumps["TEXTMAP"])
    # keep geometry lumps; drop BEHAVIOR/SCRIPTS/DIALOGUE (ACS wave script)
    out = [("MAP01", b""), ("TEXTMAP", textmap), ("DECORATE", DECORATE.encode())]
    if "ZNODES" in lumps:
        out.append(("ZNODES", lumps["ZNODES"]))  # geometry unchanged
    out.append(("MAPINFO", MAPINFO.encode()))
    out.append(("ENDMAP", b""))
    _write_wad(OUT_WAD, out)
    print(f"wrote {OUT_WAD} ({os.path.getsize(OUT_WAD)} bytes, "
          f"{len(MONSTERS)} monsters, respawn on)")
    return OUT_WAD


if __name__ == "__main__":
    main()
