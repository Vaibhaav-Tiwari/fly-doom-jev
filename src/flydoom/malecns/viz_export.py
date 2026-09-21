"""Export shared browser-visualization assets for the full connectome.

The 3D brain view needs per-neuron positions once (not per recording). We
export a compact binary bundle keyed by NEURON INDEX (the row order of the
graph arrays, which is also what step records reference):

  positions.f32   n*3 little-endian float32 xyz (soma, voxel units @8nm;
                  NaN when the neuron has no annotated soma position)
  population.i16  n int16 codes into meta.populations
  flags.u8        n bitmask: 1=photoreceptor 2=lamina 4=descending_neuron
                  8=vnc_motor 16=has_position
  ids.i64         n int64 MaleCNS bodyIds
  meta.json       counts, population names, flag legend, provenance

Positions are real MaleCNS soma annotations (141,781 of 211,577 neurons have
one); neurons without are flagged so the frontend can drop or ghost them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from flydoom.malecns.full import COARSE_POPULATIONS, FullConnectome

log = logging.getLogger(__name__)

FLAG_RETINA = 1
FLAG_LAMINA = 2
FLAG_DESCENDING = 4
FLAG_VNC_MOTOR = 8
FLAG_HAS_POSITION = 16


def export_connectome_assets(conn: FullConnectome, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = conn.n_neurons
    positions = np.asarray(conn.positions, dtype=np.float32)
    has_pos = ~np.isnan(positions[:, 0])

    flags = np.zeros(n, dtype=np.uint8)
    flags[np.asarray(conn.retina)] |= FLAG_RETINA
    flags[np.asarray(conn.lamina)] |= FLAG_LAMINA
    flags[conn.population_indices("descending_neuron")] |= FLAG_DESCENDING
    flags[conn.population_indices("vnc_motor")] |= FLAG_VNC_MOTOR
    flags[has_pos] |= FLAG_HAS_POSITION

    (out / "positions.f32").write_bytes(
        np.ascontiguousarray(positions, dtype="<f4").tobytes())
    (out / "population.i16").write_bytes(
        np.ascontiguousarray(conn.population_index, dtype="<i2").tobytes())
    (out / "flags.u8").write_bytes(flags.tobytes())
    (out / "ids.i64").write_bytes(
        np.ascontiguousarray(conn.ids, dtype="<i8").tobytes())

    meta = {
        "n_neurons": int(n),
        "arrays": {
            "positions.f32": {"dtype": "float32-le", "shape": [n, 3],
                              "nan_means": "no annotated soma position"},
            "population.i16": {"dtype": "int16-le", "shape": [n]},
            "flags.u8": {"dtype": "uint8", "shape": [n]},
            "ids.i64": {"dtype": "int64-le", "shape": [n]},
        },
        "populations": list(COARSE_POPULATIONS),
        "flags_legend": {"1": "photoreceptor", "2": "lamina",
                         "4": "descending_neuron", "8": "vnc_motor",
                         "16": "has_position"},
        "coordinate_space": "malecns EM voxels (8nm units), soma positions",
        "with_position": int(has_pos.sum()),
        "provenance": conn.provenance,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("connectome viz assets -> %s (%d neurons, %d with positions)",
             out, n, int(has_pos.sum()))
    return out
