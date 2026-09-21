"""MaleCNS data access (v1: reduced connectome only).

v1 loads a REDUCED but REAL subset of MaleCNS v1.0 when the raw feather files
are present under ``data/raw/malecns-v1.0`` (see README). When they are absent,
a deterministic, clearly-labeled fixture graph is built through the same
interface so tests and the demo still run. The fixture graph is NEVER presented
as real connectome data: ``ReducedConnectome.provenance['source']`` says which.

The full download/checksum pipeline is deferred beyond v1; the manifest in
data/manifests/ already records the real sha256 checksums for verification.
"""

from .graph import ReducedConnectome, load_connectome

__all__ = ["ReducedConnectome", "load_connectome"]
