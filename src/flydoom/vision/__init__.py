"""Visual pathway.

PhotoreceptorPathway: full-mode retinotopic input — each mapped photoreceptor
samples the RGB frame bilinearly at its UV coordinate (from MaleCNS ommatidia
columns; see PROVENANCE.md), with sRGB linearization and a Naka-Rushton-style
drive. Lamina interneurons get a tonic bias (spiking proxy for graded cells).

RetinaEncoder: the v1 tiled-mosaic encoder, kept for fixture graphs that have
no photoreceptor annotations.
"""

from .photoreceptors import PhotoreceptorPathway
from .retina import RetinaEncoder

__all__ = ["PhotoreceptorPathway", "RetinaEncoder"]
