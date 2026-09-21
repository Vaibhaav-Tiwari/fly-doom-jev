# THIRD_PARTY.md

Third-party projects inspected or reused, with their licenses. No upstream
source code is vendored in this repository; pathway *concepts* were adapted
and re-implemented (see PROVENANCE.md for exactly what was taken from where).

## Data

- **MaleCNS v1.0 connectome** — Janelia FlyEM (HHMI Janelia), University of
  Cambridge Dept. of Zoology, MRC Laboratory of Molecular Biology, Google
  Research. License: **CC-BY 4.0**. Source: https://male-cns.janelia.org/download/
  Files used (sha256 pinned in `data/manifests/malecns-v1.0.manifest.json`):
  `body-annotations-male-cns-v1.0-minconf-0.5.feather`,
  `connectome-weights-male-cns-v1.0-minconf-0.5.feather`,
  `body-neurotransmitters-male-cns-v1.0.feather`.

## Software

- **ViZDoom** — Farama Foundation, https://github.com/Farama-Foundation/ViZDoom
  License: **MIT**. Used as a library (pip package), unmodified.

- **DOOMFLY** — nftechie, https://github.com/nftechie/doomfly
  License: **MIT**. Inspected (2026-09-21). Concepts adapted (re-implemented,
  no code copied): photoreceptor retinotopic mapping via MaleCNS ommatidia
  hex columns (`assignedOlHex1/2`), bilinear luminance sampling with sRGB
  linearization, Naka-Rushton-style photoreceptor drive, tonic lamina bias,
  synaptic delay queue, event-driven compiled LIF kernel, typed
  descending-neuron readouts (DNa02 turning, DNp09/MDN forward/backward).

- **FlyBrain** — Jhongdlp, https://github.com/Jhongdlp/FlyBrain
  License: **MIT**. Inspected (2026-09-21). Concepts adapted: column
  inheritance for photoreceptors lacking hex coordinates (inherit modal column
  of strongest lamina targets), reading leg motor neurons in the ventral nerve
  cord, DNg100 (BDN2) as forward-walking descending neuron
  (Sapkal et al. 2024), DNa02 wiring asymmetry for turning
  (Rayshubskiy et al. 2020).

- **fly-connectome-template** — cobanov,
  https://github.com/cobanov/fly-connectome-template
  License: **Cobanov Template Attribution License 1.0**
  (SPDX `LicenseRef-Cobanov-Template-Attribution-1.0`). Inspected (2026-09-21).
  No code or assets reused. Note: this license requires a readable web-UI
  attribution ("Built with fly-connectome-template by Mert Cobanov", with
  links) if substantial portions are ever incorporated. If the frontend later
  reuses any of it, the attribution obligation applies to the web UI.

## Runtime dependencies

See `pyproject.toml`; all are permissively licensed (BSD/Apache/MIT-class)
Python packages installed from PyPI.
