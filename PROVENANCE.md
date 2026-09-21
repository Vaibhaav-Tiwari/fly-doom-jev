# PROVENANCE.md

Where each non-obvious part of the backend comes from. "Adapted" = the idea
was taken from the cited source and re-implemented in our own code; no upstream
files were copied.

## Visual pathway (`src/flydoom/vision/photoreceptors.py`)

- Retinotopic mapping from MaleCNS ommatidia columns (`assignedOlHex1/2` in
  the body-annotations feather) to viewport UV coordinates, per eye side
  (left eye mirrored into the left viewport half, right into the right):
  **adapted from DOOMFLY** (`doom_learning_v6/visual.py`, MIT).
- Photoreceptors lacking a direct hex annotation inherit the modal column of
  their strongest annotated lamina targets: **adapted from FlyBrain**
  (`fly/ojo.py`, MIT) — the same "let the wiring say what the annotation
  doesn't" trick both projects use.
- Bilinear sampling of the RGB frame at receptor UVs, sRGB linearization,
  Naka-Rushton-style drive `g * L / (c + L)`, tonic bias current on lamina
  interneurons (spiking proxy for graded cells): **adapted from DOOMFLY**
  (`doom/game.py:retinal_samples`, `doom/native.py`, MIT).
- R8p/R8y spectral proxy (blue/green channel) vs broadband R1–R6 luminance:
  **adapted from DOOMFLY**; the underlying R8→aMe12 excitation biology is
  attributed there to Xiao et al., Nature 2023 (doi:10.1038/s41586-023-06681-6).
  Ours is a display proxy, not calibrated spectral sensitivity (see SCIENCE.md).

## Neural engine (`src/flydoom/neural/kernel.c`, `engine_native.py`)

- Compiled-kernel architecture (C source compiled at setup with the system
  clang, cached by source sha256, CSR arrays passed by pointer):
  architecture **adapted from DOOMFLY v6** (`doom_learning_v6/brain.py`, MIT).
  Our kernel source is written from scratch and differs in dynamics
  (leak/refractory schedule, signed neurotransmitter weights).
- Synaptic transmission delay via a ring buffer: **adapted from DOOMFLY**
  (`doom/engine.py`, MIT).
- Signed synaptic weights from per-neuron neurotransmitter predictions
  (acetylcholine +, GABA −, glutamate −; others excitatory-by-default):
  our implementation over `body-neurotransmitters-male-cns-v1.0.feather`;
  the sign convention is standard Drosophila physiology, the magnitude
  normalization is an engineering choice (SCIENCE.md).

## Motor decoder (`src/flydoom/motor/decoder.py`)

- Typed descending-neuron readouts: turn = DNa02(R) − DNa02(L)
  (Rayshubskiy et al. 2020), forward = DNp09/DNg100 vs MDN (backward):
  mapping concept **adapted from DOOMFLY** (`doom/engine.py:NeuralControls`,
  MIT) and **FlyBrain** (`fly/patas.py`, MIT), with literature references as
  cited by those projects (Sapkal et al. 2024 for DNg100/BDN2). Our decoder
  keeps these mappings configurable and labeled as joystick-gain engineering,
  not biological claims.
- Population-bank decoder (v1) remains as a fallback for graphs without type
  annotations.

## Environment

- ViZDoom `basic` / `defend_the_center` scenarios ship with the ViZDoom
  package (MIT); scenario files are loaded from the installed package, not
  vendored.

## Not reused

- fly-connectome-template (Cobanov Template Attribution License 1.0):
  inspected for reference only; nothing incorporated.
