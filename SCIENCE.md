# SCIENCE.md — what this system is, and is not

## DATA

- **MaleCNS v1.0** is a real biological connectome dataset (Janelia FlyEM et
  al., CC-BY 4.0). The default mode uses the **full neuron-level graph**:
  211,577 annotated neurons and 26,028,386 edges. The raw connectome-weights
  table is segment-to-segment (151.9M rows); edges touching unannotated
  fragments are dropped, which is the standard neuron-level projection. Raw
  files are sha256-pinned in `data/manifests/malecns-v1.0.manifest.json`
  (`make verify-data`).
- **Synapse signs** come from the dataset's consensus neurotransmitter
  predictions: acetylcholine excitatory (+), GABA and glutamate inhibitory (−)
  (ionotropic convention in insects). All other predictions default to + —
  an engineering default. Edge magnitudes are raw synapse counts times one
  global synaptic efficacy (`engine.syn_gain`, default 0.3). No per-synapse
  physiology is modeled.

## MODEL

- **Neural dynamics**: leaky integrate-and-fire over the full graph, computed
  by a compiled C kernel (`engine/native/lif_kernel.c`): exponential synaptic
  drive decay (tau 5 ms), membrane tau 20 ms, 2 ms transmission delay,
  2 ms refractory, dt 1 ms. Photoreceptors and lamina interneurons are spiking
  proxies for what are in reality graded-potential cells — a declared
  approximation.
- **Jev**: an artificial probabilistic decision module. The default is a
  **deterministic mock** (fixed function of the typed environment state) so
  the system is reproducible without API keys. The mock is not an LLM and
  proves nothing about LLM-driven behavior.

## ENGINEERING ASSUMPTIONS (explicitly not biology)

- **Visual pathway**: each photoreceptor samples the RGB frame at a viewport
  coordinate derived from its ommatidium column (`assignedOlHex1/2`; receptors
  inherit the modal column of their strongest lamina targets — both DOOMFLY
  and FlyBrain use this trick, see PROVENANCE.md). Left/right eyes are
  mirrored into left/right viewport halves; the absolute orientation of the
  eye relative to the DOOM world is chosen by us. R8p/R8y sample blue/green
  channels as a display proxy for spectral sensitivity — not calibrated
  phototransduction. Lamina neurons get a tonic bias current.
- **Jev→MaleCNS bridge**: question probabilities inject current into slices of
  upstream/intermediate populations (`visual_projection`, `ol_intrinsic`,
  `cx_intrinsic`). These are **engineering hypotheses**. Injection into
  `descending_neuron` (the motor/readout population) is structurally forbidden
  (`MotorInjectionError`, tested).
- **Motor decoder**: reads known descending-neuron types — DNa02 (turning,
  Rayshubskiy et al. 2020), DNp09 forward / MDN backward, DNg100 (BDN2,
  Sapkal et al. 2024) — via configurable gains. The DN identities have
  literature support; the channel gains, thresholds, and the DNpe017 attack
  readout are **joystick mappings**, not biology.
- **Threat level** is a hand-crafted heuristic of enemy proximity/centering.
- **Stale-decision semantics**: Jev runs at ~3 Hz on a background thread; the
  last valid decision stays active until a new one arrives. The game/neural
  loop never blocks on Jev.

## OBSERVATIONS (what recordings contain)

Per controller step: typed environment state, Jev probabilities + latency,
population firing-rate summaries + sampled neurons, motor channels/readout
rates/action scores, selected action, reward, controller latency, JPEG frame
reference. Episode end: kills, reward, survival, action counts, decision
counts. See `docs/RECORDING_FORMAT.md`.

## INFERENCES — none claimed

- Temporal association between Jev decisions, neural activity, and actions is
  **not causation**; no interventional analysis is performed.
- Scoring kills in DOOM does not demonstrate biological equivalence, fly
  "understanding" of DOOM, or learning. No plasticity exists in the system.
- The causal pathway constraint (Jev → MaleCNS modulation → dynamics → typed
  DN readouts → action) is enforced in code: Jev never selects the final
  action directly and cannot inject into motor/readout neurons.

## LIMITATIONS

- LIF parameters are not fitted to fly physiology; synaptic weights are raw
  counts times one global gain; neuromodulation, gap junctions, compartments,
  and graded transmission are absent.
- The decoder reads a handful of DN types; real locomotion emerges from the
  VNC motor periphery, which is simulated but not used for control.
- The mock Jev is a heuristic. The live Jev client is a stub.
- Behavior: the controller tracks and kills some enemies, then dies
  (~1–6 kills per episode on defend_the_center). This is reported, not tuned
  away.
- The fixture environment and fixture/reduced graphs exist for tests/dev only
  and are always labeled in logs, recording headers, and the UI.
