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
- **Tonic baseline** (`neural.tonic`, chosen dynamics, not biology): a
  calibrated per-neuron tonic current holds each coarse population near a
  modest baseline rate (target 2 Hz; 0.5 Hz for motor readout populations;
  retina excluded), because a network at rest attenuates cascades within a
  few hops. Real fly neurons do have spontaneous rates, but our values are
  calibrated for signal propagation, not measured. The motor decoder reads
  evoked activity above the calibrated baseline. Targets, achieved baselines,
  and current stats are recorded in every recording header (`neural.tonic`).
- **Jev**: an artificial probabilistic decision module. In `live` mode this is
  TypeSafe's **System One** structured-probability API (`POST /v1/systemone`,
  model `jev-latest`): one request carries the typed environment state plus
  the 10-question bank as noul/score/choice questions and returns calibrated
  probabilities — matching the spec's "structured probabilistic decision
  model, NOT a generic chat model". The default remains a **deterministic
  mock** (fixed function of the typed environment state) so the system is
  reproducible without API keys. The mock is not an LLM and proves nothing
  about LLM-driven behavior.

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
- **Reward-modulated plasticity** (`plasticity`, doomfly v6 reference): shaped
  reward events (kill +1, health delta ×0.02, death −1 — the raw ViZDoom
  reward in these scenarios is living-reward only, and damage dealt is not
  directly observable, so the kill event is its proxy) pulse the connectome's
  dopaminergic neurons (PAM types for positive, PPL1 for negative) and gate a
  three-factor Hebbian update on KC→MBON synapses with eligibility traces.
  Weights are clamped to [0, 3×w0] and persist across episodes within a run
  (reset on POST /new). This is a **chosen learning rule**, not a validated
  model of mushroom-body plasticity: PAM/PPL1 identities are real MaleCNS
  annotations, but the rule, gains, and taus are engineering picks.
- **Aim gate (true geometry)**: attack fires only when an enemy is visible,
  inside a 14° cone of screen center, and within range — computed from
  ViZDoom's label-derived enemy fields (`motor/reflexes.aim_in_reticle`), not
  from a neural proxy. The gate only PERMITS attack; the neural attack channel
  still decides (raw channel rates stay in telemetry).
- **Unstuck reflex** (E1M1 profile): commanding movement with <6 map units of
  XY displacement over 2.5 s of game time forces a 1 s turn (alternating
  direction). Engineering reflex against wall-parking, not biology.
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
- The mock Jev is a heuristic. The live Jev path (System One) is real but
  shallow: one structured-probability call per decision, no retries/caching
  yet.
- Behavior: the controller tracks and kills some enemies, then dies
  (~1–6 kills per episode on defend_the_center). This is reported, not tuned
  away.
- The fixture environment and fixture/reduced graphs exist for tests/dev only
  and are always labeled in logs, recording headers, and the UI.
