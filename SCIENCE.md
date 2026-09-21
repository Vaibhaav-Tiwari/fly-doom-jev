# SCIENCE.md — what v1 is, and is not

## DATA

- **MaleCNS v1.0** is a real biological connectome dataset (Janelia FlyEM et
  al., CC-BY 4.0). v1 uses an explicit **REDUCED subset**: population-capped
  samples of `visual_projection` (800), `ol_intrinsic` (1500), central-complex
  `CX` (800), and all `descending_neuron` (1314), with the induced synaptic
  subgraph (~122k of ~152M edges). This is labeled REDUCED everywhere (logs,
  recording header, dashboard banner).
- Edge weights are synapse counts, **normalized per postsynaptic neuron**
  (each neuron's inbound weights sum to ≤ 1). That normalization is an
  engineering choice, not a biological one.

## MODEL

- **Neural dynamics**: leaky integrate-and-fire (LIF), vectorized over the
  sparse connectome. v1 treats all synapses as **excitatory**; MaleCNS
  neurotransmitter predictions are not wired in. Rates reported downstream are
  exponential moving averages of spike rate.
- **Jev**: an artificial probabilistic decision module. In v1 the default is a
  **deterministic mock** (fixed function of the typed environment state), used
  so the system is reproducible and runnable without API keys. The mock is not
  an LLM and proves nothing about LLM-driven behavior.

## ENGINEERING ASSUMPTIONS (explicitly not biology)

- **Retinal pathway**: frames are mean-pooled to a 16×12 mosaic and tiled
  round-robin onto `visual_projection` neurons. No retinotopic mapping to
  real neurons is claimed.
- **Jev→MaleCNS bridge**: question probabilities inject current into slices of
  upstream/intermediate populations (`visual_projection`, `ol_intrinsic`,
  `cx_intrinsic`). These mappings are **engineering hypotheses**. Injection
  into the motor/readout population (`descending_neuron`) is structurally
  forbidden (`MotorInjectionError`).
- **Motor decoder**: contiguous disjoint banks of descending neurons per
  action; score = bank mean firing rate, mean-centered against the whole motor
  population, softmaxed. Bank assignment is arbitrary (by index order).
- **Threat level** is a hand-crafted heuristic of enemy proximity/centering.
- **Stale-decision semantics**: Jev runs at 2–5 Hz on a background thread; the
  last valid decision stays active until a new one arrives. The game/neural
  loop never blocks on Jev.

## OBSERVATIONS (what you can see in a recording)

Per controller step: typed environment state, current Jev probabilities +
latency, population firing-rate summaries, sampled-neuron rates, motor action
scores, selected action, reward. Timelines are synchronized by a shared `t_ms`.

## INFERENCES — none claimed

- Correlation between Jev decisions, population activity, and actions in a
  recording is **temporal association, not causation**.
- Useful DOOM behavior would not demonstrate biological equivalence, nor that
  a fly understands DOOM. v1 makes no behavioral-performance claims.
- The causal pathway constraint (Jev → MaleCNS modulation → dynamics → motor
  decoder → action) is enforced in code; Jev never selects the final action
  directly and never touches motor neurons.

## LIMITATIONS (v1)

- Reduced connectome (~2% of neurons), excitatory-only synapses, no
  plasticity, no neuromodulation, LIF parameters not fitted to fly physiology.
- Mock Jev is a heuristic; live API path is an untested stub.
- Fixture environment (tests only) is a scripted 2D toy — passing integration
  tests does not validate the real ViZDoom loop. The real loop was exercised
  manually via `make run-demo` (headless ViZDoom basic scenario).
- Locomotion in ViZDoom is controlled directly by button commands; the fly's
  actual motor periphery is not modeled.
